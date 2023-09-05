import asyncio
import gc
import os
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type

import ray
import ray.exceptions
import ray.util
from ray.air import ScalingConfig
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer

from aviary.backend.llm.continuous.error_handling import InputTooLong
from aviary.backend.llm.continuous.policy import QuotaBasedTaskSelectionPolicy
from aviary.backend.llm.continuous.scheduler import InferenceScheduler
from aviary.backend.llm.continuous.tokenizer import (
    CachingTokenizer,
    TransformersTokenizer,
)
from aviary.backend.llm.continuous.tokenstream import FinishReason
from aviary.backend.llm.continuous.types import InferenceTask, Request, TGIParams
from aviary.backend.llm.engine.tgi import TextGenerationInferenceEngine, AviaryTGIInferenceWorker
from aviary.backend.llm.utils import (
    _init_torch_distributed_env_vars_only,
    init_torch_dist_process_group_async,
    initialize_node,
    merge_dicts,
)
from aviary.backend.logger import get_logger
from aviary.backend.server.models import (
    AviaryModelResponse,
    TextGenerationInferenceEngineConfig,
)
from aviary.backend.server.utils import QueuePriority
from aviary.common.models import Prompt
from aviary.conf import ENV_VARS_TO_PROPAGATE

try:
    from aviary.backend.llm.continuous.tgi.tgi_worker import (
        TGIInferenceWorker,
        TGIInferenceWorkerGroup,
    )
except ImportError as e:
    TGIInferenceWorkerGroup = e

    class TGIInferenceWorker:
        pass


logger = get_logger(__name__)
TOTAL_BATCH_TOKENS_MULTIPLIER = 0.99


class LocalTextGenerationInferenceEngine(TextGenerationInferenceEngine):
    _prediction_worker_cls: Type[AviaryTGIInferenceWorker] = AviaryTGIInferenceWorker

    def __init__(
        self,
        engine_config: Optional[TextGenerationInferenceEngineConfig],
    ) -> None:
        super().__init__(engine_config=engine_config)

    async def _start_prediction_workers(
        self, scaling_config: ScalingConfig, remote_prediction_worker_cls: type
    ):
        # Create the prediction workers.
        logger.info("Creating prediction workers...")
        worker_group = [
            remote_prediction_worker_cls.remote(
                self.engine_config, scaling_config.num_workers
            )
            for i in range(scaling_config.num_workers)
        ]

        logger.info("Initializing torch_dist process group on workers...")
        # Initialize torch distributed process group for the workers.
        local_ranks = await self._initialize_torch_dist_process_group(
            worker_group,
            backend="nccl" if scaling_config.use_gpu else "gloo",
        )

        # Initialize model on each worker.
        logger.info("Initializing model on workers...")
        await asyncio.gather(
            *[
                worker.init_model.remote(
                    local_rank,
                    num_cpus_per_worker=scaling_config.num_cpus_per_worker,
                    num_gpus_per_worker=scaling_config.num_gpus_per_worker,
                )
                for worker, local_rank in zip(worker_group, local_ranks)
            ]
        )

        logger.info("Warming up model on workers...")

        # TODO: issue with flash-attn v2
        can_infer_max_batch_total_tokens = (
            await asyncio.gather(
                worker_group[0].can_infer_max_batch_total_tokens.remote()
            )
        )[0]
        if can_infer_max_batch_total_tokens:
            max_batch_total_tokens = None
        else:
            max_batch_total_tokens = self.task_selection_policy.max_batch_total_tokens
            if not max_batch_total_tokens:
                raise ValueError(
                    f"Model {self.engine_config.model_id} cannot automatically infer max_batch_total_tokens. "
                    "Make sure to set engine_config.scheduler.policy.max_batch_total_tokens in the model "
                    "configuration yaml."
                )

        max_supported_total_tokens = await asyncio.gather(
            *[
                worker.warmup.remote(
                    max_batch_prefill_tokens=self.task_selection_policy.max_batch_prefill_tokens,
                    max_input_length=self.task_selection_policy.max_input_length,
                    max_batch_total_tokens=max_batch_total_tokens,
                )
                for worker in worker_group
            ]
        )

        max_supported_total_tokens = min(max_supported_total_tokens)

        if can_infer_max_batch_total_tokens and max_supported_total_tokens:
            self.task_selection_policy.max_batch_total_tokens = int(
                max_supported_total_tokens * TOTAL_BATCH_TOKENS_MULTIPLIER
            )

            # Warmup again with max_supported_total_tokens to ensure constant environment across workers
            max_supported_total_tokens = await asyncio.gather(
                *[
                    worker.warmup.remote(
                        max_batch_prefill_tokens=self.task_selection_policy.max_batch_prefill_tokens,
                        max_input_length=self.task_selection_policy.max_input_length,
                        max_batch_total_tokens=self.task_selection_policy.max_batch_total_tokens,
                    )
                    for worker in worker_group
                ]
            )
            max_supported_total_tokens = min(max_supported_total_tokens)

        if max_supported_total_tokens:
            self.task_selection_policy.max_batch_total_tokens = (
                max_supported_total_tokens
            )

        assert worker_group
        return worker_group

    async def _create_worker_group(
        self,
        scaling_config: ScalingConfig,
        pg_timeout_s: float = 600,
    ) -> List[ray.ObjectRef]:
        assert self.engine_config

        self.task_queue = asyncio.Queue()
        self.task_selection_policy = QuotaBasedTaskSelectionPolicy(
            **self.engine_config.scheduler.policy.dict(exclude={"type"})
        )

        llm_config = self.engine_config

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        scaling_options = dict(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        runtime_env = self._prepare_worker_runtime_env()
        remote_prediction_worker_cls = ray.remote(self._prediction_worker_cls).options(
            **scaling_options, runtime_env=runtime_env
        )
        # local aviary doesn't init node (download weights) from s3, hf_model_id points to the local model or hf url

        logger.info("Waiting for placement group to be ready...")
        # This will raise a timeout error.
        try:
            await asyncio.wait_for(self.pg.ready(), timeout=pg_timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Placement group {self.pg} did not become ready within {pg_timeout_s} seconds. "
                "This means that the cluster doesn't have the required resources to start the worker group. "
                "Please check the autoscaler logs for more information.\n"
                "This can also be caused by the model workers requiring resources that are not present in the "
                "cluster (eg. `accelerator_type_a10`). Either remove them from the model configuration yaml "
                "or add them to the cluster."
            ) from e

        # Download the tokenizer
        _ = AutoTokenizer.from_pretrained(llm_config.actual_hf_model_id)

        worker_group = await self._start_prediction_workers(
            scaling_config=scaling_config,
            remote_prediction_worker_cls=remote_prediction_worker_cls,
        )

        self.tokenizer = CachingTokenizer(
            TransformersTokenizer(ray.get(worker_group[0].get_tokenizer.remote())),
            capacity=1024,
        )
        self.inference_task_cls = ray.get(
            worker_group[0].get_inference_task_cls.remote()
        )

        self.scheduler = InferenceScheduler(
            inference_worker=TGIInferenceWorkerGroup(worker_group=worker_group),
            task_selection_policy=self.task_selection_policy,
            task_queue=self.task_queue,
        )

        return worker_group

    def process_request(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        sampling_params: Dict[str, Any],
    ) -> InferenceTask:
        num_input_tokens = self.tokenizer.get_input_length(prompt)
        if num_input_tokens > self.task_selection_policy.max_input_length:
            logger.info("Task is over the max input length.")
            InputTooLong(
                num_input_tokens, self.task_selection_policy.max_input_length
            ).raise_exception() # TODO: add truncation option

        if "stopping_sequences" in sampling_params:
            sampling_params["stop_sequences"] = sampling_params.pop(
                "stopping_sequences"
            )
        max_new_tokens = int(
            min(
                max_new_tokens or float("inf"),
                self.task_selection_policy.max_total_tokens - num_input_tokens,
            )
        )
        task = self.inference_task_cls(
            Request(
                inputs=prompt,
                input_tokens=num_input_tokens,
                truncate=self.task_selection_policy.max_input_length,
                max_new_tokens=max_new_tokens,
                params=TGIParams(**sampling_params),
            )
        )
        self.scheduler.add_task(task)
        return task

