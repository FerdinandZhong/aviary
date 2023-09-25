import os
import sys
import time
from typing import Tuple, Union

import ray
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import ServeInstanceDetails

from aviary.backend.server.models import LLMApp, TextGenerationInferenceEngineConfig
from aviary.backend.server.utils import parse_args
from aviary.conf import ENV_VARS_TO_PROPAGATE
from local_aviary.local_model_app import TextGenerationInferenceLLMDeployment


def single_llm_model(app: LLMApp):
    print("Initializing LLM app", app.json(indent=2))
    user_config = app.dict()
    deployment_config = app.deployment_config.dict()
    deployment_config = deployment_config.copy()

    if isinstance(app.engine_config, TextGenerationInferenceEngineConfig):
        deployment_cls = TextGenerationInferenceLLMDeployment
        max_concurrent_queries = deployment_config.pop("max_concurrent_queries", None)
        if max_concurrent_queries is None:
            raise ValueError(
                "deployment_config.max_concurrent_queries must be specified for continuous batching models."
            )

    deployment_config.setdefault("ray_actor_options", {})
    deployment_config["ray_actor_options"].setdefault("runtime_env", {})
    deployment_config["ray_actor_options"]["runtime_env"].setdefault("env_vars", {})
    for env_var in ENV_VARS_TO_PROPAGATE:
        if env_var in os.environ:
            deployment_config["ray_actor_options"]["runtime_env"]["env_vars"][
                env_var
            ] = os.getenv(env_var)

    return deployment_cls.options(
        name=app.engine_config.model_id.replace("/", "--").replace(".", "_"),
        max_concurrent_queries=max_concurrent_queries,
        user_config=user_config,
        **deployment_config,
    ).bind()


def _parse_config_for_router(
    engine_config: TextGenerationInferenceEngineConfig,
) -> Tuple[str, str, str]:
    deployment_name = engine_config.model_id.replace("/", "--").replace(".", "_")
    deployment_route = f"/{deployment_name}"
    return deployment_name, deployment_route


def _applications_healthy(target_name):
    address = os.environ.get("RAY_AGENT_ADDRESS", "http://localhost:52365")
    serve_status = ServeInstanceDetails(
        **ServeSubmissionClient(address).get_serve_details()
    )
    if not target_name in serve_status.applications:
        return False
    else:
        target_app = serve_status.applications[target_name]
        if target_app.status == "DEPLOY_FAILED":
            raise RuntimeError(
                f"Application {target_name } failed to deploy. "
                "Check output above and Ray Dashboard/Ray logs for more details."
            )
        return target_app.status == "RUNNING"


def run_single_model(model: Union[LLMApp, str], blocking: bool = True):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A single LLMApp object or path to yaml file defining LLMApp

    Example:
       run_single_model("models/model1.yaml") # mix and match
    """
    model = parse_args(model)[0]
    if not model:
        raise RuntimeError("No enabled models were found.")

    ray._private.usage.usage_lib.record_library_usage("aviary")
    app = single_llm_model(model)

    (
        deployment_name,
        deployment_route,
    ) = _parse_config_for_router(model.engine_config)
    app_name = deployment_name

    serve.run(
        app,
        name=app_name,
        route_prefix=deployment_route,
        host="0.0.0.0",
        _blocking=False,
    )

    if blocking:
        while not _applications_healthy(app_name):
            time.sleep(1)


if __name__ == "__main__":
    run_single_model(*sys.argv[1:])
