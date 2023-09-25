import requests
import signal
import sys
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import subprocess
from typing import List
import ray
import typer
from rich import print as rp
from rich.progress import Progress, SpinnerColumn, TextColumn

from local_aviary.run import run_single_model

__all__ = ["run"]

app = typer.Typer()

model_type = typer.Option(
    default=..., help="The model to use. You can specify multiple models."
)

true_or_false_type = typer.Option(default=False, is_flag=True)

def _exec_cmd(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )

    def terminate_subprocess(signum, _frame):
        print(f"[{signum}] terminate subprocess")
        process.terminate()
        subprocess.run("ray stop", shell=True)

    signal.signal(signal.SIGTERM, terminate_subprocess)
    signal.signal(signal.SIGINT, terminate_subprocess)

    while True:
        msg = process.stdout.readline()
        if msg == "" and process.poll() is not None:
            break

        sys.stdout.write(msg)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitcode = process.returncode

    if exitcode == 0:
        return output
    raise subprocess.CalledProcessError(exitcode, command)

def _get_text(result: dict) -> str:
    if "text" in result["choices"][0]:
        return result["choices"][0]["text"]
    elif "message" in result["choices"][0]:
        return result["choices"][0]["message"]["content"]
    elif "delta" in result["choices"][0]:
        return result["choices"][0]["delta"].get("content", "")


def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def _get_yes_or_no_input(prompt) -> bool:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "yes" or user_input == "y":
            return True
        elif user_input == "no" or user_input == "n" or user_input == "":
            return False
        else:
            print("Invalid input. Please enter 'yes / y' or 'no / n'.")


# TODO: update local run command
@app.command()
def run(
    model: str,
    blocking: bool = True,
    start_ray_cluster: bool = False
):
    """Start a model in Aviary.

    Args:
        model: Model to run.
        blocking: Whether to block the CLI until the application is ready.
        start_ray_cluster: Whether to start a basic ray cluster.
    """
    msg = (
        "Running `local_aviary run` with `start_ray_cluster' as True "
        "will start a new basic ray cluster with current machine as head "
        f"to serve the specified ({model}).\n"
        "Do you want to continue? [y/N]\n"
    )
    if start_ray_cluster:
        if not _get_yes_or_no_input(msg):
            return
        try:
            ray_command = "ray start --head --disable-usage-stats"
            _exec_cmd(ray_command)
        except Exception as e:
            print(f"Could not start the ray cluster: {str(e)}")
            return
    run_single_model(model=model, blocking=blocking)


@app.command()
def shutdown():
    """Shutdown Aviary."""
    subprocess.run("ray stop", shell=True)


if __name__ == "__main__":
    app()
