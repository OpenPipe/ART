from typing import TYPE_CHECKING
import sky
from sky.core import endpoints
import os
import semver

from art.skypilot.load_env_file import load_env_file
from .utils import is_task_created, wait_for_task_to_start

from .. import dev
from ..api import API

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class SkypilotAPI(API):
    _cluster_name: str

    def __init__(
        self,
        *,
        cluster_name: str = "art",
        resources: sky.Resources | None = None,
        art_version: str | None = None,
        env_path: str | None = None,
    ) -> None:
        self._cluster_name = cluster_name

        if resources is None:
            resources = sky.Resources(
                cloud=sky.clouds.RunPod(),
                # region="US",
                accelerators={"H100": 1},
                ports=["8080"],
            )

        # ensure ports 7999 and 8000 are open
        updated_ports = resources.ports
        if updated_ports is None:
            updated_ports = []
        updated_ports += ["7999", "8000"]
        resources = resources.copy(ports=updated_ports)

        # check if cluster already exists
        cluster_status = sky.status(cluster_names=[cluster_name])
        if (
            len(cluster_status) == 0
            or cluster_status[0]["status"] != sky.ClusterStatus.UP
        ):
            self._launch_cluster(cluster_name, resources, art_version, env_path)
        else:
            print(f"Cluster {cluster_name} exists, using it...")

        if is_task_created(cluster_name=cluster_name, task_name="art_server"):
            print("Art server task already running, using it...")
        else:
            art_server_task = sky.Task(name="art_server", run="uv run art")
            resources = sky.status(cluster_names=["art"])[0][
                "handle"
            ].launched_resources
            art_server_task.set_resources(resources)

            # run art server task
            sky.exec(
                task=art_server_task,
                cluster_name=cluster_name,
                detach_run=True,
            )
            print("Task launched, waiting for it to start...")
            wait_for_task_to_start(cluster_name=cluster_name, task_name="art_server")
            print("Art server task started")

        art_endpoint = endpoints(cluster=cluster_name, port=7999)[7999]
        base_url = f"http://{art_endpoint}"
        print(f"Using base_url: {base_url}")

        super().__init__(base_url=base_url)

    def _launch_cluster(
        self,
        cluster_name: str,
        resources: sky.Resources,
        art_version: str | None = None,
        env_path: str | None = None,
    ) -> None:
        print("Launching cluster...")

        task = sky.Task(
            name=cluster_name,
        )
        task.set_resources(resources)

        # TODO: TEST VERSIONED INSTALLATION ONCE WE'VE PUBLISHED A NEW VERSION OF ART WITH THE 'art' CLI SCRIPT

        # default to installing latest version of art
        art_installation_command = "uv pip install art"
        if art_version is not None:
            art_version_is_semver = False
            # check if art_version is valid semver
            if art_version is not None:
                try:
                    semver.Version.parse(art_version)
                    art_version_is_semver = True
                except Exception:
                    pass

            if art_version_is_semver:
                art_installation_command = f"uv pip install art=={art_version}"
            elif os.path.exists(art_version):
                # copy the contents of the art_path onto the new machine
                task.set_file_mounts(
                    {
                        "~/sky_workdir": art_version,
                    }
                )
                art_installation_command = ""
            else:
                raise ValueError(
                    f"Invalid art_version: {art_version}. Must be a semver or a path to a local directory."
                )

        setup_script = f"""
    curl -LsSf https://astral.sh/uv/install.sh | sh

    source $HOME/.local/bin/env

    {art_installation_command}
    uv add awscli
    uv sync
    """

        task.setup = setup_script

        if env_path is not None:
            envs = load_env_file(env_path)
            print(f"Loading envs from {env_path}")
            print(f"{len(envs)} environment variables found")
            task.update_envs(envs)

        print(task)

        try:
            sky.launch(task=task, cluster_name=cluster_name)
        except Exception as e:
            print(f"Error launching cluster: {e}")
            print()
            raise e

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the API for logging and/or training.

        Args:
            model: An art.Model instance.
        """

        print("Registering model with server")
        print(f"To view logs, run: 'uv run sky logs {self._cluster_name}'")
        await super().register(model)

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        response = await self._client.post(
            "/_prepare_backend_for_training",
            json={"model": model.model_dump(), "config": config},
            timeout=600,
        )
        response.raise_for_status()
        [_, api_key] = tuple(response.json())

        vllm_endpoint = endpoints(cluster=self._cluster_name, port=8000)[8000]
        base_url = f"http://{vllm_endpoint}/v1"

        return [base_url, api_key]

    def down(self) -> None:
        sky.down(cluster_name=self._cluster_name)
