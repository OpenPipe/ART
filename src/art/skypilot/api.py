from typing import TYPE_CHECKING
import sky
import os
import semver
from dotenv import dotenv_values

from .utils import (
    is_task_created,
    to_thread_typed,
    wait_for_art_server_to_start,
    get_art_server_base_url,
    get_vllm_base_url,
)

from .. import dev
from ..api import API

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class SkyPilotAPI(API):
    _cluster_name: str

    @classmethod
    async def initialize_cluster(
        cls,
        *,
        cluster_name: str = "art",
        gpu: str | None = None,
        resources: sky.Resources | None = None,
        art_version: str | None = None,
        env_path: str | None = None,
    ) -> None:
        self = cls.__new__(cls)
        self._cluster_name = cluster_name

        if gpu is None and resources is None:
            raise ValueError("Either gpu or resources must be provided")

        if resources is None:
            resources = sky.Resources(
                cloud=sky.clouds.RunPod(),
                accelerators={"H100": 1},
                ports=[],
            )

        if gpu is not None:
            resources = resources.copy(accelerators={gpu: 1})

        # ensure ports 7999 and 8000 are open
        updated_ports = resources.ports
        if updated_ports is None:
            updated_ports = []
        updated_ports += ["7999", "8000"]
        resources = resources.copy(ports=updated_ports)

        # check if cluster already exists
        cluster_status = await to_thread_typed(
            lambda: sky.status(cluster_names=[self._cluster_name])
        )
        if (
            len(cluster_status) == 0
            or cluster_status[0]["status"] != sky.ClusterStatus.UP
        ):
            await self._launch_cluster(resources, art_version, env_path)
        else:
            print(f"Cluster {self._cluster_name} exists, using it...")

        if await is_task_created(
            cluster_name=self._cluster_name, task_name="art_server"
        ):
            print("Art server task already running, using it...")
        else:
            art_server_task = sky.Task(name="art_server", run="uv run art")
            resources = await to_thread_typed(
                lambda: sky.status(cluster_names=[self._cluster_name])[0][
                    "handle"
                ].launched_resources
            )
            art_server_task.set_resources(resources)

            # run art server task
            await to_thread_typed(
                lambda: sky.exec(
                    task=art_server_task,
                    cluster_name=self._cluster_name,
                    detach_run=True,
                )
            )
            print("Task launched, waiting for it to start...")
            await wait_for_art_server_to_start(cluster_name=self._cluster_name)
            print("Art server task started")

        base_url = await get_art_server_base_url(self._cluster_name)
        print(f"Using base_url: {base_url}")

        # Manually call the real __init__ now that base_url is ready
        super(SkyPilotAPI, self).__init__(base_url=base_url)

        return self

    async def _launch_cluster(
        self,
        resources: sky.Resources,
        art_version: str | None = None,
        env_path: str | None = None,
    ) -> None:
        print("Launching cluster...")

        task = sky.Task(
            name=self._cluster_name,
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

    git config --global --add safe.directory /root/sky_workdir

    {art_installation_command}
    uv sync
    """

        task.setup = setup_script

        if env_path is not None:
            envs = dotenv_values(env_path)
            print(f"Loading envs from {env_path}")
            print(f"{len(envs)} environment variables found")
            task.update_envs(envs)

        print(task)

        try:
            await to_thread_typed(
                lambda: sky.launch(task=task, cluster_name=self._cluster_name)
            )
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
            timeout=1200,
        )
        response.raise_for_status()
        [_, api_key] = tuple(response.json())

        vllm_base_url = await get_vllm_base_url(self._cluster_name)

        return [vllm_base_url, api_key]

    async def down(self) -> None:
        await to_thread_typed(lambda: sky.down(cluster_name=self._cluster_name))
