import asyncio
import os
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, cast

import semver
import sky
import sky.clouds
from dotenv import dotenv_values

from .. import dev
from ..backend import Backend
from .utils import (
    get_art_server_base_url,
    get_task_job_id,
    get_vllm_base_url,
    is_task_created,
    to_thread_typed,
    wait_for_art_server_to_start,
)

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class SkyPilotBackend(Backend):
    _cluster_name: str
    _envs: dict[str, str]
    _art_server_job_id: int | None

    @classmethod
    async def initialize_cluster(
        cls,
        *,
        cluster_name: str = "art",
        gpu: str | None = None,
        resources: sky.Resources | None = None,
        art_version: str | None = None,
        env_path: str | None = None,
        force_restart: bool = False,
        tail_logs: bool = True,
    ) -> "SkyPilotBackend":
        self = cls.__new__(cls)
        self._cluster_name = cluster_name
        self._envs = {}
        self._art_server_job_id = None

        if env_path is not None:
            self._envs = {
                k: v for k, v in dotenv_values(env_path).items() if v is not None
            }
            print(f"Loading envs from {env_path}")
            print(f"{len(self._envs)} environment variables found")

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
            lambda: sky.stream_and_get(sky.status(cluster_names=[self._cluster_name]))
        )
        if (
            len(cluster_status) == 0
            or cluster_status[0]["status"] != sky.ClusterStatus.UP
        ):
            await self._launch_cluster(resources, art_version)

        else:
            print(f"Cluster {self._cluster_name} exists, using it...")

        art_server_running = await is_task_created(
            cluster_name=self._cluster_name, task_name="art_server"
        )

        if art_server_running and force_restart:
            print("force_restart=True; cancelling existing art_server task…")
            await to_thread_typed(
                lambda: sky.cancel(cluster_name=self._cluster_name, all=True)
            )
            # wait 5 seconds to ensure the server finishes winding down
            await asyncio.sleep(5)
            art_server_running = False

        if art_server_running:
            self._art_server_job_id = await get_task_job_id(
                cluster_name=self._cluster_name, task_name="art_server"
            )
            print("Art server task already running, using it…")
        else:
            art_server_task = sky.Task(name="art_server", run="uv run art")

            clusters = await to_thread_typed(
                lambda: sky.stream_and_get(
                    sky.status(cluster_names=[self._cluster_name])
                )
            )
            resources = clusters[0]["handle"].launched_resources
            if resources is None:
                raise ValueError("Cluster handle has no launched resources")

            # For some reason, skypilot doesn't support the region and zone set
            resources = resources.copy(region=None, zone=None)

            # If a local path was provided for art_version, ensure it is mounted so the latest
            # code is synced to the remote cluster every time we (re)launch the art_server task.
            if art_version is not None and os.path.exists(art_version):
                art_server_task.workdir = art_version

            # print(clusters[0]["handle"].launched_resources)
            art_server_task.set_resources(cast(sky.Resources, resources))
            art_server_task.update_envs(self._envs)

            # run art server task
            job_id, _ = await to_thread_typed(
                lambda: sky.stream_and_get(
                    sky.exec(
                        task=art_server_task,
                        cluster_name=self._cluster_name,
                    )
                )
            )
            self._art_server_job_id = job_id

            print("Task launched, waiting for it to start...")
            await wait_for_art_server_to_start(cluster_name=self._cluster_name)
            print("Art server task started")

        base_url = await get_art_server_base_url(self._cluster_name)
        print(f"Using base_url: {base_url}")

        # Manually call the real __init__ now that base_url is ready
        super(cls, self).__init__(base_url=base_url)

        if self._art_server_job_id is not None and tail_logs:
            await asyncio.to_thread(
                sky.tail_logs,
                cluster_name=self._cluster_name,
                job_id=self._art_server_job_id,
                follow=True,
            )
            print(
                "Tailing logs. This process will only automatically exit after the backend is down. To exit before then, you'll have to manually close the process (e.g. ctrl+C)."
            )
        return self

    async def _launch_cluster(
        self,
        resources: sky.Resources,
        art_version: str | None = None,
    ) -> None:
        print("Launching cluster...")

        task = sky.Task(
            name=self._cluster_name,
        )
        task.set_resources(resources)
        task.update_envs(self._envs)

        # default to installing the version of art that is used by the client
        if art_version is None:
            try:
                art_version = version("openpipe-art")
            except PackageNotFoundError:
                raise ValueError(
                    "No version of openpipe-art installed in project. Please provide an art_version."
                )

        art_version_is_semver = False
        # check if art_version is valid semver
        try:
            semver.Version.parse(art_version)
            art_version_is_semver = True
        except Exception:
            pass

        if art_version_is_semver:
            art_installation_command = (
                f"uv pip install openpipe-art[backend]=={art_version}"
            )
        elif os.path.exists(art_version):
            # copy the contents of the art_path onto the new machine
            task.workdir = art_version
            art_installation_command = "uv sync --extra backend"
        else:
            raise ValueError(
                f"Invalid art_version: {art_version}. Must be a semver or a path to a local directory."
            )

        setup_script = f"""
    curl -LsSf https://astral.sh/uv/install.sh | sh

    source $HOME/.local/bin/env

    git config --global --add safe.directory /root/sky_workdir

    {art_installation_command}
    """
        task.setup = setup_script

        try:
            job_id, _ = await to_thread_typed(
                lambda: sky.stream_and_get(
                    sky.launch(
                        task=task, cluster_name=self._cluster_name, retry_until_up=True
                    )
                )
            )

            await to_thread_typed(
                lambda: sky.tail_logs(
                    cluster_name=self._cluster_name,
                    job_id=job_id,
                    follow=True,
                )
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
        Registers a model with the Backend for logging and/or training.

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

        return (vllm_base_url, api_key)

    async def down(self) -> None:
        await to_thread_typed(
            lambda: sky.stream_and_get(sky.down(cluster_name=self._cluster_name))
        )
