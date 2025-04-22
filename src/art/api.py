import httpx
import json
from tqdm import auto as tqdm
from typing import AsyncIterator, TYPE_CHECKING
import sky
from sky.core import endpoints

from art.utils.skypilot import is_task_created, wait_for_task_to_start

from . import dev
from .trajectories import TrajectoryGroup
from .types import TrainConfig

if TYPE_CHECKING:
    from .model import Model, TrainableModel


class API:
    def __init__(
        self,
        *,
        base_url: str = "http://0.0.0.0:7999",
        cluster_name: str | None = None,
    ) -> None:
        if cluster_name:
            self._cluster_name = cluster_name

            # check if cluster already exists
            cluster_status = sky.status(cluster_names=[cluster_name])
            if (
                len(cluster_status) == 0
                or cluster_status[0]["status"] != sky.ClusterStatus.UP
            ):
                # self._launch_cluster(cluster_name)
                raise ValueError(f"Cluster {cluster_name} does not exist or is not up")
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
                wait_for_task_to_start(
                    cluster_name=cluster_name, task_name="art_server"
                )
                print("Art server task started")

            art_endpoint = endpoints(cluster=cluster_name, port=7999)[7999]
            # override base_url if one is provided
            base_url = f"http://{art_endpoint}"
            print(f"Using base_url: {base_url}")

        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url)

    # TODO: fix this
    # def _launch_cluster(self, cluster_name: str) -> None:
    #     print("Launching cluster...")

    #     task = sky.Task.from_yaml(
    #         "../../skypilot-config.yaml",
    #     )
    #     # load .env file into dict
    #     env_dict = {}
    #     with open("../../.env", "r") as f:
    #         for line in f:
    #             # skip empty lines and comments
    #             if line.strip() == "" or line.strip().startswith("#"):
    #                 continue
    #             key, value = line.strip().split("=")
    #             env_dict[key] = value
    #     task.update_envs(env_dict)

    #     try:
    #         sky.launch(task=task, cluster_name=cluster_name)
    #     except Exception as e:
    #         print(f"Error launching cluster: {e}")
    #         print()
    #         raise e

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the API for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        response = await self._client.post("/register", json=model.model_dump())
        response.raise_for_status()

    async def _get_step(self, model: "TrainableModel") -> int:
        response = await self._client.post("/_get_step", json=model.model_dump())
        response.raise_for_status()
        return response.json()

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        response = await self._client.post(
            "/_delete_checkpoints",
            json=model.model_dump(),
            params={"benchmark": benchmark, "benchmark_smoothing": benchmark_smoothing},
        )
        response.raise_for_status()

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
        [base_url, api_key] = tuple(response.json())

        # override base_url if one is provided
        vllm_endpoint = endpoints(cluster=self._cluster_name, port=8000)[8000]
        base_url = f"http://{vllm_endpoint}/v1"

        return [base_url, api_key]

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        response = await self._client.post(
            "/_log",
            json={
                "model": model.model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "split": split,
            },
        )
        response.raise_for_status()

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        async with self._client.stream(
            "POST",
            "/_train_model",
            json={
                "model": model.model_dump(),
                "trajectory_groups": [tg.model_dump() for tg in trajectory_groups],
                "config": config.model_dump(),
                "dev_config": dev_config,
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            pbar: tqdm.tqdm | None = None
            async for line in response.aiter_lines():
                result = json.loads(line)
                yield result
                num_steps = result.pop("num_steps")
                if pbar is None:
                    pbar = tqdm.tqdm(total=num_steps, desc="train")
                pbar.update(1)
                pbar.set_postfix(result)
            if pbar is not None:
                pbar.close()
