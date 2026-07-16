import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, cast
import uuid

import httpx

import art
from art import dev
from art.distributed import (
    ArtRuntime,
    ClusterSpec,
    EndpointSpec,
    HostSpec,
    ModelServiceMemberSpec,
    ModelServiceSpec,
    NcclTransportSpec,
    VllmParallelSpec,
    compile_topology,
)
from art.distributed.monarch_bootstrap import attach_controller
from art.megatron import MegatronBackend
from art.megatron.distributed_service import DistributedMegatronService

REDUCED_MODEL = "/mnt/ws_pvc/ws/projects/worktrees/art/glm52_cp/scratch/glm52_e2e_hf_home/hub/models--zai-org--GLM-5.2/snapshots/e2e_12l_full_dims"
MODEL_NAME = "glm52-vllm-recovery"
HOSTS = ("10.0.9.49", "10.0.7.93")
MONARCH_PORT = int(os.environ.get("ART_MONARCH_PORT", "22242"))
SERVICE_PORT = int(os.environ.get("ART_SERVICE_PORT", "18114"))
ROOT = Path(
    os.environ.get(
        "ART_RECOVERY_ROOT", Path(__file__).parent / "live_vllm_recovery_state"
    )
)
READY = Path(
    os.environ.get(
        "ART_RECOVERY_READY", Path(__file__).parent / "vllm_recovery_ready.json"
    )
)
OUTPUT = Path(
    os.environ.get(
        "ART_RECOVERY_OUTPUT", Path(__file__).parent / "vllm_recovery_result.json"
    )
)


def _cluster() -> ClusterSpec:
    return ClusterSpec(
        hosts=tuple(
            HostSpec(
                host_id=f"host{rank}",
                node_rank=rank,
                worker_address=f"tcp://{host}:{MONARCH_PORT}",
                cpu_slots=8,
                gpu_ids=tuple(range(4, 8)),
            )
            for rank, host in enumerate(HOSTS)
        ),
        controller_host_id="host0",
        artifact_root=str(ROOT),
        nccl_transport=NcclTransportSpec(net_name="IB"),
        startup_timeout_s=1800,
        rpc_timeout_s=300,
    )


def _service() -> ModelServiceSpec:
    return ModelServiceSpec(
        name=MODEL_NAME,
        members=tuple(
            ModelServiceMemberSpec(
                member_id=f"node{rank}",
                host_id=f"host{rank}",
                node_rank=rank,
                gpu_ids=tuple(range(4, 8)),
            )
            for rank in range(2)
        ),
        leader_endpoint=EndpointSpec(host=HOSTS[0], port=SERVICE_PORT),
        rendezvous=EndpointSpec(host=HOSTS[0], port=SERVICE_PORT + 11510),
        base_model=REDUCED_MODEL,
        model_revision="local-e2e-12l-full-dims",
        runtime_fingerprint="glm52-e2e-12l-vllm-recovery",
        parallel=VllmParallelSpec(
            tp=4,
            pp=2,
            enable_expert_parallel=True,
        ),
        update_mode="lora",
    )


async def _completion(client: httpx.AsyncClient, base_url: str, model: str) -> Any:
    response = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Return one token."}],
            "temperature": 0,
            "seed": 314159,
            "max_tokens": 1,
            "logprobs": True,
        },
    )
    response.raise_for_status()
    return response.json()


async def main(hosts: Any | None = None) -> None:
    os.environ.pop("WANDB_API_KEY", None)
    ROOT.mkdir(parents=True, exist_ok=True)
    READY.unlink(missing_ok=True)
    OUTPUT.unlink(missing_ok=True)
    owns_host_mesh = hosts is None
    if hosts is None:
        hosts = await attach_controller(
            [f"tcp://{host}:{MONARCH_PORT}" for host in HOSTS],
            name=f"glm52_vllm_recovery_{uuid.uuid4().hex}",
        )
    runtime = await ArtRuntime.start(
        hosts,
        compile_topology(cluster=_cluster(), model_services=(_service(),)),
        owns_host_mesh=owns_host_mesh,
    )
    backend = MegatronBackend(
        runtime=runtime, path=str(ROOT), enable_expert_replay=False
    )
    model = art.TrainableModel(
        name=MODEL_NAME,
        project="multinode",
        base_model=REDUCED_MODEL,
        _internal_config=cast(
            dev.InternalModelConfig,
            {
                "allow_unvalidated_arch": True,
                "init_args": {"max_seq_length": 122880},
                "engine_args": {
                    "distributed_executor_backend": "mp",
                    "max_model_len": 122880,
                    "gpu_memory_utilization": 0.75,
                    "enable_prefix_caching": True,
                    "max_num_batched_tokens": 16384,
                    "max_num_seqs": 64,
                },
            },
        ),
    )
    try:
        await model.register(backend)
        service = cast(DistributedMegatronService, backend._services[MODEL_NAME])
        manager = runtime.model_service(MODEL_NAME)
        initial = manager.state
        base_url = str(model.inference_base_url).removesuffix("/v1")
        inference_name = model.inference_model_name
        assert inference_name is not None
        headers = {"Authorization": f"Bearer {model.inference_api_key}"}
        async with httpx.AsyncClient(timeout=300, headers=headers) as client:
            before = await _completion(client, base_url, inference_name)
            READY.write_text(
                json.dumps(
                    {
                        "service_name": MODEL_NAME,
                        "generation": initial.generation,
                        "members": [
                            member.model_dump(mode="json") for member in initial.members
                        ],
                    },
                    indent=2,
                )
            )
            deadline = time.monotonic() + 1200
            while manager.state.generation == initial.generation or (
                manager.state.phase != "ready"
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"model service did not recover: {manager.state.model_dump()}"
                    )
                await asyncio.sleep(0.25)
            while service._recovery_tasks:
                if time.monotonic() >= deadline:
                    raise TimeoutError("model-service recovery task did not finish")
                await asyncio.sleep(0.25)
            after = await _completion(client, base_url, inference_name)
            state = (await client.get(f"{base_url}/art/state")).json()
        result = {
            "before": before,
            "after": after,
            "initial_generation": initial.generation,
            "recovered_generation": manager.state.generation,
            "state": state,
        }
        OUTPUT.write_text(json.dumps(result, indent=2))
        print("VLLM_MEMBER_RECOVERY_PASS", OUTPUT, manager.state.generation)
    finally:
        await backend.close()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
