import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, cast
import uuid

import httpx
from transformers import AutoTokenizer

import art
from art.distributed import (
    ArtRuntime,
    ClusterSpec,
    EndpointSpec,
    HostSpec,
    ModelServiceMemberSpec,
    ModelServiceReplicaSpec,
    ModelServiceSpec,
    VllmParallelSpec,
    compile_topology,
)
from art.distributed.monarch_bootstrap import attach_controller
from art.distributed.vllm_router import (
    RoutingInput,
    canonical_block_hash,
    vllm_request_block_hashes,
)
from art.megatron import MegatronBackend
from art.megatron.distributed_service import DistributedMegatronService

REDUCED_MODEL = "/mnt/ws_pvc/ws/projects/worktrees/art/glm52_cp/scratch/glm52_e2e_hf_home/hub/models--zai-org--GLM-5.2/snapshots/e2e_12l_full_dims"
MODEL_NAME = "glm52-prefix-affinity"
ROOT = Path(__file__).parent / "live_prefix_affinity_state"
OUTPUT = Path(__file__).parent / "glm_prefix_affinity.json"
HOSTS = ("10.0.9.49", "10.0.7.93")


def _replica(rank: int) -> ModelServiceReplicaSpec:
    endpoint_port = 18120 + 10 * rank
    return ModelServiceReplicaSpec(
        service_name=MODEL_NAME,
        replica_id=f"glm52-tp4-replica-{rank}",
        members=(
            ModelServiceMemberSpec(
                member_id=f"node{rank}",
                host_id=f"host{rank}",
                node_rank=0,
                gpu_ids=tuple(range(4)),
                leader=True,
            ),
        ),
        leader_endpoint=EndpointSpec(host=HOSTS[rank], port=endpoint_port),
        rendezvous=EndpointSpec(host=HOSTS[rank], port=29630 + rank),
        base_model=REDUCED_MODEL,
        model_revision="local-e2e-12l-full-dims",
        runtime_fingerprint="glm52-e2e-12l-two-replica-tp4",
        parallel=VllmParallelSpec(tp=4, enable_expert_parallel=True),
        update_mode="lora",
        kv_event_port=18220 + 10 * rank,
        kv_replay_port=18320 + 10 * rank,
    )


def _cluster() -> ClusterSpec:
    return ClusterSpec(
        hosts=tuple(
            HostSpec(
                host_id=f"host{rank}",
                node_rank=rank,
                worker_address=f"tcp://{host}:22227",
                cpu_slots=8,
                gpu_ids=tuple(range(8)),
            )
            for rank, host in enumerate(HOSTS)
        ),
        controller_host_id="host0",
        artifact_root=str(ROOT),
        startup_timeout_s=1800,
        rpc_timeout_s=300,
    )


async def _metrics(
    client: httpx.AsyncClient, rank: int, headers: dict[str, str]
) -> dict[str, float]:
    response = await client.get(
        f"http://{HOSTS[rank]}:{18120 + 10 * rank}/art/metrics", headers=headers
    )
    response.raise_for_status()
    return {key: float(value) for key, value in response.json()["metrics"].items()}


async def _wait_for_owner(
    service: DistributedMegatronService, request: RoutingInput
) -> tuple[str, dict[str, int]]:
    gateway = service._gateway
    assert gateway is not None
    deadline = time.monotonic() + 30
    while True:
        token_prefixes = gateway.router._token_prefixes(request)
        matches = {
            replica.replica_id: gateway.router._prefix_match(
                replica, request, token_prefixes
            )
            for replica in gateway.router.table.replicas
        }
        owners = [replica_id for replica_id, matched in matches.items() if matched]
        if len(owners) == 1:
            return owners[0], matches
        if time.monotonic() >= deadline:
            config = gateway.router.table.prefix_hash
            variants = {}
            assert config is not None
            for block_size in (16, 64):
                for lora_name in (
                    config.lora_name,
                    None,
                    f"{MODEL_NAME}@0",
                    f"{MODEL_NAME}:active",
                ):
                    candidate = config.model_copy(
                        update={"block_size": block_size, "lora_name": lora_name}
                    )
                    base = vllm_request_block_hashes(
                        request.prompt_token_ids or (), candidate
                    )
                    hashes = (
                        tuple(canonical_block_hash(value) for value in base)
                        if block_size == 64
                        else tuple(
                            canonical_block_hash(b"".join(base[start : start + 4]))
                            for start in range(0, len(base) - 3, 4)
                        )
                    )
                    variants[f"{block_size}:{lora_name}"] = {
                        str(key): next(
                            (
                                index
                                for index, value in enumerate(hashes)
                                if value
                                not in blocks.blocks.get((config.version, 64, 0), set())
                            ),
                            len(hashes),
                        )
                        for key, blocks in gateway.router._kv.items()
                    }
            indexes = {
                str(key): {
                    "generation": index.generation,
                    "next_sequence": index.next_sequence,
                    "groups": {
                        str(group): len(blocks)
                        for group, blocks in index.blocks.items()
                    },
                }
                for key, index in gateway.router._kv.items()
            }
            subscribers = {
                subscriber.source.replica_id: {
                    "next_sequence": subscriber._next_sequence,
                    "done": subscriber._task.done() if subscriber._task else None,
                    "exception": (
                        repr(subscriber._task.exception())
                        if subscriber._task and subscriber._task.done()
                        else None
                    ),
                }
                for subscriber in gateway._kv_subscribers
            }
            raise TimeoutError(
                "exactly one KV owner was not observed: "
                f"matches={matches}, config={config}, variants={variants}, "
                f"indexes={indexes}, subscribers={subscribers}"
            )
        await asyncio.sleep(0.1)


async def main() -> None:
    os.environ.pop("WANDB_API_KEY", None)
    service_spec = ModelServiceSpec(
        name=MODEL_NAME, replicas=tuple(_replica(rank) for rank in range(2))
    )
    hosts = await attach_controller(
        [f"tcp://{host}:22227" for host in HOSTS],
        name=f"glm52_prefix_affinity_{uuid.uuid4().hex}",
    )
    runtime = await ArtRuntime.start(
        hosts,
        compile_topology(cluster=_cluster(), model_services=(service_spec,)),
        owns_host_mesh=False,
    )
    backend = MegatronBackend(
        runtime=runtime, path=str(ROOT), enable_expert_replay=False
    )
    model = art.TrainableModel(
        name=MODEL_NAME,
        project="multinode",
        base_model=REDUCED_MODEL,
        _internal_config={
            "allow_unvalidated_arch": True,
            "init_args": {"max_seq_length": 122880},
            "engine_args": {
                "distributed_executor_backend": "mp",
                "max_model_len": 122880,
                "gpu_memory_utilization": 0.77,
                "enable_prefix_caching": True,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 64,
            },
        },
    )
    try:
        await model.register(backend)
        service = cast(DistributedMegatronService, backend._services[MODEL_NAME])
        gateway = service._gateway
        assert gateway is not None
        tokenizer = AutoTokenizer.from_pretrained(REDUCED_MODEL, trust_remote_code=True)
        prompt_token_ids = tuple(
            tokenizer.encode("Shared cache-routing context. " * 4096)[:8192]
        )
        assert len(prompt_token_ids) >= 4096
        routing = RoutingInput(
            policy_version=gateway.router.table.policy_version,
            policy_digest=gateway.router.table.policy_digest,
            prompt_token_ids=prompt_token_ids,
        )
        payload: dict[str, Any] = {
            "model": model.inference_model_name,
            "prompt": prompt_token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 314159,
            "art_routing": {"prompt_token_ids": prompt_token_ids},
        }
        base_url = str(model.inference_base_url).removesuffix("/v1")
        headers = {"Authorization": f"Bearer {model.inference_api_key}"}
        async with httpx.AsyncClient(timeout=600) as client:
            before = [await _metrics(client, rank, headers) for rank in range(2)]
            first = await client.post(
                f"{base_url}/v1/completions", json=payload, headers=headers
            )
            first.raise_for_status()
            owner, matches = await _wait_for_owner(service, routing)
            reservation = await gateway.router.acquire(routing, timeout_s=5)
            try:
                selected = reservation.replica.replica_id
            finally:
                await reservation.release()
            assert selected == owner
            middle = [await _metrics(client, rank, headers) for rank in range(2)]
            second = await client.post(
                f"{base_url}/v1/completions", json=payload, headers=headers
            )
            second.raise_for_status()
            after = [await _metrics(client, rank, headers) for rank in range(2)]
        owner_rank = int(owner.rsplit("-", 1)[-1])
        other_rank = 1 - owner_rank
        owner_hits = (
            after[owner_rank]["prefix_cache_hits_total"]
            - middle[owner_rank]["prefix_cache_hits_total"]
        )
        other_prompts = (
            after[other_rank]["prompt_tokens_total"]
            - middle[other_rank]["prompt_tokens_total"]
        )
        assert owner_hits >= len(prompt_token_ids) - 64
        assert other_prompts == 0
        result = {
            "owner": owner,
            "matches": matches,
            "selected": selected,
            "prompt_tokens": len(prompt_token_ids),
            "owner_second_request_cache_hits": owner_hits,
            "other_second_request_prompt_tokens": other_prompts,
            "before": before,
            "middle": middle,
            "after": after,
            "first": first.json(),
            "second": second.json(),
        }
        OUTPUT.write_text(json.dumps(result, indent=2))
        print("TWO_REPLICA_PREFIX_AFFINITY_PASS", OUTPUT, owner, owner_hits)
    finally:
        await backend.close()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
