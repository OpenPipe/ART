import asyncio
import time

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_router import (
    KvCacheEvent,
    PrefixBlockHashes,
    ReplicaRouter,
    ReplicaTelemetry,
    RoutableReplica,
    RoutingInput,
    RoutingTable,
)


async def main() -> None:
    hashes = tuple(str(index) for index in range(7680))
    now = time.monotonic()
    replicas = tuple(
        RoutableReplica(
            replica_id=f"r{index}",
            endpoint=EndpointSpec(host="127.0.0.1", port=8000 + index),
            phase="ready",
            generation=0,
            generation_digest=f"g{index}",
            committed_version="1",
            policy_digest="d",
            update_identity="u",
            telemetry=ReplicaTelemetry(observed_at=now, in_flight=index, capacity=256),
        )
        for index in range(4)
    )
    router = ReplicaRouter(
        RoutingTable(
            policy_generation=1,
            policy_version="1",
            policy_digest="d",
            update_identity="u",
            replicas=replicas,
        )
    )
    for replica in replicas:
        router.apply_kv_event(
            KvCacheEvent(
                replica_id=replica.replica_id,
                generation=0,
                sequence=0,
                version="cache",
                block_size=16,
                operation="store",
                block_hashes=hashes,
            )
        )
    request = RoutingInput(
        policy_version="1",
        policy_digest="d",
        prefix=PrefixBlockHashes(version="cache", block_size=16, hashes=hashes),
    )
    start = time.perf_counter()
    for _ in range(100):
        reservation = await router.acquire(request, timeout_s=1)
        await reservation.release()
    elapsed = time.perf_counter() - start
    print(f"{elapsed / 100 * 1e3:.3f} ms/decision")


asyncio.run(main())
