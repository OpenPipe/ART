"""Utilities that require replies from every vLLM engine core."""

import asyncio
from typing import Any


async def query_engine_cores(
    engine_client: Any, method: str, *args: Any
) -> tuple[Any, ...]:
    core = engine_client.engine_core
    data_parallel_size = int(
        engine_client.vllm_config.parallel_config.data_parallel_size
    )
    if data_parallel_size == 1:
        return (await core.call_utility_async(method, *args),)

    engines = getattr(core, "core_engines", ())
    call = getattr(core, "_call_utility_async", None)
    if len(engines) != data_parallel_size or not callable(call):
        raise RuntimeError("vLLM client does not expose every DP engine core")
    return tuple(
        await asyncio.gather(
            *(call(method, *args, engine=engine) for engine in engines)
        )
    )
