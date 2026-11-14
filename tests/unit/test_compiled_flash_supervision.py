"""Verify the exact Monarch startup body installs capability after ownership."""

import ast
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from art.megatron.flex_attn import compiled


def test_monarch_installs_compiled_capability_only_after_mesh_ownership():
    source = Path(compiled.__file__).parents[1] / "runtime/monarch.py"
    nodes = ast.parse(source.read_text()).body
    actor = next(
        n
        for n in nodes
        if isinstance(n, ast.ClassDef) and n.name == "MonarchTrainerActor"
    )
    methods = [
        n
        for n in actor.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"enable_compiled_flash_supervision", "__cleanup__"}
    ]
    for method in methods:
        method.decorator_list = []
    spawn = next(
        n
        for n in nodes
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "spawn_monarch_trainer_actors"
    )
    unit = ast.parse("from __future__ import annotations")
    unit.body.extend([*methods, spawn])
    events = []
    runtime = SimpleNamespace(compiled_flash_cohort_supervisor_pid=None)
    actor_instance = SimpleNamespace(
        _runtime=runtime,
        _stop_cp_lookahead=lambda: None,
        _executor=SimpleNamespace(close=lambda: None),
    )
    ns = dict(
        os=os,
        _TrainerSPMDActor=object(),
        MonarchTrainerActor=object(),
        _TrainerRankReady=SimpleNamespace(model_validate=lambda value: value),
    )
    exec(compile(unit, str(source), "exec"), ns)

    async def mesh_name(name):
        events.append(name)
        return name

    async def initialized():
        events.append("initialized")

    async def start_store(_):
        return "localhost", 1234

    async def setup_env(*args):
        pass

    async def enable():
        assert events == ["spmd", "own:spmd", "actors", "own:actors", "initialized"]
        assert runtime.compiled_flash_cohort_supervisor_pid is None
        ns["enable_compiled_flash_supervision"](actor_instance)
        events.append("enabled")

    async def ready():
        assert runtime.compiled_flash_cohort_supervisor_pid == os.getpid()
        return {0: SimpleNamespace(rank=0, host_id="host", gpu_id=0)}

    async def ports():
        return {0: None}

    async def run():
        spmd = SimpleNamespace(
            _name=mesh_name("spmd"),
            slice=lambda **kw: SimpleNamespace(
                start_store=SimpleNamespace(call_one=start_store)
            ),
            setup_agent_store_env=SimpleNamespace(call=setup_env),
        )
        actors = SimpleNamespace(
            _name=mesh_name("actors"),
            initialized=initialized(),
            enable_compiled_flash_supervision=SimpleNamespace(call=enable),
            ready=SimpleNamespace(call=ready),
            cp_lookahead_port=SimpleNamespace(call=ports),
        )
        meshes = iter((spmd, actors))
        proc_mesh = SimpleNamespace(_labels=["rank"], spawn=lambda *a: next(meshes))
        supervision = SimpleNamespace(
            token="token",
            run_id="run",
            own_mesh=lambda name: events.append(f"own:{name}"),
        )
        spec = SimpleNamespace(
            model_dump_json=lambda: "{}",
            trainer_mesh=SimpleNamespace(
                ranks=[SimpleNamespace(host_id="host", gpu_id=0)],
                topology=SimpleNamespace(cp=1),
            ),
        )
        await ns["spawn_monarch_trainer_actors"](proc_mesh, spec, supervision)
        assert events[-1] == "enabled"
        ns["__cleanup__"](actor_instance, None)
        assert runtime.compiled_flash_cohort_supervisor_pid is None

    asyncio.run(run())
