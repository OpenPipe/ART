"""CPU lifecycle/admission probes; execute the real trainer method bodies.

Native trainer construction, planner metadata, telemetry and allocation accounting
are stubbed. Dynamo cache transitions, reentrant checkpoint backward, Gloo and
calibration phases are real. No CUDA/Megatron installation is needed.
"""

import ast
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import checkpoint

from art.megatron.flex_attn import compiled
from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _memory as memory


def _body(relative, name, namespace):
    source = Path(compiled.__file__).parents[2] / relative
    nodes = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(nodes) == 1
    node = nodes[0]
    node.decorator_list = []  # Only the unrelated backward-time meter.
    unit = ast.parse("from __future__ import annotations")
    unit.body.append(node)
    exec(compile(unit, str(source), "exec"), namespace)
    return namespace[name]


def _entries(entry):
    originals = {
        "dense": compiled._forced_flex_attention_dense
        if entry == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"}),
        "sparse": compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"}),
    }
    return originals, {
        kind: torch.compile(fn, backend="eager", dynamic=False)
        for kind, fn in originals.items()
    }


def _plan(entries):
    kernels = [
        compiled.compiled_flex_attention_family(
            fn, device=torch.device("cpu"), head_dim=64
        )
        for fn in entries.values()
    ]
    workload = json.dumps(
        {
            "execution_qualification": {
                "backend": "FLASH",
                "kernels": kernels,
            }
        }
    )
    return SimpleNamespace(caller_cp_workload=(workload, ((1, 1),) * 8))


def _iterators(monkeypatch, entry, checkpointed):
    torch._dynamo.reset()
    monkeypatch.setattr(
        compiled,
        "_FORCED_FLEX_EXECUTION_STATE",
        {
            "required": False,
            "fallback": False,
        },
    )
    eager = []

    def dense_math(q, *args, **kwargs):
        if not torch.compiler.is_compiling():
            eager.append(True)
        return q + 1

    def sparse_math(q, *args, **kwargs):
        return dense_math(q), q.sum(-1), q.sum(-1)

    monkeypatch.setattr(compiled, "flex_attention", dense_math)
    monkeypatch.setattr(compiled, "flex_attention_hop", sparse_math)
    originals, entries = _entries(entry)
    monkeypatch.setattr(
        memory, "_cp_flex_entrypoints", lambda rank: tuple(entries.values())
    )
    plan_data = vars(_plan(entries))

    class Plan(SimpleNamespace):
        pass

    flat = Plan(
        **plan_data, packed_tokens=16384, logical_tokens=16384, subforward_count=1
    )
    telemetry = lambda *a, **kw: nullcontext()
    impl = SimpleNamespace(
        torch=torch,
        _FlatForwardPlan=Plan,
        _materialize=lambda x: x,
        _flatten=lambda items: iter(items),
        _unflatten=lambda item, outputs: next(outputs),
        _telemetry_phase=telemetry,
        MicroBatch=lambda **kw: SimpleNamespace(**kw),
        MicroBatchStats=lambda **kw: SimpleNamespace(**kw),
    )
    namespace = dict(
        _impl=impl,
        _memory=memory,
        _protocol=protocol,
        torch=torch,
        _telemetry_phase=telemetry,
    )
    micro = _body(
        "trainer_rank/_micro_batch_planner.py", "_forward_micro_batches", namespace
    )
    run_flat = _body(
        "trainer_rank/_impl.py", "_run_flat_plan_with_memory_tracking", namespace
    )
    monkeypatch.setattr(memory, "_mark_subfloor_cp_sample", lambda *a: None)
    monkeypatch.setattr(memory, "_try_update_caller_memory_profile", lambda *a: False)
    rank = SimpleNamespace(
        runtime=SimpleNamespace(compiled_flash_cohort_supervisor_pid=os.getpid()),
        device=torch.device("cpu"),
    )
    candidate = SimpleNamespace(
        plan=flat,
        check=SimpleNamespace(estimated_required_bytes=1, available_bytes=2),
        inputs=[object()],
        indices=[0],
        stats_global_count=1,
        rejected_candidates=0,
        cold_start=False,
    )
    for name in (
        "_validate_replicated_top_level_count",
        "_forward_item",
        "_reset_planning_telemetry",
        "_snapshot_planning_telemetry",
        "_begin_planner_observation",
    ):
        setattr(rank, name, lambda *a, **kw: None)
    rank._backward_work = lambda: None
    rank._group_active_request_indices = lambda *a, **kw: [(None, [0])]
    rank._select_next_micro_batch = lambda *a, **kw: candidate
    rank._telemetry_signature = lambda plan: None
    rank._telemetry_plan_signature = lambda plan: None
    attention = entries["sparse" if entry == "sparse" else "dense"]
    original = originals["sparse" if entry == "sparse" else "dense"]
    mask = (
        create_block_mask(
            lambda b, h, q, k: q >= 0,
            B=1,
            H=1,
            Q_LEN=128,
            KV_LEN=128,
            device="cpu",
        )
        if entry == "sparse"
        else None
    )
    options = dict(block_mask=mask, scale=1.0, enable_gqa=False)

    def compute(q):
        output = attention(q, q, q, **options)
        return output[0] if entry == "sparse" else output

    rank._execute_flat_plan = lambda plan: (
        [
            checkpoint(
                compute,
                torch.ones(1, 1, 2, 64, requires_grad=True),
                use_reentrant=True,
            )
        ]
        if checkpointed
        else [object()]
    )
    rank._run_flat_plan_with_memory_tracking = lambda plan, **kw: run_flat(
        rank, plan, **kw
    )
    rank._release_cached_memory_for_backward = lambda *a, **kw: None
    return (
        micro(rank, [object()], checkpoint=None, yield_empty=False),
        micro(rank, [object()], checkpoint=None, yield_empty=False),
        entries,
        original,
        compute,
        options,
        eager,
    )


@pytest.mark.parametrize("entry", ["dense", "sparse", "forced_dense"])
@pytest.mark.parametrize("finish", ["close", "complete", "throw"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_out_of_order_qualified_iterators(monkeypatch, entry, finish, checkpointed):
    a, b, entries, original, compute, options, eager = _iterators(
        monkeypatch, entry, checkpointed
    )
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            next(a)
            wave = next(b)
            if not checkpointed:
                compute(torch.zeros(1, 1, 2, 64))
            if finish == "close":
                a.close()
            elif finish == "complete":
                with pytest.raises(StopIteration):
                    next(a)
            else:
                with pytest.raises(ValueError, match="caller failed"):
                    a.throw(ValueError("caller failed"))
            assert all(
                compiled._flex_execution_state(fn)["required"]
                for fn in entries.values()
            )
            with pytest.raises(RuntimeError, match="refusing eager FLASH fallback"):
                if checkpointed:
                    wave.outputs[0].sum().backward()
                else:
                    compute(torch.zeros(1, 1, 3, 64))
            assert not eager
        b.close()
        for fn in entries.values():
            state = compiled._flex_execution_state(fn)
            assert not state["required"]
            assert state["owners"] == state["supervised_owners"] == 0
        # After the last owner closes, ordinary eager math is allowed as on main.
        q = torch.ones(1, 1, 2, 64)
        original(q, q, q, **options)
        assert eager == [True]
    finally:
        a.close()
        b.close()
        torch._dynamo.reset()


def _unsupervised_rank(rank_index, rendezvous, directory, entry, capability, split):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank_index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        _, entries = _entries(entry)
        plan = _plan(entries)
        memory._cp_flex_entrypoints = lambda rank: tuple(entries.values())
        runtime = SimpleNamespace()
        if capability == "present" or (capability == "mixed" and rank_index == 0):
            runtime.compiled_flash_cohort_supervisor_pid = os.getpid()
        elif capability == "stale_pid":
            runtime.compiled_flash_cohort_supervisor_pid = -1
        rank = SimpleNamespace(runtime=runtime, device=torch.device("cpu"))
        rank._forward_collective_device = lambda: "cpu"
        reduce = _body(
            "trainer_rank/_impl.py",
            "_recovery_reduce",
            dict(
                dist=dist,
                torch=torch,
                _calibration_protocol=protocol,
            ),
        )
        rank._recovery_reduce = lambda *a, **kw: reduce(rank, *a, **kw)
        flat = plan
        if split:
            plan = SimpleNamespace(subforwards=(flat, flat))
        # Test both ordinary retained-v2 execution and the calibration driver.
        entered = 0
        for driver in (nullcontext(), protocol.Driver(rank)):
            with driver:
                if capability == "present":
                    # Capability is a supervisor stand-in; do not trigger fatal
                    # fallback in these raw, independently managed processes.
                    with memory._compiled_cp_execution(rank, plan):
                        assert all(
                            compiled._flex_execution_state(fn)["required"]
                            for fn in entries.values()
                        )
                        entered += 1
                else:
                    with pytest.raises(
                        RuntimeError, match="requires cohort supervision"
                    ):
                        with memory._compiled_cp_execution(rank, plan):
                            raise AssertionError("unsupervised qualified model entered")
            assert all(
                not compiled._flex_execution_state(fn)["required"]
                for fn in entries.values()
            )
        # The low-level guard also refuses distributed callers without an explicit
        # supervisor assertion, before it can arm any entrypoint or run the model.
        with pytest.raises(RuntimeError, match="requires cohort supervision"):
            with compiled.require_compiled_flex_attention(*entries.values()):
                raise AssertionError("unsupervised low-level model entered")
        # The group still works on both ranks: no fatal refusal or stranded peer.
        dist.barrier()
        Path(directory, f"rank{rank_index}.json").write_text(
            json.dumps({"refused": capability != "present", "entered": entered})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("entry", ["dense", "sparse", "forced_dense"])
@pytest.mark.parametrize("capability", ["absent", "mixed", "stale_pid", "present"])
@pytest.mark.parametrize("split", [False, True])
def test_qualified_admission_requires_collective_supervision(
    tmp_path, entry, capability, split
):
    # Raw Process objects deliberately provide no ProcessContext.join reaper.
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_unsupervised_rank,
            args=(
                index,
                f"file://{tmp_path}/rendezvous",
                str(tmp_path),
                entry,
                capability,
                split,
            ),
        )
        for index in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(12)
        assert all(not process.is_alive() for process in processes), "admission hung"
        assert [process.exitcode for process in processes] == [0, 0]
        assert [
            json.loads((tmp_path / f"rank{index}.json").read_text())
            for index in range(2)
        ] == [
            {
                "refused": capability != "present",
                "entered": 2 if capability == "present" else 0,
            }
        ] * 2
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()


def test_local_qualified_admission_requires_explicit_capability(monkeypatch):
    _, entries = _entries("dense")
    monkeypatch.setattr(
        memory, "_cp_flex_entrypoints", lambda rank: tuple(entries.values())
    )
    rank = SimpleNamespace(runtime=SimpleNamespace())
    with pytest.raises(RuntimeError, match="requires cohort supervision"):
        with memory._compiled_cp_execution(rank, _plan(entries)):
            pytest.fail("unsupervised qualified model entered")
    # Unqualified and TRITON plans keep main's execution behavior.
    for plan in (
        SimpleNamespace(caller_cp_workload=None),
        SimpleNamespace(
            caller_cp_workload=(
                json.dumps(
                    {
                        "execution_qualification": {
                            "backend": "TRITON",
                            "kernels": [
                                json.dumps(
                                    {
                                        "version": 1,
                                        "kind": kind,
                                        "options": {"BACKEND": "TRITON"},
                                    }
                                )
                                for kind in ("dense", "sparse")
                            ],
                        }
                    }
                ),
                (),
            ),
        ),
    ):
        with memory._compiled_cp_execution(rank, plan):
            assert all(
                not compiled._flex_execution_state(fn)["required"]
                for fn in entries.values()
            )


def test_partial_requirement_acquisition_releases_its_owners():
    _, entries = _entries("dense")
    fn = entries["dense"]
    with pytest.raises(ValueError, match="Unknown compiled Flex"):
        with compiled.require_compiled_flex_attention(fn, fn, object()):
            pytest.fail("invalid scope entered")
    state = compiled._flex_execution_state(fn)
    assert not state["required"]
    assert state["owners"] == state["supervised_owners"] == 0
