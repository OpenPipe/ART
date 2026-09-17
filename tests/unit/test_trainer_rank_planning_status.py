"""Pure planning failures must reach every WORLD participant before admission."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("primary_type", (ValueError, KeyboardInterrupt))
@pytest.mark.parametrize("exchange_fails", (False, True))
def test_planning_status_preserves_primary_chain(
    enabled: bool, primary_type: type[BaseException], exchange_fails: bool
) -> None:
    from art.trainer_rank import TrainerRank

    rank = TrainerRank.__new__(TrainerRank)
    primary = primary_type("local planner failed")
    cause, context = LookupError("cause"), KeyError("context")
    primary.__cause__, primary.__context__ = cause, context
    primary.__suppress_context__ = True
    calls = []

    def exchange(succeeded: bool) -> bool:
        calls.append(succeeded)
        if exchange_fails:
            raise OSError("collective failed")
        return succeeded

    rank._all_ranks_true = exchange
    with pytest.raises(primary_type) as captured:
        with rank._planning_status(enabled):
            raise primary
    assert captured.value is primary
    assert primary.__cause__ is cause and primary.__context__ is context
    assert primary.__suppress_context__
    assert calls == ([False] if enabled else [])


def test_planning_failures_and_empty_ranks_use_aligned_status(tmp_path: Path) -> None:
    children = []
    logs = []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [sys.executable, "-B", __file__, str(rank), str(tmp_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for rank, child in enumerate(children):
            assert child.wait(timeout=60) == 0, (
                tmp_path / f"rank-{rank}.log"
            ).read_text()
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in logs:
            log.close()


def _worker(index: int, directory: Path) -> None:
    import torch
    import torch.distributed as dist

    from art.trainer_rank import ForwardInput, TrainerRank, _impl
    from art.trainer_rank._prefix_tree_planner import (
        build_canonical_prefix_tree,
        prefix_tree_layout_candidates,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=index,
        world_size=2,
        init_method=f"file://{directory / 'gloo'}",
        timeout=timedelta(seconds=10),
    )
    try:
        for mode in (
            "estimate",
            "materialize",
            "price",
            "cp_plan",
            "empty",
            "unequal",
            "unavailable",
        ):
            rank = TrainerRank.__new__(TrainerRank)
            rank.device = torch.device("cpu")
            rank._padded_vocab_size = None
            rank._moe_layers = rank._gdn_layers = 0
            rank._slot_stack = []
            rank._default_slot_ref = None
            rank._planning_seconds_accum = 0.0
            rank._dp_rank_and_size = lambda: (index, 2)
            rank._physical_tokens = lambda tokens: tokens
            rank._estimate_group_request_output_bytes = lambda requests: 0
            rank._memory_signature_from_requests = lambda *args, **kwargs: None
            rank._forward_item = lambda request: SimpleNamespace(
                input_ids=request.input_tokens, request=request
            )
            rank._forward_output_metadata = lambda *args, **kwargs: (None, True)

            def slots(selections):
                assert tuple(selections) == ()
                dist.barrier()

            rank._ensure_checkpoint_slots = slots

            def layout(ids, **kwargs):
                tree = build_canonical_prefix_tree(ids)
                return tree, prefix_tree_layout_candidates(tree)[0].layout

            rank._select_group_layout = layout
            primary = ValueError("rank-local planning failure")
            cause, context = LookupError("cause"), KeyError("context")
            primary.__cause__, primary.__context__ = cause, context
            primary.__suppress_context__ = True
            original_estimate = _impl.estimate_prefix_tree_packed_tokens
            original_materialize = _impl.materialize_prefix_tree_layout

            def estimate(*args, **kwargs):
                if index == 0 and mode == "estimate":
                    raise primary
                if index == 0 and mode == "unavailable":
                    return None
                return original_estimate(*args, **kwargs)

            def materialize(*args, **kwargs):
                if index == 0 and mode == "materialize":
                    raise primary
                return original_materialize(*args, **kwargs)

            def price(**kwargs):
                if index == 0 and mode == "price":
                    raise primary
                return 0

            rank._estimate_required_memory_bytes_from_values = price

            def retained_tokens(plan):
                if index == 0 and mode == "cp_plan":
                    raise primary
                return plan.packed_tokens

            rank._plan_retained_tokens = retained_tokens
            patches = pytest.MonkeyPatch()
            patches.setattr(_impl, "estimate_prefix_tree_packed_tokens", estimate)
            patches.setattr(_impl, "materialize_prefix_tree_layout", materialize)
            requests = [
                ForwardInput(
                    input_tokens=torch.tensor([1, 2]), hidden_states=True, no_grad=True
                )
            ]
            if mode == "empty" and index == 1:
                requests = []
            if mode == "unequal" and index == 0:
                requests.append(
                    ForwardInput(
                        input_tokens=torch.tensor([3, 4]),
                        hidden_states=True,
                        no_grad=False,
                    )
                )
            try:
                error = None
                try:
                    if mode == "estimate":
                        # Enter the ordinary scheduler, before its first memory check.
                        rank._select_next_micro_batch(requests * 2, 0)
                        raise AssertionError("Planner failure unexpectedly returned")
                    values = rank._estimate_flat_forward(
                        requests, sync_planning_errors=True
                    )
                    if mode == "unavailable":
                        assert not rank._all_ranks_true(values is not None)
                    plan = rank._plan_flat_forward(requests, sync_planning_errors=True)
                    if mode in ("price", "cp_plan"):
                        rank._memory_check(
                            plan, sync_across_dp=True, sync_planning_errors=True
                        )
                    assert plan.request_count == len(requests)
                except BaseException as caught:
                    error = caught
                if mode in ("estimate", "materialize", "price", "cp_plan"):
                    if index == 0:
                        assert error is primary, (mode, repr(error))
                        assert error.__cause__ is cause and error.__context__ is context
                    else:
                        assert type(error) is RuntimeError
                        assert str(error) == "Local planning failed on another DP rank"
                else:
                    assert error is None, repr(error)
            finally:
                patches.undo()
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Direct child execution uses this checkout even without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    _worker(int(sys.argv[1]), Path(sys.argv[2]))
