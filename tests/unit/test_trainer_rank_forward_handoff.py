"""A local handoff failure must precede the next model-parallel forward."""

import asyncio
from dataclasses import replace
import gc
import traceback
import weakref

import pytest
from test_trainer_rank_slot_graph_lifetime import _prepare_forward
import torch
import torch.distributed as dist
from trainer_rank_test_support import gloo_group, megatron_topology, spawn_and_join

from art.trainer_rank import ForwardOutput
from art.trainer_rank._impl import (
    _FlatForwardPlan,
    _MemoryCheck,
    _MemorySignature,
    _SplitForwardPlan,
)


@pytest.mark.parametrize("split", [False, True], ids=["groups", "split"])
def test_failed_handoff_precedes_next_physical_forward(tmp_path, monkeypatch, split):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    spawn_and_join(
        _handoff_worker,
        (f"file://{tmp_path / 'handoff'}", split),
        timeout=180,
        failure="Handoff failure stranded a peer in the next physical forward",
    )


def _handoff_worker(physical, rendezvous, split):
    with pytest.MonkeyPatch.context() as patch:
        # Load the real native LoRA types before the callback topology shim.
        trainer, ref, weight, group = _prepare_forward(patch, "cpu", "cpu")
        with (
            gloo_group(physical, rendezvous, timeout=10),
            megatron_topology(physical, dp_size=1, tp_size=2),
        ):
            calls = 0

            def forward(items, prepared):
                nonlocal calls
                calls += 1
                # This real collective models the next TP/CP model boundary.
                value = torch.ones(1)
                dist.all_reduce(value)
                assert value.item() == 2
                return [ForwardOutput(weight.square(), None, None, None)]

            patch.setattr(trainer, "_forward_packed", forward)
            signature = _MemorySignature(
                (1, 2, 1, 1), (0, None), 3, (), True, (True,) * 3
            )
            flat = _FlatForwardPlan(
                3,
                (("student", False),) * 3,
                tuple(replace(group, request_indices=(i,)) for i in range(3)),
                6,
                6,
                12,
                signature,
            )
            plan = (
                _SplitForwardPlan(
                    tuple(
                        replace(
                            flat,
                            request_count=1,
                            output_metadata=(("student", False),),
                            groups=(group,),
                        )
                        for _ in range(3)
                    ),
                    ((0,), (1,), (2,)),
                    3,
                )
                if split
                else flat
            )
            patch.setattr(
                trainer,
                "_plan_admissible_forward",
                lambda *a, **k: (plan, _MemoryCheck(12, 100, True)),
            )
            inputs = [group.items[0].request] * 3
            cache = trainer._forward_graph_cache()
            original_to = torch.Tensor.to
            for fail_at, kind in ((1, MemoryError), (2, asyncio.CancelledError)):
                preserved = trainer._execute_graph_group(group)[0].target_logprobs
                assert preserved is not None
                original_handles = cache.handles()
                calls = 0
                primary = kind("injected correction copy failure")
                primary.__cause__ = cause = ValueError("original cause")
                references = []
                observed = set()

                def copy(value, *args, **kwargs):
                    if args == ("cpu",) and kwargs.get("copy"):
                        for handle in (
                            set(cache.handles()) - set(original_handles) - observed
                        ):
                            record = cache._records[handle]
                            references.extend(record.saved or ())
                            references.extend(
                                weakref.ref(t) for t in record.outputs or ()
                            )
                            observed.add(handle)
                        if physical == 1 and calls == fail_at:
                            raise primary
                    return original_to(value, *args, **kwargs)

                with patch.context() as allocation:
                    allocation.setattr(torch.Tensor, "to", copy)
                    with pytest.raises(
                        kind if physical == 1 else RuntimeError
                    ) as caught:
                        trainer.forward(inputs)
                assert calls == fail_at, (calls, fail_at, repr(caught.value))
                if physical == 1:
                    assert caught.value is primary and primary.__cause__ is cause
                else:
                    assert "another rank" in str(caught.value).lower(), repr(
                        caught.value
                    )
                assert cache.handles() == original_handles
                assert trainer._has_live_slot_graph(ref)
                assert len(trainer._slot_graphs()[ref]) == 1
                assert observed and all(reference() is None for reference in references)
                traceback.clear_frames(caught.value.__traceback__)
                gc.collect()
                assert weight.grad is None
                calls = 0
                outputs = trainer.forward(inputs)
                assert calls == 3
                targets = [output.target_logprobs for output in outputs]
                assert all(value is not None and value.item() == 4 for value in targets)
                trainer.backward(
                    torch.stack([value for value in targets if value is not None]).sum()
                )
                torch.testing.assert_close(weight.grad, torch.tensor(12.0))
                assert cache.handles() == original_handles
                trainer.backward(preserved)
                torch.testing.assert_close(weight.grad, torch.tensor(16.0))
                trainer.zero_grad()
                # The fixture's analytic parameter is independent of checkpoint slots.
                weight.grad = None
                assert not cache.handles() and not trainer._has_live_slot_graph(ref)

            single = replace(
                flat,
                request_count=1,
                output_metadata=(("student", False),),
                groups=(group,),
            )
            patch.setattr(
                trainer,
                "_plan_admissible_forward",
                lambda *a, **k: (single, _MemoryCheck(4, 100, True)),
            )
            patch.setattr(
                trainer,
                "_recovery_reduce",
                lambda *a, **k: pytest.fail("single forward added a handoff reduction"),
            )
            output = trainer.forward(inputs[:1])[0].target_logprobs
            assert output is not None and output.item() == 4
            trainer.backward(output)
            torch.testing.assert_close(weight.grad, torch.tensor(4.0))
            assert not cache.handles()
