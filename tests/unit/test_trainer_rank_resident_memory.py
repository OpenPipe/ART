"""CPU placement accounts for contexts which retain their original tensor storage."""

from dataclasses import replace

from test_trainer_rank_memory_admission import _requests, rank  # noqa: F401

from art.trainer_rank import ForwardOptions
from art.trainer_rank._memory_policy import ForwardMemoryCost, placement_cost


def test_partial_offload_retains_residual_for_every_complete_root_child():
    cost = ForwardMemoryCost(100, 80, 10, cpu_resident_bytes=40)
    cpu = placement_cost([cost] * 3, backward_state="cpu", output_device="cpu")
    replay = placement_cost([cost] * 3, backward_state="replay", output_device="cpu")
    assert cpu.gpu_retained_bytes == 120
    assert cpu.gpu_required_bytes == 180
    assert replay.gpu_retained_bytes == 0 and replay.gpu_required_bytes == 100


def test_cp_residency_requires_matching_layout_and_keeps_cold_bound(rank, monkeypatch):
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 1, 2, 1))
    plan = rank._plan_flat_forward(
        _requests(ForwardOptions(backward_state="cpu", output_device="cpu"))[:1]
    )
    assert tuple(rank._graph_memory_units(plan))[0][2].cpu_resident_bytes == 80
    group = plan.groups[0]
    rank._graph_residency = {rank._graph_residency_key(group): 40}
    measured = tuple(rank._graph_memory_units(plan))[0][2]
    assert 40 <= measured.cpu_resident_bytes < 80
    other = replace(
        plan,
        groups=(replace(group, packed=replace(group.packed, segments=())),),
    )
    assert tuple(rank._graph_memory_units(other))[0][2].cpu_resident_bytes == 80
    branched = replace(
        group,
        packed=replace(
            group.packed,
            segments=tuple(
                replace(segment, parent_id=segment.parent_id + 1)
                for segment in group.packed.segments
            ),
        ),
    )
    assert rank._graph_residency_key(branched) != rank._graph_residency_key(group)
    rank._graph_residency = {rank._graph_residency_key(group): 120}
    underestimated = tuple(rank._graph_memory_units(plan))[0][2]
    assert underestimated.retained_bytes >= 120
    assert underestimated.peak_bytes == underestimated.retained_bytes + 20
