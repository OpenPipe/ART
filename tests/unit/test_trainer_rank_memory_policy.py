from pathlib import Path
from typing import Literal

import pytest

from art.trainer_rank._memory_policy import (
    ForwardMemoryCost,
    HostMemoryBudget,
    MemoryScope,
    choose_memory_placement,
    choose_output_placements,
    host_memory_budget,
    local_rank_count,
    placement_cost,
)


def test_aggregate_outputs_reserve_explicit_model_before_auto():
    assert choose_output_placements(
        [(60, "auto"), (60, "model"), (20, "auto"), (100, "cpu")],
        gpu_available_bytes=100,
    ) == ("cpu", "model", "model", "cpu")


def test_aggregate_outputs_refuse_explicit_model_before_any_copy():
    with pytest.raises(MemoryError, match="require 120 GPU bytes"):
        choose_output_placements(
            [(60, "model"), (60, "model")], gpu_available_bytes=100
        )


def test_aggregate_outputs_fresh_headroom_accounts_previous_waves():
    outputs: list[tuple[int, Literal["auto", "model", "cpu"]]] = [
        (60, "auto"),
        (20, "auto"),
    ]
    assert choose_output_placements(outputs, gpu_available_bytes=100) == (
        "model",
        "model",
    )
    assert choose_output_placements(outputs, gpu_available_bytes=20) == ("cpu", "model")


def test_staged_gradients_accumulate_across_children_beside_restore_workspace():
    placement = placement_cost(
        [ForwardMemoryCost(100, 80, 10, gradient_staging_bytes=30)] * 3,
        backward_state="cpu",
        output_device="cpu",
    )
    assert placement.gpu_retained_bytes == 30
    assert placement.gpu_backward_bytes == 90
    assert placement.gpu_required_bytes == 30 + 90 + 90


def _host(tmp_path: Path, *, v2=True, namespace=False):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 800 kB\n")
    mount = tmp_path / "memory"
    child = mount / "pod" / "rank"
    child.mkdir(parents=True)
    root = "/delegated" if namespace else "/"
    member = root.rstrip("/") + "/pod/rank"
    (proc / "self/cgroup").write_text(
        f"0::{member}\n" if v2 else f"2:cpu,memory:{member}\n"
    )
    (proc / "self/mountinfo").write_text(
        f"30 1 0:29 {root} {mount} rw - "
        + ("cgroup2 cgroup rw\n" if v2 else "cgroup cgroup rw,memory\n")
    )
    limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
    used_name = "memory.current" if v2 else "memory.usage_in_bytes"
    for path, limit, used in (
        (mount, 800_000, 100_000),
        (child.parent, 500_000, 300_000),
        (child, 400_000, 100_000),
    ):
        (path / limit_name).write_text(str(limit))
        (path / used_name).write_text(str(used))
    return proc, mount, child, limit_name, used_name


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("namespace", [True, False])
def test_shared_host_and_ancestor_cgroup_budgets(tmp_path, v2, namespace):
    proc, _, child, _, _ = _host(tmp_path, v2=v2, namespace=namespace)
    budget = host_memory_budget(local_world_size=4, proc_root=proc)
    # The pod ancestor binds: (500000 - 300000 - 10% reserve) / 4.
    assert budget.available_bytes == 37_500
    assert len(budget.scopes) == 4
    assert min(budget.scopes, key=lambda s: s.per_rank_available_bytes).name == str(
        child.parent
    )
    assert (
        budget.available_bytes * 4
        == host_memory_budget(local_world_size=1, proc_root=proc).available_bytes
    )


def test_fresh_usage_reduces_additional_offload_credit(tmp_path):
    proc, _, child, _, used_name = _host(tmp_path)
    first = host_memory_budget(local_world_size=2, proc_root=proc)
    (child.parent / used_name).write_text("400000")
    second = host_memory_budget(local_world_size=2, proc_root=proc)
    assert first.available_bytes == 75_000
    assert second.available_bytes == 25_000


def test_topology_cache_observes_membership_and_mount_changes_immediately(tmp_path):
    proc, mount, child, limit_name, used_name = _host(tmp_path)
    assert (
        host_memory_budget(local_world_size=4, proc_root=proc).available_bytes == 37_500
    )
    moved = child.parent / "moved"
    moved.mkdir()
    (moved / limit_name).write_text("10000")
    (moved / used_name).write_text("0")
    (proc / "self/cgroup").write_text("0::/pod/moved\n")
    assert (
        host_memory_budget(local_world_size=4, proc_root=proc).available_bytes == 2_250
    )
    remount = tmp_path / "remounted"
    replacement = remount / "pod/moved"
    replacement.mkdir(parents=True)
    (replacement / limit_name).write_text("12000")
    (replacement / used_name).write_text("2000")
    mountinfo = proc / "self/mountinfo"
    mountinfo.write_text(mountinfo.read_text().replace(str(mount), str(remount)))
    assert (
        host_memory_budget(local_world_size=4, proc_root=proc).available_bytes == 2_200
    )


@pytest.mark.parametrize("used", [None, "not-a-counter", "500000"])
def test_missing_or_exhausted_cgroup_usage_grants_no_credit(tmp_path, used):
    proc, _, child, _, used_name = _host(tmp_path)
    path = child / used_name
    if used is None:
        path.unlink()
    else:
        path.write_text(used)
    assert host_memory_budget(local_world_size=2, proc_root=proc).available_bytes == 0


def test_unlimited_cgroup_still_obeys_host_and_parent(tmp_path):
    proc, mount, child, limit_name, _ = _host(tmp_path)
    for path in (mount, child):
        (path / limit_name).write_text("max")
    budget = host_memory_budget(local_world_size=2, proc_root=proc)
    assert len(budget.scopes) == 2
    assert budget.available_bytes == 75_000


def test_unknown_host_budget_fails_closed_and_empty_scopes_grant_nothing(tmp_path):
    assert (
        host_memory_budget(local_world_size=1, proc_root=tmp_path).available_bytes == 0
    )
    assert HostMemoryBudget(()).available_bytes == 0


@pytest.mark.parametrize("ranks", [0, -1])
def test_invalid_rank_count_is_rejected(ranks):
    with pytest.raises(ValueError, match="positive"):
        host_memory_budget(local_world_size=ranks)
    with pytest.raises(ValueError, match="positive"):
        _ = MemoryScope("host", 100, 100, ranks).per_rank_available_bytes


def test_local_rank_count_does_not_use_gpu_count(monkeypatch):
    for name in ("LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert local_rank_count() == 16
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert local_rank_count() == 4


ROOT = (ForwardMemoryCost(100, 80, 10, 5),) * 3


@pytest.mark.parametrize(
    "state,device,gpu,cpu,retained",
    [
        ("gpu", "model", 290, 15, 270),
        ("gpu", "cpu", 260, 45, 240),
        ("cpu", "model", 150, 225, 60),
        ("cpu", "cpu", 120, 255, 30),
        ("replay", "model", 130, 15, 30),
        ("replay", "cpu", 100, 45, 0),
    ],
)
def test_root_cost_keeps_outputs_and_backward_restore_workspace(
    state, device, gpu, cpu, retained
):
    placement = placement_cost(ROOT, backward_state=state, output_device=device)
    assert placement.gpu_required_bytes == gpu
    assert placement.cpu_required_bytes == cpu
    assert placement.gpu_retained_bytes == retained


@pytest.mark.parametrize(
    "gpu,cpu,state,device",
    [
        (290, 15, "gpu", "model"),
        (259, 225, "cpu", "model"),
        (130, 224, "replay", "model"),
        (129, 45, "replay", "cpu"),
    ],
)
def test_admission_prefers_retention_and_admits_previously_refused_roots(
    gpu, cpu, state, device
):
    placement = choose_memory_placement(
        ROOT, gpu_available_bytes=gpu, cpu_available_bytes=cpu, output_device="auto"
    )
    assert placement is not None
    assert (placement.backward_state, placement.output_device) == (state, device)
    assert placement.gpu_required_bytes <= gpu
    assert placement.cpu_required_bytes <= cpu


@pytest.mark.parametrize("gpu,cpu", [(99, 1000), (259, 14), (129, 44)])
def test_policy_never_admits_without_cpu_capacity_or_one_child_workspace(gpu, cpu):
    assert (
        choose_memory_placement(
            ROOT, gpu_available_bytes=gpu, cpu_available_bytes=cpu, output_device="auto"
        )
        is None
    )


def test_disable_levels_and_explicit_output_opt_out_are_authoritative():
    assert (
        choose_memory_placement(
            ROOT,
            gpu_available_bytes=129,
            cpu_available_bytes=1000,
            allow_cpu_offload=False,
            output_device="model",
        )
        is None
    )
    assert (
        choose_memory_placement(
            ROOT,
            gpu_available_bytes=130,
            cpu_available_bytes=224,
            allow_replay=False,
        )
        is None
    )
    forced = choose_memory_placement(
        ROOT,
        gpu_available_bytes=1000,
        cpu_available_bytes=1000,
        backward_state="replay",
        output_device="cpu",
    )
    assert forced is not None
    assert (forced.backward_state, forced.output_device) == ("replay", "cpu")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backward_state": "cpu", "allow_cpu_offload": False},
        {"backward_state": "replay", "allow_replay": False},
        {"backward_state": "typo"},
        {"output_device": "typo"},
    ],
)
def test_invalid_policy_raises_before_admission(kwargs):
    with pytest.raises(ValueError):
        choose_memory_placement(
            ROOT, gpu_available_bytes=1000, cpu_available_bytes=1000, **kwargs
        )


def test_no_grad_cpu_outputs_release_gpu_storage_without_replay():
    costs = (ForwardMemoryCost(100, 80, 80, backward_required=False),) * 3
    placement = choose_memory_placement(
        costs,
        gpu_available_bytes=100,
        cpu_available_bytes=240,
        allow_cpu_offload=False,
        allow_replay=False,
        output_device="auto",
    )
    assert placement is not None
    assert (placement.backward_state, placement.output_device) == ("gpu", "cpu")
    assert placement.gpu_required_bytes == 100
    assert placement.cpu_required_bytes == 240


def test_cost_is_independent_of_child_execution_order():
    costs = [*ROOT, ForwardMemoryCost(300, 60, 5, 10)]
    for state in ("gpu", "cpu", "replay"):
        assert placement_cost(costs, backward_state=state, output_device="cpu") == (
            placement_cost(costs[::-1], backward_state=state, output_device="cpu")
        )


def test_current_probability_correction_reserves_a_forward_beside_old_graphs():
    cost = ForwardMemoryCost(100, 80, 10, correction_workspace_bytes=100)
    placement = placement_cost((cost,) * 3, backward_state="gpu", output_device="model")
    assert placement.gpu_required_bytes == 370
