"""Independent CPU checks: real profile/update/admission; no GPU allocations."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import pickle
import random
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import _impl as tr
from art.trainer_rank import _memory as m


def sig(cp):
    return m._ScopedMemorySignature(
        topology=(1, 1, cp, 1),
        planner_coefficients=(2, "independent"),
        slot_group_count=1,
        request_mix=("target:single",),
        grad_enabled=True,
        grad_modes=(True,),
        bounded_caller_peak=True,
    )


def plan(s, n):
    return SimpleNamespace(
        signature=s, packed_tokens=n, active_logical_tokens=n, output_bytes=0
    )


def add(r, s, n, p, complete):
    m._update_memory_profile(
        r, plan(s, n), p, retained_bytes=None, caller_phase=complete
    )


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_adversarial_random_histories(cp):
    checks = 0
    for seed in range(12):
        rng = random.Random(seed)
        r = SimpleNamespace(_memory_profiles={})
        s = sig(cp)
        extent = rate = floor = 0
        observations = [
            (128, 1_075_000_000, True),
            (8192, 4_000_000_000, False),
            (8192, 4_000_000_000, True),
            (128, 12_000_000_000, True),
            (32768, 20_000_000_000, False),
        ]
        observations += [
            (
                rng.choice((64, 128, 512, 2048, 8192, 9216, 32768, 65536)),
                rng.randrange(1_000_000_000, 40_000_000_000),
                bool(rng.randrange(2)),
            )
            for _ in range(80)
        ]
        for n, p, complete in observations:
            before = r._memory_profiles.get(s)
            add(r, s, n, p, complete)
            rate = max(rate, p / n)
            floor = max(floor, p)
            if complete:
                extent = max(extent, n)
            profile = r._memory_profiles[s]
            assert profile.bytes_per_token == rate
            assert profile.compute_peak_floor == floor
            assert (
                max((n for n, _ in profile.caller_compute_peaks), default=0) == extent
            )
            for length in (64, 128, 1024, 8192, 9216, 32768, 65536, 131072):
                estimate = profile.compute_peak(length)
                assert estimate >= floor
                if length > extent:
                    assert estimate >= rate * length
                if not complete and before:
                    assert estimate >= before.compute_peak(length)
                checks += 1
            r._memory_profiles = pickle.loads(pickle.dumps(r._memory_profiles))
            assert replace(s, topology=(1, 1, cp * 2, 1)) not in r._memory_profiles
    assert checks == 12 * 85 * 8


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_exact_global_length_and_uneven_floor(cp):
    from test_trainer_rank_split_peak import _rank, _requests

    from art.trainer_rank import _micro_batch_planner as planner

    r = _rank()
    p = r._plan_flat_forward(_requests(1, 9216))
    assert p.packed_tokens == p.logical_tokens == 9216
    p = replace(p, signature=sig(cp))
    loads = tuple([9216 - (cp - 1) * 128] + [128] * (cp - 1))
    r._topology = lambda: ParallelTopology(cp=cp)
    r._max_rank_model_tokens = lambda batch, topology: max(loads)
    assert planner._plan_group_rows(r, p) == ((max(loads), True),)
    assert r._physical_tokens(9217) == 9217  # TP1 does not divide by CP.


def collective_worker(rank, cp, init_path, out):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=cp,
        timeout=timedelta(seconds=60),
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        r = cast(
            Any,
            SimpleNamespace(
                _memory_profiles={}, device=torch.device("cpu"), _peak_resets=1
            ),
        )
        s = sig(cp)
        r._update_memory_profile = MethodType(m._update_memory_profile, r)
        r._forward_memory_group = lambda: dist.group.WORLD
        r._forward_collective_device = lambda: "cpu"
        r._available_memory_bytes = lambda: 30_000_000_000
        add(r, s, 128, 1_000_000_000, True)
        # Unequal local peaks, same global complete coverage.
        monkeypatch.setattr(
            torch.cuda,
            "max_memory_allocated",
            lambda device: 4_000_000_000 + rank * 100_000_000,
        )
        m._update_peak_memory_profile(
            r, plan(s, 8192), 0, caller_phase=True, interval=(1, 1)
        )
        local = r._memory_profiles[s].compute_peak(8192)
        assert local == 4_000_000_000 + rank * 100_000_000  # no raw peak MAX
        check = m._memory_check_required(r, int(local))
        assert (
            check.fits
            and check.estimated_required_bytes == 4_000_000_000 + (cp - 1) * 100_000_000
        )
        # On one rank a reset invalidates the whole caller reading.
        if rank == cp - 1:
            r._peak_resets += 1
        monkeypatch.setattr(
            torch.cuda,
            "max_memory_allocated",
            lambda device: 5_000_000_000 + rank * 100_000_000,
        )
        m._update_peak_memory_profile(
            r, plan(s, 32768), 0, caller_phase=True, interval=(1, 1)
        )
        profile = r._memory_profiles[s]
        extent = max(n for n, _ in profile.caller_compute_peaks)
        assert extent == (8192 if rank == cp - 1 else 32768)
        local = int(profile.compute_peak(32768))
        check = m._memory_check_required(r, local)
        assert not check.fits
        assert check.estimated_required_bytes == 256_000_000_000
        # With unequal budgets every rank still must fit its own requirement.
        r._available_memory_bytes = lambda: local - 1 if rank == cp - 1 else local + 1
        assert not m._memory_check_required(r, local).fits
        (Path(out) / f"collective-cp{cp}-rank{rank}.json").write_text(
            json.dumps(
                {
                    "cp": cp,
                    "rank": rank,
                    "complete_extent": extent,
                    "local_requirement": local,
                    "reduced_requirement": check.estimated_required_bytes,
                    "fits": check.fits,
                }
            )
        )
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_asymmetric_interval_and_admission(cp, tmp_path):
    mp.start_processes(
        collective_worker,
        args=(cp, str(tmp_path / "gloo"), str(tmp_path)),
        nprocs=cp,
        join=True,
        start_method="fork",
    )
