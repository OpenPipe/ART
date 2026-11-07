"""Apply the companion patch to a temporary copy and check real encode/decode/merge."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from art.trainer_rank import _impl, _memory, _memory_model

OUT = Path(__file__).parent
SOURCE = OUT.parents[2] / "caladan/src/caladan/stark/trainer/calibration.py"


def profile(cls, points):
    hull = _memory_model.merge_points(points)
    fixed, slope = _memory_model.fit_points(hull)
    return cls(
        bytes_per_token=max(y / n for n, y in points),
        packed_tokens=max(n for n, y in points),
        peak_model_version=2,
        compute_observations=hull,
        fixed_compute_bytes=fixed,
        compute_slope=slope,
    )


def main():
    checked = 0
    with tempfile.TemporaryDirectory(prefix="art-memory-caladan-") as temp:
        path = Path(temp) / "src/caladan/stark/trainer/calibration.py"
        path.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, path)
        subprocess.run(
            [
                "patch",
                "-p1",
                "-d",
                temp,
                "-i",
                str((OUT / "caladan-v4.patch").resolve()),
            ],
            check=True,
        )
        spec = importlib.util.spec_from_file_location(
            "calibration_fixed_slope_check", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for cls in (
            _impl._MemoryProfile,
            _memory._CallerMemoryProfile,
            _memory._BoundedMemoryProfile,
        ):
            signature = _memory._ScopedMemorySignature(
                topology=(1, 1, 8, 1),
                planner_coefficients=(1, None),
                slot_group_count=1,
                request_mix=("target:single",),
                grad_enabled=True,
                grad_modes=(True,),
            )
            a = profile(cls, ((1, 1000), (8192, 2000)))
            b = profile(cls, ((128, 1200), (32768, 3000)))
            rows = module.encode({signature: a})
            restored = module.decode(json.loads(json.dumps(rows)))
            assert restored == {signature: a}
            checked += 1
            module.merge(restored, {signature: b})
            merged = restored[signature]
            reverse = {signature: b}
            module.merge(reverse, {signature: a})
            assert reverse[signature] == merged
            checked += 1
            assert merged.compute_slope == (3000, 32768)
            for n, peak in (*a.compute_observations, *b.compute_observations):
                assert _memory._profile_compute_peak(merged, n) >= peak
            checked += 1
            old_rows = module.encode({signature: _impl._MemoryProfile(100, 8192)})
            for key in [
                "peak_model_version",
                "compute_observations",
                "fixed_compute_bytes",
                "compute_slope",
                "legacy_bytes_per_token",
            ]:
                del old_rows[0]["profile"][key]
            assert module.decode(old_rows)[signature].peak_model_version == 1
            checked += 1
            corrupt = json.loads(json.dumps(rows))
            corrupt[0]["profile"]["compute_slope"] = [1, 0]
            try:
                module.decode(corrupt)
            except ValueError:
                pass
            else:
                raise AssertionError("corrupt slope accepted")
            checked += 1
        assert '"version": 4' in path.read_text()
        checked += 1
    result = {
        "passed": checked,
        "source_modified": False,
        "checks": "real Caladan patch round-trip, order-independent merge, observation coverage, v3 field defaults, invalid slope rejection, v4 key",
    }
    (OUT / "caladan-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
