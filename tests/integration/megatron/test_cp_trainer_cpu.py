"""CP regressions through native tiny GLM, TrainerRank, packing and Gloo plans."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("megatron.bridge")

from tests.integration.megatron.cp_trainer_compare import compare


def _probe(output: Path, cp: int) -> None:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHON_DOTENV_DISABLED": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={cp}",
                "--module",
                "tests.integration.megatron.cp_trainer_cpu",
                "--model",
                "glm",
                "--lengths",
                "503",
                "775",
                "--output",
                str(output),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )


@pytest.fixture(scope="module")
def cp1_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("trainer-cp") / "cp1"
    _probe(path, 1)
    return path


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_glm_trainer_cp_alignment_preserves_logprobs(
    cp1_reference: Path, cp: int
) -> None:
    output = cp1_reference.parent / f"cp{cp}"
    _probe(output, cp)
    results = compare(cp1_reference, output)
    # Include every KDA/DSA/MoE/mHC layer capture as well as target logprobs.
    # Tied top-k ordering may permute the same selected set; outputs must agree.
    outputs = {
        name: row for name, row in results.items() if not name.endswith("indices")
    }
    assert all(row["close"] for row in outputs.values()), outputs
