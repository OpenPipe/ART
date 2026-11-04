import subprocess

import pytest

from art import vllm_runtime

OLD_SMI = (
    "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0     |"
)
NEW_SMI = (
    "| NVIDIA-SMI 617.02.01   KMD Version: 617.02.01   CUDA UMD Version: 13.4     |"
)
CUDA12_SMI = (
    "| NVIDIA-SMI 570.86.15   Driver Version: 570.86.15   CUDA Version: 12.8     |"
)


def fake_run(outputs):
    def run(command, **kwargs):
        name = "nvcc" if "nvcc" in command[0] else command[0]
        if name not in outputs:
            raise FileNotFoundError(name)
        return subprocess.CompletedProcess(command, 0, stdout=outputs[name], stderr="")

    return run


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ({"nvidia-smi": OLD_SMI}, "cuda13"),
        ({"nvidia-smi": NEW_SMI}, "cuda13"),
        ({"nvidia-smi": CUDA12_SMI}, "cuda12"),
        ({"nvcc": "Cuda compilation tools, release 13.0, V13.0.48"}, "cuda13"),
        ({}, "cuda12"),
    ],
)
def test_managed_runtime_extra_reads_cuda_version(monkeypatch, outputs, expected):
    monkeypatch.delenv("ART_VLLM_RUNTIME_CUDA_PROFILE", raising=False)
    monkeypatch.setattr(vllm_runtime.subprocess, "run", fake_run(outputs))
    assert vllm_runtime._managed_runtime_extra() == expected
