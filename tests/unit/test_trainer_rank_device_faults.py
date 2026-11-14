"""Device faults are fatal; only non-device caller samples are auxiliary."""

import pytest
from test_trainer_rank_caller_sampling import _iterator
from test_trainer_rank_split_peak import _counter_split
import torch

from art.trainer_rank import _impl
from art.trainer_rank import _memory as memory

DEVICE_CASES = ["runtime"]
DEVICE_CASES += [
    name
    for owner, name in (
        (torch.cuda, "OutOfMemoryError"),
        (torch, "AcceleratorError"),
        (torch.cuda, "CudaError"),
        (torch.cuda, "DeferredCudaCallError"),
    )
    if isinstance(getattr(owner, name, None), type)
]


def _error(name):
    if name == "runtime":
        return RuntimeError("CUDA error: illegal memory access")
    owner = torch if name == "AcceleratorError" else torch.cuda
    kind = getattr(owner, name)
    # CudaError's initializer calls cudart. Construct only its Python exception
    # object here; no runtime or device initialization is needed for this test.
    error = kind.__new__(kind)
    BaseException.__init__(error, "statistics unavailable")
    return error


@pytest.mark.parametrize("name", DEVICE_CASES)
def test_device_types_are_classified_without_message_hints(name):
    assert memory._is_device_fault(_error(name))


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: launch failed",
        "CUDA out of memory",
        "device-side assert triggered",
        "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUDA driver initialization failed",
        "Accelerator operation failed",
        "NCCL unhandled system error",
        "an illegal memory access was encountered",
        "unspecified launch failure",
        "HIP error: launch failed",
        "rocBLAS execution error",
        "device map lookup failed",  # Ambiguous device errors fail fast.
        "out of memory",  # An unqualified runtime OOM is ambiguous too.
        "OUT_OF_MEMORY",
        "XPU error: launch failed",
        "MPS backend allocation failed",
        "SYCL exception: execution failed",
        "Metal error: command buffer failed",
        "",
    ],
)
def test_runtime_device_signatures_are_fatal(message):
    assert memory._is_device_fault(RuntimeError(message))


@pytest.mark.parametrize(
    "error",
    [
        ValueError("sample unavailable"),
        TypeError("invalid profile field"),
        KeyError("peak"),
        RuntimeError("profile metadata unavailable"),
        RuntimeError("missing clock timestamps"),
        MemoryError("host metadata allocation failed"),
        KeyboardInterrupt(),
    ],
)
def test_non_device_exceptions_are_not_device_faults(error):
    assert not memory._is_device_fault(error)


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_wrapped_device_faults_are_fatal(link):
    outer = ValueError("profile helper failed")
    setattr(outer, link, _error("runtime"))
    assert memory._is_device_fault(outer)


def test_grouped_device_faults_are_fatal():
    assert memory._is_device_fault(
        ExceptionGroup("profile errors", [ValueError("metadata"), _error("runtime")])
    )


def test_classifier_handles_context_cycles():
    first, second = ValueError("metadata"), RuntimeError("sample unavailable")
    first.__context__, second.__context__ = second, first
    assert not memory._is_device_fault(first)
    second.__cause__ = _error("runtime")
    assert memory._is_device_fault(first)


def test_optional_torch_types_can_be_absent(monkeypatch):
    for owner, name in (
        (torch, "OutOfMemoryError"),
        (torch, "AcceleratorError"),
        (torch.cuda, "OutOfMemoryError"),
        (torch.cuda, "CudaError"),
        (torch.cuda, "DeferredCudaCallError"),
    ):
        monkeypatch.delattr(owner, name, raising=False)
    assert memory._is_device_fault(RuntimeError("CUDA driver error"))
    assert not memory._is_device_fault(ValueError("sample unavailable"))


def test_unknown_cuda_exception_type_is_fatal():
    class NewCudaError(Exception):
        __module__ = "torch.cuda.errors"

    assert memory._is_device_fault(NewCudaError("unavailable"))


def test_opaque_runtime_failure_is_fatal():
    class Opaque(RuntimeError):
        def __str__(self):
            raise ValueError("message unavailable")

    assert memory._is_device_fault(Opaque())


@pytest.mark.parametrize("name", DEVICE_CASES)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("stage", ["sample", "update"])
def test_device_fault_propagates_without_suppression(
    monkeypatch, caplog, name, split, stage
):
    if split:
        # This fixture seeds its admitted split using v1 counter profiles.
        # Enable collection only after its forward has yielded, before the
        # fault injection; caller failure handling is independent of version.
        rank, requests, _ = _counter_split(monkeypatch)
        iterator = rank.forward_micro_batches([requests], yield_empty=True)
        next(iterator)
    with memory.fixed_slope_memory_profiles():
        if not split:
            rank, _, iterator = _iterator(monkeypatch)
        profiles, floors, reading = (
            dict(rank._memory_profiles),
            dict(rank._split_memory_floors),
            rank._peak_reading,
        )
        error = _error(name)

        def fail(*args, **kwargs):
            raise error

        if stage == "sample":
            monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail)
        else:
            field = "_record_split_memory_floor" if split else "_update_memory_profile"
            original = getattr(rank, field)

            def update(*args, **kwargs):
                original(*args, **kwargs)
                raise error

            monkeypatch.setattr(rank, field, update)
        with pytest.raises(type(error)) as caught:
            next(iterator)
        assert caught.value is error
        assert rank._memory_profiles == profiles
        assert rank._split_memory_floors == floors
        assert rank._peak_reading == reading
        assert rank._suppressed_caller_memory_samples == 0
        assert not rank._caller_memory_sample_warning_logged
        assert not any("caller memory sample" in r.getMessage() for r in caplog.records)


def test_device_fault_during_warning_is_fatal(monkeypatch):
    rank, _, iterator = _iterator(monkeypatch)
    error = _error("runtime")

    def sample(*args, **kwargs):
        raise ValueError("sample unavailable")

    def warn(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", sample)
    monkeypatch.setattr(_impl.logger, "warning", warn)
    with pytest.raises(RuntimeError) as caught:
        next(iterator)
    assert caught.value is error
    assert rank._suppressed_caller_memory_samples == 0
    assert not rank._caller_memory_sample_warning_logged


def test_non_device_runtime_error_is_still_suppressed(monkeypatch):
    rank, _, iterator = _iterator(monkeypatch)

    def sample(*args, **kwargs):
        raise RuntimeError("profile metadata unavailable")

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", sample)
    assert list(iterator) == []
    assert rank._suppressed_caller_memory_samples == 1
    assert rank._caller_memory_sample_warning_logged
