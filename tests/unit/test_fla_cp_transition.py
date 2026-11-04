"""Compile the review's resource repro and interpret its large-shard offsets.

Neither check launches a GPU kernel or allocates the multi-gigabyte shard.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("fla.ops.utils.op")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.interpreter import (  # noqa: E402
    InterpretedFunction,
    InterpreterError,
    TensorHandle,
    interpreter_builder,
)

from art.megatron.gdn import fla_cp_kernels as kernels  # noqa: E402


def _capture_launch(monkeypatch, width, arch, shared_limit, packed, reverse):
    calls = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, kwargs))

            return launch

    monkeypatch.setattr(kernels, "_summary_transition_kernel", Capture())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=arch // 10,
            minor=arch % 10,
            shared_memory_per_block_optin=shared_limit,
        ),
    )
    key = torch.empty(1, 64, 48, width)
    kernels.fill_summary_transition(
        torch.empty(1, 48, width, 37 + width),
        k=key,
        w=key,
        g=torch.empty(1, 64, 48),
        cu_seqlens=torch.tensor([0, 64]) if packed else None,
        reverse=reverse,
    )
    ((grid, kwargs),) = calls
    return grid, kwargs


@pytest.mark.parametrize(
    "arch,shared_limit",
    [
        (90, 232448),
        (100, 232448),
        (103, 232448),
        (120, 101376),
        (90, 100352),
        (120, 100352),
    ],
)
@pytest.mark.parametrize("width", [16, 64, 128, 256])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_transition_launch_fits_device(
    monkeypatch, arch, shared_limit, width, packed, reverse
):
    kernel = kernels._summary_transition_kernel
    grid, kwargs = _capture_launch(
        monkeypatch, width, arch, shared_limit, packed, reverse
    )
    assert grid == (triton.cdiv(width, kwargs["BC"]), 48, 1)
    constants = {name: kwargs[name] for name in ("BK", "BC", "IS_VARLEN", "REVERSE")}
    if "BT" in kwargs:
        constants["BT"] = kwargs["BT"]
    source = ASTSource(
        kernel,
        signature={
            "K_ptr": "*bf16",
            "W_ptr": "*bf16",
            "G_ptr": "*fp32",
            "Summary": "*fp32",
            "Cu": "*i64",
            "T": "i32",
        },
        constexprs={"H": 48, "K": width, "V": 37, **constants},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(5)},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", arch, 32),
        options={
            name: value for name, value in kwargs.items() if name.startswith("num_")
        },
    )
    assert compiled.metadata.shared <= shared_limit


def _interpret_kernel(monkeypatch):
    # Nested JIT helpers must use the interpreter too, without setting global
    # TRITON_INTERPRET (the resource checks above need the real compiler).
    monkeypatch.setattr(tl, "cdiv", InterpretedFunction(tl.cdiv.fn))
    monkeypatch.setattr(kernels, "exp", InterpretedFunction(kernels.exp.fn))
    return InterpretedFunction(kernels._summary_transition_kernel.fn)


@pytest.mark.parametrize("packed_dtype", [None, torch.int32, torch.int64])
def test_large_shard_key_offsets_do_not_overflow(monkeypatch, packed_dtype):
    # Reverse traversal starts at token 349504, so row 22 is the review repro.
    # Stop before dereferencing the first key load, after real kernel arithmetic.
    key = torch.empty(1)
    pointers = []
    original = interpreter_builder.create_masked_load

    class Captured(Exception):
        pass

    def capture(ptrs, *args):
        if ptrs.data.ndim == 2:
            pointers.append(ptrs.data.copy())
            raise Captured
        if ptrs.get_element_ty() == tl.float32:
            return TensorHandle(np.zeros_like(ptrs.data, dtype=np.float32), tl.float32)
        return original(ptrs, *args)

    monkeypatch.setattr(interpreter_builder, "create_masked_load", capture)
    kernel = _interpret_kernel(monkeypatch)
    kwargs = dict(
        H=48,
        K=128,
        V=37,
        BK=128,
        BC=32,
        IS_VARLEN=packed_dtype is not None,
        REVERSE=True,
    )
    if "BT" in kernel.arg_names:
        kwargs["BT"] = 64
    with pytest.raises(InterpreterError, match="Captured"):
        kernel[(1, 1, 1)](
            key,
            key,
            key,
            key,
            None
            if packed_dtype is None
            else torch.tensor([0, 349568], dtype=packed_dtype),
            349568,
            **kwargs,
        )
    (addresses,) = pointers
    actual = (
        int(addresses.astype(np.int64)[22, 0]) - key.data_ptr()
    ) // key.element_size()
    assert actual == 2147487744, f"token 349526, H=48, K=128: {actual}"


@pytest.mark.parametrize("width", [1, 8, 16, 31, 64, 128, 256])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("token_block", [16, 32, 64])
def test_transition_matches_fp64(monkeypatch, width, packed, reverse, token_block):
    """The review's nonzero updates, partial chunks and both directions on CPU."""
    generator = torch.Generator().manual_seed(7123)
    lengths = [1, 63, 64, 65, 129] if packed else [193]
    total, heads, value_dim = sum(lengths), 2, 37
    key, wy = [
        (torch.randn(1, total, heads, width, generator=generator) * 0.07).bfloat16()
        for _ in range(2)
    ]
    gates = torch.zeros(1, total, heads)
    cursor = 0
    for length in lengths:
        for start in range(0, length, 64):
            end = min(start + 64, length)
            gates[:, cursor + start : cursor + end] = (
                -torch.rand(1, end - start, heads, generator=generator) * 0.003
            ).cumsum(1)
        cursor += length
    summary = torch.full((len(lengths), heads, width, value_dim + width), -222.0)
    cu = (
        torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]) if packed else None
    )
    kernel = _interpret_kernel(monkeypatch)
    kwargs = dict(
        H=heads,
        K=width,
        V=value_dim,
        BK=max(16, triton.next_power_of_2(width)),
        BC=32,
        IS_VARLEN=packed,
        REVERSE=reverse,
    )
    if "BT" in kernel.arg_names:
        kwargs["BT"] = token_block
    kernel[(triton.cdiv(width, 32), heads, len(lengths))](
        key, wy, gates, summary, cu, total, **kwargs
    )
    assert (summary[..., :value_dim] == -222.0).all()
    key, wy, gates = key.double()[0], wy.double()[0], gates.double()[0]
    cursor = 0
    for sequence, length in enumerate(lengths):
        reference = torch.eye(width).expand(heads, width, width).double().clone()
        chunks = list(range(0, length, 64))
        for start in reversed(chunks) if reverse else chunks:
            end = min(start + 64, length)
            last = gates[cursor + end - 1]
            decayed_key = (
                key[cursor + start : cursor + end]
                * (last - gates[cursor + start : cursor + end]).exp()[..., None]
            )
            wk = wy[cursor + start : cursor + end]
            product = (
                torch.einsum("thk,thv->hkv", wk, decayed_key)
                if reverse
                else torch.einsum("thk,thv->hkv", decayed_key, wk)
            )
            step = last.exp()[:, None, None] * torch.eye(width) - product
            reference = step @ reference
        torch.testing.assert_close(
            summary[sequence, ..., value_dim:].double(), reference, atol=1e-6, rtol=3e-5
        )
        cursor += length
