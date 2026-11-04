"""ctypes wrapper with per-call AlgoCheck and pointer-capability validation."""

import ctypes as C
import hashlib
from pathlib import Path


def describe(t):
    return dict(
        shape=list(t.shape), stride=list(t.stride()), alignment=t.data_ptr() % 256
    )


class Unsupported(RuntimeError):
    pass


class Binding:
    def __init__(self, library):
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        sigs = {
            "lt_new": ([C.c_int] * 7 + [C.POINTER(C.c_void_p)], C.c_int),
            "lt_delete": ([C.c_void_p], None),
            "lt_discover": (
                [C.c_void_p, C.c_size_t, C.c_void_p, C.POINTER(C.c_int)],
                C.c_int,
            ),
            "lt_nosplit": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
            "lt_config": ([C.c_void_p, C.c_int, C.POINTER(C.c_int)], C.c_int),
            "lt_check": ([C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
            "lt_alignment": ([C.c_void_p, C.c_int, C.POINTER(C.c_int)], C.c_int),
            "lt_run": ([C.c_void_p] * 6 + [C.c_size_t, C.c_void_p], C.c_int),
            "lt_version": ([], C.c_size_t),
            "lt_abi_bytes": ([], C.c_int),
        }
        for name, (args, result) in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result
        if self.lib.lt_abi_bytes() != 64 or self.lib.lt_version() != 120804:
            raise Unsupported(
                "requires cuBLASLt 12.8.4 / 64-byte descriptor ABI; no fallback"
            )

    def checked(self, status, where):
        if status:
            raise Unsupported(f"{where}: cuBLASLt status {status}; no fallback")

    def context(self, a, b):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise Unsupported("expected [M,K] @ [K,N]")
        if (
            a.dtype != b.dtype
            or str(a.dtype) not in ("torch.bfloat16", "torch.float32")
            or a.device != b.device
        ):
            raise Unsupported("matching CUDA BF16/FP32 operands required")
        if a.device.type != "cuda":
            raise Unsupported("CUDA operands required")
        if a.stride(1) != 1:
            raise Unsupported("A columns must have unit stride")
        trans = b.stride(0) == 1
        if not trans and b.stride(1) != 1:
            raise Unsupported("unsupported B strides")
        ctx = C.c_void_p()
        status = self.lib.lt_new(
            len(a),
            b.shape[1],
            a.shape[1],
            int(str(a.dtype) == "torch.bfloat16"),
            int(trans),
            a.stride(0),
            b.stride(1) if trans else b.stride(0),
            C.byref(ctx),
        )
        if status:
            self.lib.lt_delete(ctx)
            self.checked(status, "descriptor creation")
        return ctx

    def discover(self, a, b, workspace_bytes=8 << 20):
        ctx = self.context(a, b)
        try:
            buf = (C.c_ubyte * (64 * 4))()
            count = C.c_int()
            self.checked(
                self.lib.lt_discover(ctx, workspace_bytes, buf, C.byref(count)),
                "discovery",
            )
            out = []
            for i in range(count.value):
                algo = (C.c_ubyte * 64).from_buffer_copy(
                    bytes(buf)[i * 64 : (i + 1) * 64]
                )
                accepted = C.c_int()
                if self.lib.lt_nosplit(algo, C.byref(accepted)):
                    continue
                required = C.c_size_t()
                if (
                    self.lib.lt_check(ctx, algo, C.byref(required))
                    or required.value > workspace_bytes
                ):
                    continue
                out.append(bytes(algo))
            return out
        finally:
            self.lib.lt_delete(ctx)

    def config(self, raw, dtype):
        buf = (C.c_ubyte * 64).from_buffer_copy(raw)
        out = {}
        for i, name in enumerate(
            [
                "id",
                "tile",
                "split_k",
                "reduction_scheme",
                "cta_swizzle",
                "custom",
                "stages",
                "inner_shape",
                "cluster_shape",
            ]
        ):
            v = C.c_int()
            status = self.lib.lt_config(buf, i, C.byref(v))
            out[name] = {"status": status, "value": v.value if status == 0 else None}
        assert out["split_k"]["value"] == 1 and out["reduction_scheme"]["value"] == 0
        out.update(
            opaque_hex=raw.hex(),
            sha256=hashlib.sha256(raw).hexdigest(),
            compute="CUBLAS_COMPUTE_32F"
            if dtype == "bfloat16"
            else "CUBLAS_COMPUTE_32F_PEDANTIC",
            scalar="FP32",
            epilogue="DEFAULT",
            cublasLt_version=self.lib.lt_version(),
        )
        return out

    def run(self, a, b, raw, workspace):
        import torch

        ctx = self.context(a, b)
        algo = (C.c_ubyte * 64).from_buffer_copy(raw)
        try:
            required = C.c_size_t()
            self.checked(self.lib.lt_check(ctx, algo, C.byref(required)), "AlgoCheck")
            if required.value > workspace.numel():
                raise Unsupported("workspace exceeds pinned limit")
            out = torch.empty((len(a), b.shape[1]), dtype=a.dtype, device=a.device)
            for i, tensor in enumerate([a, b, out, out]):
                alignment = C.c_int()
                self.checked(
                    self.lib.lt_alignment(algo, i, C.byref(alignment)),
                    "alignment capability",
                )
                if alignment.value <= 0 or tensor.data_ptr() % alignment.value:
                    raise Unsupported(
                        f"operand {i} pointer violates {alignment.value}-byte alignment"
                    )
            self.checked(
                self.lib.lt_run(
                    ctx,
                    algo,
                    a.data_ptr(),
                    b.data_ptr(),
                    out.data_ptr(),
                    workspace.data_ptr(),
                    workspace.numel(),
                    torch.cuda.current_stream().cuda_stream,
                ),
                "Matmul",
            )
            return out, dict(
                algo_check="PASS",
                workspace_required=required.value,
                operands=[describe(a), describe(b), describe(out)],
            )
        finally:
            self.lib.lt_delete(ctx)
