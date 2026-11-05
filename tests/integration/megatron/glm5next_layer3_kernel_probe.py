"""Sealable, bounded saved-input kernel check. No model load or new parity gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import ModuleType


def worker(seal: dict, output: Path, recording: bool) -> None:
    from safetensors.torch import load_file, save_file
    import torch

    from art.megatron.glm5next.kpool_indexer import _pool_scores, compress_keys
    from tests.integration.megatron.glm5next_layer3_capture import (
        sampler_fp8,
        sampler_pool_scores,
    )

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cpu" if recording else "cuda"
    kernel = None
    if not recording:
        # Execute the exact inert image kernel bytes with the local runtime.
        # Only its triton import is adapted; no vLLM engine is constructed.
        import triton
        import triton.language as tl

        shim = ModuleType("vllm.triton_utils")
        setattr(shim, "triton", triton)
        setattr(shim, "tl", tl)
        sys.modules["vllm.triton_utils"] = shim
        spec = importlib.util.spec_from_file_location(
            "audited_kpool", seal["sampler_kernel"]
        )
        assert spec is not None and spec.loader is not None
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
    rows = []
    for path in seal["probe_inputs"]:
        data = load_file(path, device=device)
        outputs = {}
        item = {"input": path, "recording": recording, "gpu_measured": not recording}
        for side in ("native", "hf"):
            q, k, w, g = [
                data[side + "." + key] for key in ("q", "keys", "weights", "gates")
            ]
            # Glm52SelfAttention.forward is compiler-disabled; compression and
            # latent RMSNorm keep eager casts even inside compiled layers.
            pools = compress_keys(
                k, g, data["ape"], torch.arange(len(k) // 4, device=device) * 4
            )
            scores = _pool_scores(q, pools, w)
            visible = (
                torch.arange(len(pools), device=device)[None] * 4 + 3
                <= data["positions"][:, None]
            )
            scores.masked_fill_(~visible, -torch.inf)
            outputs[side + ".scores"] = scores.cpu()
            outputs[side + ".pool_ids"] = scores.topk(512).indices.cpu()
            if not bool(torch.isfinite(scores[visible]).all()):
                raise ValueError("Nonfinite native/HF score")
        q, k, g = [data["hf." + key] for key in ("q", "keys", "gates")]
        reference = sampler_pool_scores(
            q.cpu(), k.cpu(), data["sampler.weights"].cpu(), g.cpu(), data["ape"].cpu()
        )
        reference.masked_fill_(~visible.cpu(), -torch.inf)
        outputs["sampler.reference_scores"] = reference
        if kernel is not None:
            q8, qs = kernel.fwht128_quant_fp8(q.reshape(-1, 128).contiguous())
            count = len(k) // 4
            cache = torch.empty((1, 64, 132), dtype=torch.uint8, device=device)
            k8, ks = kernel.kpool_compress_and_write_cache(
                cache,
                k[: count * 4].reshape(count, 4, 128),
                g[: count * 4].reshape(count, 4, 128),
                data["ape"],
                torch.arange(count, device=device),
                pool_size=4,
                return_compressed=True,
                write_cache=False,
            )
            qref, sref = sampler_fp8(q.cpu())
            item["query_fp8_value_mismatches"] = int(
                (q8.cpu().float().reshape_as(qref) != qref.float()).sum()
            )
            item["query_scale_max_abs_delta"] = float(
                (qs.cpu().reshape_as(sref) - sref).abs().max()
            )
            # This is a score equation check using actual image quantization
            # kernels, not an execution of DeepGEMM or a full vLLM forward.
            dots = (
                (q8.float().reshape_as(q) @ k8.float().T)
                * qs.reshape(*q.shape[:2], 1)
                * ks[None, None]
            )
            scores = (
                dots.relu() * data["sampler.weights"][..., None] * (32 * 128) ** -0.5
            ).sum(1)
            scores.masked_fill_(~visible, -torch.inf)
            outputs["sampler.kernel_scores"] = scores.cpu()
            outputs["sampler.kernel_pool_ids"] = scores.topk(512).indices.cpu()
            if not bool(torch.isfinite(scores[visible]).all()):
                raise ValueError("Nonfinite sampler score")
        save_file(outputs, str(output / (Path(path).stem + "-scores.safetensors")))
        rows.append(item)
    (output / "result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "rows": rows,
                "cuda_initialized": torch.cuda.is_initialized(),
                "policy": "finite blocking; all score/set/margin differences report-only; historical parity FAILs unchanged",
                "limitations": "Local torch/Triton and CPU-prepared projections; not the full image, DeepGEMM, or end-to-end trainer/sampler parity",
            },
            indent=2,
        )
        + "\n"
    )
    if recording:
        assert not torch.cuda.is_initialized()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--approved-seal", required=True)
    parser.add_argument("--controller-owner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.seal.read_text()), args.output, args.recording)
        return
    if not args.execute:
        parser.error("--execute required (add --recording for CPU only)")
    from tests.integration.megatron.glm5next_qualification_launch import (
        BOARD,
        claim_controller,
        has_grant,
        idle,
        preflight,
        run,
        sha,
        write,
    )

    started = time.monotonic()
    seal = preflight(args.seal, args.approved_seal)
    if not args.recording and not has_grant(
        seal, args.approved_seal, BOARD.read_text()
    ):
        raise ValueError("Fresh literal exact-seal workstation GRANT required")
    args.output.mkdir(parents=True, exist_ok=False)
    claim = claim_controller(
        seal,
        args.approved_seal,
        args.controller_owner,
        (args.output if args.recording else args.seal.parent) / "claims",
        args.output,
    )
    source = args.output / "source"
    with tarfile.open(seal["source_archive"]) as archive:
        archive.extractall(source, filter="data")
    if sha(Path(seal["source_archive"])) != seal["source_sha256"]:
        raise ValueError("Source archive changed")
    receipt = {
        "status": "fail",
        "recording": args.recording,
        "seal_sha256": args.approved_seal,
        "claim": claim,
        "gpu_measured": False,
        "cleanup_done": False,
    }
    try:
        if not args.recording:
            for check in range(3):
                idle(seal["uuids"])
                if check < 2:
                    time.sleep(10)
        for gpu in seal["physical_gpus"]:
            out = args.output / f"gpu{gpu}"
            out.mkdir()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="" if args.recording else str(gpu),
                PYTHONPATH=str(source / "src") + os.pathsep + str(source),
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                PYTHON_DOTENV_DISABLED="1",
                TORCHINDUCTOR_CACHE_DIR=str(out / "inductor"),
                TRITON_CACHE_DIR=str(out / "triton"),
            )
            remaining = int(
                seal["window_seconds"]
                - seal["cleanup_reserve_seconds"]
                - (time.monotonic() - started)
            )
            if remaining <= 0:
                raise TimeoutError("No remaining probe time")
            argv = [
                seal["native_python"],
                "-m",
                "tests.integration.megatron.glm5next_layer3_kernel_probe",
                "--worker",
                "--seal",
                str(args.seal),
                "--approved-seal",
                args.approved_seal,
                "--controller-owner",
                args.controller_owner,
                "--output",
                str(out),
            ]
            if args.recording:
                argv.append("--recording")
            receipt[f"gpu{gpu}"] = run(
                argv,
                min(300, remaining),
                out / "worker.log",
                env=env,
                cwd=source,
                pss_limit=seal["pss_limit_bytes"],
            )
        receipt.update(status="pass", gpu_measured=not args.recording)
    finally:
        if not args.recording:
            idle(seal["uuids"])
        receipt.update(cleanup_done=True, seconds=time.monotonic() - started)
        write(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
