"""Check dense and CP-stage sparse attention/LSE gradients on a native GPU."""

import json
import math

import torch
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.functional as F

from art.megatron.flex_attn.compiled import (
    flex_backend_for_head_dims,
    get_sparse_compiled_flex_attention,
)


def main():
    torch.manual_seed(921)
    for dim in (128, 256):
        for length in (128, 512):
            q = (
                torch.randn(1, 24, length, dim, device="cuda", dtype=torch.bfloat16)
                * 0.2
            ).requires_grad_()
            k = torch.randn(
                1,
                4,
                length,
                dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            v = torch.randn_like(k, requires_grad=True)
            refq, refk, refv = [x.detach().float().requires_grad_() for x in (q, k, v)]
            scale = 1 / math.sqrt(dim)
            mask = create_block_mask(
                lambda b, h, q, k: q >= k,
                B=1,
                H=1,
                Q_LEN=length,
                KV_LEN=length,
                device="cuda",
            )
            backend = flex_backend_for_head_dims(
                head_dim=dim, head_dim_v=dim, device=q.device
            )
            kernel = get_sparse_compiled_flex_attention(
                family_key="check",
                backend=backend,
                head_dim=dim,
                head_dim_v=dim,
                device=q.device,
            )
            out, lse = kernel(q, k, v, block_mask=mask, scale=scale, enable_gqa=True)
            loss = out.float().square().mean() + lse.mean() * 0.01
            grads = torch.autograd.grad(loss, (q, k, v))
            score = refq @ refk.repeat_interleave(6, 1).transpose(-1, -2) * scale
            score = score.masked_fill(
                torch.ones(length, length, device="cuda", dtype=torch.bool).triu(1),
                float("-inf"),
            )
            reflse = score.logsumexp(-1)
            refout = score.softmax(-1) @ refv.repeat_interleave(6, 1)
            refgrads = torch.autograd.grad(
                refout.square().mean() + reflse.mean() * 0.01, (refq, refk, refv)
            )
            metrics = []
            for value, reference in zip(
                (out, lse, *grads), (refout, reflse, *refgrads), strict=True
            ):
                a, b = (
                    value.detach().float().flatten(),
                    reference.detach().float().flatten(),
                )
                relative = float((a - b).norm() / b.norm())
                cosine = float(F.cosine_similarity(a, b, dim=0))
                assert relative < 0.02 and cosine > 0.9995, (
                    dim,
                    length,
                    relative,
                    cosine,
                )
                metrics.append(dict(relative_l2=relative, cosine=cosine))
            print(
                json.dumps(
                    dict(head_dim=dim, length=length, backend=backend, metrics=metrics)
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
