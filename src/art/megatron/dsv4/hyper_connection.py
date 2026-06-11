from typing import Any, cast

import einops
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
import torch
from torch import Tensor
import torch.nn.functional as F


def _unweighted_rms_norm(x: Tensor, eps: float) -> Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class HCHeadParams(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        cfg = cast(Any, config)
        hc_mult = int(cfg.dsv4_hc_mult)
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = torch.nn.Parameter(
            torch.empty(hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))
        self._keep_fp32_parameters = (
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        )
        for param in (self.hc_head_fn, self.hc_head_base, self.hc_head_scale):
            setattr(param, "_keep_fp32", True)

    def forward(self):
        raise NotImplementedError


class DeepSeekV4HyperConnectionUtil:
    """DeepSeek-V4 manifold-constrained hyper-connection math.

    This implements the HF reference equations directly in PyTorch. TileKernels
    MHC currently requires a newer CUDA toolchain than this ART Megatron env
    provides, so production training keeps the exact eager math here.
    """

    def __init__(self, config: TransformerConfig):
        cfg = cast(Any, config)
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = int(cfg.dsv4_hc_mult)
        self.hc_sinkhorn_iters = int(cfg.dsv4_hc_sinkhorn_iters)
        self.hc_eps = float(cfg.dsv4_hc_eps)

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = x.dtype
        hc = self.hc_mult
        flat = _unweighted_rms_norm(x.flatten(start_dim=2).float(), self.norm_eps)
        pre_w, post_w, comb_w = F.linear(flat, hc_fn.float()).split(
            [hc, hc, hc * hc], dim=-1
        )
        pre_b, post_b, comb_b = hc_base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = hc_scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(
            *comb_w.shape[:-1], hc, hc
        ) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        layer_input = (pre.unsqueeze(-1) * x).sum(dim=2).to(dtype)
        return layer_input, post, comb

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        dtype = residual.dtype
        return post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        dtype = x.dtype
        flat = _unweighted_rms_norm(x.flatten(start_dim=2).float(), self.norm_eps)
        mixes = F.linear(flat, hc_fn.float())
        pre = (
            torch.sigmoid(mixes * hc_scale.float().reshape(1) + hc_base.float())
            + self.hc_eps
        )
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(dtype)

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x, post, comb = self.hc_pre_raw(
            x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base
        )
        return einops.rearrange(x, "b s d -> s b d"), post, comb

    def layer_post(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        out = einops.rearrange(out, "s b d -> b s d")
        residual_bshd = einops.rearrange(residual, "s b hc d -> b s hc d")
        hidden_states = self.hc_post_raw(
            x=out, residual=residual_bshd, post=post, comb=comb
        )
        return einops.rearrange(hidden_states, "b s hc d -> s b hc d")

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        return einops.repeat(hidden_states, "s b d -> s b hc d", hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        x = einops.rearrange(hidden_states, "s b hc d -> b s hc d")
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return einops.rearrange(x, "b s d -> s b d")
