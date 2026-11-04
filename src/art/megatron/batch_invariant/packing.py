"""Canonical per-sequence reduction geometry at the FLA recurrence boundary."""

import torch


def canonical_contiguous(tensor):
    # contiguous() can preserve a packed parent stride on singleton dimensions.
    return tensor.contiguous().view(tensor.shape)


def root_local_gdn(
    function,
    q,
    k,
    v,
    *,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    **kwargs,
):
    """Use B=1 dense FLA for each segment, including its captured backward.

    FLA's scalar cumsum autotunes on B and IS_VARLEN. Correct segment-relative
    chunk indices alone do not keep that reduction tree fixed. Never call this
    for CP state exchange: that path has separate collective semantics.
    """
    if cu_seqlens is None and q.shape[0] == 1:
        return function(
            canonical_contiguous(q),
            canonical_contiguous(k),
            canonical_contiguous(v),
            g=canonical_contiguous(g),
            beta=canonical_contiguous(beta),
            initial_state=None
            if initial_state is None
            else canonical_contiguous(initial_state),
            **kwargs,
        )
    if cu_seqlens is None:
        parts = [(b, 0, q.shape[1]) for b in range(q.shape[0])]
        dim = 0
    else:
        if q.shape[0] != 1:
            raise ValueError("packed GDN requires batch-one streams")
        cu = cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu
        bounds = cu.detach().cpu().tolist()
        if (
            bounds[0] != 0
            or bounds[-1] != q.shape[1]
            or any(b <= a for a, b in zip(bounds[:-1], bounds[1:], strict=True))
        ):
            raise ValueError("GDN boundaries must cover nonempty segments")
        parts = [(0, a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        dim = 1
    outputs, finals = [], []
    for i, (b, start, stop) in enumerate(parts):

        def local(t):
            return canonical_contiguous(t[b : b + 1, start:stop])

        out, final = function(
            local(q),
            local(k),
            local(v),
            g=local(g),
            beta=local(beta),
            initial_state=None
            if initial_state is None
            else canonical_contiguous(initial_state[i : i + 1]),
            **kwargs,
        )
        outputs.append(out)
        if final is not None:
            finals.append(final)
    return torch.cat(outputs, dim=dim), torch.cat(finals) if finals else None
