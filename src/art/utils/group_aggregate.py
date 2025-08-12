from typing import Literal

import torch

Reduction = Literal["sum", "mean", "var", "std", "count", "min", "max"]


def group_aggregate(
    values: torch.Tensor,
    *,
    by: torch.Tensor,
    reduce: Reduction = "mean",
    eps: float = 1e-8,
    broadcast: bool = True,
) -> torch.Tensor:
    """Vectorised group-wise reduction over sequence dimension.

    Parameters
    ----------
    values: Tensor of shape ``[B, S, *F]`` where ``*F`` can be empty or any
        number of trailing feature dimensions. Tokens live along the ``S``
        axis and will be aggregated per *group*.
    by: Tensor broadcast-compatible with the first two dimensions of
        *values*, i.e. shape ``[B, S]`` or ``[B, S, 1, …]``.  Values are
        interpreted as categorical labels – their actual numeric content is
        irrelevant.  Any dtype accepted by ``torch.unique`` is supported.
    reduce: Which reduction to apply: "sum", "mean", "var", "std", "count".
    eps: Numerical stability term used for mean/var/std.
    broadcast: If True (default) the aggregated statistic is scattered back to
        the token dimension so that the return shape equals *values.shape*.
        If False the function returns the per-batch-per-group tensor of shape
        [B, num_groups].

    Returns
    -------
    Tensor with aggregated values, shape depends on `broadcast`:
        - If True: same shape as `values`
        - If False: [B, num_groups, *F]

    Examples
    --------
    >>> # Compute mean logits per conversation turn
    >>> logits = torch.randn(2, 10, 768)  # [batch, seq, hidden]
    >>> turn_ids = torch.tensor([[0,0,0,1,1,1,2,2,2,2], [0,0,1,1,1,2,2,2,3,3]])
    >>> mean_per_turn = group_aggregate(logits, turn_ids, reduce="mean")

    >>> # Get variance of rewards per group, compact form
    >>> rewards = torch.randn(4, 20)  # [batch, seq]
    >>> groups = torch.randint(0, 5, (4, 20))
    >>> var_per_group = group_aggregate(rewards, groups, reduce="var", broadcast=False)
    >>> # Returns shape [4, 5] - variance for each of 5 groups per batch

    Notes
    -----
    The implementation relies exclusively on *scatter_add_* and is fully
    differentiable w.r.t. *values*.
    """
    if values.dim() < 2:
        raise ValueError("`values` must have at least 2 dimensions [B, S].")

    if by.shape[:2] != values.shape[:2]:
        raise ValueError("`by` must match the first two dimensions of `values` (B, S).")

    B, S, *feat_dims = values.shape
    device, dtype = values.device, values.dtype

    # Handle edge cases
    if values.numel() == 0:
        # Empty tensor - return same shape
        return (
            values.clone()
            if broadcast
            else torch.empty(B, 0, *feat_dims, device=device, dtype=dtype)
        )

    if by.numel() == 0:
        raise ValueError("`by` cannot be empty when values is non-empty")

    # Treat *by* as a categorical variable: map each unique id (within the whole
    # minibatch) to an integer rank starting at 0.  We rely on `torch.unique(return_inverse=True)`
    # which provides this mapping in a fully-vectorised manner.

    # Flatten all dims to obtain the mapping once for the entire mini-batch.
    flat_ids = by.reshape(-1)
    _, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
    mapped_ids = inverse.view_as(by[:, :])  # shape [B, S]

    num_groups = int(mapped_ids.max().item()) + 1  # contiguous 0..K-1

    # Collapse trailing feature dims to 1 for scatter-add efficiency, then restore later.
    values_flat = values.reshape(B, S, -1)  # [B, S, P]
    P = values_flat.size(-1)

    # Use expand_as for memory efficiency (creates view, not copy)
    mapped_ids_exp = mapped_ids.unsqueeze(-1).expand_as(values_flat)  # [B, S, P]

    # --- Scatter sums -----------------------------------------------------
    sum_per_gid = torch.zeros(B, num_groups, P, device=device, dtype=dtype)
    sum_per_gid.scatter_add_(1, mapped_ids_exp, values_flat)

    if reduce == "sum":
        agg = sum_per_gid
    else:
        # Optimize: for count, we don't need to create a full ones tensor
        if reduce == "count":
            count_per_gid = torch.zeros(B, num_groups, 1, device=device, dtype=dtype)
            count_per_gid.scatter_add_(
                1,
                mapped_ids.unsqueeze(-1),
                torch.ones(B, S, 1, device=device, dtype=dtype),
            )
            agg = count_per_gid.expand(-1, -1, P)
        else:
            count_per_gid = torch.zeros_like(sum_per_gid)
            ones = torch.ones_like(values_flat)
            count_per_gid.scatter_add_(1, mapped_ids_exp, ones)

            if reduce == "mean":
                agg = sum_per_gid / (count_per_gid + eps)
            elif reduce in ("var", "std"):
                # E[X^2] - E[X]^2
                sumsq_per_gid = torch.zeros_like(sum_per_gid)
                sumsq_per_gid.scatter_add_(1, mapped_ids_exp, values_flat * values_flat)
                mean = sum_per_gid / (count_per_gid + eps)
                mean_sq = sumsq_per_gid / (count_per_gid + eps)
                var = mean_sq - mean * mean
                var = torch.clamp(var, min=0.0)  # numerical stability
                agg = var if reduce == "var" else torch.sqrt(var + eps)
            elif reduce in ("min", "max"):
                # Initialize with appropriate fill values
                fill_val = float("inf") if reduce == "min" else float("-inf")
                agg = torch.full(
                    (B, num_groups, P), fill_val, device=device, dtype=dtype
                )
                agg = torch.scatter_reduce(
                    agg, 1, mapped_ids_exp, values_flat, reduce=reduce
                )
            else:
                raise ValueError(f"Unsupported reduce type: {reduce}")

    if broadcast:
        out = agg.gather(1, mapped_ids_exp)  # [B, S, P]
        return out.reshape(values.shape)

    # Return compact form: [B, num_groups, *F]
    return agg.reshape(B, num_groups, *feat_dims)
