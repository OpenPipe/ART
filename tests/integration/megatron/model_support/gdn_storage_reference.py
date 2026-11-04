"""CPU FP64 WY algebra with explicit, selectable intermediate storage rounding.

This isolates storage precision; it does not emulate Triton reduction/exp2 order.
"""

import torch


def chunk(q, k, v, g, beta, stage_dtype):
    """First zero-state 64-token WY chunk with explicit storage boundaries.

    This models storage rounding, not Triton reduction trees or exp2 rounding.
    All contractions and the triangular solve use CPU FP64.
    """
    q, k, v = (x.double().transpose(0, 1) for x in (q, k, v))
    g, beta = (x.double().T for x in (g, beta))
    n = q.shape[1]
    assert n <= 64

    def store(x):
        return x.to(stage_dtype).double()

    cumulative = g.cumsum(-1)
    decay = (cumulative[:, :, None] - cumulative[:, None, :]).exp().tril()
    lower = ((k @ k.transpose(-1, -2)) * beta[:, :, None] * decay).tril(-1)
    identity = torch.eye(n, dtype=torch.float64).expand_as(lower)
    inverse = store(
        torch.linalg.solve_triangular(identity + lower, identity, upper=False)
    )
    u = store(inverse @ store(v * beta[:, :, None]))
    attention = store(((q @ k.transpose(-1, -2)) * decay).tril())
    return store((attention @ u) * q.shape[-1] ** -0.5).transpose(0, 1)


def recurrent(q, k, v, g, beta):
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    h = q.new_zeros(k.shape[1], k.shape[-1], v.shape[-1])
    outputs = []
    for i in range(len(q)):
        h = h * g[i].exp()[:, None, None]
        update = beta[i, :, None] * (v[i] - (h * k[i, :, :, None]).sum(1))
        h = h + k[i, :, :, None] * update[:, None, :]
        outputs.append((h * q[i, :, :, None]).sum(1) * q.shape[-1] ** -0.5)
    return torch.stack(outputs)
