"""Real CPU Gloo adjoints with different default and owning-linear TP groups.

Only CUDA allocation devices in Megatron's mappings are adapted for CPU; its
collective autograd Functions and ART's production forwards execute unchanged.
TE's base GEMM is zeroed to isolate adapter gradients. Native TE is not loaded.
"""

import ast
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_megatron_lora_input_gradients import load_wrappers
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CASES = (
    ("ComponentwiseColumnParallelLinearLoRA", False),
    ("SharedExpertsLinearFC1LoRA", False),
    ("ComponentwiseColumnParallelLinearLoRA", True),
)


def _cpu_mappings(source, default_group):
    tree = ast.parse(Path(source).read_text())
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]

    class CPUAllocation(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == "torch.cuda.current_device()":
                return ast.copy_location(
                    ast.parse("input_.device", mode="eval").body, node
                )
            return node

    module = CPUAllocation().visit(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(
        torch=torch,
        get_tensor_model_parallel_group_if_none=lambda group: (
            default_group if group is None else group
        ),
        dist_all_gather_func=dist.all_gather_into_tensor,
        dist_reduce_scatter_func=dist.reduce_scatter_tensor,
    )
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(module),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


def _worker(rank, root, source):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        default_group = linear_group = None
        for ranks in ([0, 1], [2, 3]):
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                default_group = group
        for ranks in ([0, 2], [1, 3]):
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                linear_group = group
        mappings = _cpu_mappings(source, default_group)
        namespace, _ = load_wrappers(
            mappings["copy_to_tensor_model_parallel_region"],
            mappings["gather_from_sequence_parallel_region"],
        )
        results = []
        for family, fused in CASES:
            for sp in (False, True):
                # Distinct SP shards also expose gathering from the wrong peers.
                x = torch.full(
                    (1, 1, 1),
                    float(rank + 1) if sp else 1.0,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                a = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
                b = torch.full(
                    (1, 1), float(rank + 1), dtype=torch.float64, requires_grad=True
                )
                lora = SimpleNamespace(
                    is_expert=False, active_lora_tensors=lambda: (a, b, 1.0)
                )

                def adapter(z):
                    return namespace["LoRA"](lora, z)

                class Linear:
                    tp_size = 2
                    tp_group = linear_group
                    sequence_parallel = sp
                    parallel_mode = "column"

                    def __call__(self, z):
                        base = z.new_zeros((2 if sp else 1, 1, 1))
                        return ((base, z) if fused else base), None

                linear = Linear()
                obj = SimpleNamespace(
                    in_proj=linear,
                    linear_fc1=linear,
                    lora=adapter,
                    up_lora=adapter,
                    non_gated=True,
                )
                output, _ = namespace[family](obj, x)
                dx, da, db = torch.autograd.grad(output.sum(), (x, a, b))
                results.append(
                    dict(
                        output=output.detach().flatten().tolist(),
                        dx=dx.item(),
                        da=da.item(),
                        db=db.item(),
                    )
                )
        Path(root, f"rank-{rank}.json").write_text(json.dumps(results))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def custom_group_results(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    try:
        source = distribution("megatron-core").locate_file(
            "megatron/core/tensor_parallel/mappings.py"
        )
    except PackageNotFoundError:
        pytest.skip("requires megatron-core's mapping source (no CUDA imports)")
    root = tmp_path_factory.mktemp("lora-custom-tp")
    mp.start_processes(
        _worker,
        args=(root, source),
        nprocs=4,
        start_method="spawn",
        join=True,
    )
    return [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(4)]


@pytest.mark.parametrize(
    "case",
    range(len(CASES)),
    ids=["componentwise-nonfused", "shared-experts-nonfused", "returned-norm-control"],
)
@pytest.mark.parametrize("sp", [False, True], ids=["sp-off", "sp-on"])
def test_custom_tp_group_adjoint(custom_group_results, case, sp):
    for rank, results in enumerate(custom_group_results):
        actual = results[case * 2 + int(sp)]
        members = [rank % 2, rank % 2 + 2]
        full_input = [float(r + 1) for r in members] if sp else [1.0]
        # Independent unsharded FP64 reference over this linear's output shards.
        ref_x = torch.tensor(full_input, dtype=torch.float64, requires_grad=True)
        ref_a = torch.ones(2, dtype=torch.float64, requires_grad=True)
        ref_b = torch.tensor(
            [float(r + 1) for r in members], dtype=torch.float64, requires_grad=True
        )
        ref_outputs = [ref_x * ref_a[i] * ref_b[i] for i in range(2)]
        dx, da, db = torch.autograd.grad(
            torch.stack(ref_outputs).sum(), (ref_x, ref_a, ref_b)
        )
        local = members.index(rank)
        # Check dx first so omission of the group reports the reviewed 3/7 vs 4/6.
        expected = dict(
            dx=dx[local if sp else 0].item(),
            output=ref_outputs[local].detach().tolist(),
            da=da[local].item(),
            db=db[local].item(),
        )
        for key, value in expected.items():
            torch.testing.assert_close(
                torch.tensor(actual[key], dtype=torch.float64),
                torch.tensor(value, dtype=torch.float64),
                atol=1e-12,
                rtol=1e-12,
                msg=lambda msg: f"rank={rank}, {key}: {msg}",
            )
    assert not torch.cuda.is_initialized()


def _no_collectives():
    return load_wrappers(
        lambda *a, **k: pytest.fail("unexpected copy"),
        lambda *a, **k: pytest.fail("unexpected gather"),
    )[0]["_column_parallel_lora_input"]


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("group_state", ["absent", "none", "wrong-size"])
def test_nonfused_requires_initialized_matching_group(sp, group_state):
    linear = SimpleNamespace(tp_size=2, sequence_parallel=sp, parallel_mode="column")
    if group_state != "absent":
        linear.tp_group = (
            None if group_state == "none" else SimpleNamespace(size=lambda: 3)
        )
    with pytest.raises(RuntimeError, match="initialized TP group"):
        _no_collectives()(torch.ones(1, dtype=torch.float64), linear)


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("returned_norm", [False, True])
def test_tp1_without_group_is_identity(sp, returned_norm):
    x = torch.ones(1, dtype=torch.float64)
    linear = SimpleNamespace(tp_size=1, sequence_parallel=sp, parallel_mode="column")
    assert _no_collectives()(x, linear, returned_norm=returned_norm) is x


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("returned_norm", [False, True])
@pytest.mark.parametrize("owner", ["outer-mode", "outer-expert"])
def test_external_owner_without_group_is_identity(sp, returned_norm, owner):
    x = torch.ones(1, dtype=torch.float64)
    linear = SimpleNamespace(
        tp_size=2,
        sequence_parallel=sp,
        parallel_mode=None if owner == "outer-mode" else "column",
        explicit_expert_comm=owner == "outer-expert",
    )
    assert _no_collectives()(x, linear, returned_norm=returned_norm) is x
