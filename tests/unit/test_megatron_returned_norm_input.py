"""Exercise the actual helper/forwards without importing the CUDA runtime."""

import ast
from copy import deepcopy
from itertools import product
from math import prod, sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "src/art/megatron/lora.py"
FAMILIES = (
    "SelfAttentionLinearQKVLoRA",
    "GatedDeltaNetInProjLoRA",
    "ComponentwiseColumnParallelLinearLoRA",
    "SharedExpertsLinearFC1LoRA",
)


class Tensor:
    def __init__(self, shape, *, requires_grad=True):
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.dtype = "bf16"

    def reshape(self, *shape):
        assert prod(shape) == prod(self.shape)
        return Tensor(shape, requires_grad=self.requires_grad)

    def flatten(self, start):
        start %= len(self.shape)
        return self.reshape(*self.shape[:start], prod(self.shape[start:]))

    def narrow(self, dim, start, size):
        shape = list(self.shape)
        assert start + size <= shape[dim]
        shape[dim] = size
        return Tensor(shape, requires_grad=self.requires_grad)

    def new_zeros(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(shape, requires_grad=False)

    def clone(self):
        return Tensor(self.shape, requires_grad=self.requires_grad)

    def numel(self):
        return prod(self.shape)

    def to(self, **kwargs):
        return self

    def sum(self):
        return Tensor((), requires_grad=self.requires_grad)

    def expand(self, *shape):
        return Tensor(shape, requires_grad=self.requires_grad)

    def __mul__(self, other):
        return self.clone()

    def __add__(self, other):
        assert self.shape == other.shape
        return Tensor(
            self.shape, requires_grad=self.requires_grad or other.requires_grad
        )


def cat(values, dim):
    shape = list(values[0].shape)
    dim %= len(shape)
    assert all(
        value.shape[:dim] + value.shape[dim + 1 :]
        == tuple(shape[:dim] + shape[dim + 1 :])
        for value in values
    )
    shape[dim] = sum(value.shape[dim] for value in values)
    return Tensor(shape, requires_grad=any(value.requires_grad for value in values))


def namespace():
    tree = ast.parse(SOURCE.read_text())
    calls, boundaries = [], []

    def copy(x, *, group=None):
        calls.append(("copy", group, None))
        return x.clone()

    def gather(x, *, group=None, tensor_parallel_output_grad=True):
        calls.append(("gather", group, tensor_parallel_output_grad))
        size = group.size() if group is not None else 2
        return Tensor((x.shape[0] * size, *x.shape[1:]), requires_grad=x.requires_grad)

    def disable(function):
        boundaries.append(function)
        return function

    names = {
        "_linear_disables_tensor_parallel_comm",
        "_compile_disabled_collective",
        "_column_parallel_lora_input",
    }
    aliases = {
        "_gather_lora_sequence_parallel_region",
        "_copy_lora_tensor_model_parallel_region",
    }
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in aliases
            for target in node.targets
        )
    ]
    result = {
        "torch": SimpleNamespace(
            Tensor=Tensor, cat=cat, compiler=SimpleNamespace(disable=disable)
        ),
        "cast": cast,
        "_F": Any,
        "copy_to_tensor_model_parallel_region": copy,
        "gather_from_sequence_parallel_region": gather,
    }
    exec(
        compile(
            "from __future__ import annotations\n"
            + ast.unparse(ast.Module(body=nodes, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        result,
    )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in FAMILIES:
            forward = deepcopy(
                next(
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "forward"
                )
            )
            forward.name = node.name
            exec(
                compile(
                    "from __future__ import annotations\n" + ast.unparse(forward),
                    str(SOURCE),
                    "exec",
                ),
                result,
            )
    return tree, result, calls, boundaries, copy, gather


class ReturnedNormInputTests(unittest.TestCase):
    def test_norm_vjp_has_one_tp_owner(self):
        # Model the public collective adjoints and TE's local returned-norm
        # contract, then check the input VJP against a separate loss finite
        # difference. This is binary64 CPU math, not a BF16/native oracle.
        tree, ns, calls, _, _, _ = namespace()
        flags = {
            node.name: next(
                assignment.value
                for assignment in ast.walk(node)
                if isinstance(assignment, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "return_layernorm_output_gathered"
                    for target in assignment.targets
                )
            )
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in FAMILIES
        }
        rows = ((0.7, -1.2), (1.5, 0.2), (-0.4, 0.9), (0.1, -0.8))
        dense = ((0.4, -0.2), (-0.8, 0.3))
        adapter = ((0.7, 0.5), (0.2, -0.6))

        def add(*values):
            return tuple(map(sum, zip(*values, strict=True)))

        def norm_vjp(row, cotangent):
            inv = 1 / sqrt(sum(x * x for x in row) / len(row) + 1e-5)
            dot = sum(x * g for x, g in zip(row, cotangent, strict=True))
            return tuple(
                inv * g - x * inv**3 * dot / len(row)
                for x, g in zip(row, cotangent, strict=True)
            )

        for family, tp, sp, outer, arm in product(
            FAMILIES,
            (1, 2),
            (False, True),
            (False, True),
            ("dense", "adapter", "combined"),
        ):
            with self.subTest(family=family, tp=tp, sp=sp, outer=outer, arm=arm):
                group = SimpleNamespace(size=lambda: tp)
                linear = SimpleNamespace(
                    tp_size=tp,
                    tp_group=group,
                    sequence_parallel=sp,
                    parallel_mode=None if outer else "column",
                )
                gathered = eval(
                    compile(ast.Expression(flags[family]), str(SOURCE), "eval"),
                    {"linear_qkv": linear, "in_proj": linear, "linear_fc1": linear},
                )
                calls.clear()
                local_rows = len(rows) // tp if sp and not outer else len(rows)
                ns["_column_parallel_lora_input"](
                    Tensor((local_rows, 1, 2)), linear, returned_norm=True
                )
                dc = dense[:tp] if arm != "adapter" else ((0.0, 0.0),) * tp
                ac = adapter[:tp] if arm != "dense" else ((0.0, 0.0),) * tp
                coefficients = [add(d, a) for d, a in zip(dc, ac, strict=True)]

                def loss(values):
                    return sum(
                        sum(x * c for x, c in zip(row, coefficient, strict=True))
                        / sqrt(sum(x * x for x in row) / len(row) + 1e-5)
                        for row in values
                        for coefficient in coefficients
                    )

                summed_adapter = bool(calls) and (
                    calls[0][0] == "copy" or calls[0][2] is True
                )
                # Ordinary TE reduces dense dgrad before adding this branch.
                # Overlap's outer edge reduces the whole local norm VJP.
                cots = [
                    add(
                        dc[rank] if outer else add(*dc),
                        (0.0, 0.0)
                        if gathered
                        else add(*ac)
                        if summed_adapter
                        else ac[rank],
                    )
                    for rank in range(tp)
                ]
                for rank in range(tp):
                    actual = [
                        add(*(norm_vjp(row, cot) for cot in cots))
                        if outer
                        else norm_vjp(row, cots[rank])
                        for row in rows
                    ]
                    indices = (
                        range(rank * len(rows) // tp, (rank + 1) * len(rows) // tp)
                        if sp
                        else range(len(rows))
                    )
                    for i in indices:
                        for j in range(len(rows[i])):
                            plus, minus = (
                                [list(row) for row in rows],
                                [list(row) for row in rows],
                            )
                            plus[i][j] += 1e-6
                            minus[i][j] -= 1e-6
                            expected = (loss(plus) - loss(minus)) / 2e-6
                            self.assertAlmostEqual(actual[i][j], expected, delta=2e-9)

    def test_local_return_and_compile_boundaries(self):
        tree, _, _, boundaries, copy, gather = namespace()
        flags = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "return_layernorm_output_gathered"
                for target in node.targets
            )
        ]
        self.assertEqual(len(flags), 4)
        self.assertTrue(
            all(
                isinstance(value, ast.Constant) and value.value is False
                for value in flags
            )
        )
        self.assertEqual(boundaries, [gather, copy])

    def test_helper_modes_group_and_grad_participation(self):
        _, ns, calls, _, _, _ = namespace()
        group = SimpleNamespace(size=lambda: 2)
        for tp in (1, 2):
            for sp in (False, True):
                for disabled in (False, True):
                    for grad in (False, True):
                        with self.subTest(tp=tp, sp=sp, disabled=disabled, grad=grad):
                            calls.clear()
                            x = Tensor((4, 1, 4), requires_grad=grad)
                            linear = SimpleNamespace(
                                tp_size=tp,
                                sequence_parallel=sp,
                                parallel_mode=None if disabled else "column",
                                tp_group=None if disabled or tp == 1 else group,
                            )
                            mapped = ns["_column_parallel_lora_input"](
                                x, linear, returned_norm=True
                            )
                            expected = (
                                []
                                if disabled or tp == 1
                                else [("gather", group, True)]
                                if sp
                                else [("copy", group, None)]
                            )
                            self.assertEqual(calls, expected)
                            self.assertEqual(
                                mapped.shape[0],
                                8 if sp and tp == 2 and not disabled else 4,
                            )
                            self.assertEqual(mapped.requires_grad, grad)
                            if not expected:
                                self.assertIs(mapped, x)
        for bad in (None, SimpleNamespace(size=lambda: 3)):
            for sp in (False, True):
                with (
                    self.subTest(bad_group=bad, sp=sp),
                    self.assertRaisesRegex(RuntimeError, "initialized TP group"),
                ):
                    ns["_column_parallel_lora_input"](
                        Tensor((4, 1, 4)),
                        SimpleNamespace(
                            tp_size=2,
                            sequence_parallel=sp,
                            parallel_mode="column",
                            tp_group=bad,
                        ),
                        returned_norm=True,
                    )
        linear = SimpleNamespace(
            tp_size=2,
            sequence_parallel=True,
            parallel_mode="column",
            explicit_expert_comm=True,
        )
        calls.clear()
        x = Tensor((8, 1, 4))
        self.assertIs(
            ns["_column_parallel_lora_input"](x, linear, returned_norm=True), x
        )
        self.assertEqual(calls, [])

    def test_actual_four_forwards_share_one_edge(self):
        for family in FAMILIES:
            for tp in (1, 2):
                for sp in (False, True):
                    for mode in ("column", None):
                        with self.subTest(family=family, tp=tp, sp=sp, mode=mode):
                            _, ns, calls, _, _, _ = namespace()
                            group = SimpleNamespace(size=lambda: 2)
                            rows = 4 if tp == 2 and sp and mode == "column" else 8
                            norm, base = (
                                Tensor((rows, 1, 4)),
                                Tensor((8, 1, 12 if family == FAMILIES[1] else 6)),
                            )
                            seen = []

                            def adapter(width):
                                def run(x):
                                    seen.append(x)
                                    return Tensor(
                                        (*x.shape[:-1], width),
                                        requires_grad=x.requires_grad,
                                    )

                                setattr(run, "A_T", Tensor((4, 1)))
                                setattr(run, "B_T", Tensor((1, width)))
                                return run

                            linear = SimpleNamespace(
                                tp_size=tp,
                                tp_group=group,
                                sequence_parallel=sp,
                                parallel_mode=mode,
                            )

                            class Inner:
                                def __call__(self, x):
                                    return (base, norm), None

                            inner = Inner()
                            inner.__dict__.update(vars(linear))
                            obj = SimpleNamespace(
                                linear_qkv=inner,
                                in_proj=inner,
                                linear_fc1=inner,
                                q_proj_lora=adapter(2),
                                k_proj_lora=adapter(2),
                                v_proj_lora=adapter(2),
                                qkv_lora=adapter(6),
                                z_lora=adapter(2),
                                lora=adapter(6),
                                gate_lora=adapter(3),
                                up_lora=adapter(3),
                                non_gated=False,
                                out_features=6,
                                num_value_heads_per_partition=2,
                                num_query_groups_per_partition=1,
                                num_attention_heads_per_group=1,
                                attention_output_gate=False,
                                hidden_size_per_attention_head=2,
                                replicated_qkv=False,
                            )
                            obj._qkv_lora_output = lambda lora, x, width: (
                                lora(x)
                                if lora is not None
                                else x.new_zeros((*x.shape[:-1], width))
                            )
                            obj.q_and_gate_out_features_per_rank = (
                                obj.kv_out_features_per_rank
                            ) = 2
                            result, bias = ns[family](obj, Tensor((rows, 1, 4)))
                            self.assertEqual(result.shape, base.shape)
                            self.assertIsNone(bias)
                            self.assertTrue(
                                seen
                                and all(
                                    x is seen[0] and x.shape == (8, 1, 4) for x in seen
                                )
                            )
                            self.assertEqual(
                                calls,
                                []
                                if mode is None or tp == 1
                                else [("gather", group, True)]
                                if sp
                                else [("copy", group, None)],
                            )

    def test_nonfused_defaults_and_shape_refusal(self):
        _, ns, calls, _, _, _ = namespace()
        for sp in (False, True):
            calls.clear()
            x = Tensor((4, 1, 4))
            linear = SimpleNamespace(
                tp_size=2, sequence_parallel=sp, parallel_mode="column"
            )
            result = ns["_column_parallel_lora_input"](x, linear)
            self.assertEqual(calls, [("gather", None, True)] if sp else [])
            self.assertEqual(result.shape[0], 8 if sp else 4)
        inner = SimpleNamespace(
            tp_size=2,
            sequence_parallel=False,
            parallel_mode="column",
            tp_group=SimpleNamespace(size=lambda: 2),
        )

        class Inner:
            def __call__(self, x):
                return (Tensor((8, 1, 6)), x), None

        projection = Inner()
        projection.__dict__.update(vars(inner))
        wrong = lambda x: Tensor((*x.shape[:-1], 5))
        setattr(wrong, "adapter_model_prefix", "shape-refusal")
        with self.assertRaisesRegex(RuntimeError, "does not match base"):
            ns[FAMILIES[2]](
                SimpleNamespace(in_proj=projection, lora=wrong), Tensor((8, 1, 4))
            )

    def test_shared_non_gated_and_empty_bypass(self):
        _, ns, calls, _, _, _ = namespace()
        norm = Tensor((8, 1, 4))

        class Inner:
            tp_size = 2
            sequence_parallel = False
            parallel_mode = None

            def __call__(self, x):
                return (Tensor((8, 1, 6)), norm), None

        seen = []

        def up(x):
            seen.append(x)
            return Tensor((*x.shape[:-1], 6))

        setattr(up, "A_T", Tensor((4, 1)))
        setattr(up, "B_T", Tensor((1, 6)))
        obj = SimpleNamespace(
            linear_fc1=Inner(), up_lora=up, non_gated=True, out_features=6
        )
        output, bias = ns[FAMILIES[3]](obj, norm)
        self.assertEqual(output.shape, (8, 1, 6))
        self.assertEqual(seen, [norm])
        self.assertIsNone(bias)
        seen.clear()
        output, bias = ns[FAMILIES[3]](obj, Tensor((0, 1, 4)))
        self.assertEqual(output.shape, (0, 1, 6))
        self.assertTrue(output.requires_grad)
        self.assertEqual(seen, [])
        self.assertEqual(calls, [])
        self.assertIsNone(bias)


if __name__ == "__main__":
    unittest.main()
