"""Binary64 input VJP regression; declared TP adjoints, no CUDA/framework import."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "src/art/megatron/lora.py"


class Value:
    def __init__(self, value, parents=()):
        self.value, self.parents, self.grad = float(value), parents, 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value + other.value, ((self, 1), (other, 1)))

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.value * other.value, ((self, other.value), (other, self.value))
        )

    __rmul__ = __mul__


def backward(loss):
    order, seen = [], set()

    def visit(node):
        if id(node) not in seen:
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            order.append(node)

    visit(loss)
    loss.grad = 1
    for node in reversed(order):
        for parent, derivative in node.parents:
            parent.grad += node.grad * derivative


class Matrix:
    def __init__(self, rows, stage="input"):
        self.rows, self.stage = rows, stage
        self.shape = (len(rows), len(rows[0]))

    def numel(self):
        return self.shape[0] * self.shape[1]

    def __matmul__(self, other):
        return Matrix(
            [
                [
                    sum(a * b for a, b in zip(row, col, strict=True))
                    for col in zip(*other.rows, strict=True)
                ]
                for row in self.rows
            ]
        )

    def __add__(self, other):
        return Matrix(
            [
                [a + b for a, b in zip(x, y, strict=True)]
                for x, y in zip(self.rows, other.rows, strict=True)
            ]
        )

    def __mul__(self, value):
        return Matrix([[x * value for x in row] for row in self.rows])


def constants(rows):
    return Matrix([[Value(x) for x in row] for row in rows])


def namespace(collectives):
    tree = ast.parse(SOURCE.read_text())
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
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in names
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in aliases for t in n.targets)
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in (
            "LoRA",
            "SharedExpertsLinearFC1LoRA",
        ):
            forward = deepcopy(
                next(
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "forward"
                )
            )
            forward.name = node.name + "_forward"
            nodes.append(forward)
    ns = {
        "torch": SimpleNamespace(
            Tensor=Matrix,
            compiler=SimpleNamespace(disable=lambda f: f),
            cat=lambda xs, dim: Matrix(
                [list(a) + list(b) for a, b in zip(xs[0].rows, xs[1].rows, strict=True)]
            ),
        ),
        "cast": cast,
        "_F": Any,
        "copy_to_tensor_model_parallel_region": collectives.copy,
        "gather_from_sequence_parallel_region": collectives.gather,
    }
    text = "from __future__ import annotations\n" + ast.unparse(
        ast.Module(body=nodes, type_ignores=[])
    )
    exec(compile(text, str(SOURCE), "exec"), ns)
    return ns


class CollectiveAdjoints:
    def __init__(self, inputs):
        self.inputs, self.stages, self.calls = inputs, {"input": inputs}, []

    def copy(self, x, *, group=None):
        self.calls.append("copy")
        peers = self.stages[x.stage]
        return Matrix(
            [
                [
                    Value(value.value, tuple((p.rows[i][j], 1) for p in peers))
                    for j, value in enumerate(row)
                ]
                for i, row in enumerate(x.rows)
            ]
        )

    def gather(self, x, *, group=None, tensor_parallel_output_grad=True):
        assert tensor_parallel_output_grad is True
        self.calls.append("gather")
        # Shared parents implement all-gather's SUM/reduce-scatter adjoint.
        return Matrix([row for peer in self.inputs for row in peer.rows])


class ColumnInputVJPTests(unittest.TestCase):
    def test_nonfused_input_vjp_has_one_tp_owner(self):
        rows = ((1.0, -2.0), (-0.5, 3.0), (2.0, 0.25), (-1.0, 1.5))
        weights = (((1.0, 2.0), (-3.0, 1.0)), ((-2.0, 1.0), (1.0, 4.0)))
        aa = ((2.0, -1.0), (-1.0, 3.0))
        bb = ((1.0, -2.0), (3.0, 1.0))
        cot = (
            ((1.0, -2.0), (3.0, 1.0), (-1.0, 2.0), (2.0, 3.0)),
            ((-2.0, 1.0), (1.0, -3.0), (2.0, 1.0), (-1.0, -2.0)),
        )
        for sp in (False, True):
            for owner in ("inner", "outer_mode", "outer_expert"):
                for arm in ("base_only", "adapter_only", "total"):
                    with self.subTest(SP=sp, owner=owner, arm=arm):
                        inputs = [
                            constants(rows[r * 2 : (r + 1) * 2] if sp else rows)
                            for r in range(2)
                        ]
                        comm = CollectiveAdjoints(inputs)
                        ns = namespace(comm)
                        outer = owner != "inner"
                        feed = [
                            (comm.gather(x) if sp else comm.copy(x)) if outer else x
                            for x in inputs
                        ]
                        if outer:
                            for x in feed:
                                x.stage = "outer"
                            comm.stages["outer"] = feed
                        local_w = [
                            tuple(
                                tuple(0.0 if arm == "adapter_only" else v for v in row)
                                for row in w
                            )
                            for w in weights
                        ]
                        local_b = [
                            tuple(0.0 if arm == "base_only" else v for v in b)
                            for b in bb
                        ]
                        outputs = []
                        for rank, x in enumerate(feed):

                            class Inner:
                                tp_size = 2
                                sequence_parallel = sp
                                parallel_mode = (
                                    None if owner == "outer_mode" else "column"
                                )
                                explicit_expert_comm = owner == "outer_expert"

                                def __call__(self, x):
                                    mapped = (
                                        x
                                        if outer
                                        else comm.gather(x)
                                        if sp
                                        else comm.copy(x)
                                    )
                                    return mapped @ constants(local_w[rank]), None

                            class Adapter:
                                def __init__(self, column):
                                    self.active = (
                                        constants([[v] for v in aa[column]]),
                                        constants([[local_b[rank][column]]]),
                                        1.0,
                                    )

                                def active_lora_tensors(self):
                                    return self.active

                                def __call__(self, x):
                                    return ns["LoRA_forward"](self, x)

                            obj = SimpleNamespace(
                                linear_fc1=Inner(),
                                non_gated=False,
                                gate_lora=Adapter(0),
                                up_lora=Adapter(1),
                            )
                            output, bias = ns["SharedExpertsLinearFC1LoRA_forward"](
                                obj, x
                            )
                            self.assertIsNone(bias)
                            outputs.append(output)
                        loss = sum(
                            value * cot[r][i][j]
                            for r, y in enumerate(outputs)
                            for i, row in enumerate(y.rows)
                            for j, value in enumerate(row)
                        )
                        backward(loss)

                        def reference(values):
                            return sum(
                                cot[r][i][j]
                                * sum(
                                    values[i][k]
                                    * (local_w[r][k][j] + aa[j][k] * local_b[r][j])
                                    for k in range(2)
                                )
                                for r in range(2)
                                for i in range(4)
                                for j in range(2)
                            )

                        for rank, x in enumerate(inputs):
                            for i, row in enumerate(x.rows):
                                global_i = rank * 2 + i if sp else i
                                for j, value in enumerate(row):
                                    plus, minus = (
                                        [list(z) for z in rows],
                                        [list(z) for z in rows],
                                    )
                                    plus[global_i][j] += 1e-5
                                    minus[global_i][j] -= 1e-5
                                    expected = (
                                        reference(plus) - reference(minus)
                                    ) / 2e-5
                                    self.assertAlmostEqual(
                                        value.grad, expected, delta=2e-8
                                    )
                        self.assertAlmostEqual(loss.value, reference(rows), delta=1e-10)

    def test_single_rank_and_external_comm_identity(self):
        x = constants(((1.0, 2.0),))
        comm = CollectiveAdjoints([x])
        helper = namespace(comm)["_column_parallel_lora_input"]
        for linear in (
            SimpleNamespace(tp_size=1, sequence_parallel=False, parallel_mode="column"),
            SimpleNamespace(tp_size=2, sequence_parallel=True, parallel_mode=None),
            SimpleNamespace(
                tp_size=2,
                sequence_parallel=False,
                parallel_mode="column",
                explicit_expert_comm=True,
            ),
        ):
            self.assertIs(helper(x, linear), x)
        self.assertEqual(comm.calls, [])


if __name__ == "__main__":
    unittest.main()
