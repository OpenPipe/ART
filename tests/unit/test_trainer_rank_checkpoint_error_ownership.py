"""Discover exception-to-collective paths and their calibration error guards."""

import ast
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2] / "src" / "art" / "trainer_rank"
PRIMITIVES = {
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "gather",
    "gather_object",
    "broadcast",
    "broadcast_object_list",
    "barrier",
    "monitored_barrier",
    "reduce",
    "reduce_scatter",
    "all_to_all",
    "all_to_all_single",
    "send",
    "recv",
    "new_group",
}


def analyze(sources: dict[str, str]) -> dict[str, Any]:
    functions: dict[str, Any] = {}
    parents: dict[Any, Any] = {}
    modules: dict[str, str] = {}

    def index(node: Any, module: str, prefix: str = ""):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = module + "." + prefix + child.name
                functions[q] = child
                modules[q] = module
                index(child, module, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                index(child, module, prefix + child.name + ".")
            else:
                index(child, module, prefix)

    for module, source in sources.items():
        index(ast.parse(source), module)

    def owned_nodes(fn: Any):
        def walk(node: Any):
            yield node
            for c in ast.iter_child_nodes(node):
                if not isinstance(
                    c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    yield from walk(c)

        return walk(fn)

    def name(node: Any) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return name(node.value) + "." + node.attr
        return ""

    def resolve(key: str, call: Any) -> str | None:
        n = name(call.func)
        module = modules[key]
        if (
            n.startswith(("dist.", "torch.distributed."))
            and n.rsplit(".", 1)[-1] in PRIMITIVES
        ):
            return "@wire"
        if n == "_calibration_protocol.all_gather_object":
            return "@wire"
        if n.startswith("_calibration_protocol."):
            return None  # the phase owner, not a nested error exchange
        if n.startswith(("self.", "trainer.")):
            candidate = "_impl.TrainerRank." + n.split(".", 1)[1]
            return candidate if candidate in functions else None
        if n.startswith("_") and "." in n:
            candidate = n
            if candidate in functions:
                return candidate
        prefix = key.rsplit(".", 1)[0]
        while prefix.startswith(module):
            candidate = prefix + "." + n
            if candidate in functions:
                return candidate
            if "." not in prefix:
                break
            prefix = prefix.rsplit(".", 1)[0]
        candidate = module + "." + n
        return candidate if candidate in functions else None

    calls = {
        key: [(c, resolve(key, c)) for c in owned_nodes(fn) if isinstance(c, ast.Call)]
        for key, fn in functions.items()
    }

    # Calls already enclosed by the calibration owner's collective boundary are not independent exchanges.
    def boundary(call: Any) -> bool:
        node = call
        while node in parents:
            node = parents[node]
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "_calibration_protocol.collective"
            ):
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        return False

    starters = {
        key
        for key, cs in calls.items()
        if any(target == "@wire" and not boundary(c) for c, target in cs)
    }
    while True:
        expanded = starters | {
            key
            for key, cs in calls.items()
            if any(target in starters for c, target in cs)
        }
        if expanded == starters:
            break
        starters = expanded

    def active_guard(node: Any) -> bool:
        return (
            isinstance(node, ast.If)
            and any(
                isinstance(c, ast.Compare)
                and isinstance(c.left, ast.Call)
                and name(c.left.func) == "_calibration_protocol.active"
                and len(c.ops) == 1
                and isinstance(c.ops[0], ast.IsNot)
                and len(c.comparators) == 1
                and isinstance(c.comparators[0], ast.Constant)
                and c.comparators[0].value is None
                for c in ast.walk(node.test)
            )
            and any(isinstance(c, ast.Raise) for s in node.body for c in ast.walk(s))
        )

    gateways = set()
    for key, fn in functions.items():
        if key not in starters:
            continue
        for statement in fn.body:
            if active_guard(statement):
                gateways.add(key)
                break
            if any(
                isinstance(c, ast.Call) and resolve(key, c) in starters | {"@wire"}
                for c in ast.walk(statement)
            ):
                break
    rows: list[dict[str, Any]] = []
    for key, fn in functions.items():
        if modules[key] == "_calibration_protocol":
            continue
        for t in owned_nodes(fn):
            if not isinstance(t, ast.Try):
                continue
            # Direct except/finally paths, plus continuation after an exception is retained.
            for handler in t.handlers:
                aliases = {handler.name} if handler.name else set()
                for a in ast.walk(handler):
                    if (
                        isinstance(a, ast.Assign)
                        and isinstance(a.value, ast.Name)
                        and a.value.id in aliases
                    ):
                        aliases.update(
                            v.id for v in a.targets if isinstance(v, ast.Name)
                        )
                terminal = bool(handler.body) and isinstance(
                    handler.body[-1], (ast.Raise, ast.Return)
                )
                regions = [handler.body]
                child = t
                p = parents.get(child)
                while p is not None and not terminal:
                    for field in ("body", "orelse", "finalbody"):
                        seq = getattr(p, field, [])
                        if child in seq:
                            regions.append(seq[seq.index(child) + 1 :])
                            break
                    if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        break
                    child = p
                    p = parents.get(child)
                guarded = any(active_guard(s) for s in handler.body)
                consumed = False
                for region in regions:
                    if consumed:
                        break
                    for s in region:
                        if active_guard(s):
                            guarded = True
                        for c in ast.walk(s):
                            if not isinstance(c, ast.Call):
                                continue
                            target = resolve(key, c)
                            if target != "@wire" and target not in starters:
                                continue
                            # Captured errors may be forwarded to a centrally guarded error gateway.
                            forwarded = target in gateways and any(
                                isinstance(v, ast.Name) and v.id in aliases
                                for arg in c.args[:1]
                                for v in ast.walk(arg)
                            )
                            rows.append(
                                dict(
                                    function=key,
                                    handler_line=handler.lineno,
                                    call_line=c.lineno,
                                    target=target,
                                    aliases=sorted(aliases),
                                    guarded=guarded or forwarded,
                                )
                            )
                            if forwarded:
                                consumed = True
                                break
                        if consumed:
                            break
            # Finalizers must either forward their close error through the guarded gateway,
            # or avoid collectives while a primary body error is propagating.
            if t.finalbody:
                aliases = set()
                for h in t.handlers:
                    if h.name:
                        aliases.add(h.name)
                        for a in ast.walk(h):
                            if (
                                isinstance(a, ast.Assign)
                                and isinstance(a.value, ast.Name)
                                and a.value.id in aliases
                            ):
                                aliases.update(
                                    v.id for v in a.targets if isinstance(v, ast.Name)
                                )
                for s in t.finalbody:
                    for c in ast.walk(s):
                        if not isinstance(c, ast.Call):
                            continue
                        target = resolve(key, c)
                        if target != "@wire" and target not in starters:
                            continue
                        # An error gateway checks the active owner for errors from its close action.
                        guarded = target in gateways or any(
                            active_guard(v) and v.lineno < c.lineno for v in t.finalbody
                        )
                        rows.append(
                            dict(
                                function=key,
                                handler_line=t.lineno,
                                call_line=c.lineno,
                                target=target,
                                aliases=sorted(aliases),
                                guarded=guarded,
                                finally_path=True,
                            )
                        )
    return dict(
        gateways=sorted(gateways),
        paths=rows,
        violations=[r for r in rows if not r["guarded"]],
    )


def _sources():
    return {p.stem: p.read_text() for p in ROOT.glob("*.py")}


def test_error_collective_wrappers_have_a_calibration_owner_guard():
    audit = analyze(_sources())
    assert audit["gateways"] and audit["paths"]
    assert not audit["violations"], audit["violations"]


@pytest.mark.parametrize(
    "body",
    [
        "try:\n    action()\nexcept BaseException:\n    dist.all_reduce(value)",
        "try:\n    action()\nfinally:\n    dist.all_gather_object([], value)",
        "error = None\ntry:\n    action()\nexcept BaseException as exc:\n    error = exc\ndist.all_reduce(value)",
        'try:\n    action()\nexcept BaseException as exc:\n    error = exc\n_checkpoint._phase(lambda: None, "cleanup", None)',
        "try:\n    action()\nfinally:\n    dist.all_reduce(value)\n    if _calibration_protocol.active() is not None:\n        raise RuntimeError()",
        "try:\n    action()\nexcept BaseException:\n    if _calibration_protocol.active() is None:\n        raise\n    dist.all_reduce(value)",
    ],
    ids=[
        "except",
        "finally",
        "retained-error",
        "nested-helper",
        "late-guard",
        "inverted-guard",
    ],
)
def test_guard_rejects_a_planted_unowned_error_collective(body):
    sources = _sources()
    sources["planted"] = "def planted():\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    violations = analyze(sources)["violations"]
    assert any(row["function"] == "planted.planted" for row in violations), violations


def test_removing_a_gateway_guard_is_detected():
    sources = _sources()
    audit = analyze(sources)
    for gateway in audit["gateways"]:
        module, qualified = gateway.split(".", 1)
        tree = ast.parse(sources[module])
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == qualified.split(".")[-1]
            ):
                node.body = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.If)
                        and any(
                            isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Attribute)
                            and c.func.attr == "active"
                            for c in ast.walk(s.test)
                        )
                    )
                ]
        sources[module] = ast.unparse(tree)
    assert analyze(sources)["violations"]
