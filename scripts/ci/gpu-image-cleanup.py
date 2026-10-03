"""Bounded cleanup receipts for temporary, attempt-labelled image resources."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

spec = importlib.util.spec_from_file_location(
    "gpu_ci", Path(__file__).with_name("trainer-rank-gpu.py")
)
assert spec is not None and spec.loader is not None
gpu_ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_ci)


def cleanup(context, namespace, selector, kinds, receipt, seconds=60):
    if not 0 < seconds <= 120 or not kinds or set(kinds) - {"pod", "service"}:
        raise ValueError("Bounded temporary Pod/Service cleanup required")
    labels = dict(part.split("=", 1) for part in selector.split(","))
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", namespace)
        or not any(
            key in labels
            for key in (
                "art.openpipe/build-run",
                "art.openpipe/prewarm-run",
                "art.openpipe/image-run",
            )
        )
        or any(not key or not value for key, value in labels.items())
    ):
        raise ValueError("Exact ownership labels required")
    deadline = time.monotonic() + seconds
    observed = []
    result = dict(
        context=context,
        namespace=namespace,
        selector=selector,
        kinds=kinds,
        scope="attempt_label_census",
        creator_quiescence="UNKNOWN",
        outcome="UNKNOWN",
        observed=observed,
        remaining=None,
    )

    def command(*args, body=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Image cleanup deadline expired")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            argv = [
                "kubectl",
                "--context",
                context,
                f"--request-timeout={remaining}s",
                *args,
            ]
            if body is not None:
                source = Path(directory) / "delete.json"
                source.write_text(json.dumps(body))
                argv += ["-f", str(source)]
            try:
                with (Path(directory) / "stdout").open("w") as stream:
                    gpu_ci.run_child(argv, remaining, output, stdout=stream)
            except subprocess.CalledProcessError:
                if body is None or not any(
                    error in output.read_text()
                    for error in ("(NotFound)", "(Conflict)")
                ):
                    raise
            return (Path(directory) / "stdout").read_text()

    try:
        while True:
            found = []
            for kind in kinds:
                items = json.loads(
                    command("get", kind, "-n", namespace, "-l", selector, "-o", "json")
                )["items"]
                for item in items:
                    metadata = item["metadata"]
                    name, uid = metadata["name"], metadata["uid"]
                    if (
                        metadata.get("namespace") != namespace
                        or item.get("kind", "").lower() != kind
                        or any(
                            metadata.get("labels", {}).get(key) != value
                            for key, value in labels.items()
                        )
                        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", name)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", uid)
                    ):
                        raise ValueError("Kubernetes returned an unowned resource")
                    identity = dict(kind=kind, name=name, uid=uid)
                    found.append(identity)
                    if identity not in observed:
                        observed.append(identity)
            result["remaining"] = found
            if not found:
                result["outcome"] = "ABSENT"
                return result
            for item in found:
                command(
                    "delete",
                    "--raw",
                    f"/api/v1/namespaces/{namespace}/{item['kind']}s/{item['name']}",
                    body=dict(
                        apiVersion="v1",
                        kind="DeleteOptions",
                        preconditions=dict(uid=item["uid"]),
                    ),
                )
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    except BaseException as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        gpu_ci.write_json(receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--selector", required=True)
    parser.add_argument("--kinds", nargs="+", default=["pod"])
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    print(
        json.dumps(
            cleanup(
                args.context,
                args.namespace,
                args.selector,
                args.kinds,
                args.receipt,
                args.seconds,
            )
        )
    )


if __name__ == "__main__":
    main()
