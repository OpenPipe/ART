"""Exercise the actual shell builder with a private, provider-free kubectl fake."""

import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[2]


@pytest.fixture
def builder(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, signal, sys, time
import yaml
root = pathlib.Path(os.environ['FAKE_BUILD_ROOT'])
args = sys.argv[1:]
with (root/'calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\n')
state = root/'pod.json'
mode = os.environ.get('FAKE_BUILD_MODE', 'success')
if 'create' in args:
    raw = pathlib.Path(args[args.index('-f')+1]).read_text()
    (root/'manifest.yaml').write_text(raw)
    pod = yaml.safe_load(raw)
    pod['metadata'].update(uid='owned-builder', resourceVersion='1', namespace='default')
    state.write_text(json.dumps(pod))
elif 'wait' in args:
    if mode == 'wait_hang': time.sleep(60)
    if mode == 'wait_ignores_term':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
elif 'cp' in args:
    if args[-2].endswith(':/tmp/art-build.log'):
        if mode == 'log_hang': time.sleep(60)
        pathlib.Path(args[-1]).write_text('#1 pushing manifest for registry.invalid/art:latest@sha256:'+'a'*64+' 0.1s done\n')
elif 'exec' in args:
    if '/tmp/art-build.exit' in args[-1] and args[-1].startswith('if '):
        if mode in ('success', 'processor_hang', 'final_processor_hang'): print('0')
        elif mode in ('failed', 'failed_final_processor_hang'): print('42')
        elif mode != 'pending': raise AssertionError(mode)
elif 'get' in args:
    selector = args[args.index('-l')+1]
    items = [json.loads(state.read_text())] if state.exists() and selector == 'art.openpipe/build-run=offline-1' else []
    print(json.dumps({'items': items}))
elif 'delete' in args:
    body = json.loads(pathlib.Path(args[args.index('-f')+1]).read_text())
    assert body['preconditions'] == {'uid': 'owned-builder', 'resourceVersion': '1'}
    state.unlink()
else:
    raise AssertionError(args)
"""
    )
    kubectl.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + r"""
import os, pathlib, sys, time
root = pathlib.Path(os.environ['FAKE_BUILD_ROOT'])
if len(sys.argv) > 6 and 'art-gpu-build-log.' in sys.argv[5]:
    counter = root/'processor-calls'
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    mode = os.environ.get('FAKE_BUILD_MODE')
    if mode == 'processor_hang' or (mode in ('final_processor_hang', 'failed_final_processor_hang') and count == 2):
        time.sleep(60)
os.execv(sys.executable, [sys.executable, *sys.argv[4:]])
"""
    )
    uv.chmod(0o755)
    for name in ("gh", "curl", "docker", "sky"):
        deny = bin_dir / name
        deny.write_text('#!/bin/sh\necho "Unexpected external command" >&2\nexit 99\n')
        deny.chmod(0o755)

    def run(**changes):
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_BUILD_ROOT": str(tmp_path),
            "DOCKER_CONFIG_PATH": str(tmp_path / "absent.json"),
            "REGISTRY_AUTH_JSON_B64": base64.b64encode(b"{}").decode(),
            "GPU_IMAGE_CLEANUP_ROOT": str(tmp_path / "cleanup"),
            "PREWARM_RUN_UID": "offline-1",
            "BUILDKIT_TIMEOUT_SECONDS": "30",
            **changes,
        }
        command = [
            "bash",
            str(ROOT / "scripts/build-gpu-image.sh"),
            "--cluster-name",
            "offline-builder",
            "--infra",
            "k8s/offline",
            "--image-repo",
            "registry.invalid/art",
            "--no-prewarm-nodes",
            "--no-prewarm-modal",
        ]
        with subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as child:
            try:
                out, err = child.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                raise
        return subprocess.CompletedProcess(command, child.returncode, out, err)

    return tmp_path, run


def test_builder_renders_requests_and_native_deadline(builder):
    root, run = builder
    result = run(BUILDKIT_TIMEOUT_SECONDS="21600")
    assert result.returncode == 0, result.stderr
    spec = yaml.safe_load((root / "manifest.yaml").read_text())["spec"]
    assert spec["activeDeadlineSeconds"] == 21600
    assert spec["restartPolicy"] == "Never"
    assert spec["containers"][0]["resources"] == {
        "requests": {"cpu": "2", "memory": "64Gi", "ephemeral-storage": "80Gi"}
    }
    assert not (root / "pod.json").exists()
    assert (
        json.loads((root / "cleanup/trap-builder.json").read_text())["outcome"]
        == "ABSENT"
    )


@pytest.mark.parametrize(
    "seconds", ["0", "-1", "21601", "1.5", "nope", "99999999999999999999"]
)
def test_builder_refuses_invalid_lifetime_before_provider_effects(builder, seconds):
    root, run = builder
    result = run(BUILDKIT_TIMEOUT_SECONDS=seconds)
    assert result.returncode != 0
    assert "BUILDKIT_TIMEOUT_SECONDS" in result.stderr
    assert not (root / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("pending", 124),
        ("wait_hang", 124),
        ("log_hang", 124),
        ("wait_ignores_term", 137),
    ],
)
def test_builder_deadline_reaches_owned_cleanup_without_exit_file(builder, mode, code):
    root, run = builder
    result = run(BUILDKIT_TIMEOUT_SECONDS="1", FAKE_BUILD_MODE=mode)
    assert result.returncode == code, result.stderr
    assert not (root / "pod.json").exists()
    assert (
        json.loads((root / "cleanup/trap-builder.json").read_text())["outcome"]
        == "ABSENT"
    )


def test_builder_preserves_build_failure_and_cleanup(builder):
    root, run = builder
    result = run(FAKE_BUILD_MODE="failed")
    assert result.returncode == 42, result.stderr
    assert not (root / "pod.json").exists()


@pytest.mark.parametrize(
    "mode,code,calls",
    [
        ("processor_hang", 124, 1),
        ("final_processor_hang", 124, 2),
        ("failed_final_processor_hang", 42, 2),
    ],
)
def test_builder_bounds_log_processor_and_preserves_known_failure(
    builder, mode, code, calls
):
    root, run = builder
    result = run(BUILDKIT_TIMEOUT_SECONDS="3", FAKE_BUILD_MODE=mode)
    assert result.returncode == code, result.stderr
    assert int((root / "processor-calls").read_text()) == calls
    assert not (root / "pod.json").exists()
    assert (
        json.loads((root / "cleanup/trap-builder.json").read_text())["outcome"]
        == "ABSENT"
    )
