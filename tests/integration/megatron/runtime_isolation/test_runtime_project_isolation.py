import json
from pathlib import Path
import subprocess

from art.vllm_runtime import _vllm_runtime_subprocess_env

ROOT = Path(__file__).resolve().parents[4]


def _runtime_python(source: str, artifact_dir: Path, name: str) -> str:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            source,
        ],
        cwd=ROOT,
        env=_vllm_runtime_subprocess_env(),
        capture_output=True,
        text=True,
    )
    (artifact_dir / f"{name}_stdout.txt").write_text(result.stdout)
    (artifact_dir / f"{name}_stderr.txt").write_text(result.stderr)
    result.check_returncode()
    return result.stdout.strip()


def test_runtime_project_imports_in_its_own_project_env(artifact_dir: Path) -> None:
    payload = json.loads(
        _runtime_python(
            "import importlib.util, json; "
            "import art_vllm_runtime; "
            "print(json.dumps({"
            "'runtime_ok': True, "
            "'has_vllm': importlib.util.find_spec('vllm') is not None"
            "}))",
            artifact_dir,
            "runtime_import",
        )
    )
    assert payload == {"runtime_ok": True, "has_vllm": True}


def test_runtime_server_source_contains_only_required_custom_routes() -> None:
    source = (
        ROOT / "vllm_runtime" / "src" / "art_vllm_runtime" / "dedicated_server.py"
    ).read_text()
    for route in ("/sleep", "/wake_up", "/is_sleeping", "/art/set_served_model_name"):
        assert route in source


def test_runtime_patch_always_returns_token_ids(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.patches import apply_vllm_runtime_patches; "
        "apply_vllm_runtime_patches(); "
        "from vllm.entrypoints.openai.chat_completion import protocol; "
        "request = protocol.ChatCompletionRequest("
        "model='m', messages=[{'role': 'user', 'content': 'x'}]"
        "); "
        "print(json.dumps({"
        "'logprobs': request.logprobs, "
        "'top_logprobs': request.top_logprobs, "
        "'return_token_ids': request.return_token_ids"
        "}))",
        artifact_dir,
        "route_token_ids",
    )
    assert json.loads(payload) == {
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
    }


def test_runtime_lora_updates_linearize_request_admission(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import asyncio
import json
from types import SimpleNamespace
from art_vllm_runtime.policy_spans import (
    LoraUpdateCoordinator,
    _apply_lora_alias_policy_cache_salt,
    publish_lora_slot_policy,
    register_lora_alias,
)

async def main():
    slot = "model:active"
    old = SimpleNamespace(lora_name=slot, lora_path="old")
    new = SimpleNamespace(lora_name=slot, lora_path="new")
    models = SimpleNamespace(lora_requests={slot: new})
    register_lora_alias(models, public_model_name="model@4", lora_slot=slot)
    publish_lora_slot_policy(models, lora_slot=slot, policy_version=5)
    request = SimpleNamespace(model="model@4", cache_salt=None)
    _apply_lora_alias_policy_cache_salt(models, request, new)

    coordinator = LoraUpdateCoordinator()
    await coordinator.begin_update(slot)
    await coordinator.commit_update(slot, 4, old)
    await coordinator.begin_update(slot)

    async def admit():
        async with coordinator.admission(slot) as state:
            return state

    admission = asyncio.create_task(admit())
    await asyncio.sleep(0)
    blocked = not admission.done()
    await coordinator.commit_update(slot, 5, new)
    version, admitted_lora = await admission
    print(json.dumps({
        "blocked": blocked,
        "cache_salt": request.cache_salt,
        "policy_version": version,
        "lora_path": admitted_lora.lora_path,
    }, sort_keys=True))

asyncio.run(main())
""",
        artifact_dir,
        "lora_update_admission",
    )
    assert json.loads(payload) == {
        "blocked": True,
        "cache_salt": "art_policy_cache_salt=model:active:5",
        "lora_path": "new",
        "policy_version": 5,
    }


def test_runtime_policy_spans_survive_parallel_sample_aggregation(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace

from art_vllm_runtime.patches import _patch_openai_namespace_tool_import

_patch_openai_namespace_tool_import()
from vllm.v1.engine.output_processor import RequestState


def aggregate_final_outputs(self, new_token_ids, *args, **kwargs):
    if not self.finished:
        return None
    self.parent_req.outputs[self.request_index] = SimpleNamespace(
        index=self.request_index
    )
    self.parent_req.finished += 1
    if self.parent_req.finished < len(self.parent_req.outputs):
        return None
    return SimpleNamespace(outputs=self.parent_req.outputs)


RequestState.make_request_output = aggregate_final_outputs
import art_vllm_runtime.policy_spans as policy_spans

policy_spans._patch_output_processor_policy_span_accumulation()
parent = SimpleNamespace(outputs=[None] * 4, finished=0)
none_count = 0
final_output = None
for choice_index in range(4):
    detokenizer = SimpleNamespace(num_output_tokens=lambda: 0)
    state = SimpleNamespace(
        request_id=f"child-{choice_index}",
        request_index=choice_index,
        parent_req=parent,
        detokenizer=detokenizer,
        finished=False,
    )
    for token_index in range(5):
        detokenizer.num_output_tokens = lambda count=token_index + 1: count
        state.finished = token_index == 4
        policy_spans._CURRENT_ENGINE_POLICY_SPANS = {
            state.request_id: [{
                "start_token": 0,
                "end_token": 1,
                "policy_version": 7,
                "lora_slot": "model:active",
                "update_seq": 3,
            }]
        }
        output = RequestState.make_request_output(state, [100 + token_index])
        if output is None:
            none_count += 1
        else:
            final_output = output

print(json.dumps({
    "none_count": none_count,
    "spans": [
        getattr(output, policy_spans.ART_POLICY_TOKEN_SPANS_FIELD, None)
        for output in final_output.outputs
    ],
}, sort_keys=True))
""",
        artifact_dir,
        "parallel_sample_policy_spans",
    )
    assert json.loads(payload) == {
        "none_count": 19,
        "spans": [
            [
                {
                    "end_token": 5,
                    "lora_slot": "model:active",
                    "policy_version": 7,
                    "start_token": 0,
                    "update_seq": 3,
                }
            ]
        ]
        * 4,
    }


def test_runtime_general_plugin_loads_full_patch_set() -> None:
    pyproject = (ROOT / "vllm_runtime" / "pyproject.toml").read_text()
    assert 'art = "art_vllm_runtime.patches:apply_vllm_runtime_patches"' in pyproject


def test_runtime_patch_selects_checkpoint_weight_update_lifecycle(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace

from art_vllm_runtime.patches import apply_vllm_runtime_patches

apply_vllm_runtime_patches()
from vllm.v1.worker.gpu_worker import Worker


class Engine:
    def __init__(self):
        self.starts = self.updates = self.finishes = 0

    def start_weight_update(self):
        self.starts += 1

    def update_weights(self, update_info):
        self.updates += 1

    def finish_weight_update(self):
        self.finishes += 1


def exercise(architecture):
    engine = Engine()
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        ),
        weight_transfer_engine=engine,
        _weight_update_active=False,
        _check_weight_transfer_engine=lambda: None,
    )
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": []})
    Worker.finish_weight_update(worker)
    return {
        "starts": engine.starts,
        "updates": engine.updates,
        "finishes": engine.finishes,
        "active": worker._weight_update_active,
    }


print(json.dumps({
    "dense": exercise("Qwen3ForCausalLM"),
    "gemma4": exercise("Gemma4ForConditionalGeneration"),
}, sort_keys=True))
""",
        artifact_dir,
        "checkpoint_weight_update_lifecycle",
    )
    assert json.loads(payload) == {
        "dense": {"active": False, "finishes": 1, "starts": 1, "updates": 1},
        "gemma4": {"active": False, "finishes": 0, "starts": 0, "updates": 1},
    }


def test_runtime_patch_set_does_not_install_lora_monkey_patches() -> None:
    source = (
        ROOT / "vllm_runtime" / "src" / "art_vllm_runtime" / "patches.py"
    ).read_text()
    assert "patch_punica_ep_moe_lora_alignment" not in source
    assert "patch_lora_duplicate_module_aliases" not in source
    assert "patch_fused_moe_ep_lora_support" not in source


def test_runtime_cli_serializes_lora_target_modules_as_single_nargs_vector(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.dedicated_server import _append_cli_arg; "
        "args = []; "
        "_append_cli_arg(args, 'lora_target_modules', ['a', 'b']); "
        "print(json.dumps(args))",
        artifact_dir,
        "lora_target_modules",
    )
    assert json.loads(payload) == ["--lora-target-modules", "a", "b"]


def test_runtime_project_restores_nccl_unique_id_from_raw_bytes(
    artifact_dir: Path,
) -> None:
    payload = json.loads(
        _runtime_python(
            "import ctypes, json; "
            "from art_vllm_runtime.patches import _restore_nccl_unique_id_payload; "
            "from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId; "
            "payload = bytes(range(128)); "
            "restored = _restore_nccl_unique_id_payload(payload, ncclUniqueId()); "
            "print(json.dumps({"
            "'type': type(restored).__name__, "
            "'matches': ctypes.string_at(ctypes.byref(restored), ctypes.sizeof(restored)).hex() == payload.hex()"
            "}))",
            artifact_dir,
            "restore",
        )
    )
    assert payload == {"type": "ncclUniqueId", "matches": True}


def test_runtime_project_nccl_wrapper_accepts_raw_bytes(artifact_dir: Path) -> None:
    payload = json.loads(
        _runtime_python(
            "import json; "
            "from art_vllm_runtime.patches import _normalize_nccl_comm_init_rank_unique_id; "
            "FakeLibrary = type('FakeLibrary', (), {'unique_id_from_bytes': lambda self, data: {'restored': len(data)}}); "
            "restored = _normalize_nccl_comm_init_rank_unique_id(FakeLibrary(), bytes(range(128))); "
            "print(json.dumps(restored))",
            artifact_dir,
            "nccl_wrapper",
        )
    )
    assert payload == {"restored": 128}
