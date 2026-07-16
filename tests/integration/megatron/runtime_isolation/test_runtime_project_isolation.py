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


def test_parallel_sampling_preserves_every_child_policy_span(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace
import art_vllm_runtime.policy_spans as policy
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestState
from vllm.v1.engine.parallel_sampling import ParentRequest

policy._patch_output_processor_policy_span_accumulation()

class Detokenizer:
    output_token_ids = [10, 20]
    def num_output_tokens(self): return 2
    def get_next_output_text(self, finished, delta): return "done"

class Logprobs:
    logprobs = cumulative_logprob = prompt_logprobs = None

parent = ParentRequest.__new__(ParentRequest)
parent.external_req_id = "parent"
parent.child_requests = {"0_parent", "1_parent"}
parent.output_aggregator = [None, None]
parent.sampling_params = SimpleNamespace(
    output_kind=RequestOutputKind.FINAL_ONLY, n=2
)

def finish_child(index, policy_version):
    state = RequestState(
        request_id=f"{index}_parent", external_req_id="parent",
        parent_req=parent, request_index=index, lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY, prompt="p",
        prompt_token_ids=[1], prompt_embeds=None,
        logprobs_processor=Logprobs(), detokenizer=Detokenizer(),
        max_tokens_param=2, arrival_time=0.0, queue=None,
        log_stats=False, stream_interval=1,
    )
    policy._CURRENT_ENGINE_POLICY_SPANS = {state.request_id: [{
        "start_token": 0, "end_token": 2,
        "policy_version": policy_version,
        "lora_slot": "model:active", "update_seq": policy_version,
    }]}
    return state.make_request_output([10, 20], None, "stop", None)

assert finish_child(0, 3) is None
result = finish_child(1, 4)
print(json.dumps([
    output.art_policy_token_spans[0]["policy_version"]
    for output in result.outputs
]))
""",
        artifact_dir,
        "parallel_sampling_policy_spans",
    )
    assert json.loads(payload) == [3, 4]


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
    _set_policy_cache_salt,
)

async def main():
    slot = "model:active"
    old = SimpleNamespace(lora_name=slot, lora_path="old")
    new = SimpleNamespace(lora_name=slot, lora_path="new")
    request = SimpleNamespace(model=slot, cache_salt=None)
    _set_policy_cache_salt(request, lora_slot=slot, policy_version=5)

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

    async with coordinator.admission(slot):
        cancelled_update = asyncio.create_task(coordinator.begin_update(slot))
        await asyncio.sleep(0)
        cancelled_update.cancel()
        try:
            await cancelled_update
        except asyncio.CancelledError:
            pass
    async with coordinator.admission(slot) as recovered_state:
        recovered = recovered_state[0] == 5

    print(json.dumps({
        "blocked": blocked,
        "cache_salt": request.cache_salt,
        "policy_version": version,
        "lora_path": admitted_lora.lora_path,
        "recovered_after_cancel": recovered,
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
        "recovered_after_cancel": True,
    }


def test_runtime_general_plugin_loads_full_patch_set() -> None:
    pyproject = (ROOT / "vllm_runtime" / "pyproject.toml").read_text()
    assert 'art = "art_vllm_runtime.patches:apply_vllm_runtime_patches"' in pyproject


def test_runtime_patch_adds_gemma4_moe_topk_alias(artifact_dir: Path) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.patches import apply_vllm_runtime_patches; "
        "apply_vllm_runtime_patches(); "
        "from transformers import Gemma4TextConfig; "
        "config = Gemma4TextConfig(enable_moe_block=True, top_k_experts=8); "
        "print(json.dumps({'num_experts_per_tok': config.num_experts_per_tok}))",
        artifact_dir,
        "gemma4_topk_alias",
    )
    assert json.loads(payload) == {"num_experts_per_tok": 8}


def test_runtime_patch_skips_gemma4_layerwise_weight_update_reload(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.patches import apply_vllm_runtime_patches; "
        "apply_vllm_runtime_patches(); "
        "from vllm.v1.worker.gpu_worker import Worker; "
        "HfConfig = type('HfConfig', (), {"
        "'architectures': ['Gemma4ForConditionalGeneration']"
        "}); "
        "ModelConfig = type('ModelConfig', (), {'hf_config': HfConfig()}); "
        "DummyWorker = type('DummyWorker', (), {"
        "'model_config': ModelConfig(), "
        "'_weight_update_active': False, "
        "'_is_checkpoint_format': True, "
        "'checks': 0, "
        "'_check_weight_transfer_engine': "
        "lambda self: setattr(self, 'checks', self.checks + 1)"
        "}); "
        "dummy = DummyWorker(); "
        "Worker.start_weight_update(dummy, is_checkpoint_format=True); "
        "active_after_start = dummy._weight_update_active; "
        "Worker.finish_weight_update(dummy); "
        "print(json.dumps({"
        "'active_after_start': active_after_start, "
        "'active_after_finish': dummy._weight_update_active, "
        "'is_checkpoint_format': dummy._is_checkpoint_format, "
        "'checks': dummy.checks"
        "}))",
        artifact_dir,
        "gemma4_weight_update_reload",
    )
    assert json.loads(payload) == {
        "active_after_start": True,
        "active_after_finish": False,
        "is_checkpoint_format": True,
        "checks": 2,
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
