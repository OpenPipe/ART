import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def test_runtime_project_imports_in_its_own_project_env(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import importlib.util, json; "
                "import art_vllm_runtime; "
                "print(json.dumps({"
                "'runtime_ok': True, "
                "'has_vllm': importlib.util.find_spec('vllm') is not None"
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "stdout.txt").write_text(result.stdout)
    (artifact_dir / "stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
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
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, os; "
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
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "route_token_ids_stdout.txt").write_text(result.stdout)
    (artifact_dir / "route_token_ids_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == {
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
    }


def test_runtime_policy_spans_read_final_request_output(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, pickle; "
                "from types import SimpleNamespace; "
                "from art_vllm_runtime.policy_spans import ("
                "ART_POLICY_TOKEN_SPANS_FIELD, "
                "_policy_spans_by_choice_from_final_output"
                "); "
                "spans = [{'start_token': 0, 'end_token': 3, "
                "'policy_version': 4, 'lora_slot': 'm:active', "
                "'update_seq': 2}]; "
                "choice = SimpleNamespace(index=1); "
                "setattr(choice, ART_POLICY_TOKEN_SPANS_FIELD, spans); "
                "payload = _policy_spans_by_choice_from_final_output("
                "SimpleNamespace(outputs=[choice])"
                "); "
                "print(json.dumps(payload))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "policy_spans_stdout.txt").write_text(result.stdout)
    (artifact_dir / "policy_spans_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == {
        "1": [
            {
                "start_token": 0,
                "end_token": 3,
                "policy_version": 4,
                "lora_slot": "m:active",
                "update_seq": 2,
            }
        ]
    }


def test_runtime_policy_spans_accumulate_engine_core_outputs(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, pickle; "
                "from types import SimpleNamespace; "
                "from art_vllm_runtime.policy_spans import ("
                "ART_POLICY_TOKEN_SPANS_FIELD, "
                "_engine_core_policy_spans_by_request"
                "); "
                "spans = [{'start_token': 0, 'end_token': 1, "
                "'policy_version': 2, 'lora_slot': 'm:active', "
                "'update_seq': 5}]; "
                "flat = SimpleNamespace(request_id='flat'); "
                "nested = SimpleNamespace(request_id='nested'); "
                "setattr(flat, ART_POLICY_TOKEN_SPANS_FIELD, spans); "
                "setattr(nested, ART_POLICY_TOKEN_SPANS_FIELD, spans); "
                "payload = _engine_core_policy_spans_by_request(["
                "flat, SimpleNamespace(outputs=[nested])"
                "]); "
                "print(json.dumps(payload, sort_keys=True))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "engine_core_policy_spans_stdout.txt").write_text(result.stdout)
    (artifact_dir / "engine_core_policy_spans_stderr.txt").write_text(result.stderr)
    expected = [
        {
            "start_token": 0,
            "end_token": 1,
            "policy_version": 2,
            "lora_slot": "m:active",
            "update_seq": 5,
        }
    ]
    assert json.loads(result.stdout.strip()) == {
        "flat": expected,
        "nested": expected,
    }


def test_runtime_policy_spans_declares_model_runner_output_field(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json, pickle; "
                "from dataclasses import fields; "
                "from art_vllm_runtime.policy_spans import ("
                "ART_POLICY_TOKEN_SPANS_FIELD, patch_policy_token_spans"
                "); "
                "patch_policy_token_spans(); "
                "import vllm.v1.outputs as outputs_mod; "
                "import vllm.v1.worker.gpu_model_runner as active_runner; "
                "import vllm.v1.worker.gpu_worker as gpu_worker; "
                "from vllm.v1.outputs import ModelRunnerOutput; "
                "spans = {'r': [{'start_token': 0, 'end_token': 2, "
                "'policy_version': 7, 'lora_slot': 'm:active', "
                "'update_seq': 3}]}; "
                "output = ModelRunnerOutput("
                "req_ids=['r'], req_id_to_index={'r': 0}, "
                "sampled_token_ids=[[1, 2]], art_policy_token_spans=spans"
                "); "
                "roundtrip = pickle.loads(pickle.dumps(output)); "
                "print(json.dumps({"
                "'active_async_patched': getattr("
                "active_runner.AsyncGPUModelRunnerOutput.get_output, "
                "'__art_policy_spans_patched__', False), "
                "'active_sample_patched': getattr("
                "active_runner.GPUModelRunner.sample_tokens, "
                "'__art_policy_spans_patched__', False), "
                "'class_shared': active_runner.ModelRunnerOutput is "
                "ModelRunnerOutput and gpu_worker.ModelRunnerOutput is "
                "ModelRunnerOutput, "
                "'empty_shared': active_runner.EMPTY_MODEL_RUNNER_OUTPUT is "
                "outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT, "
                "'has_field': ART_POLICY_TOKEN_SPANS_FIELD in "
                "[field.name for field in fields(ModelRunnerOutput)], "
                "'spans': getattr(output, ART_POLICY_TOKEN_SPANS_FIELD), "
                "'roundtrip_spans': getattr(roundtrip, ART_POLICY_TOKEN_SPANS_FIELD)"
                "}, sort_keys=True))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "model_runner_policy_spans_stdout.txt").write_text(result.stdout)
    (artifact_dir / "model_runner_policy_spans_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == {
        "active_async_patched": True,
        "active_sample_patched": True,
        "class_shared": True,
        "empty_shared": True,
        "has_field": True,
        "spans": {
            "r": [
                {
                    "start_token": 0,
                    "end_token": 2,
                    "policy_version": 7,
                    "lora_slot": "m:active",
                    "update_seq": 3,
                }
            ]
        },
        "roundtrip_spans": {
            "r": [
                {
                    "start_token": 0,
                    "end_token": 2,
                    "policy_version": 7,
                    "lora_slot": "m:active",
                    "update_seq": 3,
                }
            ]
        },
    }


def test_runtime_policy_spans_reads_active_input_batch_lora_mapping(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json; "
                "from types import SimpleNamespace; "
                "from art_vllm_runtime.policy_spans import "
                "_policy_context_from_runner; "
                "lora = SimpleNamespace("
                "lora_int_id=11, lora_name='active@6', "
                "lora_path='/tmp/checkpoints/0006'"
                "); "
                "input_batch = SimpleNamespace("
                "req_ids=['req'], req_id_to_index={'req': 0}, "
                "request_lora_mapping=[11], lora_id_to_lora_request={11: lora}"
                "); "
                "direct = _policy_context_from_runner("
                "SimpleNamespace(input_batch=input_batch)"
                "); "
                "state = _policy_context_from_runner("
                "SimpleNamespace("
                "input_batch=None, "
                "execute_model_state=SimpleNamespace(input_batch=input_batch)"
                ")"
                "); "
                "print(json.dumps({'direct': direct, 'state': state}, sort_keys=True))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "policy_span_input_batch_stdout.txt").write_text(result.stdout)
    (artifact_dir / "policy_span_input_batch_stderr.txt").write_text(result.stderr)
    expected = {
        "req": {
            "lora_slot": "active@6",
            "policy_version": 6,
            "update_seq": 1,
        }
    }
    assert json.loads(result.stdout.strip()) == {
        "direct": expected,
        "state": expected,
    }


def test_runtime_general_plugin_loads_full_patch_set() -> None:
    pyproject = (ROOT / "vllm_runtime" / "pyproject.toml").read_text()
    assert 'art = "art_vllm_runtime.patches:apply_vllm_runtime_patches"' in pyproject


def test_runtime_patch_adds_gemma4_moe_topk_alias(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json; "
                "from art_vllm_runtime.patches import apply_vllm_runtime_patches; "
                "apply_vllm_runtime_patches(); "
                "from transformers import Gemma4TextConfig; "
                "config = Gemma4TextConfig(enable_moe_block=True, top_k_experts=8); "
                "print(json.dumps({'num_experts_per_tok': config.num_experts_per_tok}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "gemma4_topk_alias_stdout.txt").write_text(result.stdout)
    (artifact_dir / "gemma4_topk_alias_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == {"num_experts_per_tok": 8}


def test_runtime_patch_skips_gemma4_layerwise_weight_update_reload(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
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
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "gemma4_weight_update_reload_stdout.txt").write_text(result.stdout)
    (artifact_dir / "gemma4_weight_update_reload_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == {
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
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json; "
                "from art_vllm_runtime.dedicated_server import _append_cli_arg; "
                "args = []; "
                "_append_cli_arg(args, 'lora_target_modules', ['a', 'b']); "
                "print(json.dumps(args))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "lora_target_modules_stdout.txt").write_text(result.stdout)
    (artifact_dir / "lora_target_modules_stderr.txt").write_text(result.stderr)
    assert json.loads(result.stdout.strip()) == ["--lora-target-modules", "a", "b"]


def test_runtime_project_restores_nccl_unique_id_from_raw_bytes(
    artifact_dir: Path,
) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import ctypes, json; "
                "from art_vllm_runtime.patches import _restore_nccl_unique_id_payload; "
                "from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId; "
                "payload = bytes(range(128)); "
                "restored = _restore_nccl_unique_id_payload(payload, ncclUniqueId()); "
                "print(json.dumps({"
                "'type': type(restored).__name__, "
                "'matches': ctypes.string_at(ctypes.byref(restored), ctypes.sizeof(restored)).hex() == payload.hex()"
                "}))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "restore_stdout.txt").write_text(result.stdout)
    (artifact_dir / "restore_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {"type": "ncclUniqueId", "matches": True}


def test_runtime_project_nccl_wrapper_accepts_raw_bytes(artifact_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            (
                "import json; "
                "from art_vllm_runtime.patches import _normalize_nccl_comm_init_rank_unique_id; "
                "FakeLibrary = type('FakeLibrary', (), {'unique_id_from_bytes': lambda self, data: {'restored': len(data)}}); "
                "restored = _normalize_nccl_comm_init_rank_unique_id(FakeLibrary(), bytes(range(128))); "
                "print(json.dumps(restored))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "nccl_wrapper_stdout.txt").write_text(result.stdout)
    (artifact_dir / "nccl_wrapper_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip())
    assert payload == {"restored": 128}
