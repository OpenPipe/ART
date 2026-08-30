from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import socket
import struct
import subprocess
import sys
from typing import Any, AsyncIterator, Iterator, cast
import uuid

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field

from art.dev.model import InternalModelConfig
from art.megatron.prefix_tree import parse_prefix_tree
from art.preprocessing.moe_routing import (
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
    choice_moe_routing_metadata,
)
from art.preprocessing.pack import DiskPackedTensors
from art.preprocessing.policy_spans import (
    choice_policy_token_spans,
    validate_complete_policy_token_spans,
)
from art.preprocessing.vllm_tokens import choice_vllm_token_metadata

from .artifacts import ARTIFACTS_ROOT_ENV, REPO_ROOT
from .output_parity import (
    TOP_K,
    LogicalToken,
    LogicalTokenMap,
    PairComparison,
    RolloutMode,
    ScoreBundle,
    TokenTopK,
    TopKComparison,
    TrainInfOutputParityConfig,
    WeightState,
    _build_deterministic_nonzero_lora,
    _collect_full_lora_state,
    _configure_lora_target_modules,
    _configure_provider,
    _extract_scores_from_logits,
    _lora_target_modules,
    _read_json,
    _run_logits,
    _save_vllm_lora_adapter,
    _set_seed,
    _write_json,
    build_logical_token_map,
    compare_pair,
    compare_topk,
    fwd_mean_abs_pct_limit_for_model,
    model_support_is_moe,
    score_context_parallel_runtime,
    top20_kl_candidate_to_target_limit_for_model,
)


class RealPathConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_parity: TrainInfOutputParityConfig = Field(
        default_factory=TrainInfOutputParityConfig
    )
    prompt_count: int = 4
    rollouts_per_prompt: int = 2
    max_completion_tokens: int = 16
    prompt_sentence_count: int = 2
    prompt_target_tokens: int | None = None
    sliding_window: int | None = None
    diagnose_base: bool = False
    trace_layers: bool = False
    trace_enforce_eager: bool = False
    adapter_cache_dir: str | None = None


class RealPathMegatronWorkerRequest(BaseModel):
    config: TrainInfOutputParityConfig
    artifact_dir: str
    disk_packed_tensors: DiskPackedTensors
    logical_map_path: str
    weight_state: WeightState
    adapter_path: str | None = None
    moe_routing_replay_path: str | None = None
    global_grad_accumulation_sequences: int
    forward_trace_dir: str | None = None


class RealPathMegatronWorkerResult(BaseModel):
    score_path: str
    adapter_path: str | None = None


class RealPathBaseDiagnosticBundle(BaseModel):
    vllm_scores: ScoreBundle
    megatron_scores: ScoreBundle
    megatron_score_path: str
    vllm_score_path: str
    logical_prompt_count: int
    logical_token_count: int
    moe_routing_packed_tokens: int
    vllm_forward_trace_dir: str | None = None
    megatron_forward_trace_dir: str | None = None


@contextmanager
def _temporary_env(updates: dict[str, str] | None) -> Iterator[None]:
    if not updates:
        yield
        return
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class RealPathTrainInfReport(BaseModel):
    base_model: str
    artifact_dir: str
    logical_prompt_count: int
    logical_token_count: int
    base_logical_prompt_count: int | None = None
    base_logical_token_count: int | None = None
    base_moe_routing_packed_tokens: int | None = None
    adapter_path: str
    adapter_cache_key: str
    adapter_cache_hit: bool
    megatron_base_scores: str | None = None
    vllm_base_scores: str | None = None
    megatron_lora_scores: str
    vllm_lora_scores: str
    base: PairComparison | None = None
    base_topk: TopKComparison | None = None
    lora: PairComparison
    lora_topk: TopKComparison
    moe_routing_packed_tokens: int
    prompt_tree_depth: int = 0
    prompt_tree_branch_count: int = 0
    mean_abs_pct_limit: float
    top20_kl_candidate_to_target_limit: float
    passed: bool


class AdapterCacheResult(BaseModel):
    path: str
    cache_key: str
    cache_hit: bool


class ResidentTrainInfAttempt(BaseModel):
    attempt: int
    logical_prompt_count: int
    logical_token_count: int
    lora: PairComparison
    lora_topk: TopKComparison
    moe_routing_packed_tokens: int
    batch_fingerprint: str
    passed: bool


class ResidentTrainInfReport(BaseModel):
    base_model: str
    artifact_dir: str
    run_id: str
    policy_step: int
    generation_id: str
    attempt_count: int
    max_attempts: int
    attempts: list[ResidentTrainInfAttempt]
    mean_abs_pct_limit: float
    top20_kl_candidate_to_target_limit: float
    passed: bool


def _real_path_rollout_mode(config: TrainInfOutputParityConfig) -> RolloutMode:
    return config.rollout_modes[0]


def _packed_batch_fingerprint(packed_tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("tokens", "group_ids", "parent_ids", "input_pos", "assistant_mask"):
        tensor = packed_tensors[name].detach().cpu().contiguous()
        for value in (name, str(tuple(tensor.shape)), str(tensor.dtype)):
            payload = value.encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        payload = tensor.numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    replay = packed_tensors.get("moe_routing_replay")
    if replay is not None:
        tensor = replay.expert_indices.detach().cpu().contiguous()
        for value in (
            "moe_routing_replay",
            str(tuple(tensor.shape)),
            str(tensor.dtype),
        ):
            payload = value.encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        payload = tensor.numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


_PROMPT_SENTENCES = [
    "A careful systems engineer checks assumptions before changing thresholds.",
    "The training batch contains prefix treees and divergent completions.",
    "Numerical parity should be measured on the exact tokens used by the policy.",
    "Sparse expert routing can create discontinuous output differences.",
    "A reproducible test writes enough artifacts to explain every comparison.",
    "LoRA adapters must be active and nonzero during both inference and training.",
    "The prompt should be realistic enough to exercise ordinary tokenizer paths.",
    "Packed Megatron inputs use prefix treees while vLLM receives flat requests.",
    "If tokenization diverges, the mismatch should fail as early as possible.",
    "The report includes target logprobs and top token overlap for diagnosis.",
    "Routing replay should use vLLM expert ids captured from real rollouts.",
    "The model is asked to continue a compact technical note with concrete facts.",
    "Every rollout in a group starts from the same prompt and then branches.",
    "The comparison avoids hidden fallbacks that mask training inference drift.",
    "Strict tests should make incorrect assumptions visible instead of tolerating them.",
    "A small live probe can still cover important module and routing behavior.",
    "The artifact bundle records the packed tensor layout used by training.",
    "Inference responses provide logprobs for generated assistant tokens.",
    "Megatron replay receives expert ids before each router executes.",
    "The same adapter checkpoint should drive the served and trained policy.",
    "Top-k overlap is useful because sampling behavior depends on ranking.",
    "Mean absolute percent follows the support branch elementwise convention.",
    "The run should not update weights just to measure a forward mismatch.",
    "Validation code belongs in tests unless production needs the behavior.",
]
_PROMPT_TOKENS_PER_SENTENCE_ESTIMATE = 12


def _shared_prompt_segment(text: str) -> str:
    return " ".join((text,) * 4)


_PROMPT_TREE_ROOT = _shared_prompt_segment(
    "Validate this training and serving comparison with concrete evidence about "
    "tokenization, packed execution, routed experts, adapter weights, and output scores."
)
_PROMPT_TREE_MIDS = (
    _shared_prompt_segment(
        "Branch alpha emphasizes runtime validation, packed training inputs, "
        "deterministic metadata, and exact policy tokens for every comparison."
    ),
    _shared_prompt_segment(
        "Branch beta emphasizes serving parity, reproducible artifacts, adapter "
        "loading, and measured output differences for every comparison."
    ),
)
_PROMPT_TREE_LEAVES = (
    _shared_prompt_segment(
        "Case one follows a prefix-tree branch into the first completion and records "
        "each relevant execution decision."
    ),
    _shared_prompt_segment(
        "Case two follows the same intermediate branch into a second prompt leaf and "
        "records each relevant execution decision."
    ),
    _shared_prompt_segment(
        "Case three follows another intermediate branch into a different expert route "
        "and records each relevant execution decision."
    ),
    _shared_prompt_segment(
        "Case four follows that intermediate branch into its final prompt leaf and "
        "records each relevant execution decision."
    ),
)


def config_from_env() -> RealPathConfig:
    from art.megatron.model_support.registry import get_model_support_spec

    from .output_parity import config_from_env as output_config_from_env

    config = RealPathConfig(output_parity=output_config_from_env())
    if raw := os.environ.get("ART_REAL_PATH_PROMPT_COUNT"):
        config.prompt_count = int(raw)
    elif (
        get_model_support_spec(
            config.output_parity.base_model,
            allow_unvalidated_arch=config.output_parity.allow_unvalidated_arch,
        ).handler_key
        == "dsv4"
    ):
        config.prompt_count = 2
    if raw := os.environ.get("ART_REAL_PATH_ROLLOUTS_PER_PROMPT"):
        config.rollouts_per_prompt = int(raw)
    if raw := os.environ.get("ART_REAL_PATH_MAX_COMPLETION_TOKENS"):
        config.max_completion_tokens = int(raw)
    if raw := os.environ.get("ART_REAL_PATH_PROMPT_SENTENCE_COUNT"):
        config.prompt_sentence_count = int(raw)
    if raw := os.environ.get("ART_REAL_PATH_PROMPT_TARGET_TOKENS"):
        config.prompt_target_tokens = int(raw)
    if raw := os.environ.get("ART_REAL_PATH_DIAGNOSE_BASE"):
        config.diagnose_base = raw == "1"
    if raw := os.environ.get("ART_REAL_PATH_TRACE_LAYERS"):
        config.trace_layers = raw == "1"
        if config.trace_layers:
            config.diagnose_base = True
    if raw := os.environ.get("ART_REAL_PATH_TRACE_ENFORCE_EAGER"):
        config.trace_enforce_eager = raw == "1"
    if raw := os.environ.get("ART_TRAIN_INF_MISMATCH_ADAPTER_CACHE_DIR"):
        config.adapter_cache_dir = raw
    return config


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _config_sliding_window(config: TrainInfOutputParityConfig) -> int | None:
    from huggingface_hub import hf_hub_download

    local_config_path = Path(config.base_model) / "config.json"
    config_path = (
        local_config_path
        if local_config_path.exists()
        else Path(hf_hub_download(config.base_model, "config.json"))
    )
    hf_config = _read_json(config_path)
    text_config = hf_config.get("text_config", hf_config)
    if not isinstance(text_config, dict):
        return None
    layer_types = tuple(str(value) for value in text_config.get("layer_types", ()))
    if not any("sliding" in layer_type for layer_type in layer_types):
        return None
    window = text_config.get("sliding_window")
    if window is None:
        return None
    return int(window)


def _apply_sliding_window_prompt_defaults(config: RealPathConfig) -> None:
    window = _config_sliding_window(config.output_parity)
    if window is None:
        return
    config.sliding_window = window
    if config.prompt_target_tokens is None:
        config.prompt_target_tokens = 2 * window
    config.prompt_sentence_count = max(
        config.prompt_sentence_count,
        (config.prompt_target_tokens + _PROMPT_TOKENS_PER_SENTENCE_ESTIMATE - 1)
        // _PROMPT_TOKENS_PER_SENTENCE_ESTIMATE,
    )
    min_sequence_length = _round_up(
        config.prompt_target_tokens + config.max_completion_tokens + 256,
        128,
    )
    if config.output_parity.packed.sequence_length < min_sequence_length:
        config.output_parity.packed.sequence_length = min_sequence_length


def _build_prompt_from_sentences(index: int, sentences: list[str]) -> str:
    mid = _PROMPT_TREE_MIDS[(index // 2) % len(_PROMPT_TREE_MIDS)]
    leaf = _PROMPT_TREE_LEAVES[index % len(_PROMPT_TREE_LEAVES)]
    return f"{_PROMPT_TREE_ROOT}\n\n{mid}\n\n{leaf}\n\nNotes: " + " ".join(sentences)


def _prompt_tree_shape(packed_tensors: dict[str, Any]) -> tuple[int, int]:
    segments = tuple(
        segment
        for row in parse_prefix_tree(
            group_ids=packed_tensors["group_ids"],
            parent_ids=packed_tensors["parent_ids"],
        )
        for segment in row.segments
    )
    return (
        max((segment.depth for segment in segments), default=0),
        sum(segment.depth > 0 for segment in segments),
    )


def _dsv4_aligned_prompts(config: RealPathConfig, tokenizer: Any) -> list[str]:
    if config.prompt_count != 2:
        raise ValueError("DSV4 train/inf mismatch requires exactly two aligned prompts")
    root = "Calibration root." + " common" * 249 + "\n"
    prompts = [root + " alpha" * 254, root + " beta" * 254]
    token_ids = [
        list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        for prompt in prompts
    ]
    shared = next(
        index
        for index, pair in enumerate(zip(*token_ids, strict=True))
        if pair[0] != pair[1]
    )
    if shared != 256 or any(len(tokens) != 512 for tokens in token_ids):
        raise RuntimeError(
            "DSV4 aligned fixture tokenization changed: "
            f"shared={shared}, lengths={[len(tokens) for tokens in token_ids]}"
        )
    return prompts


def _build_prompts(config: RealPathConfig, tokenizer: Any) -> list[str]:
    from art.megatron.model_support.registry import get_model_support_spec

    if (
        get_model_support_spec(
            config.output_parity.base_model,
            allow_unvalidated_arch=config.output_parity.allow_unvalidated_arch,
        ).handler_key
        == "dsv4"
    ):
        # FP4 vLLM rescoring varies enough for DSV4 that arbitrary short tails
        # dominate train/inf mismatch. Exact 256-token blocks maximize stable KV
        # reuse while retaining a real shared-root, divergent-branch packed tree.
        return _dsv4_aligned_prompts(config, tokenizer)
    rng = random.Random(config.output_parity.seed)
    prompts: list[str] = []
    for index in range(config.prompt_count):
        sentences = [
            rng.choice(_PROMPT_SENTENCES) for _ in range(config.prompt_sentence_count)
        ]
        prompts.append(_build_prompt_from_sentences(index, sentences))
    return prompts


def _real_path_max_model_len(
    config: RealPathConfig,
    *,
    tokenizer: Any,
    prompts: list[str],
    chat_template_kwargs: dict[str, Any],
) -> int:
    def rendered_tokens(messages: list[dict[str, str]], **kwargs: Any) -> int:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            **chat_template_kwargs,
            **kwargs,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        shape: Any = getattr(token_ids, "shape", ())
        if len(shape) > 1:
            return int(shape[-1])
        if len(token_ids) == 1 and isinstance(token_ids[0], list):
            return len(token_ids[0])
        return len(token_ids)

    def completed_tokens(prompt: str) -> int:
        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = rendered_tokens(messages, add_generation_prompt=True)
        closed_tokens = rendered_tokens(
            [*messages, {"role": "assistant", "content": ""}],
            add_generation_prompt=False,
        )
        return max(prompt_tokens, closed_tokens) + config.max_completion_tokens

    return max(
        config.output_parity.packed.sequence_length, *map(completed_tokens, prompts)
    )


def _prepare_real_path_prompts(
    config: RealPathConfig,
) -> tuple[list[str], dict[str, Any] | None, int]:
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from art.megatron.model_support.tokenizer import (
        configure_tokenizer_for_model_support,
    )

    loaded_tokenizer = AutoTokenizer.from_pretrained(config.output_parity.base_model)
    assert isinstance(loaded_tokenizer, PreTrainedTokenizerBase)
    tokenizer = configure_tokenizer_for_model_support(
        loaded_tokenizer,
        base_model=config.output_parity.base_model,
        internal_config={
            "allow_unvalidated_arch": config.output_parity.allow_unvalidated_arch
        },
    )
    chat_template_kwargs: dict[str, Any] = {}
    if isinstance(tokenizer.chat_template, str):
        if "enable_thinking" in tokenizer.chat_template:
            chat_template_kwargs["enable_thinking"] = False
        if "preserve_thinking" in tokenizer.chat_template:
            chat_template_kwargs["preserve_thinking"] = True
    prompts = _build_prompts(config, tokenizer)
    max_model_len = _real_path_max_model_len(
        config,
        tokenizer=tokenizer,
        prompts=prompts,
        chat_template_kwargs=chat_template_kwargs,
    )
    max_model_len = _round_up(max_model_len, 128)
    config.output_parity.packed.sequence_length = max_model_len
    return (
        prompts,
        (
            {"chat_template_kwargs": chat_template_kwargs}
            if chat_template_kwargs
            else None
        ),
        max_model_len,
    )


async def _rollout(
    *,
    model: Any,
    prompt: str,
    max_completion_tokens: int,
    reward: float,
    seed: int,
    extra_body: dict[str, Any] | None,
    policy_step: int | None = None,
) -> Any:
    import art

    messages = [{"role": "user", "content": prompt}]
    request_kwargs: dict[str, Any] = {}
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body
    response = await model.openai_client().chat.completions.create(
        model=model.get_inference_name(step=policy_step),
        messages=messages,
        max_tokens=max_completion_tokens,
        temperature=0.8,
        seed=seed,
        logprobs=True,
        top_logprobs=TOP_K,
        **request_kwargs,
    )
    choice = response.choices[0]
    logprobs = choice.logprobs
    completion_tokens = len(logprobs.content or []) if logprobs is not None else 0
    if policy_step is not None:
        validate_complete_policy_token_spans(
            choice, completion_tokens=completion_tokens
        )
        if any(
            span.policy_version != policy_step
            for span in choice_policy_token_spans(choice)
        ):
            raise RuntimeError(
                f"policy-{policy_step} parity rollout returned another policy"
            )
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward,
        metrics={"completion_tokens": completion_tokens},
        initial_policy_version=policy_step,
        final_policy_version=policy_step,
    )


async def _collect_real_trajectory_groups(
    *,
    model: Any,
    config: RealPathConfig,
    prompts: list[str],
    extra_body: dict[str, Any] | None,
    policy_step: int | None = None,
) -> list[Any]:
    import art

    if config.rollouts_per_prompt < 2:
        raise ValueError("real-path mismatch requires at least two rollouts per prompt")
    groups = [
        art.TrajectoryGroup(
            [
                _rollout(
                    model=model,
                    prompt=prompt,
                    max_completion_tokens=config.max_completion_tokens,
                    reward=float(rollout_index % 2),
                    seed=(
                        config.output_parity.seed
                        + prompt_index * config.rollouts_per_prompt
                        + rollout_index
                    ),
                    extra_body=extra_body,
                    policy_step=policy_step,
                )
                for rollout_index in range(config.rollouts_per_prompt)
            ]
        )
        for prompt_index, prompt in enumerate(prompts)
    ]
    return await art.gather_trajectory_groups(
        cast(Any, groups),
        pbar_desc="real-path-rollouts",
    )


def _parse_token_id(raw: str | None) -> int:
    if raw is None:
        raise RuntimeError("vLLM logprob entry is missing token id")
    if raw.startswith("token_id:"):
        return int(raw.split(":", 1)[1])
    raise RuntimeError(
        "Expected vLLM logprob token strings to use token_id:<id>; got "
        f"{raw!r}. Ensure return_tokens_as_token_ids is enabled."
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cuda_visible_devices_for_slots(gpu_ids: list[int]) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [value.strip() for value in visible.split(",") if value.strip()]
        if visible is not None
        else None
    )
    if devices is None:
        return ",".join(map(str, gpu_ids))
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= len(devices)]
    if invalid:
        raise ValueError(f"GPU slots {invalid} exceed CUDA_VISIBLE_DEVICES={visible!r}")
    return ",".join(devices[gpu_id] for gpu_id in gpu_ids)


def _choice_score_index(
    trajectory_groups: list[Any],
    *,
    require_routing_metadata: bool,
) -> dict[tuple[int, ...], list[Choice]]:
    indexed: dict[tuple[int, ...], list[Choice]] = {}
    for group in trajectory_groups:
        for trajectory in group:
            for item in trajectory.messages_and_choices:
                if not isinstance(item, Choice):
                    continue
                if (
                    require_routing_metadata
                    and choice_moe_routing_metadata(item) is None
                ):
                    raise RuntimeError("Real-path trajectory choice is missing routes")
                token_metadata = choice_vllm_token_metadata(item)
                if token_metadata is None:
                    raise RuntimeError(
                        "Real-path trajectory choice is missing exact vLLM token metadata"
                    )
                prompt_ids, completion_ids = token_metadata
                indexed.setdefault(tuple(prompt_ids + completion_ids), []).append(item)
    return indexed


@asynccontextmanager
async def _direct_vllm_runtime(
    *,
    config: TrainInfOutputParityConfig,
    artifact_dir: Path,
    served_model_name: str,
    lora_path: str | None,
    engine_args: dict[str, Any],
    server_args: dict[str, Any] | None = None,
    forward_trace_dir: Path | None = None,
) -> AsyncIterator[tuple[str, int]]:
    import art.vllm_runtime as runtime

    port = _free_port()
    launch_config = runtime.VllmRuntimeLaunchConfig(
        base_model=config.base_model,
        port=port,
        host="127.0.0.1",
        cuda_visible_devices=_cuda_visible_devices_for_slots(config.inference_gpu_ids),
        lora_path=lora_path,
        served_model_name=served_model_name,
        engine_args=engine_args,
        server_args={
            "return_tokens_as_token_ids": True,
            **(server_args or {}),
            **config.server_args,
        },
    )
    command = runtime.build_vllm_runtime_server_cmd(launch_config)
    log_path = artifact_dir / f"real_path_vllm_{served_model_name}.log"
    env = runtime._vllm_runtime_subprocess_env(command)
    env["PYTHONUNBUFFERED"] = "1"
    if forward_trace_dir is not None:
        trace_site = Path(__file__).resolve().parent / "vllm_forward_trace_site"
        env["ART_VLLM_FORWARD_TRACE_DIR"] = str(forward_trace_dir)
        env["ART_VLLM_FORWARD_TRACE_DETAIL"] = "1"
        env["PYTHONPATH"] = (
            str(trace_site)
            if not env.get("PYTHONPATH")
            else f"{trace_site}{os.pathsep}{env['PYTHONPATH']}"
        )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(runtime.get_vllm_runtime_working_dir()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        await runtime.wait_for_vllm_runtime(
            process=process,
            host=launch_config.host,
            port=launch_config.port,
            timeout=float(
                os.environ.get("ART_TRAIN_INF_MISMATCH_VLLM_TIMEOUT", "1200")
            ),
        )
        yield launch_config.host, launch_config.port
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def _topk_from_chat_logprob(entry: Any) -> TokenTopK:
    if entry.top_logprobs is None:
        raise RuntimeError("vLLM logprob entry is missing top_logprobs")
    parsed: list[tuple[int, float]] = []
    for top in entry.top_logprobs:
        parsed.append((_parse_token_id(top.token), float(top.logprob)))
    parsed.sort(key=lambda item: item[1], reverse=True)
    return TokenTopK(
        token_ids=[token_id for token_id, _logprob in parsed[:TOP_K]],
        logprobs=[logprob for _token_id, logprob in parsed[:TOP_K]],
    )


def _vllm_scores_from_real_choices(
    *,
    trajectory_groups: list[Any],
    logical_map: LogicalTokenMap,
    require_routing_metadata: bool,
    weight_state: WeightState,
    rollout_mode: RolloutMode,
) -> ScoreBundle:
    choices_by_tokens = _choice_score_index(
        trajectory_groups,
        require_routing_metadata=require_routing_metadata,
    )
    prompt_by_id = {prompt.prompt_id: prompt for prompt in logical_map.prompts}
    tokens_by_leaf: dict[tuple[int, int, int], list[LogicalToken]] = {}
    for token in logical_map.tokens:
        tokens_by_leaf.setdefault(
            (token.sample_id, token.family_id, token.completion_id), []
        ).append(token)
    choice_by_leaf: dict[tuple[int, int, int], Choice] = {}
    for leaf, tokens in tokens_by_leaf.items():
        prompt = prompt_by_id[tokens[0].prompt_id]
        choices = choices_by_tokens.get(tuple(prompt.token_ids))
        if not choices:
            raise RuntimeError(
                "Could not find captured vLLM choice for logical prompt "
                f"{prompt.prompt_id}"
            )
        has_source_logprobs = all(token.source_logprob is not None for token in tokens)
        if not has_source_logprobs and len(choices) != 1:
            raise RuntimeError(
                "Duplicate vLLM token paths require packed source logprobs"
            )
        matching = choices
        if has_source_logprobs:
            matching = []
            for choice in choices:
                entries = (
                    choice.logprobs.content
                    if choice.logprobs is not None
                    and choice.logprobs.content is not None
                    else []
                )
                for token in tokens:
                    index = (
                        token.vllm_prompt_token_index - prompt.scored_token_start_index
                    )
                    if (
                        index < 0
                        or index >= len(entries)
                        or _parse_token_id(entries[index].token) != token.token_id
                        or struct.pack("!f", float(entries[index].logprob))
                        != struct.pack("!f", cast(float, token.source_logprob))
                    ):
                        break
                else:
                    matching.append(choice)
        if not matching:
            raise RuntimeError(
                "No captured vLLM choice matches packed source logprobs for "
                f"prompt {prompt.prompt_id}"
            )
        choice = matching[0]
        choices.remove(choice)
        choice_by_leaf[leaf] = choice
    target_logprobs: list[float] = []
    topk: list[TokenTopK] = []
    for token in logical_map.tokens:
        prompt = prompt_by_id[token.prompt_id]
        choice = choice_by_leaf[(token.sample_id, token.family_id, token.completion_id)]
        metadata = choice_moe_routing_metadata(choice)
        vllm_prompt_len = prompt.scored_token_start_index
        if (
            metadata is not None
            and len(metadata["prompt_token_ids"]) != vllm_prompt_len
        ):
            raise RuntimeError(
                "vLLM routed prompt length does not match ART packed request: "
                f"prompt_id={prompt.prompt_id}, art={vllm_prompt_len}, "
                f"vllm={len(metadata['prompt_token_ids'])}"
            )
        token_logprobs = (
            choice.logprobs.content
            if choice.logprobs is not None and choice.logprobs.content is not None
            else []
        )
        completion_index = token.vllm_prompt_token_index - vllm_prompt_len
        if completion_index < 0 or completion_index >= len(token_logprobs):
            raise RuntimeError(
                "Logical token is outside captured vLLM completion logprobs: "
                f"prompt_id={prompt.prompt_id}, index={token.vllm_prompt_token_index}"
            )
        entry = token_logprobs[completion_index]
        returned_token_id = _parse_token_id(entry.token)
        if returned_token_id != token.token_id:
            raise RuntimeError(
                "Captured vLLM token id does not match logical token: "
                f"expected={token.token_id}, returned={returned_token_id}"
            )
        target_logprobs.append(float(entry.logprob))
        topk.append(_topk_from_chat_logprob(entry))
    return ScoreBundle(
        side="vllm",
        weight_state=weight_state,
        rollout_mode=rollout_mode,
        target_logprobs=target_logprobs,
        topk=topk,
    )


async def _score_base_real_generation_path(
    *,
    config: RealPathConfig,
    artifact_dir: Path,
    is_moe: bool,
    prompts: list[str],
    extra_body: dict[str, Any] | None,
    max_model_len: int,
) -> RealPathBaseDiagnosticBundle:
    import art
    from art.megatron.backend import MegatronBackend
    from art.megatron.model_support import get_model_support_handler
    from art.preprocessing.pack import packed_tensors_to_dir

    parity_config = config.output_parity
    handler = get_model_support_handler(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    served_name = f"train_inf_real_base_{uuid.uuid4().hex[:8]}"
    engine_args = {
        "tensor_parallel_size": len(parity_config.inference_gpu_ids),
        "enable_expert_parallel": is_moe and len(parity_config.inference_gpu_ids) > 1,
        "max_model_len": max_model_len,
        "max_logprobs": TOP_K,
        **parity_config.engine_args,
    }
    for key, value in handler.vllm_engine_args().items():
        engine_args.setdefault(key, value)
    engine_args.setdefault("generation_config", "vllm")
    engine_args.pop("enable_lora", None)
    engine_args.pop("max_loras", None)
    engine_args.pop("lora_target_modules", None)
    if is_moe:
        engine_args["enable_return_routed_experts"] = True
    vllm_forward_trace_dir = (
        artifact_dir / "real_path_base_vllm_forward_trace"
        if config.trace_layers
        else None
    )
    megatron_forward_trace_dir = (
        artifact_dir / "real_path_base_megatron_forward_trace"
        if config.trace_layers
        else None
    )
    if config.trace_enforce_eager:
        engine_args["enforce_eager"] = True

    async with _direct_vllm_runtime(
        config=parity_config,
        artifact_dir=artifact_dir,
        served_model_name=served_name,
        lora_path=None,
        engine_args=engine_args,
        server_args={
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            **handler.vllm_server_args(),
        },
        forward_trace_dir=vllm_forward_trace_dir,
    ) as (host, port):
        model = art.TrainableModel(
            run_name=f"{served_name}_client",
            name=f"{served_name}_client",
            project="train_inf_mismatch",
            base_model=parity_config.base_model,
            _internal_config={
                "init_args": {
                    "max_seq_length": parity_config.packed.sequence_length,
                },
            },
        )
        object.__setattr__(model, "inference_base_url", f"http://{host}:{port}/v1")
        object.__setattr__(model, "inference_api_key", "EMPTY")
        object.__setattr__(model, "inference_model_name", served_name)
        trajectory_groups = await _collect_real_trajectory_groups(
            model=model,
            config=config,
            prompts=prompts,
            extra_body=extra_body,
        )

    packing_backend = MegatronBackend(
        path=str(artifact_dir / "base_art_path"),
        enable_expert_replay=is_moe,
    )
    packed_tensors = packing_backend._get_packed_tensors(
        model,
        trajectory_groups,
        advantage_balance=0.0,
        allow_training_without_logprobs=False,
        scale_rewards=True,
        plot_tensors=False,
        packed_sequence_length=parity_config.packed.sequence_length,
        logprob_calculation_chunk_size=1024,
        include_moe_routing=is_moe,
    )
    if packed_tensors is None:
        raise RuntimeError("Base diagnostic ART path produced no packed tensors")
    logical_map = build_logical_token_map(cast(dict[str, Any], packed_tensors))
    logical_map_path = artifact_dir / "real_path_base_logical_token_map.json"
    _write_json(logical_map_path, logical_map.model_dump(mode="json"))

    vllm_base = _vllm_scores_from_real_choices(
        trajectory_groups=trajectory_groups,
        logical_map=logical_map,
        require_routing_metadata=is_moe,
        weight_state="base",
        rollout_mode="native_lora",
    )
    vllm_score_path = artifact_dir / "real_path_vllm_base_scores.json"
    _write_json(vllm_score_path, vllm_base.model_dump(mode="json"))

    routing_replay_path: str | None = None
    global_grad_accumulation_sequences = int(packed_tensors["tokens"].shape[0])
    if is_moe:
        routing_replay_dir = artifact_dir / "real_path_base_moe_routing_replay"
        _build_real_path_moe_routing_replay_bundle(
            packed_tensors=packed_tensors,
            config=parity_config,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        ).to_dir(routing_replay_dir)
        routing_replay_path = str(routing_replay_dir)
        routing_replay = cast(
            PackedMoeRoutingReplay, packed_tensors["moe_routing_replay"]
        )
        stats = routing_replay.pack_stats
    else:
        stats = MoeRoutingPackStats()

    disk_packed_tensors = packed_tensors_to_dir(
        packed_tensors,
        str(artifact_dir / "real_path_base_packed_tensors"),
    )
    _write_json(
        artifact_dir / "real_path_base_disk_packed_tensors.json",
        cast(dict[str, Any], disk_packed_tensors),
    )
    worker_result = _run_real_path_megatron_worker(
        RealPathMegatronWorkerRequest(
            config=parity_config,
            artifact_dir=str(artifact_dir),
            disk_packed_tensors=disk_packed_tensors,
            logical_map_path=str(logical_map_path),
            weight_state="base",
            adapter_path=None,
            moe_routing_replay_path=routing_replay_path,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            forward_trace_dir=(
                str(megatron_forward_trace_dir)
                if megatron_forward_trace_dir is not None
                else None
            ),
        )
    )
    megatron_base = ScoreBundle.model_validate(
        _read_json(Path(worker_result.score_path))
    )
    return RealPathBaseDiagnosticBundle(
        vllm_scores=vllm_base,
        megatron_scores=megatron_base,
        megatron_score_path=worker_result.score_path,
        vllm_score_path=str(vllm_score_path),
        logical_prompt_count=len(logical_map.prompts),
        logical_token_count=len(logical_map.tokens),
        moe_routing_packed_tokens=int(stats.packed_tokens),
        vllm_forward_trace_dir=(
            str(vllm_forward_trace_dir) if vllm_forward_trace_dir is not None else None
        ),
        megatron_forward_trace_dir=(
            str(megatron_forward_trace_dir)
            if megatron_forward_trace_dir is not None
            else None
        ),
    )


def _move_adapter_to_step_zero(*, adapter_path: str, model: Any, backend: Any) -> str:
    from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

    model_dir = get_model_dir(model=model, art_path=backend._path)
    step_zero = get_step_checkpoint_dir(model_dir, 0)
    os.makedirs(step_zero, exist_ok=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy(Path(adapter_path) / filename, Path(step_zero) / filename)
    return step_zero


def _routing_topology_from_config(config: TrainInfOutputParityConfig) -> Any:
    from art.megatron.routing_replay import ParallelTopology

    return ParallelTopology(
        tp=config.topology.tp,
        ep=config.topology.ep,
        etp=config.topology.etp,
        dp=config.topology.dp,
        sp=config.topology.tp > 1,
        cp=config.topology.cp,
        pp=config.topology.pp,
    )


def _init_art_megatron_runtime_config(config: TrainInfOutputParityConfig) -> None:
    import art

    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(
            tp=config.topology.tp,
            cp=config.topology.cp,
            ep=config.topology.ep,
            pp=config.topology.pp,
            etp=config.topology.etp,
        ),
        packed_sequence_length=config.packed.sequence_length,
        streaming_weight_offload=config.streaming_weight_offload,
    )


def _build_real_path_moe_routing_replay_bundle(
    *,
    packed_tensors: Any,
    config: TrainInfOutputParityConfig,
    global_grad_accumulation_sequences: int,
) -> Any:
    from art.megatron.routing_replay import (
        build_moe_routing_replay_bundle_from_packed_tensors,
    )

    return build_moe_routing_replay_bundle_from_packed_tensors(
        packed_tensors=packed_tensors,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        topology=_routing_topology_from_config(config),
    )


def _pack_expert_lora_tensors(
    tensors: dict[str, Any],
    groups: tuple[Any, ...],
) -> dict[str, Any]:
    import torch

    packed = dict(tensors)
    for group in groups:
        for slot in group.slots:
            suffix = f".{slot.source_projection}.{slot.source_lora}.weight"
            matches: dict[str, dict[int, tuple[str, torch.Tensor]]] = {}
            for key, tensor in tensors.items():
                if not key.endswith(suffix):
                    continue
                prefix, separator, expert = key[: -len(suffix)].rpartition(".")
                if not separator or not prefix.endswith(group.art_group_suffix):
                    continue
                try:
                    expert_index = int(expert)
                except ValueError:
                    continue
                matches.setdefault(prefix, {})[expert_index] = (key, tensor)

            for prefix, experts in matches.items():
                if sorted(experts) != list(range(len(experts))):
                    raise RuntimeError(
                        f"Non-contiguous expert LoRA tensors for {prefix}: "
                        f"{sorted(experts)}"
                    )
                joined = torch.stack([experts[index][1] for index in sorted(experts)])
                for key, _tensor in experts.values():
                    packed.pop(key)
                if slot.pack_layout == "expert_rows":
                    value = joined.flatten(0, 1)
                elif slot.pack_layout == "rank_major_expert_cols":
                    value = joined.permute(1, 2, 0).reshape(
                        joined.shape[1], joined.shape[2] * joined.shape[0]
                    )
                elif slot.pack_layout == "interleaved_gate_up_rank_major_expert_cols":
                    gate, up = joined.split(joined.shape[1] // 2, dim=1)
                    interleaved = torch.stack((gate, up), dim=2).flatten(1, 2)
                    value = interleaved.permute(1, 2, 0).reshape(
                        interleaved.shape[1],
                        interleaved.shape[2] * interleaved.shape[0],
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported expert LoRA layout: {slot.pack_layout}"
                    )
                output_key = f"{prefix}.{slot.output_suffix}"
                if output_key in packed:
                    raise RuntimeError(
                        f"Duplicate packed expert LoRA tensor: {output_key}"
                    )
                packed[output_key] = value.contiguous()
    return packed


def _make_nonzero_adapter(
    *,
    config: TrainInfOutputParityConfig,
    artifact_dir: Path,
) -> str:
    import torch

    from art.megatron.identity_lora import create_identity_lora
    from art.megatron.model_support import get_model_support_handler
    from art.megatron.model_support.lora_disk import (
        load_adapter_config,
        load_vllm_lora_tensors,
        save_vllm_lora_tensors,
    )

    from .output_parity import _adapter_config

    handler = get_model_support_handler(
        config.base_model,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    adapter_path = artifact_dir / "real_path_active_lora"
    with torch.random.fork_rng(devices=[]):
        create_identity_lora(
            config.base_model,
            str(adapter_path),
            target_modules=_lora_target_modules(config),
            random_state=config.seed,
            allow_unvalidated_arch=config.allow_unvalidated_arch,
            handler=handler,
        )
    published_config = load_adapter_config(adapter_path)
    published_tensors = load_vllm_lora_tensors(adapter_path)
    invalid_dtypes = {
        key: str(value.dtype)
        for key, value in published_tensors.items()
        if value.dtype != torch.bfloat16
    }
    if invalid_dtypes:
        raise RuntimeError(f"Identity LoRA tensors must be BF16: {invalid_dtypes}")
    templates = handler.from_vllm_lora_tensors(
        published_tensors,
        adapter_config=published_config,
    )
    if not templates:
        raise RuntimeError("Identity LoRA metadata produced no adapter tensors")
    adapter_config = _adapter_config(config)
    initialized = _build_deterministic_nonzero_lora(
        {
            key: torch.empty_like(value, device="cpu", dtype=torch.bfloat16)
            for key, value in templates.items()
        },
        seed=config.seed,
    )
    normalized, normalized_config = handler.to_vllm_lora_tensors(
        initialized,
        adapter_config=dict(adapter_config),
    )
    initialized = handler.from_vllm_lora_tensors(
        normalized,
        adapter_config=normalized_config,
    )
    tensors, published_config = handler.to_vllm_lora_tensors(
        _pack_expert_lora_tensors(
            initialized,
            tuple(handler.expert_packed_lora_groups()),
        ),
        adapter_config=adapter_config,
    )
    invalid = [
        key
        for key, value in tensors.items()
        if value.dtype != torch.bfloat16 or not torch.count_nonzero(value).item()
    ]
    if invalid:
        raise RuntimeError(f"Invalid materialized LoRA tensors: {invalid[:5]}")
    save_vllm_lora_tensors(
        adapter_path,
        {key: value.cpu().contiguous() for key, value in tensors.items()},
        published_config,
    )
    return str(adapter_path)


def _adapter_cache_key(config: TrainInfOutputParityConfig) -> str:
    from transformers import AutoConfig

    from art.megatron.model_support import (
        get_model_support_handler,
        vllm_lora_config_for_model,
    )

    from .output_parity import _adapter_config

    def jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, set):
            return [jsonable(item) for item in sorted(value, key=str)]
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    adapter_config = _adapter_config(config)
    published_adapter_config = vllm_lora_config_for_model(
        config.base_model,
        adapter_config,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    handler = get_model_support_handler(
        config.base_model,
        allow_unvalidated_arch=config.allow_unvalidated_arch,
    )
    handler_module = Path(inspect.getfile(type(handler)))
    model_config = handler.identity_lora_model_config(
        AutoConfig.from_pretrained(config.base_model, trust_remote_code=True)
    ).to_json_string(use_diff=False)
    payload = {
        "schema": 4,
        "base_model": config.base_model,
        "model_config_sha256": hashlib.sha256(model_config.encode()).hexdigest(),
        "seed": config.seed,
        "allow_unvalidated_arch": config.allow_unvalidated_arch,
        "lora_target_modules": _lora_target_modules(config),
        "adapter_config": adapter_config,
        "published_adapter_config": published_adapter_config,
        "handler_sha256": hashlib.sha256(handler_module.read_bytes()).hexdigest(),
    }
    encoded = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_adapter_cache_dir() -> Path:
    if artifacts_root := os.environ.get(ARTIFACTS_ROOT_ENV):
        return Path(artifacts_root) / "adapter_cache"
    return REPO_ROOT / "scratch" / "train_inf_mismatch_adapters"


def _adapter_cache_dir(config: RealPathConfig) -> Path:
    if config.adapter_cache_dir:
        return Path(config.adapter_cache_dir)
    model_namespace = hashlib.sha256(
        config.output_parity.base_model.encode()
    ).hexdigest()[:16]
    return _default_adapter_cache_dir() / model_namespace


def _adapter_cache_manifest_path(adapter_path: Path) -> Path:
    return adapter_path / "art_train_inf_mismatch_adapter_cache.json"


def _cached_adapter_is_valid(adapter_path: Path, *, cache_key: str) -> bool:
    manifest_path = _adapter_cache_manifest_path(adapter_path)
    model_path = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"
    if not (manifest_path.exists() and model_path.exists() and config_path.exists()):
        return False
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return False
    return (
        manifest.get("cache_key") == cache_key
        and manifest.get("non_identity") is True
        and int(manifest.get("adapter_model_bytes", 0)) == model_path.stat().st_size
        and model_path.stat().st_size > 0
    )


def _copy_adapter_dir(src: Path, dst: Path, *, cache_key: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(src / filename, dst / filename)
    _write_json(
        _adapter_cache_manifest_path(dst),
        {
            "cache_key": cache_key,
            "non_identity": True,
            "adapter_model_bytes": (dst / "adapter_model.safetensors").stat().st_size,
        },
    )


def _prune_adapter_cache_dir(cache_dir: Path, *, keep_key: str) -> None:
    if not cache_dir.exists():
        return
    cache_root = cache_dir.resolve()
    keep_path = (cache_root / keep_key).resolve()
    for candidate in cache_root.iterdir():
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if resolved == keep_path:
            continue
        resolved.relative_to(cache_root)
        if _adapter_cache_manifest_path(resolved).exists():
            shutil.rmtree(resolved)


def _make_or_reuse_nonzero_adapter(
    *,
    config: RealPathConfig,
    artifact_dir: Path,
) -> AdapterCacheResult:
    cache_key = _adapter_cache_key(config.output_parity)
    cache_dir = _adapter_cache_dir(config)
    cache_path = cache_dir / cache_key
    if _cached_adapter_is_valid(cache_path, cache_key=cache_key):
        _prune_adapter_cache_dir(cache_dir, keep_key=cache_key)
        return AdapterCacheResult(
            path=str(cache_path),
            cache_key=cache_key,
            cache_hit=True,
        )
    adapter_path = Path(
        _make_nonzero_adapter(config=config.output_parity, artifact_dir=artifact_dir)
    )
    if not adapter_path:
        raise RuntimeError("Real-path adapter worker did not create an adapter")
    _copy_adapter_dir(adapter_path, cache_path, cache_key=cache_key)
    _prune_adapter_cache_dir(cache_dir, keep_key=cache_key)
    return AdapterCacheResult(
        path=str(cache_path),
        cache_key=cache_key,
        cache_hit=False,
    )


def _run_logits_with_replay(
    *,
    runtime: Any,
    packed_tensors: dict[str, Any],
    global_grad_accumulation_sequences: int,
) -> Any:
    import torch

    if runtime.moe_routing_replay_controller is None:
        return _run_logits(runtime=runtime, packed_tensors=packed_tensors)

    logits_by_sample = []
    num_sequences = int(packed_tensors["tokens"].shape[0])
    for sample_index in range(num_sequences):
        sample_tensors = {
            key: (
                value[sample_index : sample_index + 1]
                if isinstance(value, torch.Tensor)
                and value.shape[:1] == packed_tensors["tokens"].shape[:1]
                else value
            )
            for key, value in packed_tensors.items()
        }
        step_index = sample_index // global_grad_accumulation_sequences
        runtime.moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=sample_index,
        )
        runtime.moe_routing_replay_controller.begin_micro(sample_index, sample_index)
        logits_by_sample.append(
            _run_logits(runtime=runtime, packed_tensors=sample_tensors)
        )
        runtime.moe_routing_replay_controller.finalize_step()
    return torch.cat(logits_by_sample, dim=0)


def _score_megatron_runtime(
    *,
    runtime: Any,
    packed_tensors: dict[str, Any],
    logical_map: LogicalTokenMap,
    weight_state: WeightState,
    rollout_mode: RolloutMode,
    global_grad_accumulation_sequences: int,
    forward_trace_capture: Any | None,
    forward_trace_dir: str | None,
) -> ScoreBundle:
    from megatron.core import parallel_state as ps
    import torch

    if int(ps.get_context_parallel_world_size()) > 1:
        if forward_trace_capture is not None or forward_trace_dir is not None:
            if forward_trace_capture is not None:
                forward_trace_capture.close()
            raise RuntimeError(
                "CP train/inf mismatch scoring uses sparse UID records, not gathered "
                "full logits, so forward trace logits capture is unsupported here."
            )
        return score_context_parallel_runtime(
            runtime=runtime,
            packed_tensors=packed_tensors,
            logical_map=logical_map,
            weight_state=weight_state,
            rollout_mode=rollout_mode,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )

    try:
        logits = _run_logits_with_replay(
            runtime=runtime,
            packed_tensors=packed_tensors,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        if forward_trace_capture is not None and forward_trace_dir is not None:
            trace_dir = Path(forward_trace_dir)
            forward_trace_capture.save_current_step(trace_dir)
            if (
                not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
                or torch.distributed.get_rank() == 0  # ty: ignore[possibly-missing-attribute]
            ):
                trace_dir.mkdir(parents=True, exist_ok=True)
                torch.save(logits.detach().cpu(), trace_dir / "logits.pt")
    finally:
        if forward_trace_capture is not None:
            forward_trace_capture.close()
    return _extract_scores_from_logits(
        logits=logits,
        logical_map=logical_map,
        side="megatron",
        weight_state=weight_state,
        rollout_mode=rollout_mode,
    )


def _real_path_megatron_worker(
    request: RealPathMegatronWorkerRequest,
    *,
    adapter_only: bool = False,
) -> None:
    import torch

    from art.megatron import train as megatron_train
    from art.megatron.model_support.lora_disk import load_lora_tensors_for_megatron
    from art.megatron.training.weight_offload import WeightOffloadManager
    from art.preprocessing.pack import packed_tensors_from_dir

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")  # type: ignore[possibly-missing-attribute]
    _set_seed(request.config.seed)
    os.environ.update(request.config.topology.env())

    def _configure_worker_bundle(bundle: Any) -> None:
        if request.config.lora_target_modules is not None:
            _configure_lora_target_modules(
                bundle,
                _lora_target_modules(request.config),
            )
        if not adapter_only and request.weight_state == "base":
            bundle.provider.register_pre_wrap_hook(megatron_train.freeze_model)

    runtime = megatron_train.build_training_runtime(
        model_identifier=request.config.base_model,
        provider_torch_dtype=torch.bfloat16,
        provider_bundle_configure=_configure_worker_bundle,
        provider_configure=lambda provider: _configure_provider(
            provider, request.config
        ),
        moe_routing_replay_path=request.moe_routing_replay_path,
        moe_routing_replay_strict=True,
        print_env=False,
        build_optimizer=False,
        trainable_parameter_mode=(
            "lora" if adapter_only or request.weight_state == "lora" else "base_model"
        ),
        allow_unvalidated_arch=request.config.allow_unvalidated_arch,
    )
    for chunk in runtime.model:
        chunk.eval()

    weight_offload = None
    if not adapter_only:
        weight_offload = WeightOffloadManager.from_env(
            model=runtime.model,
            rank=torch.distributed.get_rank(),  # type: ignore[possibly-missing-attribute]
            compile_enabled=runtime.transformer_layers_compiled,
        )
        weight_offload.install()
        weight_offload.after_job()
        weight_offload.before_job()

    artifact_dir = Path(request.artifact_dir)
    adapter_path: Path | None = None
    job_completed = False
    try:
        if request.weight_state == "lora":
            if request.adapter_path is None:
                initial_state = _collect_full_lora_state(cast(list[Any], runtime.model))
                rank = torch.distributed.get_rank()  # type: ignore[possibly-missing-attribute]
                initialized = (
                    _build_deterministic_nonzero_lora(
                        initial_state or {},
                        seed=request.config.seed,
                    )
                    if rank == 0
                    else None
                )
                payload = [initialized]
                torch.distributed.broadcast_object_list(  # type: ignore[possibly-missing-attribute]
                    payload,
                    src=0,
                    device=torch.device("cuda", local_rank),
                )
                initialized = cast(dict[str, Any], payload[0])
                adapter_path = artifact_dir / "real_path_active_lora"
                _save_vllm_lora_adapter(
                    lora_path=adapter_path,
                    state=initialized,
                    runtime=runtime,
                    config=request.config,
                )
                torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
            else:
                adapter_path = Path(request.adapter_path)
            adapter_model = load_lora_tensors_for_megatron(
                str(adapter_path),
                handler=runtime.model_support_handler,
                allow_unvalidated_arch=request.config.allow_unvalidated_arch,
            )
            megatron_train.load_adapter_into_model(
                runtime.model,
                adapter_model,
                model_support_handler=runtime.model_support_handler,
            )

        if adapter_only:
            if torch.distributed.get_rank() == 0:  # type: ignore[possibly-missing-attribute]
                result = RealPathMegatronWorkerResult(
                    score_path="",
                    adapter_path=str(adapter_path)
                    if adapter_path is not None
                    else None,
                )
                _write_json(
                    artifact_dir / "real_path_adapter_worker_result.json",
                    result.model_dump(mode="json"),
                )
            torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
            torch.distributed.destroy_process_group()  # type: ignore[possibly-missing-attribute]
            return

        packed_tensors = packed_tensors_from_dir(**request.disk_packed_tensors)
        logical_map = LogicalTokenMap.model_validate(
            _read_json(Path(request.logical_map_path))
        )
        forward_trace_capture = None
        if request.forward_trace_dir is not None:
            from ..model_support.forward_trace import (
                CAPTURE_NAME_TOKENS,
                ForwardTraceCapture,
            )

            forward_trace_capture = ForwardTraceCapture(
                runtime.model,
                enabled=True,
                capture_name_tokens=(*CAPTURE_NAME_TOKENS, ".decoder.final_layernorm"),
                strict_output_match=True,
            )
            forward_trace_capture.set_step(
                0,
                list(range(int(packed_tensors["tokens"].shape[0]))),
            )
        score = _score_megatron_runtime(
            runtime=runtime,
            packed_tensors=cast(dict[str, Any], packed_tensors),
            logical_map=logical_map,
            weight_state=request.weight_state,
            rollout_mode=_real_path_rollout_mode(request.config),
            global_grad_accumulation_sequences=request.global_grad_accumulation_sequences,
            forward_trace_capture=forward_trace_capture,
            forward_trace_dir=request.forward_trace_dir,
        )

        if torch.distributed.get_rank() == 0:  # type: ignore[possibly-missing-attribute]
            score_path = (
                artifact_dir / f"real_path_megatron_{request.weight_state}.json"
            )
            _write_json(score_path, score.model_dump(mode="json"))
            result = RealPathMegatronWorkerResult(
                score_path=str(score_path),
                adapter_path=str(adapter_path) if adapter_path is not None else None,
            )
            _write_json(
                artifact_dir
                / f"real_path_megatron_{request.weight_state}_worker_result.json",
                result.model_dump(mode="json"),
            )
        job_completed = True
    finally:
        if weight_offload is not None and job_completed:
            weight_offload.after_job()
    torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
    torch.distributed.destroy_process_group()  # type: ignore[possibly-missing-attribute]


def _run_real_path_megatron_worker(
    request: RealPathMegatronWorkerRequest,
    *,
    adapter_only: bool = False,
) -> RealPathMegatronWorkerResult:
    artifact_dir = Path(request.artifact_dir)
    request_name = (
        "real_path_adapter_request.json"
        if adapter_only
        else f"real_path_megatron_{request.weight_state}_request.json"
    )
    request_path = artifact_dir / request_name
    _write_json(request_path, request.model_dump(mode="json"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices_for_slots(
        request.config.trainer_gpu_ids
    )
    env.update(request.config.megatron_env)
    env["PYTHONUNBUFFERED"] = "1"
    tests_dir = str(REPO_ROOT / "tests")
    env["PYTHONPATH"] = (
        tests_dir
        if not env.get("PYTHONPATH")
        else f"{tests_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(request.config.topology.world_size()),
        "-m",
        "integration.megatron.train_inf_mismatch.real_path",
        "--worker",
        "--request",
        str(request_path),
    ]
    if adapter_only:
        command.append("--adapter-only")
    log_path = artifact_dir / (
        "real_path_adapter_worker.log"
        if adapter_only
        else f"real_path_megatron_{request.weight_state}_worker.log"
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        run = subprocess.run(
            command,
            cwd=str(REPO_ROOT / "tests"),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if run.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-120:])
        raise RuntimeError(
            f"Real-path Megatron worker failed with exit code {run.returncode}.\n{tail}"
        )
    result_name = (
        "real_path_adapter_worker_result.json"
        if adapter_only
        else f"real_path_megatron_{request.weight_state}_worker_result.json"
    )
    return RealPathMegatronWorkerResult.model_validate(
        _read_json(artifact_dir / result_name)
    )


def _delete_adapter_safetensors_on_pass(artifact_dir: Path, *, passed: bool) -> None:
    if not passed:
        return
    for path in artifact_dir.rglob("adapter_model.safetensors"):
        path.unlink()


def _resident_score_bundle(
    *,
    result: Any,
    logical_map: LogicalTokenMap,
    rollout_mode: RolloutMode,
) -> ScoreBundle:
    scores = {(score.sample_index, score.logit_index): score for score in result.scores}
    selected = []
    for token in logical_map.tokens:
        coordinate = token.sample_id, token.art_logit_index
        score = scores.get(coordinate)
        if score is None:
            raise RuntimeError(f"resident score is missing logical token {coordinate}")
        if score.target_token_id != token.token_id:
            raise RuntimeError(
                "resident score target token does not match packed input: "
                f"coordinate={coordinate}, score={score.target_token_id}, "
                f"packed={token.token_id}"
            )
        selected.append(score)
    return ScoreBundle(
        side="megatron",
        weight_state="lora",
        rollout_mode=rollout_mode,
        target_logprobs=[score.target_logprob for score in selected],
        topk=[
            TokenTopK(
                token_ids=list(score.top_token_ids),
                logprobs=list(score.top_logprobs),
            )
            for score in selected
        ],
    )


async def run_resident_train_inf_mismatch(
    *,
    backend: Any,
    model: Any,
    policy_step: int,
    config: RealPathConfig,
    artifact_dir: Path,
    max_attempts: int = 3,
) -> ResidentTrainInfReport:
    import torch

    parity_config = config.output_parity
    _apply_sliding_window_prompt_defaults(config)
    prompts, extra_body, _max_model_len = _prepare_real_path_prompts(config)
    rollout_mode = _real_path_rollout_mode(parity_config)
    is_moe = model_support_is_moe(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "resident_config.json", config.model_dump(mode="json"))
    mean_abs_pct_limit = fwd_mean_abs_pct_limit_for_model(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    top20_kl_limit = top20_kl_candidate_to_target_limit_for_model(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    attempts = []
    run_id: str | None = None
    generation_id: str | None = None
    for attempt in range(1, max_attempts + 1):
        groups = await _collect_real_trajectory_groups(
            model=model,
            config=config,
            prompts=prompts,
            extra_body=extra_body,
            policy_step=policy_step,
        )
        packed = backend._get_packed_tensors(
            model,
            groups,
            advantage_balance=0.0,
            allow_training_without_logprobs=False,
            scale_rewards=True,
            plot_tensors=False,
            packed_sequence_length=parity_config.packed.sequence_length,
            logprob_calculation_chunk_size=1024,
            include_moe_routing=is_moe,
        )
        if packed is None:
            raise RuntimeError("resident ART path produced no packed tensors")
        logical_map = build_logical_token_map(cast(dict[str, Any], packed))
        vllm = _vllm_scores_from_real_choices(
            trajectory_groups=groups,
            logical_map=logical_map,
            require_routing_metadata=is_moe,
            weight_state="lora",
            rollout_mode=rollout_mode,
        )
        result = await backend.score_resident(
            model,
            groups,
            expected_learner_version=policy_step,
            top_k=TOP_K,
        )
        fingerprint = _packed_batch_fingerprint(cast(dict[str, Any], packed))
        if result.batch_fingerprint != fingerprint:
            raise RuntimeError(
                "resident score did not use the locally reconstructed packed batch"
            )
        if run_id is not None and result.run_id != run_id:
            raise RuntimeError("resident parity attempts used different trainer runs")
        run_id = result.run_id
        generation_id = result.learner.generation_id
        megatron = _resident_score_bundle(
            result=result,
            logical_map=logical_map,
            rollout_mode=rollout_mode,
        )
        for name, value in (
            ("logical_token_map", logical_map),
            ("vllm_scores", vllm),
            ("megatron_scores", megatron),
        ):
            _write_json(
                artifact_dir / f"attempt_{attempt}_{name}.json",
                value.model_dump(mode="json"),
            )
        sequence_ids = [token.prompt_id for token in logical_map.tokens]
        comparison = compare_pair(
            candidate=torch.tensor(megatron.target_logprobs, dtype=torch.float32),
            target=torch.tensor(vllm.target_logprobs, dtype=torch.float32),
            sequence_ids=sequence_ids,
        )
        topk = compare_topk(megatron, vllm)
        passed = (
            comparison.mean_abs_pct <= mean_abs_pct_limit
            and topk.top20_intersection_kl_candidate_to_target <= top20_kl_limit
        )
        attempt_report = ResidentTrainInfAttempt(
            attempt=attempt,
            logical_prompt_count=len(logical_map.prompts),
            logical_token_count=len(logical_map.tokens),
            lora=comparison,
            lora_topk=topk,
            moe_routing_packed_tokens=result.routing_replay_packed_tokens,
            batch_fingerprint=fingerprint,
            passed=passed,
        )
        attempts.append(attempt_report)
        _write_json(
            artifact_dir / f"attempt_{attempt}.json",
            attempt_report.model_dump(mode="json"),
        )
        if passed:
            break
    assert run_id is not None and generation_id is not None
    report = ResidentTrainInfReport(
        base_model=parity_config.base_model,
        artifact_dir=str(artifact_dir),
        run_id=run_id,
        policy_step=policy_step,
        generation_id=generation_id,
        attempt_count=len(attempts),
        max_attempts=max_attempts,
        attempts=attempts,
        mean_abs_pct_limit=mean_abs_pct_limit,
        top20_kl_candidate_to_target_limit=top20_kl_limit,
        passed=attempts[-1].passed,
    )
    _write_json(
        artifact_dir / "resident_comparison_report.json",
        report.model_dump(mode="json"),
    )
    return report


async def run_real_path_train_inf_mismatch(
    *,
    config: RealPathConfig,
    artifact_dir: Path,
) -> RealPathTrainInfReport:
    import art
    from art.megatron.backend import MegatronBackend
    from art.preprocessing.pack import packed_tensors_to_dir

    parity_config = config.output_parity
    _apply_sliding_window_prompt_defaults(config)
    prompts, extra_body, max_model_len = _prepare_real_path_prompts(config)
    rollout_mode = _real_path_rollout_mode(parity_config)
    is_moe = model_support_is_moe(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    _write_json(artifact_dir / "real_path_config.json", config.model_dump(mode="json"))
    adapter = _make_or_reuse_nonzero_adapter(config=config, artifact_dir=artifact_dir)
    adapter_path = adapter.path

    _init_art_megatron_runtime_config(parity_config)
    backend = MegatronBackend(
        path=str(artifact_dir / "art_path"),
        enable_expert_replay=is_moe,
    )
    backend_open = False
    internal_config = cast(
        InternalModelConfig,
        {
            "trainer_gpu_ids": parity_config.trainer_gpu_ids,
            "inference_gpu_ids": parity_config.inference_gpu_ids,
            "allow_unvalidated_arch": parity_config.allow_unvalidated_arch,
            "engine_args": {
                "tensor_parallel_size": len(parity_config.inference_gpu_ids),
                "enable_expert_parallel": is_moe
                and len(parity_config.inference_gpu_ids) > 1,
                "max_model_len": max_model_len,
                "max_logprobs": TOP_K,
                **parity_config.engine_args,
            },
            "init_args": {
                "max_seq_length": parity_config.packed.sequence_length,
            },
        },
    )
    if parity_config.external_vllm_server_url is not None:
        internal_config["vllm_runtime"] = {
            "mode": "external",
            "server_url": parity_config.external_vllm_server_url,
            "api_key": parity_config.external_vllm_api_key,
        }
    run_name = f"train-inf-real-{uuid.uuid4().hex[:8]}"
    model = art.TrainableModel(
        name=run_name,
        run_name=run_name,
        project="train_inf_mismatch",
        base_model=parity_config.base_model,
        lora_config=(
            {"target_modules": _lora_target_modules(parity_config)}
            if parity_config.lora_target_modules is not None
            else None
        ),
        _internal_config=internal_config,
    )
    _move_adapter_to_step_zero(adapter_path=adapter_path, model=model, backend=backend)

    try:
        with _temporary_env(parity_config.megatron_env):
            await model.register(backend)
            backend_open = True
        trajectory_groups = await _collect_real_trajectory_groups(
            model=model,
            config=config,
            prompts=prompts,
            extra_body=extra_body,
        )
        packed_tensors = backend._get_packed_tensors(
            model,
            trajectory_groups,
            advantage_balance=0.0,
            allow_training_without_logprobs=False,
            scale_rewards=True,
            plot_tensors=False,
            packed_sequence_length=parity_config.packed.sequence_length,
            logprob_calculation_chunk_size=1024,
            include_moe_routing=is_moe,
        )
        if packed_tensors is None:
            raise RuntimeError("Real ART path produced no packed tensors")
        logical_map = build_logical_token_map(cast(dict[str, Any], packed_tensors))
        logical_map_path = artifact_dir / "real_path_logical_token_map.json"
        _write_json(logical_map_path, logical_map.model_dump(mode="json"))

        routing_replay_dir = artifact_dir / "real_path_moe_routing_replay"
        global_grad_accumulation_sequences = int(packed_tensors["tokens"].shape[0])
        routing_replay_path: str | None = None
        if is_moe:
            _build_real_path_moe_routing_replay_bundle(
                packed_tensors=packed_tensors,
                config=parity_config,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            ).to_dir(routing_replay_dir)
            routing_replay_path = str(routing_replay_dir)
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors,
            str(artifact_dir / "real_path_packed_tensors"),
        )
        _write_json(
            artifact_dir / "real_path_disk_packed_tensors.json",
            cast(dict[str, Any], disk_packed_tensors),
        )
        if is_moe:
            routing_replay = cast(
                PackedMoeRoutingReplay, packed_tensors["moe_routing_replay"]
            )
            stats = routing_replay.pack_stats
        else:
            stats = MoeRoutingPackStats()

        vllm_lora = _vllm_scores_from_real_choices(
            trajectory_groups=trajectory_groups,
            logical_map=logical_map,
            require_routing_metadata=is_moe,
            weight_state="lora",
            rollout_mode=rollout_mode,
        )
        _write_json(
            artifact_dir / "real_path_vllm_lora_scores.json",
            vllm_lora.model_dump(mode="json"),
        )
        # When debugging a suspected ART/Megatron scoring bug, reuse the saved
        # vLLM scores, tokens, adapter, and routing replay here and rerun only
        # the Megatron worker below. That keeps the comparison paired and avoids
        # rerunning vLLM just to iterate on training-side scoring.
        await backend.close()
        backend_open = False

        base_diagnostic: RealPathBaseDiagnosticBundle | None = None
        megatron_base: ScoreBundle | None = None
        vllm_base: ScoreBundle | None = None
        base_comparison: PairComparison | None = None
        base_topk_comparison: TopKComparison | None = None
        if config.diagnose_base:
            base_diagnostic = await _score_base_real_generation_path(
                config=config,
                artifact_dir=artifact_dir,
                is_moe=is_moe,
                prompts=prompts,
                extra_body=extra_body,
                max_model_len=max_model_len,
            )
            megatron_base = base_diagnostic.megatron_scores
            vllm_base = base_diagnostic.vllm_scores

        worker_result = _run_real_path_megatron_worker(
            RealPathMegatronWorkerRequest(
                config=parity_config,
                artifact_dir=str(artifact_dir),
                disk_packed_tensors=disk_packed_tensors,
                logical_map_path=str(logical_map_path),
                weight_state="lora",
                adapter_path=adapter_path,
                moe_routing_replay_path=routing_replay_path,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
        )
        megatron_lora = ScoreBundle.model_validate(
            _read_json(Path(worker_result.score_path))
        )
        import torch

        sequence_ids = [token.prompt_id for token in logical_map.tokens]
        if megatron_base is not None and vllm_base is not None:
            base_comparison = compare_pair(
                candidate=torch.tensor(
                    megatron_base.target_logprobs, dtype=torch.float32
                ),
                target=torch.tensor(vllm_base.target_logprobs, dtype=torch.float32),
                sequence_ids=sequence_ids,
            )
            base_topk_comparison = compare_topk(megatron_base, vllm_base)
        comparison = compare_pair(
            candidate=torch.tensor(megatron_lora.target_logprobs, dtype=torch.float32),
            target=torch.tensor(vllm_lora.target_logprobs, dtype=torch.float32),
            sequence_ids=sequence_ids,
        )
        topk_comparison = compare_topk(megatron_lora, vllm_lora)
        mean_abs_pct_limit = fwd_mean_abs_pct_limit_for_model(
            parity_config.base_model,
            allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
        )
        top20_kl_limit = top20_kl_candidate_to_target_limit_for_model(
            parity_config.base_model,
            allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
        )
        prompt_tree_depth, prompt_tree_branch_count = _prompt_tree_shape(
            cast(dict[str, Any], packed_tensors)
        )
        passed = (
            comparison.mean_abs_pct <= mean_abs_pct_limit
            and topk_comparison.top20_intersection_kl_candidate_to_target
            <= top20_kl_limit
        )
        report = RealPathTrainInfReport(
            base_model=parity_config.base_model,
            artifact_dir=str(artifact_dir),
            logical_prompt_count=len(logical_map.prompts),
            logical_token_count=len(logical_map.tokens),
            base_logical_prompt_count=(
                base_diagnostic.logical_prompt_count
                if base_diagnostic is not None
                else None
            ),
            base_logical_token_count=(
                base_diagnostic.logical_token_count
                if base_diagnostic is not None
                else None
            ),
            base_moe_routing_packed_tokens=(
                base_diagnostic.moe_routing_packed_tokens
                if base_diagnostic is not None
                else None
            ),
            adapter_path=adapter_path,
            adapter_cache_key=adapter.cache_key,
            adapter_cache_hit=adapter.cache_hit,
            megatron_base_scores=(
                base_diagnostic.megatron_score_path
                if base_diagnostic is not None
                else None
            ),
            vllm_base_scores=(
                base_diagnostic.vllm_score_path if base_diagnostic is not None else None
            ),
            megatron_lora_scores=worker_result.score_path,
            vllm_lora_scores=str(artifact_dir / "real_path_vllm_lora_scores.json"),
            base=base_comparison,
            base_topk=base_topk_comparison,
            lora=comparison,
            lora_topk=topk_comparison,
            moe_routing_packed_tokens=int(stats.packed_tokens),
            prompt_tree_depth=prompt_tree_depth,
            prompt_tree_branch_count=prompt_tree_branch_count,
            mean_abs_pct_limit=mean_abs_pct_limit,
            top20_kl_candidate_to_target_limit=top20_kl_limit,
            passed=passed,
        )
        _write_json(
            artifact_dir / "real_path_comparison_report.json",
            report.model_dump(mode="json"),
        )
        _delete_adapter_safetensors_on_pass(artifact_dir, passed=report.passed)
        return report
    finally:
        if backend_open:
            await backend.close()


def _worker_cli(request_path: Path, *, adapter_only: bool) -> None:
    request = RealPathMegatronWorkerRequest.model_validate(_read_json(request_path))
    _real_path_megatron_worker(request, adapter_only=adapter_only)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--request", type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.worker:
        if args.request is None:
            raise ValueError("--worker requires --request")
        _worker_cli(args.request, adapter_only=bool(args.adapter_only))
        return 0
    raise ValueError("This module is intended to be run through pytest or --worker")


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
