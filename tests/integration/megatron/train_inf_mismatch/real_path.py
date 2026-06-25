from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
from typing import Any, AsyncIterator, cast
import uuid

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field

from art.dev.model import RolloutWeightsMode
from art.preprocessing.moe_routing import (
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
    choice_moe_routing_metadata,
)
from art.preprocessing.pack import DiskPackedTensors

from .artifacts import REPO_ROOT
from .output_parity import (
    TOP_K,
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
    prompt_count: int = 2
    rollouts_per_prompt: int = 2
    max_completion_tokens: int = 16
    prompt_sentence_count: int = 28
    prompt_target_tokens: int | None = None
    sliding_window: int | None = None
    diagnose_base: bool = False
    trace_layers: bool = False
    trace_enforce_eager: bool = False


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
    moe_routing_shared_prefix_rows: int
    moe_routing_shared_prefix_conflict_rows: int
    moe_routing_shared_prefix_conflict_slots: int
    moe_routing_shared_prefix_compared_slots: int
    vllm_forward_trace_dir: str | None = None
    megatron_forward_trace_dir: str | None = None


class RealPathTrainInfReport(BaseModel):
    base_model: str
    artifact_dir: str
    logical_prompt_count: int
    logical_token_count: int
    base_logical_prompt_count: int | None = None
    base_logical_token_count: int | None = None
    base_moe_routing_packed_tokens: int | None = None
    base_moe_routing_shared_prefix_conflict_rows: int | None = None
    base_moe_routing_shared_prefix_conflict_slots: int | None = None
    adapter_path: str
    megatron_base_scores: str | None = None
    vllm_base_scores: str | None = None
    megatron_lora_scores: str
    vllm_lora_scores: str
    base: PairComparison | None = None
    base_topk: TopKComparison | None = None
    lora: PairComparison
    lora_topk: TopKComparison
    moe_routing_packed_tokens: int
    moe_routing_shared_prefix_rows: int
    moe_routing_shared_prefix_conflict_rows: int
    moe_routing_shared_prefix_conflict_slots: int
    moe_routing_shared_prefix_compared_slots: int
    mean_abs_pct_limit: float
    top20_kl_candidate_to_target_limit: float
    passed: bool


def _real_path_rollout_mode(config: TrainInfOutputParityConfig) -> RolloutMode:
    return config.rollout_modes[0]


def _real_path_rollout_weights_mode(
    config: TrainInfOutputParityConfig,
) -> RolloutWeightsMode:
    return "lora" if _real_path_rollout_mode(config) == "native_lora" else "merged"


_PROMPT_SENTENCES = [
    "A careful systems engineer checks assumptions before changing thresholds.",
    "The training batch contains shared prefixes and divergent completions.",
    "Numerical parity should be measured on the exact tokens used by the policy.",
    "Sparse expert routing can create discontinuous output differences.",
    "A reproducible test writes enough artifacts to explain every comparison.",
    "LoRA adapters must be active and nonzero during both inference and training.",
    "The prompt should be realistic enough to exercise ordinary tokenizer paths.",
    "Packed Megatron inputs use shared prefixes while vLLM receives flat requests.",
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


def config_from_env() -> RealPathConfig:
    from .output_parity import config_from_env as output_config_from_env

    config = RealPathConfig(output_parity=output_config_from_env())
    if raw := os.environ.get("ART_REAL_PATH_PROMPT_COUNT"):
        config.prompt_count = int(raw)
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
    return (
        "Write a concise continuation for probe "
        f"{index}. Preserve the technical tone.\n\n" + " ".join(sentences)
    )


def _build_prompts(config: RealPathConfig) -> list[str]:
    rng = random.Random(config.output_parity.seed)
    prompts: list[str] = []
    for index in range(config.prompt_count):
        sentences = [
            rng.choice(_PROMPT_SENTENCES) for _ in range(config.prompt_sentence_count)
        ]
        prompts.append(_build_prompt_from_sentences(index, sentences))
    return prompts


async def _rollout(
    *,
    model: Any,
    prompt: str,
    max_completion_tokens: int,
    reward: float,
    extra_body: dict[str, Any] | None,
) -> Any:
    import art

    messages = [{"role": "user", "content": prompt}]

    async def _request() -> None:
        request_kwargs: dict[str, Any] = {}
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body
        response = await model.openai_client().chat.completions.create(
            model=model.get_inference_name(),
            messages=messages,
            max_tokens=max_completion_tokens,
            temperature=0.8,
            logprobs=True,
            top_logprobs=TOP_K,
            **request_kwargs,
        )
        if trajectory := art.auto_trajectory():
            logprobs = response.choices[0].logprobs
            trajectory.reward = reward
            trajectory.metrics["completion_tokens"] = (
                len(logprobs.content or []) if logprobs is not None else 0
            )

    return await art.capture_auto_trajectory(_request())


async def _collect_real_trajectory_groups(
    *,
    model: Any,
    config: RealPathConfig,
) -> list[Any]:
    from transformers import AutoTokenizer

    import art

    if config.rollouts_per_prompt < 2:
        raise ValueError("real-path mismatch requires at least two rollouts per prompt")
    tokenizer = AutoTokenizer.from_pretrained(config.output_parity.base_model)
    chat_template_kwargs: dict[str, Any] = {}
    if isinstance(tokenizer.chat_template, str):
        if "enable_thinking" in tokenizer.chat_template:
            chat_template_kwargs["enable_thinking"] = False
        if "preserve_thinking" in tokenizer.chat_template:
            chat_template_kwargs["preserve_thinking"] = True
    extra_body = (
        {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None
    )
    prompts = _build_prompts(config)
    groups = [
        art.TrajectoryGroup(
            [
                _rollout(
                    model=model,
                    prompt=prompt,
                    max_completion_tokens=config.max_completion_tokens,
                    reward=float(rollout_index % 2),
                    extra_body=extra_body,
                )
                for rollout_index in range(config.rollouts_per_prompt)
            ]
        )
        for prompt in prompts
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


def _choice_score_index(
    trajectory_groups: list[Any],
    *,
    require_routing_metadata: bool,
) -> dict[tuple[int, ...], Choice]:
    indexed: dict[tuple[int, ...], Choice] = {}
    for group in trajectory_groups:
        for trajectory in group:
            for item in trajectory.messages_and_choices:
                if not isinstance(item, Choice):
                    continue
                metadata = choice_moe_routing_metadata(item)
                if metadata is None:
                    if require_routing_metadata:
                        raise RuntimeError(
                            "Real-path trajectory choice is missing routes"
                        )
                    token_logprobs = (
                        item.logprobs.content
                        if item.logprobs is not None
                        and item.logprobs.content is not None
                        else []
                    )
                    indexed.setdefault(
                        tuple(_parse_token_id(entry.token) for entry in token_logprobs),
                        item,
                    )
                    continue
                prompt_ids = [int(value) for value in metadata["prompt_token_ids"]]
                completion_ids = [
                    int(value)
                    for value in (
                        metadata.get("completion_token_ids")
                        or metadata.get("token_ids")
                        or []
                    )
                ]
                indexed.setdefault(tuple(prompt_ids + completion_ids), item)
    return indexed


@asynccontextmanager
async def _direct_vllm_runtime(
    *,
    config: TrainInfOutputParityConfig,
    artifact_dir: Path,
    served_model_name: str,
    lora_path: str,
    rollout_weights_mode: str,
    engine_args: dict[str, Any],
    forward_trace_dir: Path | None = None,
) -> AsyncIterator[tuple[str, int]]:
    import art.vllm_runtime as runtime

    port = _free_port()
    launch_config = runtime.VllmRuntimeLaunchConfig(
        base_model=config.base_model,
        port=port,
        host="127.0.0.1",
        cuda_visible_devices=",".join(str(value) for value in config.inference_gpu_ids),
        lora_path=lora_path,
        served_model_name=served_model_name,
        rollout_weights_mode=cast(Any, rollout_weights_mode),
        engine_args=engine_args,
        server_args={"return_tokens_as_token_ids": True, **config.server_args},
    )
    command = runtime.build_vllm_runtime_server_cmd(launch_config)
    log_path = artifact_dir / f"real_path_vllm_{served_model_name}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if forward_trace_dir is not None:
        trace_site = Path(__file__).resolve().parent / "vllm_forward_trace_site"
        env["ART_VLLM_FORWARD_TRACE_DIR"] = str(forward_trace_dir)
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
    choice_by_prompt_id: dict[int, Choice] = {}
    for prompt in logical_map.prompts:
        key = (
            tuple(prompt.token_ids)
            if require_routing_metadata
            else tuple(prompt.token_ids[prompt.scored_token_start_index :])
        )
        choice = choices_by_tokens.get(key)
        if choice is None:
            raise RuntimeError(
                "Could not find captured vLLM choice for logical prompt "
                f"{prompt.prompt_id}"
            )
        choice_by_prompt_id[prompt.prompt_id] = choice
    target_logprobs: list[float] = []
    topk: list[TokenTopK] = []
    for token in logical_map.tokens:
        prompt = prompt_by_id[token.prompt_id]
        choice = choice_by_prompt_id[token.prompt_id]
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
) -> RealPathBaseDiagnosticBundle:
    import art
    from art.megatron.backend import MegatronBackend
    from art.preprocessing.pack import packed_tensors_to_dir

    parity_config = config.output_parity
    served_name = f"train_inf_real_base_{uuid.uuid4().hex[:8]}"
    placeholder_lora = artifact_dir / "unused_base_lora_placeholder"
    placeholder_lora.mkdir(exist_ok=True)
    engine_args = {
        "tensor_parallel_size": len(parity_config.inference_gpu_ids),
        "enable_expert_parallel": is_moe and len(parity_config.inference_gpu_ids) > 1,
        "max_model_len": parity_config.packed.sequence_length + 8,
        "max_logprobs": TOP_K,
        **parity_config.engine_args,
    }
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
        lora_path=str(placeholder_lora),
        rollout_weights_mode="merged",
        engine_args=engine_args,
        forward_trace_dir=vllm_forward_trace_dir,
    ) as (host, port):
        model = art.TrainableModel(
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
        rollout_mode="merged",
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
        moe_routing_shared_prefix_rows=int(stats.shared_prefix_rows),
        moe_routing_shared_prefix_conflict_rows=int(stats.shared_prefix_conflict_rows),
        moe_routing_shared_prefix_conflict_slots=int(
            stats.shared_prefix_conflict_slots
        ),
        moe_routing_shared_prefix_compared_slots=int(
            stats.shared_prefix_compared_slots
        ),
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


def _make_nonzero_adapter(
    *,
    config: TrainInfOutputParityConfig,
    artifact_dir: Path,
) -> str:
    request = RealPathMegatronWorkerRequest(
        config=config,
        artifact_dir=str(artifact_dir),
        disk_packed_tensors=cast(
            DiskPackedTensors,
            {
                "dir": str(artifact_dir / "unused"),
                "num_sequences": 1,
                "sequence_length": 1,
            },
        ),
        logical_map_path=str(artifact_dir / "unused_logical_map.json"),
        weight_state="lora",
        adapter_path=None,
        moe_routing_replay_path=None,
        global_grad_accumulation_sequences=1,
        forward_trace_dir=None,
    )
    return _run_real_path_megatron_worker(request, adapter_only=True).adapter_path or ""


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
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
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

    artifact_dir = Path(request.artifact_dir)
    adapter_path: Path | None = None
    if request.weight_state == "lora":
        if request.adapter_path is None:
            initial_state = _collect_full_lora_state(cast(list[Any], runtime.model))
            if torch.distributed.get_rank() == 0:  # type: ignore[possibly-missing-attribute]
                adapter_path = artifact_dir / "real_path_active_lora"
                initialized = _build_deterministic_nonzero_lora(
                    initial_state or {},
                    seed=request.config.seed,
                )
                _save_vllm_lora_adapter(
                    lora_path=adapter_path,
                    state=initialized,
                    runtime=runtime,
                    config=request.config,
                )
            torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
            adapter_path = artifact_dir / "real_path_active_lora"
        else:
            adapter_path = Path(request.adapter_path)
        adapter_model = load_lora_tensors_for_megatron(
            str(adapter_path),
            handler=runtime.model_support_handler,
            allow_unvalidated_arch=request.config.allow_unvalidated_arch,
        )
        megatron_train.load_adapter_into_model(runtime.model, adapter_model)

    if adapter_only:
        if torch.distributed.get_rank() == 0:  # type: ignore[possibly-missing-attribute]
            result = RealPathMegatronWorkerResult(
                score_path="",
                adapter_path=str(adapter_path) if adapter_path is not None else None,
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
        score_path = artifact_dir / f"real_path_megatron_{request.weight_state}.json"
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
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(value) for value in request.config.trainer_gpu_ids
    )
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
    rollout_mode = _real_path_rollout_mode(parity_config)
    is_moe = model_support_is_moe(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    )
    _write_json(artifact_dir / "real_path_config.json", config.model_dump(mode="json"))
    adapter_path = _make_nonzero_adapter(
        config=parity_config, artifact_dir=artifact_dir
    )
    if not adapter_path:
        raise RuntimeError("Real-path adapter worker did not create an adapter")

    _init_art_megatron_runtime_config(parity_config)
    backend = MegatronBackend(
        path=str(artifact_dir / "art_path"),
        enable_expert_replay=is_moe,
    )
    backend_open = False
    model = art.TrainableModel(
        name=f"train-inf-real-{uuid.uuid4().hex[:8]}",
        project="train_inf_mismatch",
        base_model=parity_config.base_model,
        lora_config=(
            {"target_modules": _lora_target_modules(parity_config)}
            if parity_config.lora_target_modules is not None
            else None
        ),
        _internal_config={
            "trainer_gpu_ids": parity_config.trainer_gpu_ids,
            "inference_gpu_ids": parity_config.inference_gpu_ids,
            "rollout_weights_mode": _real_path_rollout_weights_mode(parity_config),
            "allow_unvalidated_arch": parity_config.allow_unvalidated_arch,
            "engine_args": {
                "tensor_parallel_size": len(parity_config.inference_gpu_ids),
                "enable_expert_parallel": is_moe
                and len(parity_config.inference_gpu_ids) > 1,
                "max_model_len": parity_config.packed.sequence_length + 8,
                "max_logprobs": TOP_K,
                **parity_config.engine_args,
            },
            "init_args": {
                "max_seq_length": parity_config.packed.sequence_length,
            },
        },
    )
    _move_adapter_to_step_zero(adapter_path=adapter_path, model=model, backend=backend)

    try:
        await model.register(backend)
        backend_open = True
        trajectory_groups = await _collect_real_trajectory_groups(
            model=model,
            config=config,
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
            base_moe_routing_shared_prefix_conflict_rows=(
                base_diagnostic.moe_routing_shared_prefix_conflict_rows
                if base_diagnostic is not None
                else None
            ),
            base_moe_routing_shared_prefix_conflict_slots=(
                base_diagnostic.moe_routing_shared_prefix_conflict_slots
                if base_diagnostic is not None
                else None
            ),
            adapter_path=adapter_path,
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
            moe_routing_shared_prefix_rows=int(stats.shared_prefix_rows),
            moe_routing_shared_prefix_conflict_rows=int(
                stats.shared_prefix_conflict_rows
            ),
            moe_routing_shared_prefix_conflict_slots=int(
                stats.shared_prefix_conflict_slots
            ),
            moe_routing_shared_prefix_compared_slots=int(
                stats.shared_prefix_compared_slots
            ),
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
