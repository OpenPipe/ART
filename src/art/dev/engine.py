from typing import Any, Literal, Tuple

from typing_extensions import TypedDict


class EngineArgs(TypedDict, total=False):
    model: str
    served_model_name: str | list[str] | None
    tokenizer: str | None
    # task: TaskOption
    skip_tokenizer_init: bool
    tokenizer_mode: str
    trust_remote_code: bool
    allowed_local_media_path: str
    download_dir: str | None
    load_format: str
    # config_format: ConfigFormat
    dtype: str
    kv_cache_dtype: str
    seed: int
    max_model_len: int | None
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # distributed_executor_backend: str | type[ExecutorBase] | None
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int
    tensor_parallel_size: int
    max_parallel_loading_workers: int | None
    block_size: int | None
    enable_prefix_caching: bool | None
    disable_sliding_window: bool
    use_v2_block_manager: bool
    swap_space: float  # GiB
    cpu_offload_gb: float  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int | None
    max_num_partial_prefills: int | None
    max_long_partial_prefills: int | None
    long_prefill_token_threshold: int | None
    max_num_seqs: int | None
    max_logprobs: int  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool
    revision: str | None
    code_revision: str | None
    rope_scaling: dict[str, Any] | None
    rope_theta: float | None
    # hf_overrides: HfOverrides | None
    tokenizer_revision: str | None
    quantization: str | None
    enforce_eager: bool | None
    max_seq_len_to_capture: int
    disable_custom_all_reduce: bool
    tokenizer_pool_size: int
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # tokenizer_pool_type: str | type["BaseTokenizerGroup"]
    tokenizer_pool_extra_config: dict[str, Any] | None
    limit_mm_per_prompt: dict[str, int] | None
    mm_processor_kwargs: dict[str, Any] | None
    disable_mm_preprocessor_cache: bool
    enable_lora: bool
    enable_lora_bias: bool
    max_loras: int
    max_lora_rank: int
    enable_prompt_adapter: bool
    max_prompt_adapters: int
    max_prompt_adapter_token: int
    fully_sharded_loras: bool
    lora_extra_vocab_size: int
    long_lora_scaling_factors: Tuple[float] | None
    lora_dtype: str | None
    max_cpu_loras: int | None
    device: str
    num_scheduler_steps: int
    multi_step_stream_outputs: bool
    ray_workers_use_nsight: bool
    num_gpu_blocks_override: int | None
    num_lookahead_slots: int
    model_loader_extra_config: dict | None
    ignore_patterns: str | list[str] | None
    preemption_mode: str | None

    scheduler_delay_factor: float
    enable_chunked_prefill: bool | None

    guided_decoding_backend: str
    logits_processor_pattern: str | None
    # Speculative decoding configuration.
    speculative_model: str | None
    speculative_model_quantization: str | None
    speculative_draft_tensor_parallel_size: int | None
    num_speculative_tokens: int | None
    speculative_disable_mqa_scorer: bool | None
    speculative_max_model_len: int | None
    speculative_disable_by_batch_size: int | None
    ngram_prompt_lookup_max: int | None
    ngram_prompt_lookup_min: int | None
    spec_decoding_acceptance_method: str
    typical_acceptance_sampler_posterior_threshold: float | None
    typical_acceptance_sampler_posterior_alpha: float | None
    qlora_adapter_name_or_path: str | None
    disable_logprobs_during_spec_decoding: bool | None

    otlp_traces_endpoint: str | None
    collect_detailed_traces: str | None
    disable_async_output_proc: bool
    scheduling_policy: Literal["fcfs", "priority"]
    scheduler_cls: str | type[object]

    override_neuron_config: dict[str, Any] | None
    # override_pooler_config: PoolerConfig | None
    # compilation_config: CompilationConfig | None
    worker_cls: str

    # kv_transfer_config: KVTransferConfig | None

    generation_config: str | None
    override_generation_config: dict[str, Any] | None
    enable_sleep_mode: bool
    model_impl: str

    calculate_kv_scales: bool | None

    additional_config: dict[str, Any] | None

    disable_log_requests: (
        bool  # Deprecated in vLLM 0.13+, use enable_log_requests instead
    )
    enable_log_requests: bool  # New in vLLM 0.13+
