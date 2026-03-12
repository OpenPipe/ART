def get_renderer_name(base_model: str) -> str:
    if base_model.startswith("meta-llama/"):
        return "llama3"
    elif base_model.startswith("Qwen/Qwen3.5-"):
        print("Defaulting to Qwen3.5 renderer with thinking for", base_model)
        print(renderer_name_message)
        return "qwen3_5"
    elif base_model.startswith("Qwen/Qwen3-"):
        if "Instruct" in base_model:
            return "qwen3_instruct"
        else:
            print("Defaulting to Qwen3 renderer without thinking for", base_model)
            print(renderer_name_message)
            return "qwen3_disable_thinking"
    elif base_model.startswith("moonshotai/Kimi-K2.5"):
        print("Defaulting to Kimi K2.5 renderer with thinking for", base_model)
        print(renderer_name_message)
        return "kimi_k25"
    elif base_model.startswith("moonshotai/Kimi-K2"):
        print("Defaulting to Kimi K2 renderer with thinking for", base_model)
        print(renderer_name_message)
        return "kimi_k2"
    elif base_model.startswith("deepseek-ai/DeepSeek-V3"):
        print("Defaulting to DeepSeekV3 renderer without thinking for", base_model)
        print(renderer_name_message)
        return "deepseekv3_disable_thinking"
    elif base_model.startswith("openai/gpt-oss"):
        print("Defaulting to GPT-OSS renderer without system prompt for", base_model)
        print(renderer_name_message)
        return "gpt_oss_no_sysprompt"
    else:
        raise ValueError(f"Unknown base model: {base_model}")


renderer_name_message = """
To manually specify a renderer (and silence this message), you can set the "renderer_name" field like so:

model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen3-8B",
    _internal_config=art.dev.InternalModelConfig(
        tinker_args=art.dev.TinkerArgs(renderer_name="qwen3_disable_thinking"),
    ),
)

Valid renderer names are:

- role_colon
- llama3
- qwen3
- qwen3_vl
- qwen3_vl_instruct
- qwen3_disable_thinking
- qwen3_instruct
- qwen3_5
- qwen3_5_disable_thinking
- deepseekv3
- deepseekv3_disable_thinking
- deepseekv3_thinking
- kimi_k2
- kimi_k25
- kimi_k25_disable_thinking
- gpt_oss_no_sysprompt
- gpt_oss_low_reasoning
- gpt_oss_medium_reasoning
- gpt_oss_high_reasoning
""".strip()
