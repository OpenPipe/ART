def get_renderer_name(base_model: str) -> str:
    if base_model.startswith("meta-llama/"):
        return "llama3"
    elif base_model.startswith("Qwen/Qwen3-"):
        if "Instruct" in base_model:
            return "qwen3_instruct"
        else:
            print("Defaulting to Qwen3 renderer without thinking for", base_model)
            print(renderer_name_message)
            return "qwen3_disable_thinking"
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

- llama3
- qwen3
- qwen3_disable_thinking
- qwen3_instruct
- deepseekv3
- deepseekv3_disable_thinking
- gpt_oss_no_sysprompt
- gpt_oss_low_reasoning
- gpt_oss_medium_reasoning
- gpt_oss_high_reasoning
""".strip()
