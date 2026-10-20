def is_art_adapter_param_name(name: str) -> bool:
    return any(
        segment in name
        for segment in (
            ".lora.",
            ".q_proj_lora.",
            ".k_proj_lora.",
            ".v_proj_lora.",
            ".qkv_lora.",
            ".q_lora.",
            ".k_lora.",
            ".v_lora.",
            ".z_lora.",
            ".gate_lora.",
            ".up_lora.",
        )
    )


def canonical_art_param_name(name: str) -> str:
    segments = name.split(".")
    while segments and segments[0] == "module":
        segments = segments[1:]

    canonical: list[str] = []
    i = 0
    while i < len(segments):
        if segments[i] == "_orig_mod":
            i += 1
            continue
        if i + 1 < len(segments):
            current = segments[i]
            nxt = segments[i + 1]
            if (
                current
                in {
                    "linear_proj",
                    "linear_qkv",
                    "in_proj",
                    "linear_fc1",
                    "linear_fc2",
                }
                and nxt == current
            ):
                canonical.append(current)
                i += 2
                continue
            if current == "out_proj" and nxt == "linear_proj":
                canonical.append(current)
                i += 2
                continue
            if current == "row_parallel_lora" and nxt == "linear_proj":
                i += 2
                continue
        canonical.append(segments[i])
        i += 1
    return ".".join(canonical)
