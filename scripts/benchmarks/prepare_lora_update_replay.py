#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

BONNIE_ARTIFACT = "wb-training/bench/bonnie:v1"
BONNIE_TRAIN_SHA256 = "fd26c97423de94a158d5a54e38485ec55b3fd3504f5f0346850fcfa79ad3ce66"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_dir = args.output / "bonnie-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    wandb.Api().artifact(BONNIE_ARTIFACT).download(root=str(artifact_dir))
    train_path = artifact_dir / "train.jsonl"
    digest = _sha256(train_path)
    if digest != BONNIE_TRAIN_SHA256:
        raise RuntimeError(
            f"Bonnie train artifact checksum is {digest}; "
            f"expected {BONNIE_TRAIN_SHA256}"
        )
    request_path = args.output / "requests.jsonl"
    requests: list[dict[str, object]] = []
    with train_path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = [dict(message) for message in row["messages"]]
            for index, message in enumerate(messages):
                if index > 0 and message.get("role") == "system":
                    message["role"] = "user"
                    message["content"] = "[System note]\n" + str(
                        message.get("content") or ""
                    )
            request: dict[str, object] = {
                "messages": messages,
                "n": 8,
                "temperature": 0.8,
                "max_tokens": 256,
                "logprobs": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if row.get("tools"):
                request["tools"] = row["tools"]
            requests.append(
                {
                    "id": str(
                        row.get("id")
                        or (
                            f"mar31_v2:train:{row.get('trace_id', source_index)}:"
                            f"assistant-{row.get('assistant_turn_index', 0)}"
                        )
                    ),
                    "request": request,
                }
            )
            if len(requests) == 64:
                break
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests)
    )
    print(f"BONNIE_TRAIN_JSONL={train_path}")
    print(f"REQUEST_JSONL={request_path}")


if __name__ == "__main__":
    main()
