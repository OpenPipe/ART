from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tests.integration.megatron.model_support.workflow_throughput import (
    _sized_prompt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scenario-id", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=3838)
    parser.add_argument("--completion-tokens", type=int, default=64)
    parser.add_argument("--bad-word", action="append", default=[])
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    token = os.environ["ART_PRIVATE_DISPATCH_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompt = _sized_prompt(tokenizer, target_tokens=args.prompt_tokens).replace(
        "00000000", f"{args.scenario_id:08d}", 1
    )
    extra_body: dict[str, Any] = {
        "ignore_eos": True,
        "min_tokens": args.completion_tokens,
    }
    if args.bad_word:
        extra_body["bad_words"] = args.bad_word

    client = AsyncOpenAI(base_url=args.base_url, api_key=token)
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=args.model,
            max_tokens=args.completion_tokens,
            n=4,
            temperature=1.0,
            seed=args.scenario_id,
            logprobs=True,
            top_logprobs=0,
            timeout=1200.0,
            extra_body=extra_body,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "scenario_id": args.scenario_id,
                    "bad_words": args.bad_word,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=True,
            )
        )
        raise
    finally:
        await client.close()

    print(
        json.dumps(
            {
                "scenario_id": args.scenario_id,
                "bad_words": args.bad_word,
                "choice_count": len(response.choices),
                "completion_tokens": response.usage.completion_tokens
                if response.usage is not None
                else None,
                "finish_reasons": [choice.finish_reason for choice in response.choices],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
