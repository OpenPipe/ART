from typing import Union
from dataclasses import dataclass
import torch
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from peft.peft_model import PeftModel


@dataclass
class MandT:
    model: Union[AutoModelForSequenceClassification, PeftModel]
    tokenizer: PreTrainedTokenizer


ModelOrPath = Union[MandT, str]


def load_model(model_or_path: ModelOrPath) -> MandT:
    if isinstance(model_or_path, str):
        return load_peft_model(model_or_path, merge=True)
    else:
        return model_or_path


def load_peft_model(model_path: str, merge: bool = False) -> MandT:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
    )

    if merge:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        _apply_liger_kernel_to_instance(model)
    else:
        _apply_liger_kernel_to_instance(model)
        model = PeftModel.from_pretrained(model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return MandT(model, tokenizer)


def run_inference_transformers(
    prompts: list[str],
    model_or_path: Union[MandT, str],
    batch_size: int = 4,
) -> list[float]:
    mandt = load_model(model_or_path)

    model = mandt.model
    tokenizer = mandt.tokenizer

    # Tokenize all prompts
    tokenized_prompts = [
        tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts
    ]

    # Sort prompts by length (number of tokens)
    sorted_indices = sorted(
        range(len(tokenized_prompts)), key=lambda i: -len(tokenized_prompts[i])
    )
    sorted_prompts = [prompts[i] for i in sorted_indices]

    results = []
    for i in tqdm(
        range(0, len(sorted_prompts), batch_size),
        total=len(sorted_prompts) // batch_size,
    ):
        batch = sorted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        results.extend(logits.cpu().tolist())

    # Reorder results to match original prompt order
    original_order_results = [0.0] * len(prompts)
    for i, result in zip(sorted_indices, results):
        original_order_results[i] = result

    return original_order_results
