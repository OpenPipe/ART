from dataclasses import dataclass, field, fields
import torch
from typing import Iterable, Optional, Union

ignore_labels_cache: dict[
    tuple[torch.Size, Union[int, float], torch.dtype, torch.device], torch.Tensor
] = {}


def shift_tensor(
    labels: torch.Tensor, ignore_label: Optional[Union[int, float]] = None
) -> torch.Tensor:
    if ignore_label is None:
        ignore_label = (
            -100
            if labels.dtype in (torch.int32, torch.int64, torch.int16, torch.int8)
            else float("nan")
        )

    # Create a tensor of ignore labels every time if we are compiling, otherwise cache it
    if torch.compiler.is_compiling():
        ignore_labels = torch.full(
            (labels.shape[0], 1), ignore_label, dtype=labels.dtype, device=labels.device
        )
    else:
        key = (labels.shape[-1:], ignore_label, labels.dtype, labels.device)
        if key not in ignore_labels_cache:
            ignore_labels_cache[key] = torch.full(
                (labels.shape[0], 1),
                ignore_label,
                dtype=labels.dtype,
                device=labels.device,
            )
        ignore_labels = ignore_labels_cache[key]

    # Shift labels to compute loss
    return torch.cat((labels[..., 1:], ignore_labels), dim=1)


@dataclass
class GRPOResult:
    num_tokens: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    policy_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    entropy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    kl_div: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    entropy_weight: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    kl_weight: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def named_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def per_token(self) -> "GRPOResult":
        return GRPOResult(
            **{name: tensor / self.num_tokens for name, tensor in self.named_tensors()}
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        return (tensor for _, tensor in self.named_tensors())

    def to(self, target: Union[torch.device, torch.dtype]) -> "GRPOResult":
        return GRPOResult(
            **{name: tensor.to(target) for name, tensor in self.named_tensors()}
        )

    def __iadd__(self, other: "GRPOResult") -> "GRPOResult":
        for tensor, other_tensor in zip(self.tensors(), other.tensors()):
            tensor += other_tensor.to(tensor.device)
        return self

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            self.policy_loss
            - self.entropy * self.entropy_weight / self.num_tokens
            + torch.nan_to_num(self.kl_div, 0.0) * self.kl_weight / self.num_tokens
        )


class GRPO(torch.nn.Module):
    def __init__(
        self, clip_epsilon: float = 0.2, entropy_coef: float = 0.0, kl_coef: float = 0.0
    ) -> None:
        """
        Initialize the GRPO loss.

        Args:
            clip_epsilon (float): The epsilon value for clipping the policy ratio.
            entropy_coef (float): The coefficient for the entropy bonus.
            kl_coef (float): The coefficient for the KL divergence penalty.
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef

    def forward(
        self,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor | None,
        mask: torch.Tensor,
        weights: torch.Tensor,
        bos_id: int,
    ) -> GRPOResult:
        """
        Computes the GRPO loss for sequence data, supporting both regular and chunked inputs.

        Args:
            logits (Union[Tensor, List[Tensor]]):
                Either a single tensor of shape (batch_size, sequence_length, vocab_size)
                or a list of chunked tensors, each of shape
                (batch_size, sequence_length/num_chunks, vocab_size).
            tokens (Tensor):
                Shape: (batch_size, sequence_length)
                Token indices.
            advantages (Tensor):
                Shape: (batch_size, sequence_length)
                Token advantages.
            logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Token log probabilities.
            reference_logprobs (Tensor | None):
                Shape: (batch_size, sequence_length)
                Reference token log probabilities.
            mask (Tensor):
                Shape: (batch_size, sequence_length)
                Boolean mask specifying positions where loss should be computed.
            weights (Tensor):
                Shape: (batch_size, sequence_length)
                Weights for each token.
            bos_id (int):
                Index of the beginning of sequence token in the vocabulary.

        Returns:
            GRPOResult: The combined loss results across all chunks.
        """
        if isinstance(logits, list):
            result = GRPOResult().to(logits[0].device)
            num_chunks = len(logits)
            for chunked_args in zip(
                logits,
                tokens.chunk(num_chunks, dim=1),
                advantages.chunk(num_chunks, dim=1),
                logprobs.chunk(num_chunks, dim=1),
                (
                    reference_logprobs.chunk(num_chunks, dim=1)
                    if reference_logprobs is not None
                    else [None] * num_chunks
                ),
                mask.chunk(num_chunks, dim=1),
                weights.chunk(num_chunks, dim=1),
            ):
                result += self._forward_chunk(*chunked_args, bos_id=bos_id)
            return result

        return self._forward_chunk(
            logits,
            tokens,
            advantages,
            logprobs,
            reference_logprobs,
            mask,
            weights,
            bos_id,
        )

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor | None,
        mask: torch.Tensor,
        weights: torch.Tensor,
        bos_id: int,
    ) -> GRPOResult:
        """
        Processes a single chunk of the GRPO loss computation.
        """
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        tokens = shift_tensor(tokens, bos_id).view(
            -1
        )  # (batch_size * sequence_length,)
        advantages = shift_tensor(advantages, 0).view(
            -1
        )  # (batch_size * sequence_length,)
        logprobs = shift_tensor(logprobs, 0).view(-1)  # (batch_size * sequence_length,)
        if reference_logprobs is not None:
            reference_logprobs = shift_tensor(reference_logprobs, 0).view(-1)
        mask = shift_tensor(mask, False).view(-1)  # (batch_size * sequence_length,)
        weights = shift_tensor(weights, 0).view(-1)  # (batch_size * sequence_length,)
        num_tokens = mask.sum()
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()[mask]
        new_logprobs = dist.log_prob(tokens)[mask]
        logprobs = logprobs[mask]
        logprobs = torch.where(torch.isnan(logprobs), new_logprobs, logprobs)
        if reference_logprobs is not None:
            reference_logprobs = reference_logprobs[mask]
        advantages = advantages[mask]
        diff = new_logprobs - logprobs
        prob_ratio = torch.exp(diff)
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages,
        )
        if reference_logprobs is not None:
            kl_div = torch.nn.functional.kl_div(
                new_logprobs,
                reference_logprobs,
                reduction="none",
                log_target=True,
            )
        else:
            kl_div = torch.tensor(torch.nan, device=logits.device)
        weights = weights[mask]
        return GRPOResult(
            num_tokens=num_tokens,
            policy_loss=policy_loss.mul(weights).sum(),
            entropy=entropy.mul(weights).sum(),
            kl_div=kl_div.mul(weights).sum(),
            entropy_weight=self.entropy_coef * num_tokens,
            kl_weight=self.kl_coef * num_tokens,
        )
