"""Pinned vLLM transport for forced-continuation teacher scoring."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import httpx

from art.serving_capabilities import ServingCapabilities

from .scorer import (
    RankedTokenLogprob,
    RetryableTeacherScoringError,
    ScoredPosition,
    TeacherScoringRequest,
    TeacherScoringResult,
)


class VLLMScoringError(RuntimeError):
    """The vLLM endpoint cannot produce a trustworthy teacher target."""


class VLLMRetryableScoringError(
    VLLMScoringError,
    RetryableTeacherScoringError,
):
    """Transient vLLM failure safe to retry with the identical request."""


# vLLM emits prompt log-probabilities from float32 model outputs. A sum of
# ``width`` positive float32 probabilities can exceed one by at most the
# standard sequential-accumulation bound gamma_width = width*u/(1-width*u),
# where u is binary32 unit roundoff. Keep this local to the wire parser: the
# prepared target remains an ordinary normalized distribution.
_FLOAT32_UNIT_ROUNDOFF = 2.0**-24


def _float32_accumulation_tolerance(width: int) -> float:
    scaled_roundoff = width * _FLOAT32_UNIT_ROUNDOFF
    return scaled_roundoff / (1.0 - scaled_roundoff)


class VLLMTeacherScorer:
    """Score exact continuation IDs through vLLM 0.23 prompt log-probabilities.

    The caller owns any supplied ``httpx.AsyncClient``. When no client is
    supplied, each call uses a short-lived client so this object itself has no
    lifecycle requirement.
    """

    def __init__(
        self,
        *,
        base_url: str,
        capabilities: ServingCapabilities,
        model_name: str | None = None,
        render_model_name: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._capabilities = capabilities
        self._model_name = model_name
        self._render_model_name = render_model_name
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._client = client

    async def score(
        self,
        request: TeacherScoringRequest,
    ) -> TeacherScoringResult:
        """Render a chat view, append forced IDs, and parse selected positions."""

        self._validate_capabilities(request)
        if request.teacher_view.protocol != "chat_completions":
            raise VLLMScoringError(
                "vLLM teacher scoring currently requires a chat-completions view"
            )
        teacher_view = request.teacher_view.request()
        if not isinstance(teacher_view, dict):
            raise VLLMScoringError("teacher chat view must be a JSON object")
        truncate_prompt_tokens = teacher_view.get("truncate_prompt_tokens")
        if truncate_prompt_tokens is not None and truncate_prompt_tokens is not False:
            raise VLLMScoringError(
                "vLLM teacher scoring does not allow prompt truncation"
            )

        if self._client is not None:
            return await self._score_with_client(self._client, request)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._score_with_client(client, request)

    def _validate_capabilities(self, request: TeacherScoringRequest) -> None:
        capabilities = self._capabilities
        try:
            capabilities.require(
                "prompt_token_distributions",
                operation="vLLM teacher scoring",
            )
        except RuntimeError as exc:
            raise VLLMScoringError(str(exc)) from exc
        if capabilities.prompt_token_distribution_version != 1:
            raise VLLMScoringError(
                "vLLM teacher scoring requires distribution protocol version 1"
            )
        if request.target.temperature != 1.0:
            raise VLLMScoringError(
                "vLLM 0.23 prompt distributions support temperature 1.0 only"
            )
        if capabilities.prompt_distribution_temperature != "unit_only":
            raise VLLMScoringError(
                "serving runtime does not advertise unit-temperature distributions"
            )
        if capabilities.token_space_fingerprint != request.token_space_fingerprint:
            raise VLLMScoringError(
                "teacher and learner token-space fingerprints do not match"
            )
        if capabilities.logical_vocab_size != request.logical_vocab_size:
            raise VLLMScoringError(
                "teacher and learner logical vocabulary sizes do not match"
            )
        capacity = capabilities.max_prompt_logprobs
        if capacity is None or request.target.k > capacity:
            raise VLLMScoringError(
                f"requested top-k width {request.target.k} exceeds the serving "
                f"capacity {capacity}"
            )

    async def _score_with_client(
        self,
        client: httpx.AsyncClient,
        request: TeacherScoringRequest,
    ) -> TeacherScoringResult:
        model_name = self._model_name or request.teacher_name
        render_model_name = self._render_model_name or model_name
        render_body = request.teacher_view.request()
        if not isinstance(render_body, dict):
            raise VLLMScoringError("teacher chat view must be a JSON object")
        render_body = dict(render_body)
        render_body["model"] = render_model_name
        render_body["stream"] = False
        render_body["max_completion_tokens"] = 1
        render_body.pop("max_tokens", None)

        rendered = await self._post_json(
            client,
            "/v1/chat/completions/render",
            render_body,
        )
        self._validate_observable_identity(
            rendered,
            expected_model=render_model_name,
            expected_revision=request.teacher_revision,
            phase="render",
        )

        prefix_ids = _integer_list(
            rendered.get("token_ids"), label="rendered token IDs"
        )
        generation_body = dict(rendered)
        generation_body["request_id"] = request.request_id
        generation_body["model"] = model_name
        generation_body["token_ids"] = [*prefix_ids, *request.forced_token_ids]
        generation_body["stream"] = False
        sampling_params = generation_body.get("sampling_params")
        if not isinstance(sampling_params, dict):
            raise VLLMScoringError("render response omitted sampling parameters")
        sampling_params = dict(sampling_params)
        sampling_params.update(
            {
                "detokenize": False,
                "max_tokens": 1,
                "n": 1,
                "prompt_logprobs": request.target.k,
                # Prefix-cache hits omit cached prompt distributions from vLLM's
                # response. Exact distillation needs one row for every forced
                # continuation token, including deterministic retries.
                "skip_reading_prefix_cache": True,
                "temperature": 0.0,
            }
        )
        generation_body["sampling_params"] = sampling_params

        generated = await self._post_json(
            client,
            "/inference/v1/generate",
            generation_body,
        )
        self._validate_observable_identity(
            generated,
            expected_model=model_name,
            expected_revision=request.teacher_revision,
            phase="generate",
        )
        choices = generated.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise VLLMScoringError(
                "generate response must contain exactly one discarded decode"
            )

        raw_distributions = generated.get("prompt_logprobs")
        if not isinstance(raw_distributions, list):
            raise VLLMScoringError(
                "generate response omitted prompt token distributions"
            )
        expected_length = len(prefix_ids) + len(request.forced_token_ids)
        if len(raw_distributions) != expected_length:
            raise VLLMScoringError(
                "prompt distribution count does not match scored token sequence"
            )

        rows = tuple(
            _parse_position(
                raw_distributions[len(prefix_ids) + position],
                position=position,
                forced_token_id=request.forced_token_ids[position],
                k=request.target.k,
                logical_vocab_size=request.logical_vocab_size,
            )
            for position in request.selected_positions
        )
        return TeacherScoringResult.create(request=request, positions=rows)

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await client.post(
                f"{self._base_url}{path}",
                headers=self._headers,
                json=body,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise VLLMRetryableScoringError(
                f"vLLM request to {path} failed transiently"
            ) from exc
        if response.status_code in {408, 425, 429} or response.status_code >= 500:
            raise VLLMRetryableScoringError(
                f"vLLM request to {path} failed transiently"
            )
        try:
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            raise VLLMScoringError(f"vLLM request to {path} failed") from exc
        except ValueError as exc:
            raise VLLMScoringError(
                f"vLLM response from {path} was not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise VLLMScoringError(f"vLLM response from {path} was not an object")
        return value

    @staticmethod
    def _validate_observable_identity(
        response: Mapping[str, Any],
        *,
        expected_model: str,
        expected_revision: int | str,
        phase: str,
    ) -> None:
        model = response.get("model")
        if model is not None and model != expected_model:
            raise VLLMScoringError(f"{phase} response model does not match request")
        revision = response.get("revision", response.get("teacher_revision"))
        if revision is not None and revision != expected_revision:
            raise VLLMScoringError(f"{phase} response revision does not match request")


def _integer_list(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value
        )
    ):
        raise VLLMScoringError(f"{label} must be a non-empty list of token IDs")
    return value


def _parse_position(
    value: Any,
    *,
    position: int,
    forced_token_id: int,
    k: int,
    logical_vocab_size: int,
) -> ScoredPosition:
    if not isinstance(value, dict) or not value:
        raise VLLMScoringError(
            f"prompt distribution at continuation position {position} is missing"
        )

    ranked: dict[int, tuple[int, float]] = {}
    forced_token_observed = False
    for raw_token_id, raw_entry in value.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as exc:
            raise VLLMScoringError(
                "prompt distribution contains a malformed token ID"
            ) from exc
        if str(token_id) != str(raw_token_id) or not 0 <= token_id < logical_vocab_size:
            raise VLLMScoringError("prompt distribution contains an invalid token ID")
        if not isinstance(raw_entry, dict):
            raise VLLMScoringError("prompt distribution entry must be an object")
        rank = raw_entry.get("rank")
        logprob = raw_entry.get("logprob")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or not isinstance(logprob, int | float)
            or isinstance(logprob, bool)
        ):
            raise VLLMScoringError(
                "prompt distribution contains a malformed rank or mass"
            )
        logprob = float(logprob)
        if not math.isfinite(logprob) or logprob > 0.0:
            raise VLLMScoringError(
                "prompt distribution contains non-finite or positive mass"
            )
        if token_id == forced_token_id:
            forced_token_observed = True
        if rank <= k:
            if rank in ranked:
                raise VLLMScoringError("prompt distribution contains duplicate ranks")
            ranked[rank] = (token_id, logprob)

    if not forced_token_observed:
        raise VLLMScoringError(
            "prompt distribution omitted the forced token at its scored position"
        )
    expected_width = min(k, logical_vocab_size)
    expected_ranks = set(range(1, expected_width + 1))
    if set(ranked) != expected_ranks:
        raise VLLMScoringError(
            "prompt distribution does not contain a complete contiguous top-k"
        )

    ordered = tuple(ranked[rank] for rank in range(1, expected_width + 1))
    explicit_mass = math.fsum(math.exp(logprob) for _, logprob in ordered)
    if expected_width == logical_vocab_size:
        if not math.isclose(explicit_mass, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise VLLMScoringError("full prompt distribution does not normalize to one")
        log_normalizer = math.log(explicit_mass)
        ordered = tuple(
            (token_id, logprob - log_normalizer) for token_id, logprob in ordered
        )
        tail_logprob = None
    else:
        tail_mass = 1.0 - explicit_mass
        if not math.isfinite(tail_mass):
            raise VLLMScoringError(
                "sparse prompt distribution has invalid residual mass"
            )
        if tail_mass > 0.0:
            tail_logprob = math.log(tail_mass)
        else:
            over_mass = -tail_mass
            tolerance = _float32_accumulation_tolerance(expected_width)
            if over_mass > tolerance:
                raise VLLMScoringError(
                    "sparse prompt distribution has invalid residual mass"
                )

            # A sparse distribution must retain a positive tail. One binary32
            # unit roundoff is the smallest meaningful probability-scale mass
            # for this float32 wire contract. Renormalizing by a common factor
            # preserves the supplied rank order while making explicit entries
            # plus the synthetic tail a valid full-vocabulary distribution.
            synthetic_tail_mass = _FLOAT32_UNIT_ROUNDOFF
            log_normalizer = math.log(explicit_mass + synthetic_tail_mass)
            ordered = tuple(
                (token_id, logprob - log_normalizer) for token_id, logprob in ordered
            )
            tail_logprob = math.log(synthetic_tail_mass) - log_normalizer

    return ScoredPosition(
        position=position,
        forced_token_id=forced_token_id,
        entries=tuple(
            RankedTokenLogprob(rank=rank - 1, token_id=token_id, logprob=logprob)
            for rank, (token_id, logprob) in enumerate(ordered, start=1)
        ),
        tail_logprob=tail_logprob,
        logical_vocab_size=logical_vocab_size,
        temperature=1.0,
    )
