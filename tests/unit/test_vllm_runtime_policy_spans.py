from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def _load_policy_spans_module() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "vllm_runtime"
        / "src"
        / "art_vllm_runtime"
        / "policy_spans.py"
    )
    spec = importlib.util.spec_from_file_location("test_art_vllm_policy_spans", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parallel_sampling_preserves_every_child_policy_span(monkeypatch: Any) -> None:
    policy_spans = _load_policy_spans_module()

    class FakeOutputProcessor:
        def process_outputs(self, engine_core_outputs: list[Any]) -> list[Any]:
            return engine_core_outputs

    class FakeParentRequest:
        def __init__(self, count: int) -> None:
            self.outputs: list[Any | None] = [None] * count

        def add(self, index: int, output: Any) -> Any | None:
            self.outputs[index] = output
            if any(item is None for item in self.outputs):
                return None
            return SimpleNamespace(outputs=list(self.outputs))

    class FakeDetokenizer:
        def __init__(self, output_tokens: int) -> None:
            self.output_tokens = output_tokens

        def num_output_tokens(self) -> int:
            return self.output_tokens

    class FakeRequestState:
        def __init__(
            self, request_id: str, request_index: int, parent_req: Any
        ) -> None:
            self.request_id = request_id
            self.request_index = request_index
            self.parent_req = parent_req
            self.detokenizer = FakeDetokenizer(output_tokens=2)

        def _new_completion_output(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(index=self.request_index)

        def make_request_output(
            self, new_token_ids: list[int], *_args: Any, **_kwargs: Any
        ) -> Any | None:
            output = self._new_completion_output(new_token_ids)
            return self.parent_req.add(self.request_index, output)

    output_processor_module = ModuleType("vllm.v1.engine.output_processor")
    setattr(output_processor_module, "OutputProcessor", FakeOutputProcessor)
    setattr(output_processor_module, "RequestState", FakeRequestState)
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.engine.output_processor", output_processor_module
    )

    policy_spans._patch_output_processor_policy_span_accumulation()

    parent = FakeParentRequest(count=2)
    states = [
        FakeRequestState("0_parent", 0, parent),
        FakeRequestState("1_parent", 1, parent),
    ]
    for index, state in enumerate(states):
        setattr(
            policy_spans,
            "_CURRENT_ENGINE_POLICY_SPANS",
            {
                state.request_id: [
                    {
                        "start_token": 0,
                        "end_token": 2,
                        "policy_version": index,
                        "lora_slot": "train",
                        "update_seq": index,
                    }
                ]
            },
        )
        response = state.make_request_output([10, 11])
        if index == 0:
            assert response is None

    assert response is not None
    assert [
        output.art_policy_token_spans[0]["policy_version"]
        for output in response.outputs
    ] == [0, 1]
    assert all(
        output.art_policy_token_spans[0]["end_token"] == 2
        for output in response.outputs
    )
