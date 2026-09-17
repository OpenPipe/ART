"""Shared-expert output may not be consumed before its side stream finishes."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_shared_expert_handoff():
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    from art.megatron.compile_workarounds import (
        _install_shared_expert_handoff_workaround,
    )

    _install_shared_expert_handoff_workaround()

    @torch.compiler.disable
    def opaque_producer_delay():
        # Reproduce the graph break at a TE call without loading a model.
        torch.cuda._sleep(10_000_000)

    class Handoff(torch.nn.Module):
        get_output = SharedExpertMLP.get_output

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(moe_shared_expert_overlap=True)
            self.use_shared_expert_gate = False
            self.stream = torch.cuda.Stream()
            self.cached_output = None

        def forward(self, x):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                opaque_producer_delay()
                self.cached_output = x + 1
            return self.get_output() * 2

    model = torch.compile(Handoff())
    for value in range(1, 6):
        inputs = torch.full((193, 2688), float(value), device="cuda")
        torch.testing.assert_close(model(inputs), 2 * (inputs + 1), rtol=0, atol=0)
