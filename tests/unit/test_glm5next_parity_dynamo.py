"""Real GLM/mHC forward under production Dynamo policy, with CPU codegen.

The legacy observer must trigger the workstation's exact duplicate-module
assertion. The fixed observers must capture both mHC sites outside all graphs.
Only the backend changes: eager replaces CUDA Inductor; fullgraph=False and
dynamic=None remain torch.compile defaults, as in training/compile.py.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")


def legacy_pre_observer(layer):
    """The failing run-1 closure, including its OptimizedModule reference."""
    original_pre = layer.hc_util.layer_pre
    site = ["attn_hc"]

    def pre(hidden, fn, scale, base):
        site[0] = "attn_hc" if fn is layer.hc_attn_fn else "ffn_hc"
        return original_pre(hidden, fn, scale, base)

    layer.hc_util.layer_pre = pre


@pytest.mark.parametrize("legacy", [True, False], ids=["old-assertion", "fixed-hooks"])
def test_real_glm_layer_dynamo_capture(legacy):
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    from art.megatron.compile_workarounds import _isolate_moe_compile_cache
    from tests.integration.megatron.glm5next_qualification import capture_layer_outputs
    from tests.unit.test_glm5next_model import cpu_layer, tiny_config

    assert not torch.cuda.is_initialized()
    torch._dynamo.reset()
    torch.manual_seed(51)
    config = tiny_config()
    reference = Glm5NextTextModel(config).layers[0]
    layer = cpu_layer(config, reference, 0)
    x = torch.randn(9, 1, 4, 16) * 0.02
    with torch.no_grad():
        expected = layer(x)[0]

    graphs = []

    def eager_backend(graph, _inputs):
        # This is backend="eager" with graph inspection/counting. No Inductor
        # or CUDA kernels. Fail if capture/D2H accidentally enters a graph.
        graphs.append(graph)
        assert not any(
            node.kwargs.get("copy") is True
            and "cpu" in str(node.kwargs.get("device", ""))
            for node in graph.graph.nodes
        )
        return graph.forward

    _isolate_moe_compile_cache(MoELayer)
    with torch._dynamo.config.patch(
        force_parameter_static_shapes=False, suppress_errors=False
    ):
        compiled = torch.compile(layer, backend=eager_backend)
        values = {}
        handles = []
        if legacy:
            legacy_pre_observer(compiled)
        else:
            wrapped = SimpleNamespace(
                module=SimpleNamespace(decoder=SimpleNamespace(layers=[compiled]))
            )
            handles = capture_layer_outputs([wrapped], values)
        try:
            with torch.no_grad():
                if legacy:
                    with pytest.raises(
                        AssertionError,
                        match="UnspecializedNNModuleVariable.*already tracked for mutation",
                    ):
                        compiled(x)
                else:
                    for _ in range(2):
                        actual = compiled(x)[0]
                        torch.testing.assert_close(actual, expected)
                    assert len(graphs) > 0
                    for site in ("attn_hc", "ffn_hc"):
                        for field in ("collapsed", "post", "comb", "streams"):
                            assert values[f"layers.0.{site}.{field}"].shape[1] == 9
                    assert values["layers.0.streams"].shape == (1, 9, 4, 16)
        finally:
            for handle in handles:
                handle.remove()
            torch._dynamo.reset()
    assert not torch.cuda.is_initialized()
