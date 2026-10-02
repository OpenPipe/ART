"""Standard public child transformations must retire the candidate before GEMM."""

from test_frozen_grouped_base import Wrapper, cpu_packing  # noqa: F401
import torch


def test_child_dtype_roundtrip_retains_te_semantics(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    module.linear.float().bfloat16()
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    expected = module.linear(x, [1, 1])[0]
    actual = module(x, [1, 1])
    torch.testing.assert_close(actual, expected)
    assert module._grouped_shape is None
    assert module._grouped_weights == ()
    assert not cpu_packing


def test_child_shared_storage_view_change_is_not_silent(cpu_packing):
    module = Wrapper()
    with torch.no_grad():
        for weight in module.parameters():
            weight.copy_(torch.arange(64, dtype=torch.bfloat16).view(8, 8))
    module._prepare_grouped_base(True)
    module.linear._apply(lambda value: value.T)
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    expected = module.linear(x, [1, 1])[0]
    actual = module(x, [1, 1])
    torch.testing.assert_close(actual, expected)
    assert module._grouped_shape is None
    assert not cpu_packing


def test_child_move_can_prepare_new_lifetime_explicitly(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    module.linear._apply(lambda value: value.clone())
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    torch.testing.assert_close(module(x, [1, 1]), module.linear(x, [1, 1])[0])
    assert module._grouped_shape is None
    module._prepare_grouped_base(True)
    torch.testing.assert_close(module(x, [1, 1]), module.linear(x, [1, 1])[0])
    assert module._grouped_preparations == 2
    assert len(cpu_packing) == 1


def test_compiled_child_move_retires_before_cached_view(monkeypatch, cpu_packing):
    monkeypatch.setattr(
        torch,
        "_grouped_mm",
        lambda x, weight, *, offs: torch.bmm(x.unsqueeze(1), weight).squeeze(1),
    )
    module = Wrapper()
    module._prepare_grouped_base(True)
    compiled = torch.compile(module, backend="eager")
    x = torch.ones((2, 8), dtype=torch.bfloat16, requires_grad=True)
    assert compiled(x, [1, 1]).tolist() == [[8.0] * 8, [16.0] * 8]
    module.linear.float().bfloat16()
    with torch.no_grad():
        module.linear.weight1.fill_(3)
    result = compiled(x, [1, 1])
    assert result.tolist() == [[8.0] * 8, [24.0] * 8]
    result.sum().backward()
    assert x.grad.tolist() == [[8.0] * 8, [24.0] * 8]
    assert module._grouped_shape is None
