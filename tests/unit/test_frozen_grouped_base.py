"""CPU storage/caller proofs. GEMM is substituted; these are not CUDA numerics."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from art.megatron.kernels import frozen_grouped_linear as base


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_gemms = 2
        self.config = SimpleNamespace(fp8=None, fp4=None, delay_wgrad_compute=False)
        self.tp_size = 1
        self.sequence_parallel = False
        self.te_return_bias = False
        self.te_quant_params = None
        self.is_first_microbatch = True
        for i in range(2):
            weight = torch.nn.Parameter(
                torch.full((8, 8), i + 1.0, dtype=torch.bfloat16), requires_grad=False
            )
            weight.partition_dim = i
            self.register_parameter(f"weight{i}", weight)

    def forward(self, x, splits):
        pieces = x.split(splits)
        return torch.cat(
            [piece @ getattr(self, f"weight{i}").T for i, piece in enumerate(pieces)]
        ), None


class Wrapper(base.FrozenGroupedBase):
    def __init__(self):
        super().__init__()
        self.linear = Linear()

    def _grouped_linear(self):
        return self.linear

    def forward(self, x, splits):
        return self._base_forward(x, splits)[0]


@pytest.fixture
def cpu_packing(monkeypatch):
    # Only bypass unsupported CPU device selection, not storage lifecycle logic.
    monkeypatch.setattr(
        base,
        "_supported",
        lambda linear: not any(p.requires_grad for p in linear.parameters()),
    )
    calls = []

    def mm(x, weights, *, offs):
        calls.append(offs)
        counts = torch.diff(offs, prepend=offs.new_zeros(1)).tolist()
        return torch.cat(
            [piece @ weights[i] for i, piece in enumerate(x.split(counts))]
        )

    monkeypatch.setattr(torch, "_grouped_mm", mm)
    return calls


def test_pack_once_preserves_schema_metadata_and_sole_backing(cpu_packing, monkeypatch):
    module = Wrapper()
    names = tuple(module.state_dict())
    parameters = tuple(module.parameters())
    old_storages = [weakref.ref(p.untyped_storage()) for p in parameters]
    calls = []
    pack = base._pack_weights
    monkeypatch.setattr(
        base, "_pack_weights", lambda linear: (calls.append(1), pack(linear))[1]
    )
    module._prepare_grouped_base(True)
    module._prepare_grouped_base(True)
    x = torch.ones((2, 8), dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3):
        module(x, [1, 1]).sum().backward()
    assert calls == [1]
    assert tuple(module.state_dict()) == names
    assert all(a is b for a, b in zip(module.parameters(), parameters))
    assert [p.partition_dim for p in parameters] == [0, 1]
    assert not tuple(module.buffers())
    assert len({p.untyped_storage().data_ptr() for p in parameters}) == 1
    assert parameters[0].untyped_storage().nbytes() == 2 * 8 * 8 * 2
    assert module.linear.is_first_microbatch is False
    gc.collect()
    assert all(ref() is None for ref in old_storages)


def test_inplace_copy_alias_and_overlapping_offset_lifetimes(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    x = torch.ones((3, 8), dtype=torch.bfloat16, requires_grad=True)
    first = module(x, [0, 3])
    second = module(x, [2, 1])
    assert cpu_packing[0].data_ptr() != cpu_packing[1].data_ptr()
    assert cpu_packing[0].tolist() == [0, 3]
    second.sum().backward(retain_graph=True)
    first.sum().backward()
    assert x.grad.tolist() == [[24.0] * 8, [24.0] * 8, [32.0] * 8]
    with torch.no_grad():
        module.linear.weight1.fill_(7)
    assert module(torch.ones((1, 8), dtype=torch.bfloat16), [0, 1]).tolist() == [
        [56.0] * 8
    ]
    assert module._grouped_shape is not None


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("child", [False, True])
def test_load_retires_before_parameters_change(cpu_packing, assign, child):
    module = Wrapper()
    module._prepare_grouped_base(True)
    target = module.linear if child else module
    new = {
        name: torch.full_like(value, 9) for name, value in target.state_dict().items()
    }
    target.load_state_dict(new, assign=assign)
    assert module._grouped_shape is None
    assert (
        module(torch.ones((2, 8), dtype=torch.bfloat16), [1, 1]).tolist()
        == [[72.0] * 8] * 2
    )


def test_move_retires_and_explicit_new_lifetime_repacks(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    old = weakref.ref(module.linear.weight0.untyped_storage())
    module._apply(lambda tensor: tensor.clone())
    assert module._grouped_shape is None
    gc.collect()
    assert old() is None
    assert (
        module.linear.weight0.untyped_storage().data_ptr()
        != module.linear.weight1.untyped_storage().data_ptr()
    )
    module._prepare_grouped_base(True)
    assert module._grouped_shape == (2, 8, 8)


def test_trainability_and_owner_fallback_retire(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    module.requires_grad_(True)
    module._prepare_grouped_base(True)
    assert module._grouped_shape is None
    module.requires_grad_(False)
    module._prepare_grouped_base(True)
    module._prepare_grouped_base(False)
    assert module._grouped_shape is None


def test_actual_cpu_selection_is_not_enabled():
    assert not base._supported(Linear())


def test_normal_outer_compile_uses_current_weights_after_move_and_assign(
    monkeypatch, cpu_packing
):
    # Fixed two one-token experts permits a traceable CPU GEMM stand-in. Normal
    # torch.compile itself is real; no CUDA graph/private cache claim is made.
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
    old = weakref.ref(module.linear.weight0.untyped_storage())
    module._apply(lambda tensor: tensor.clone())
    with torch.no_grad():
        module.linear.weight1.fill_(3)
    assert compiled(x, [1, 1]).tolist() == [[8.0] * 8, [24.0] * 8]
    module._prepare_grouped_base(True)
    assert compiled(x, [1, 1]).tolist() == [[8.0] * 8, [24.0] * 8]
    new = {
        name: torch.full_like(value, 4) for name, value in module.state_dict().items()
    }
    module.load_state_dict(new, assign=True)
    assert compiled(x, [1, 1]).tolist() == [[32.0] * 8] * 2
    gc.collect()
    assert old() is None


@pytest.mark.parametrize(
    "offload,streaming", [(False, False), (False, True), (True, False), (True, True)]
)
def test_actual_manager_install_selects_before_streaming(
    cpu_packing, offload, streaming
):
    # Execute the exact maintained install method, substituting only streaming's
    # GPU-owning constructor. Heavy Megatron imports are unavailable locally.
    import ast
    from pathlib import Path

    path = Path(base.__file__).parents[1] / "training/weight_offload.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    install = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "install"
    )
    module = Wrapper()
    observed = []
    namespace = {
        "install_streaming_weight_offload": lambda **kwargs: observed.append(
            module._grouped_shape
        )
    }
    exec(
        compile(ast.Module(body=[install], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    value = SimpleNamespace(
        model=[torch.nn.Sequential(module)],
        offload_between_jobs=offload,
        streaming_config=SimpleNamespace(enabled=streaming),
        rank=0,
        compile_enabled=True,
    )
    namespace["install"](value)
    assert observed == [None]  # The manager never activates the resident-only route.
    base.prepare_grouped_bases(value.model)
    assert module._grouped_shape == (
        (2, 8, 8) if not offload and not streaming else None
    )


def test_initial_trainable_base_keeps_te(cpu_packing):
    module = Wrapper()
    module.linear.weight1.requires_grad_(True)
    module._prepare_grouped_base(True)
    assert module._grouped_shape is None
    module(torch.ones((2, 8), dtype=torch.bfloat16), [1, 1]).sum().backward()
    assert module.linear.weight1.grad is not None
    assert not cpu_packing


def test_ancestor_requires_grad_retains_te_gradients(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    torch.nn.Sequential(module).requires_grad_(True)
    module(torch.ones((2, 8), dtype=torch.bfloat16), [1, 1]).sum().backward()
    assert module.linear.weight0.grad is not None
    assert module.linear.weight1.grad is not None
    assert not cpu_packing


def test_actual_trainer_rank_constructor_activates_resident_base(cpu_packing):
    from art.trainer_rank import TrainerRank

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.fc = Wrapper()
            self.config = SimpleNamespace(
                hidden_size=8, num_layers=1, padded_vocab_size=32
            )
            self.decoder = object()

        def _preprocess(self, *args, **kwargs):
            return None

    model = Model()
    runtime = SimpleNamespace(
        model=[model],
        optimizer=None,
        provider=SimpleNamespace(
            hidden_size=8,
            num_layers=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )
    assert model.fc._grouped_shape is None
    rank = TrainerRank(runtime)
    assert rank.runtime is runtime
    assert model.fc._grouped_shape == (2, 8, 8)
    assert model.fc._grouped_preparations == 1


def test_nonfirst_base_trainability_keeps_te_gradient(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    module.linear.weight1.requires_grad_(True)
    x = torch.ones((2, 8), dtype=torch.bfloat16, requires_grad=True)
    module(x, [1, 1]).sum().backward()
    assert torch.equal(
        module.linear.weight1.grad, torch.ones_like(module.linear.weight1)
    )
    assert not cpu_packing


def test_assign_releases_the_current_packed_storage_and_cached_parameters(cpu_packing):
    module = Wrapper()
    module._prepare_grouped_base(True)
    storage = weakref.ref(module.linear.weight0.untyped_storage())
    old_parameter = weakref.ref(module.linear.weight0)
    new = {name: value.clone() for name, value in module.state_dict().items()}
    module.load_state_dict(new, assign=True)
    assert module._grouped_weights == ()
    gc.collect()
    assert storage() is None and old_parameter() is None


def test_different_routing_values_do_not_recompile_normal_outer_graph(
    monkeypatch, cpu_packing
):
    # Consumes offsets but substitutes only CUDA's grouped GEMM. This is a
    # frontend value-specialization witness, not a numerical/Inductor proof.
    monkeypatch.setattr(
        torch,
        "_grouped_mm",
        lambda x, w, *, offs: x @ w[0] + offs.sum().to(x.dtype) * 0,
    )
    module = Wrapper()
    module._prepare_grouped_base(True)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(module, backend=backend)
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    compiled(x, [1, 1])
    count = len(graphs)
    for splits in ([0, 2], [2, 0], [1, 1]):
        compiled(x, splits)
        assert len(graphs) == count


def test_delayed_wgrad_keeps_te_before_device_query(monkeypatch):
    linear = Linear()
    linear.config.delay_wgrad_compute = True
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_: pytest.fail("device query for excluded mode"),
    )
    assert not base._supported(linear)
