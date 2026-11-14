import importlib
from importlib.metadata import PackageNotFoundError
import logging
import sys
from types import SimpleNamespace

import pytest

from art.megatron.gdn import fla_autotune as patch

EXPECTED = tuple(
    (warps, stages, bv) for warps in (2, 4) for stages in (2, 3, 4) for bv in (32, 64)
)
SAFE = (
    (2, 2, 32),
    (2, 2, 64),
    (2, 3, 32),
    (2, 3, 64),
    (2, 4, 32),
    (2, 4, 64),
    (4, 3, 64),
    (4, 4, 32),
    (4, 4, 64),
)


def fake_kernel():
    configs = [
        SimpleNamespace(
            num_warps=warps,
            num_stages=stages,
            kwargs={"BV": bv},
            num_ctas=1,
            maxnreg=None,
            pre_hook=None,
            ir_override=None,
        )
        for warps, stages, bv in EXPECTED
    ]
    tuner = SimpleNamespace(
        fn=SimpleNamespace(
            __name__="chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            __module__="fla.ops.common.chunk_delta_h",
        ),
        configs=configs,
        cache={("previous-shape",): configs[6]},
        best_config=configs[6],
        configs_timings={"old timings": 1.0},
    )
    return SimpleNamespace(fn=tuner)


def tuples(kernel):
    return tuple(
        (config.num_warps, config.num_stages, config.kwargs["BV"])
        for config in kernel.fn.configs
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_blackwell_filters_exactly_three_configs_and_invalidates_cache(
    capability, caplog
):
    kernel = fake_kernel()
    originals = kernel.fn.configs.copy()
    with caplog.at_level(logging.DEBUG, logger=patch.__name__):
        assert patch.filter_fwd_h_configs(
            kernel, fla_version="0.5.0", capability=capability
        )
    assert tuples(kernel) == SAFE
    assert all(config in originals for config in kernel.fn.configs)
    assert kernel.fn.cache == {}
    assert not hasattr(kernel.fn, "best_config")
    assert not hasattr(kernel.fn, "configs_timings")
    assert str(SAFE) in caplog.text


@pytest.mark.parametrize("capability", [(9, 0), (8, 0), (11, 0), (12, 0)])
def test_other_architectures_keep_configs_and_cache(capability, caplog):
    kernel = fake_kernel()
    original_list = kernel.fn.configs
    original_cache = kernel.fn.cache.copy()
    assert not patch.filter_fwd_h_configs(
        kernel, fla_version="0.5.0", capability=capability
    )
    assert kernel.fn.configs is original_list
    assert tuples(kernel) == EXPECTED
    assert kernel.fn.cache == original_cache
    assert not caplog.records


def test_repeated_imports_preserve_safe_tuning_cache():
    kernel = fake_kernel()
    assert patch.filter_fwd_h_configs(kernel, fla_version="0.5.0", capability=(10, 0))
    config_list = kernel.fn.configs
    kernel.fn.cache["safe-choice"] = config_list[0]
    kernel.fn.best_config = config_list[0]
    for _ in range(3):
        reimported = importlib.reload(patch)
        assert reimported.filter_fwd_h_configs(
            kernel, fla_version="0.5.0", capability=(10, 0)
        )
    assert kernel.fn.configs is config_list
    assert tuples(kernel) == SAFE
    assert kernel.fn.cache == {"safe-choice": config_list[0]}


@pytest.mark.parametrize(
    "mismatch",
    [
        "version",
        "kernel",
        "module",
        "missing",
        "extra",
        "duplicate",
        "kwargs",
        "ctas",
        "registers",
        "hook",
        "ir",
        "cache",
        "wrapper",
        "unmarked-safe",
    ],
)
def test_signature_mismatch_warns_without_mutating(mismatch, caplog):
    kernel = fake_kernel()
    fla_version = "0.5.0"
    if mismatch == "version":
        fla_version = "0.5.1"
    elif mismatch == "kernel":
        kernel.fn.fn.__name__ = "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"
    elif mismatch == "module":
        kernel.fn.fn.__module__ = "some_other_module"
    elif mismatch == "missing":
        kernel.fn.configs.pop()
    elif mismatch == "extra":
        kernel.fn.configs.append(kernel.fn.configs[0])
    elif mismatch == "duplicate":
        kernel.fn.configs[-1] = kernel.fn.configs[0]
    elif mismatch == "kwargs":
        kernel.fn.configs[0].kwargs["BK"] = 64
    elif mismatch == "ctas":
        kernel.fn.configs[0].num_ctas = 2
    elif mismatch == "registers":
        kernel.fn.configs[0].maxnreg = 64
    elif mismatch == "hook":
        kernel.fn.configs[0].pre_hook = lambda _: None
    elif mismatch == "ir":
        kernel.fn.configs[0].ir_override = "override.ptx"
    elif mismatch == "cache":
        kernel.fn.cache = None
    elif mismatch == "wrapper":
        kernel.fn.fn = None
    elif mismatch == "unmarked-safe":
        kernel.fn.configs = [
            kernel.fn.configs[i] for i in (0, 1, 2, 3, 4, 5, 9, 10, 11)
        ]
    original_list = kernel.fn.configs
    original_cache = kernel.fn.cache
    assert not patch.filter_fwd_h_configs(
        kernel, fla_version=fla_version, capability=(10, 0)
    )
    assert kernel.fn.configs is original_list
    assert kernel.fn.cache is original_cache
    assert "Skipping ART Blackwell FLA fwd_h config filter" in caplog.text
    assert caplog.records[-1].levelno == logging.WARNING


def mock_runtime(monkeypatch, *, capability=(10, 0), available=True):
    kernel = fake_kernel()
    monkeypatch.setattr(patch.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(patch.torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(patch, "version", lambda name: "0.5.0")
    monkeypatch.setattr(
        patch.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64=kernel
        ),
    )
    return kernel


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (9, 0)])
def test_runtime_uses_active_device(monkeypatch, capability):
    kernel = mock_runtime(monkeypatch, capability=capability)
    patch.ensure_blackwell_fwd_h_configs()
    assert tuples(kernel) == (SAFE if capability[0] == 10 else EXPECTED)


def test_cpu_does_not_import_fla_or_query_cuda(monkeypatch):
    monkeypatch.setattr(patch.torch.cuda, "is_available", lambda: False)

    def unexpected(*args):
        pytest.fail("CPU must not query CUDA capability or import FLA")

    monkeypatch.setattr(patch.torch.cuda, "get_device_capability", unexpected)
    monkeypatch.setattr(patch, "version", unexpected)
    patch.ensure_blackwell_fwd_h_configs()


@pytest.mark.parametrize("distribution", ["flash-linear-attention", "fla-core"])
@pytest.mark.parametrize("installed_version", ["0.5.1", None])
def test_runtime_distribution_guard(
    monkeypatch, caplog, distribution, installed_version
):
    kernel = mock_runtime(monkeypatch)

    def get_version(name):
        if name == distribution:
            if installed_version is None:
                raise PackageNotFoundError(name)
            return installed_version
        return "0.5.0"

    monkeypatch.setattr(patch, "version", get_version)
    patch.ensure_blackwell_fwd_h_configs()
    assert tuples(kernel) == EXPECTED
    assert "Skipping ART Blackwell FLA fwd_h config filter" in caplog.text


def test_public_gdn_entry_filters_before_call(monkeypatch):
    from art.megatron.gdn import operator

    kernel = mock_runtime(monkeypatch)

    def recurrent(*args, **kwargs):
        assert tuples(kernel) == SAFE
        return "output", "state"

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.gated_delta_rule",
        SimpleNamespace(chunk_gated_delta_rule=recurrent),
    )
    assert operator._chunk_gated_delta_rule() == ("output", "state")


def test_model_support_hook_install_filters(monkeypatch):
    from art.megatron.gdn import operator

    kernel = mock_runtime(monkeypatch)
    monkeypatch.setattr(operator, "_optional_gated_delta_net_type", lambda: None)
    operator.install_prefix_tree_gdn_hooks([])
    assert tuples(kernel) == SAFE


def test_native_cp_entry_filters_before_fla_forward(monkeypatch):
    from art.megatron.gdn import fla_cp

    kernel = mock_runtime(monkeypatch)

    class ReachedFla(Exception):
        pass

    def prepare(*args, **kwargs):
        assert tuples(kernel) == SAFE
        raise ReachedFla

    modules = {
        "fla.ops.common.chunk_delta_h": {"chunk_gated_delta_rule_fwd_h": None},
        "fla.ops.common.chunk_o": {"chunk_fwd_o": None},
        "fla.ops.common.chunk_scaled_dot_kkt": {"chunk_scaled_dot_kkt_fwd": None},
        "fla.ops.gated_delta_rule.wy_fast": {"recompute_w_u_fwd": None},
        "fla.ops.utils": {
            "chunk_local_cumsum": None,
            "prepare_chunk_indices": prepare,
            "solve_tril": None,
        },
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    tensor = patch.torch.empty(0)
    with pytest.raises(ReachedFla):
        fla_cp._NativeCpChunkGatedDeltaRule.forward(
            None,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            None,
            None,
            False,
            1.0,
        )


def test_real_triton_tuner_cannot_reuse_cached_racy_config(monkeypatch):
    triton = pytest.importorskip("triton")

    @triton.jit
    def noop(T):
        pass

    noop.__name__ = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
    noop.__module__ = "fla.ops.common.chunk_delta_h"
    configs = [
        triton.Config({"BV": bv}, num_warps=warps, num_stages=stages)
        for warps, stages, bv in EXPECTED
    ]
    kernel = triton.heuristics({})(triton.autotune(configs=configs, key=["T"])(noop))
    tuner = kernel.fn
    tuner.cache[(1,)] = configs[6]
    monkeypatch.setattr(tuner, "cache_results", False)
    # Exercise the real Autotuner.run selection, with CPU substitutes for
    # benchmarking and launching only. No driver or GPU is initialized.
    monkeypatch.setattr(tuner, "_bench", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(noop, "run", lambda *args, **kwargs: kwargs)
    assert patch.filter_fwd_h_configs(kernel, fla_version="0.5.0", capability=(10, 0))
    result = tuner.run(1)
    assert (result["num_warps"], result["num_stages"], result["BV"]) in SAFE
    assert tuner.cache[(1,)] in tuner.configs


def test_real_triton_disk_cache_namespace_changes(monkeypatch):
    triton = pytest.importorskip("triton")
    from triton.compiler import compiler
    from triton.runtime import autotuner

    @triton.jit
    def noop(T):
        pass

    noop.__name__ = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
    noop.__module__ = "fla.ops.common.chunk_delta_h"
    kernel = triton.heuristics({})(
        triton.autotune(
            configs=[
                triton.Config({"BV": bv}, num_warps=warps, num_stages=stages)
                for warps, stages, bv in EXPECTED
            ],
            key=["T"],
        )(noop)
    )
    keys = []

    def get_cache(key):
        keys.append(key)
        return SimpleNamespace(
            get_file=lambda _: None, put=lambda *args, **kwargs: None
        )

    monkeypatch.setattr(autotuner, "get_cache_manager", get_cache)
    monkeypatch.setattr(autotuner, "triton_key", lambda: "cpu-test-triton")
    monkeypatch.setattr(autotuner, "get_cache_invalidating_env_vars", lambda: {})
    monkeypatch.setattr(
        autotuner,
        "driver",
        SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: None)),
    )
    monkeypatch.setattr(
        compiler, "make_backend", lambda _: SimpleNamespace(hash=lambda: "cpu")
    )
    tuner = kernel.fn

    def benchmark():
        tuner.configs_timings = {tuner.configs[0]: 1.0}

    tuner.check_disk_cache((1,), tuner.configs, benchmark)
    assert patch.filter_fwd_h_configs(kernel, fla_version="0.5.0", capability=(10, 0))
    tuner.check_disk_cache((1,), tuner.configs, benchmark)
    assert len(keys) == 2 and keys[0] != keys[1]
