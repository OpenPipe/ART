"""Reserved-GPU canary for logical copies beside an existing replay backward."""

import gc
import os

import pytest
from test_trainer_rank_custom_tensors import _trainer
from test_trainer_rank_output_memory import _output
import torch
from trainer_rank_test_support import report_measurement

from art.trainer_rank import _impl
from art.trainer_rank._commands import _Executor, _view
from art.trainer_rank._tensors import detach_tree

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="reserved GPU required"
)


def test_logical_outputs_leave_existing_replay_backward_admissible(monkeypatch):
    torch.set_num_threads(2)
    trainer, api = _trainer("student")
    trainer.device = torch.device("cuda")
    weight = api.parameter("weight", lambda: torch.ones(()), checkpoint="student")
    cache = trainer._forward_graph_cache()
    inputs = torch.ones(4 * 1024**2, device="cuda")
    workspace = 64 * 1024**2
    handle, outputs = cache.run(
        lambda values: ((values.sin() * weight).sum(),),
        inputs,
        retention="replay",
        output_device="cpu",
        execution_peak_bytes=workspace,
        checkpoint_versions=(trainer._capture_checkpoint_version("student"),),
        cuda_devices=[torch.cuda.current_device()],
    )
    loss = trainer._forward_cotangent_collector().attach(detach_tree(handle, outputs))[
        0
    ]
    view = _view(_Executor(trainer, "zero"))
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    capacity = 80 * 1024**2
    limit = baseline + capacity
    monkeypatch.setattr(
        trainer,
        "_available_memory_bytes",
        lambda: limit - torch.cuda.memory_allocated(),
    )
    large = 32 * 1024**2
    with pytest.raises(MemoryError):
        view._place_outputs([_output(large, policy="model")])
    assert cache.handles() == (handle,)
    auto = view._attach(view._place_outputs([_output(large)])[0])
    assert auto.hidden_states.device.type == "cpu"
    model = view._attach(view._place_outputs([_output(8 * 1024**2, policy="model")])[0])
    assert model.hidden_states.device.type == "cuda"
    torch.cuda.reset_peak_memory_stats()
    trainer.backward(loss)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    assert peak <= capacity
    assert weight.grad is not None
    torch.testing.assert_close(
        weight.grad.cpu(),
        inputs.numel() * torch.sin(torch.tensor(1.0)),
        rtol=1e-5,
        atol=0,
    )
    assert not cache.handles()
    report_measurement(
        "OUTPUT_BACKWARD_RESERVE",
        available_bytes=capacity,
        restore_bytes=workspace,
        rejected_copy_bytes=large,
        admitted_copy_bytes=8 * 1024**2,
        peak_bytes=peak,
    )


def _graph_loss(trainer, weight, inputs):
    handle, outputs = trainer._forward_graph_cache().run(
        lambda values: ((values.sin() * weight).sum(),),
        inputs,
        retention="gpu",
        output_device="cpu",
        execution_peak_bytes=256 * 1024**2,
        checkpoint_versions=(trainer._capture_checkpoint_version("student"),),
        cuda_devices=[torch.cuda.current_device()],
    )
    return trainer._forward_cotangent_collector().attach(detach_tree(handle, outputs))[
        0
    ]


def test_forward_cache_does_not_refuse_model_outputs():
    torch.set_num_threads(2)
    trainer, api = _trainer("student")
    trainer.device = torch.device("cuda")
    weight = api.parameter("weight", lambda: torch.ones(()), checkpoint="student")
    inputs = torch.ones(4 * 1024**2, device="cuda")
    view = _view(_Executor(trainer, "rank"))
    output = _output(8 * 1024**2, policy="model")
    # Unpressured reference for the same graph.
    trainer.backward(_graph_loss(trainer, weight, inputs))
    assert weight.grad is not None
    reference = weight.grad.clone()
    weight.grad = None
    loss = _graph_loss(trainer, weight, inputs)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Physical headroom for the pending backward plus 512 MiB, then 1 GiB of
    # transients freed into the native cache: the sample goes negative.
    free, total = torch.cuda.mem_get_info()
    pending = sum(trainer._pending_backward_memory())
    reserve = int(total * _impl._MEMORY_RESERVE_FRACTION)
    ballast = torch.empty(
        free - reserve - pending - 512 * 1024**2, dtype=torch.uint8, device="cuda"
    )
    transient = torch.empty(1024**3, dtype=torch.uint8, device="cuda")
    del transient
    refused = trainer._available_memory_bytes() - pending
    assert refused < 0
    placed = view._attach(view._place_outputs([output])[0])
    assert placed.hidden_states.device.type == "cuda"
    available = trainer._available_memory_bytes() - pending
    assert available >= 8 * 1024**2
    trainer.backward(loss)
    torch.cuda.synchronize()
    assert weight.grad is not None and torch.equal(weight.grad, reference)
    del ballast
    report_measurement(
        "OUTPUT_FORWARD_CACHE_RELEASE",
        pending_bytes=pending,
        refused_available_bytes=refused,
        released_available_bytes=available,
        copy_bytes=8 * 1024**2,
    )
