import json
from pathlib import Path
import subprocess

from art.vllm_runtime import _vllm_runtime_subprocess_env

ROOT = Path(__file__).resolve().parents[4]


def _runtime_python(source: str, artifact_dir: Path, name: str) -> str:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            source,
        ],
        cwd=ROOT,
        env=_vllm_runtime_subprocess_env(),
        capture_output=True,
        text=True,
    )
    (artifact_dir / f"{name}_stdout.txt").write_text(result.stdout)
    (artifact_dir / f"{name}_stderr.txt").write_text(result.stderr)
    result.check_returncode()
    return result.stdout.strip()


def test_runtime_project_imports_in_its_own_project_env(artifact_dir: Path) -> None:
    payload = json.loads(
        _runtime_python(
            "import importlib.util, json; "
            "import art_vllm_runtime; "
            "print(json.dumps({"
            "'runtime_ok': True, "
            "'has_vllm': importlib.util.find_spec('vllm') is not None"
            "}))",
            artifact_dir,
            "runtime_import",
        )
    )
    assert payload == {"runtime_ok": True, "has_vllm": True}


def test_runtime_server_source_contains_only_required_custom_routes() -> None:
    source = (
        ROOT / "vllm_runtime" / "src" / "art_vllm_runtime" / "dedicated_server.py"
    ).read_text()
    for route in ("/sleep", "/wake_up", "/is_sleeping", "/art/set_served_model_name"):
        assert route in source


def test_runtime_patch_always_returns_token_ids(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.patches import apply_vllm_runtime_patches; "
        "apply_vllm_runtime_patches(); "
        "from vllm.entrypoints.openai.chat_completion import protocol; "
        "request = protocol.ChatCompletionRequest("
        "model='m', messages=[{'role': 'user', 'content': 'x'}]"
        "); "
        "print(json.dumps({"
        "'logprobs': request.logprobs, "
        "'top_logprobs': request.top_logprobs, "
        "'return_token_ids': request.return_token_ids"
        "}))",
        artifact_dir,
        "route_token_ids",
    )
    assert json.loads(payload) == {
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
    }


def test_runtime_lora_updates_linearize_request_admission(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import asyncio
import json
from types import SimpleNamespace
from art_vllm_runtime.policy_spans import (
    LoraUpdateCoordinator,
    PolicyLoRARequest,
    _apply_lora_alias_policy_cache_salt,
    publish_lora_slot_policy,
    register_lora_alias,
)

async def main():
    slot = "model:active"
    old = PolicyLoRARequest(
        lora_name=slot, lora_int_id=1, lora_path="old",
        policy_version=4, update_seq=1,
    )
    new = PolicyLoRARequest(
        lora_name=slot, lora_int_id=1, lora_path="new",
        policy_version=5, update_seq=2,
    )
    models = SimpleNamespace(lora_requests={slot: new})
    register_lora_alias(models, public_model_name="model@4", lora_slot=slot)
    publish_lora_slot_policy(
        models, lora_slot=slot, policy_version=5, update_seq=2
    )
    request = SimpleNamespace(model="model@4", cache_salt=None)
    _apply_lora_alias_policy_cache_salt(models, request, new)

    coordinator = LoraUpdateCoordinator()
    assert await coordinator.begin_update(slot) == 1
    await coordinator.commit_update(slot, old)
    assert await coordinator.begin_update(slot) == 2

    async def admit():
        async with coordinator.admission(slot) as state:
            return state

    admission = asyncio.create_task(admit())
    await asyncio.sleep(0)
    blocked = not admission.done()
    await coordinator.commit_update(slot, new)
    admitted_lora = await admission
    print(json.dumps({
        "blocked": blocked,
        "cache_salt": request.cache_salt,
        "policy_version": admitted_lora.policy_version,
        "lora_path": admitted_lora.lora_path,
    }, sort_keys=True))

asyncio.run(main())
""",
        artifact_dir,
        "lora_update_admission",
    )
    result = json.loads(payload)
    cache_salt = result.pop("cache_salt")
    assert result == {
        "blocked": True,
        "lora_path": "new",
        "policy_version": 5,
    }
    assert cache_salt.startswith("art_policy_cache_salt=v1:")
    assert len(cache_salt) == len("art_policy_cache_salt=v1:") + 64


def test_runtime_parallel_admission_is_atomic_and_cancellation_safe(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import asyncio
from collections import defaultdict
import json
from types import SimpleNamespace

from art_vllm_runtime.policy_spans import (
    LoraUpdateCoordinator, PolicyLoRARequest, _patch_engine_request_admission,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor

class Output:
    abort_requests = OutputProcessor.abort_requests

    def __init__(self):
        self.request_states = {}
        self.parent_requests = {}
        self.external_req_ids = defaultdict(list)
        self.lora_states = SimpleNamespace(request_finished=lambda *_args: None)

    def add_request(self, request, _prompt, parent, _index, _queue):
        self.request_states[request.request_id] = SimpleNamespace(
            external_req_id=request.external_req_id,
            lora_name=request.lora_request.lora_name,
            parent_req=parent,
            queue=None,
        )
        self.external_req_ids[request.external_req_id].append(request.request_id)
        if parent is not None:
            self.parent_requests[parent.request_id] = parent

class Core:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.calls = []
        self.aborted = []
        self.first = asyncio.Event()
        self.release_first = asyncio.Event()
        self.second = asyncio.Event()
        self.release_second = asyncio.Event()

    async def add_request_async(self, request):
        self.calls.append((request.request_id, request.lora_request.policy_version))
        if len(self.calls) == 1:
            self.first.set()
            await self.release_first.wait()
        else:
            self.second.set()
            await self.release_second.wait()

    async def abort_requests_async(self, request_ids):
        self.aborted.extend(request_ids)

async def main():
    _patch_engine_request_admission()
    slot = "model:active"
    old = PolicyLoRARequest(
        lora_name=slot, lora_int_id=1, lora_path="old",
        policy_version=1, update_seq=1,
    )
    new = PolicyLoRARequest(
        lora_name=slot, lora_int_id=1, lora_path="new",
        policy_version=2, update_seq=2,
    )
    coordinator = LoraUpdateCoordinator()
    assert await coordinator.begin_update(slot) == 1
    await coordinator.commit_update(slot, old)
    engine = object.__new__(AsyncLLM)
    engine.engine_core = Core()
    engine.output_handler = None
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    engine.input_processor = SimpleNamespace(assign_request_id=lambda _request: None)
    engine._run_output_handler = lambda: None
    engine.output_processor = Output()
    engine.log_requests = False
    engine._art_lora_update_coordinator = coordinator
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()
    abort_finished = asyncio.Event()

    async def abort(request_id, internal=False):
        abort_started.set()
        await release_abort.wait()
        await AsyncLLM.abort(engine, request_id, internal=internal)
        abort_finished.set()

    engine.abort = abort
    params = SamplingParams(n=2, max_tokens=1)
    request = EngineCoreRequest(
        request_id="parent", external_req_id="external", prompt_token_ids=[1],
        mm_features=None, sampling_params=params, pooling_params=None,
        arrival_time=0.0, lora_request=old, cache_salt=None,
        data_parallel_rank=None,
    )
    admission = asyncio.create_task(
        engine.add_request("parent", request, params, prompt_text="x")
    )
    await engine.engine_core.first.wait()
    update = asyncio.create_task(coordinator.begin_update(slot))
    await asyncio.sleep(0)
    blocked_after_first = not update.done()
    engine.engine_core.release_first.set()
    await engine.engine_core.second.wait()
    blocked_after_second = not update.done()
    admission.cancel()
    await abort_started.wait()
    state = coordinator._states[slot]
    await state.condition.acquire()
    admission.cancel()
    await asyncio.sleep(0)
    blocked_during_abort = not admission.done() and not update.done()
    maps_during_abort = [
        len(engine.output_processor.request_states),
        len(engine.output_processor.external_req_ids),
        len(engine.output_processor.parent_requests),
    ]
    release_abort.set()
    await abort_finished.wait()
    admission.cancel()
    await asyncio.sleep(0)
    blocked_during_release = not admission.done() and not update.done()
    maps_after_abort = [
        len(engine.output_processor.request_states),
        len(engine.output_processor.external_req_ids),
        len(engine.output_processor.parent_requests),
    ]
    state.condition.release()
    try:
        await admission
    except asyncio.CancelledError:
        pass
    update_seq = await asyncio.wait_for(update, timeout=1)
    await coordinator.commit_update(slot, new)
    print(json.dumps({
        "calls": engine.engine_core.calls,
        "blocked": [blocked_after_first, blocked_after_second],
        "blocked_cleanup": [blocked_during_abort, blocked_during_release],
        "maps_during_abort": maps_during_abort,
        "maps_after_abort": maps_after_abort,
        "update_seq": update_seq,
        "aborted": sorted(engine.engine_core.aborted),
        "state_sizes": [
            len(engine.output_processor.request_states),
            len(engine.output_processor.external_req_ids),
            len(engine.output_processor.parent_requests),
        ],
    }, sort_keys=True))

asyncio.run(main())
""",
        artifact_dir,
        "parallel_lora_admission",
    )
    assert json.loads(payload.splitlines()[-1]) == {
        "aborted": ["0_parent", "1_parent"],
        "blocked": [True, True],
        "blocked_cleanup": [True, True],
        "calls": [["0_parent", 1], ["1_parent", 1]],
        "maps_after_abort": [0, 0, 0],
        "maps_during_abort": [2, 1, 1],
        "state_sizes": [0, 0, 0],
        "update_seq": 2,
    }


def test_runtime_cancelled_update_releases_admission(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import asyncio
import json
from art_vllm_runtime.policy_spans import LoraUpdateCoordinator, PolicyLoRARequest

async def main():
    coordinator = LoraUpdateCoordinator()
    slot = "model:active"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_admission():
        async with coordinator.admission(slot):
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_admission())
    await entered.wait()
    update = asyncio.create_task(coordinator.begin_update(slot))
    await asyncio.sleep(0)
    update.cancel()
    try:
        await update
    except asyncio.CancelledError:
        pass
    release.set()
    await holder
    async with asyncio.timeout(1):
        async with coordinator.admission(slot):
            admitted = True
    failed_seq = await coordinator.begin_update(slot)
    await coordinator.fail_update(slot, failed_seq)
    cancelled_retry = await coordinator.begin_update(slot)
    await coordinator.cancel_update(slot, cancelled_retry)

    async def admit_after_failure():
        async with coordinator.admission(slot):
            return True

    quarantined_admission = asyncio.create_task(admit_after_failure())
    await asyncio.sleep(0)
    quarantine_preserved = not quarantined_admission.done()
    recovery_seq = await coordinator.begin_update(slot)
    await coordinator.commit_update(slot, PolicyLoRARequest(
        lora_name=slot, lora_int_id=1, lora_path="recovered",
        policy_version=2, update_seq=recovery_seq,
    ))
    recovered = await quarantined_admission
    print(json.dumps({
        "admitted": admitted,
        "quarantine_preserved": quarantine_preserved,
        "recovered": recovered,
    }))

asyncio.run(main())
""",
        artifact_dir,
        "cancelled_lora_update",
    )
    assert json.loads(payload) == {
        "admitted": True,
        "quarantine_preserved": True,
        "recovered": True,
    }


def test_runtime_policy_history_rekeys_real_vllm_requests(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import hashlib
import json
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request
from art_vllm_runtime.policy_spans import (
    PolicyLoRARequest,
    _policy_history_from_cache_salt,
    _request_has_executed,
    _set_policy_cache_salt,
    _transition_scheduler_policy_history,
)

def hash_value(value):
    return hashlib.sha256(repr(value).encode()).digest()

init_none_hash(hash_value)
block_hasher = get_request_block_hasher(4, hash_value)
old = PolicyLoRARequest(
    lora_name="model:active", lora_int_id=1, lora_path="old",
    policy_version=4, update_seq=1,
)
new = PolicyLoRARequest(
    lora_name="model:active", lora_int_id=1, lora_path="new",
    policy_version=4, update_seq=2,
)

def make_request(request_id):
    request = Request(
        request_id, list(range(8)), SamplingParams(max_tokens=4), None,
        lora_request=old, block_hasher=block_hasher,
    )
    _set_policy_cache_salt(
        request, lora_slot=old.lora_name,
        policy_version=old.policy_version, update_seq=old.update_seq,
    )
    request.block_hashes.clear()
    request.update_block_hashes()
    return request

continued = make_request("continued")
waiting = make_request("waiting")
old_hashes = list(continued.block_hashes)
continued.num_computed_tokens = 4
scheduler = SimpleNamespace(requests={
    continued.request_id: continued,
    waiting.request_id: waiting,
})
transition = _transition_scheduler_policy_history(
    scheduler,
    lora_request=new,
    previous_policy=None,
    started_request_ids={continued.request_id},
)
continued_history = continued.cache_salt
fresh = make_request("fresh")
_set_policy_cache_salt(
    fresh, lora_slot=new.lora_name,
    policy_version=new.policy_version, update_seq=new.update_seq,
)
fresh.block_hashes.clear()
fresh.update_block_hashes()
continued.num_computed_tokens = 0
third = PolicyLoRARequest(
    lora_name="model:active", lora_int_id=1, lora_path="third",
    policy_version=4, update_seq=3,
)
old_history = _policy_history_from_cache_salt(make_request("old").cache_salt)
expected_third = make_request("expected-third")
_set_policy_cache_salt(
    expected_third, lora_slot=third.lora_name,
    policy_version=third.policy_version, update_seq=third.update_seq,
    previous_digest=old_history,
)
not_executed_after_reset = not _request_has_executed(continued)
_transition_scheduler_policy_history(
    SimpleNamespace(requests={continued.request_id: continued}),
    lora_request=third,
    previous_policy=None,
    started_request_ids=set(),
)
print(json.dumps({
    "transition": transition,
    "continued_differs": continued_history != waiting.cache_salt,
    "waiting_matches_fresh": waiting.cache_salt == fresh.cache_salt,
    "same_version_reload_differs": (
        _policy_history_from_cache_salt(waiting.cache_salt)
        != _policy_history_from_cache_salt(make_request("old").cache_salt)
    ),
    "block_hashes_changed": old_hashes != continued.block_hashes,
    "not_executed_after_reset": not_executed_after_reset,
    "skipped_policy_replaced": continued.cache_salt == expected_third.cache_salt,
}))
""",
        artifact_dir,
        "policy_history_real_request",
    )
    assert json.loads(payload) == {
        "block_hashes_changed": True,
        "continued_differs": True,
        "not_executed_after_reset": True,
        "same_version_reload_differs": True,
        "skipped_policy_replaced": True,
        "transition": {"continued_requests": 1, "updated_requests": 2},
        "waiting_matches_fresh": True,
    }


def test_runtime_policy_update_verifies_declared_identity_and_quarantines(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace
from vllm.lora.request import LoRARequest
from art_vllm_runtime.policy_spans import (
    PolicyLoRARequest,
    _apply_policy_lora_update,
    _policy_metadata_for_lora_request,
    _record_worker_lora_policy,
)

declared = PolicyLoRARequest(
    lora_name="model:active", lora_int_id=1,
    lora_path="/mapped/step-999-deadbeef", policy_version=7, update_seq=3,
)
declared_state = _record_worker_lora_policy(declared)
bootstrap_state = _record_worker_lora_policy(LoRARequest(
    lora_name="model:active", lora_int_id=2,
    lora_path="/mapped/step-999-deadbeef",
))
try:
    _policy_metadata_for_lora_request(LoRARequest(
        lora_name="model:active", lora_int_id=2,
        lora_path="/mapped/step-999-deadbeef",
    ))
except RuntimeError as error:
    undeclared_failure = str(error)

class FailingCore:
    def __init__(self):
        self.scheduler = SimpleNamespace(requests={})
        self.pause_calls = []

    def is_scheduler_paused(self):
        return True

    def collective_rpc(self, *_args, **_kwargs):
        raise RuntimeError("rank 1 failed")

    def pause_scheduler(self, mode, clear_cache):
        self.pause_calls.append((mode, clear_cache))

core = FailingCore()
try:
    _apply_policy_lora_update(core, {
        "lora_name": declared.lora_name,
        "lora_int_id": declared.lora_int_id,
        "lora_path": declared.lora_path,
        "base_model_name": None,
        "tensorizer_config_dict": None,
        "is_3d_lora_weight": False,
        "policy_version": declared.policy_version,
        "update_seq": declared.update_seq,
    })
except RuntimeError as error:
    failure = str(error)

print(json.dumps({
    "declared_version": declared_state["policy_version"],
    "bootstrap_path_not_inferred": bootstrap_state["policy_version"],
    "failure": failure,
    "pause_calls": core.pause_calls,
    "undeclared_failure": undeclared_failure,
}))
""",
        artifact_dir,
        "declared_policy_and_quarantine",
    )
    assert json.loads(payload) == {
        "bootstrap_path_not_inferred": 0,
        "declared_version": 7,
        "failure": "rank 1 failed",
        "pause_calls": [["abort", True]],
        "undeclared_failure": (
            "Mutable LoRA slot 'model:active' has no declared policy identity"
        ),
    }


def test_runtime_policy_update_pins_workers_and_normalizes_scheduler_requests(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from art_vllm_runtime.policy_spans import (
    _apply_policy_lora_update,
    _patch_policy_lora_update_rpc,
)
from vllm.lora.model_manager import AdapterLRUCache, LRUCacheLoRAModelManager
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.v1.worker.worker_base import WorkerBase

class TestAdapterManager(LRUCacheLoRAModelManager):
    def __init__(self):
        self.lora_config = SimpleNamespace(max_cpu_loras=2, max_loras=2)
        self._registered_adapters = AdapterLRUCache(2, self.deactivate_adapter)
        self._active_adapters = AdapterLRUCache(2, self._deactivate_adapter)
        self.lora_index_to_id = [None, None]
        self.modules = {}

    def _create_merged_loras_inplace(self, _lora):
        pass

class TestWorkerManager(LRUCacheWorkerLoRAManager):
    def __init__(self):
        self._adapter_manager = TestAdapterManager()
        self.loaded_paths = []

    def _load_adapter(self, request):
        path = Path(request.lora_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.loaded_paths.append(path.name)
        return SimpleNamespace(id=request.lora_int_id)

class Core:
    def __init__(self, worker):
        self.worker = worker
        self.acks = []
        initial = LoRARequest("model:active", 99, "/initial")
        request = SimpleNamespace(
            request_id="waiting", lora_request=initial, cache_salt=None,
            block_hashes=[], num_computed_tokens=0, output_token_ids=[],
            num_preemptions=0, update_block_hashes=lambda: None,
        )
        self.scheduler = SimpleNamespace(requests={request.request_id: request})

    def is_scheduler_paused(self):
        return True

    def collective_rpc(self, method, args):
        assert method == "art_load_lora_policy"
        ack = WorkerBase.art_load_lora_policy(self.worker, args[0])
        self.acks.append(ack)
        return [ack]

    def _reset_caches(self, **_kwargs):
        pass

    def pause_scheduler(self, *_args):
        raise AssertionError("successful update must not quarantine the engine")

def policy_payload(path, policy_version, update_seq):
    return {
        "lora_name": "model:active",
        "lora_int_id": 99,
        "lora_path": str(path),
        "base_model_name": None,
        "tensorizer_config_dict": None,
        "is_3d_lora_weight": False,
        "policy_version": policy_version,
        "update_seq": update_seq,
    }

def pinned(manager):
    cache = manager._adapter_manager
    return (
        99 in manager.list_adapters()
        and 99 in cache._registered_adapters.pinned_items
        and 99 in cache._active_adapters.pinned_items
    )

_patch_policy_lora_update_rpc()
manager = TestWorkerManager()
worker = SimpleNamespace(
    add_lora=manager.add_adapter,
    pin_lora=manager.pin_adapter,
    list_loras=manager.list_adapters,
)
core = Core(worker)
request = core.scheduler.requests["waiting"]
with TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    first_path = root / "active_1"
    first_path.mkdir()
    first = policy_payload(first_path, 1, 1)
    first_transition = _apply_policy_lora_update(core, first)
    first_result = core.acks[-1]
    initially_pinned = pinned(manager)
    first_path.rmdir()
    manager.add_adapter(request.lora_request)
    no_scheduler_reload = manager.loaded_paths == ["active_1"]

    for adapter_id, name in ((1, "exact"), (2, "eval")):
        path = root / name
        path.mkdir()
        manager.add_adapter(LoRARequest(name, adapter_id, str(path)))
    retained_under_pressure = pinned(manager)

    update_path = root / "active_2"
    update_path.mkdir()
    update = policy_payload(update_path, 2, 2)
    update_transition = _apply_policy_lora_update(core, update)
    update_result = core.acks[-1]
    repinned = pinned(manager)
    update_path.rmdir()
    manager.add_adapter(request.lora_request)

    pressure_path = root / "eval_after_update"
    pressure_path.mkdir()
    manager.add_adapter(LoRARequest("eval_after_update", 3, str(pressure_path)))
    retained_after_update = pinned(manager)

expected_paths = ["active_1", "exact", "eval", "active_2", "eval_after_update"]
assert first_result["loaded"] and update_result["loaded"]
assert first_transition == update_transition == {
    "continued_requests": 0, "updated_requests": 1,
}
assert manager.loaded_paths == expected_paths
assert not request.lora_request.load_inplace and no_scheduler_reload
assert initially_pinned and retained_under_pressure and repinned
assert retained_after_update
assert update_result["previous"]["update_seq"] == 1
assert update_result["current"]["update_seq"] == 2
print(json.dumps({
    "loaded_paths": manager.loaded_paths,
    "load_inplace": request.lora_request.load_inplace,
    "no_scheduler_reload": no_scheduler_reload,
    "pinned_through_update_and_pressure": True,
    "update_sequence": [
        update_result["previous"]["update_seq"],
        update_result["current"]["update_seq"],
    ],
}))
""",
        artifact_dir,
        "pinned_policy_lifetime",
    )
    assert json.loads(payload) == {
        "loaded_paths": ["active_1", "exact", "eval", "active_2", "eval_after_update"],
        "load_inplace": False,
        "no_scheduler_reload": True,
        "pinned_through_update_and_pressure": True,
        "update_sequence": [1, 2],
    }


def test_runtime_declares_launch_policy_before_admission(artifact_dir: Path) -> None:
    payload = _runtime_python(
        """
import asyncio
import json
from types import SimpleNamespace
from vllm.lora.request import LoRARequest
from art_vllm_runtime.policy_spans import (
    PolicyLoRARequest,
    declare_initial_lora_policy,
    lora_update_coordinator,
)

class Core:
    async def call_utility_async(self, method, payload):
        assert method == "art_declare_loaded_lora_policy"
        self.payload = payload
        return {"workers": 2}

async def main():
    slot = "model:active"
    models = SimpleNamespace(lora_requests={slot: LoRARequest(
        lora_name=slot, lora_int_id=3, lora_path="/initial"
    )})
    core = Core()
    engine = SimpleNamespace(engine_core=core)
    await declare_initial_lora_policy(
        models, engine, lora_slot=slot, policy_version=7
    )
    declared = models.lora_requests[slot]
    coordinator = lora_update_coordinator(models, engine)
    async with coordinator.admission(slot) as admitted:
        admitted_identity = [admitted.policy_version, admitted.update_seq]
    next_sequence = await coordinator.begin_update(slot)
    await coordinator.cancel_update(slot, next_sequence)
    return {
        "declared_type": type(declared).__name__,
        "declared_identity": [declared.policy_version, declared.update_seq],
        "worker_identity": [core.payload["policy_version"], core.payload["update_seq"]],
        "admitted_identity": admitted_identity,
        "next_sequence": next_sequence,
    }

print(json.dumps(asyncio.run(main())))
""",
        artifact_dir,
        "launch_policy_declaration",
    )
    assert json.loads(payload) == {
        "admitted_identity": [7, 1],
        "declared_identity": [7, 1],
        "declared_type": "PolicyLoRARequest",
        "next_sequence": 2,
        "worker_identity": [7, 1],
    }


def test_runtime_policy_spans_survive_parallel_sample_aggregation(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace

from vllm.v1.engine.output_processor import RequestState


def aggregate_final_outputs(self, new_token_ids, *args, **kwargs):
    if not self.finished:
        return None
    self.parent_req.outputs[self.request_index] = SimpleNamespace(
        index=self.request_index
    )
    self.parent_req.finished += 1
    if self.parent_req.finished < len(self.parent_req.outputs):
        return None
    return SimpleNamespace(outputs=self.parent_req.outputs)


RequestState.make_request_output = aggregate_final_outputs
import art_vllm_runtime.policy_spans as policy_spans

policy_spans._patch_output_processor_policy_span_accumulation()
parent = SimpleNamespace(outputs=[None] * 4, finished=0)
none_count = 0
final_output = None
for choice_index in range(4):
    detokenizer = SimpleNamespace(num_output_tokens=lambda: 0)
    state = SimpleNamespace(
        request_id=f"child-{choice_index}",
        request_index=choice_index,
        parent_req=parent,
        detokenizer=detokenizer,
        finished=False,
    )
    for token_index in range(5):
        detokenizer.num_output_tokens = lambda count=token_index + 1: count
        state.finished = token_index == 4
        policy_spans._CURRENT_ENGINE_POLICY_SPANS = {
            state.request_id: [{
                "start_token": 0,
                "end_token": 1,
                "policy_version": 7,
                "lora_slot": "model:active",
                "update_seq": 3,
            }]
        }
        output = RequestState.make_request_output(state, [100 + token_index])
        if output is None:
            none_count += 1
        else:
            final_output = output

print(json.dumps({
    "none_count": none_count,
    "spans": [
        getattr(output, policy_spans.ART_POLICY_TOKEN_SPANS_FIELD, None)
        for output in final_output.outputs
    ],
}, sort_keys=True))
""",
        artifact_dir,
        "parallel_sample_policy_spans",
    )
    assert json.loads(payload) == {
        "none_count": 19,
        "spans": [
            [
                {
                    "end_token": 5,
                    "lora_slot": "model:active",
                    "policy_version": 7,
                    "start_token": 0,
                    "update_seq": 3,
                }
            ]
        ]
        * 4,
    }


def test_runtime_general_plugin_loads_full_patch_set() -> None:
    pyproject = (ROOT / "vllm_runtime" / "pyproject.toml").read_text()
    assert 'art = "art_vllm_runtime.patches:apply_vllm_runtime_patches"' in pyproject


def test_runtime_patch_selects_checkpoint_weight_update_lifecycle(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        """
import json
from types import SimpleNamespace

from art_vllm_runtime.patches import apply_vllm_runtime_patches

apply_vllm_runtime_patches()
from vllm.v1.worker.gpu_worker import Worker


class Engine:
    def __init__(self):
        self.starts = self.updates = self.finishes = 0

    def start_weight_update(self):
        self.starts += 1

    def update_weights(self, update_info):
        self.updates += 1

    def finish_weight_update(self):
        self.finishes += 1


def exercise(architecture):
    engine = Engine()
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        ),
        weight_transfer_engine=engine,
        _weight_update_active=False,
        _check_weight_transfer_engine=lambda: None,
    )
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": []})
    Worker.finish_weight_update(worker)
    return {
        "starts": engine.starts,
        "updates": engine.updates,
        "finishes": engine.finishes,
        "active": worker._weight_update_active,
    }


print(json.dumps({
    "dense": exercise("Qwen3ForCausalLM"),
    "gemma4": exercise("Gemma4ForConditionalGeneration"),
}, sort_keys=True))
""",
        artifact_dir,
        "checkpoint_weight_update_lifecycle",
    )
    assert json.loads(payload) == {
        "dense": {"active": False, "finishes": 1, "starts": 1, "updates": 1},
        "gemma4": {"active": False, "finishes": 0, "starts": 0, "updates": 1},
    }


def test_runtime_patch_set_does_not_install_lora_monkey_patches() -> None:
    source = (
        ROOT / "vllm_runtime" / "src" / "art_vllm_runtime" / "patches.py"
    ).read_text()
    assert "patch_punica_ep_moe_lora_alignment" not in source
    assert "patch_lora_duplicate_module_aliases" not in source
    assert "patch_fused_moe_ep_lora_support" not in source


def test_runtime_cli_serializes_lora_target_modules_as_single_nargs_vector(
    artifact_dir: Path,
) -> None:
    payload = _runtime_python(
        "import json; "
        "from art_vllm_runtime.dedicated_server import _append_cli_arg; "
        "args = []; "
        "_append_cli_arg(args, 'lora_target_modules', ['a', 'b']); "
        "print(json.dumps(args))",
        artifact_dir,
        "lora_target_modules",
    )
    assert json.loads(payload) == ["--lora-target-modules", "a", "b"]
