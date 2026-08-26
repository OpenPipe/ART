from collections import deque
from threading import Lock
from types import SimpleNamespace

from art.megatron.runtime.executor import _GenerationPublisher


def test_snapshot_authorization_order_follows_reservation_not_prepare_completion() -> None:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._prepared = {}
    publisher._prepared_order = deque()

    slow = SimpleNamespace(
        operation_id="slow-first",
        entry=SimpleNamespace(consumers=[]),
        completion=object(),
    )
    fast = SimpleNamespace(
        operation_id="fast-second",
        entry=SimpleNamespace(consumers=[]),
        completion=object(),
    )

    publisher.reserve_snapshot(slow.operation_id)
    publisher.reserve_snapshot(fast.operation_id)
    publisher._register_prepared(fast)
    publisher._register_prepared(slow)

    assert tuple(publisher._prepared_order) == (slow.operation_id, fast.operation_id)
    assert slow.entry.consumers == [slow.completion]
    assert fast.entry.consumers == [fast.completion]


def test_failed_snapshot_preparation_releases_its_reservation() -> None:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._prepared = {}
    publisher._prepared_order = deque()

    publisher.reserve_snapshot("failed")
    publisher.discard("failed")

    assert not publisher._prepared_order


def test_duplicate_snapshot_reservation_cannot_discard_prior_work() -> None:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._prepared = {}
    publisher._prepared_order = deque()

    publisher.reserve_snapshot("operation")

    try:
        publisher.reserve_snapshot("operation")
    except RuntimeError as error:
        assert "already reserved" in str(error)
    else:
        raise AssertionError("duplicate snapshot reservation succeeded")
    assert tuple(publisher._prepared_order) == ("operation",)
