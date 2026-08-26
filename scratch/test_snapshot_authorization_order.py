from collections import deque
from threading import Condition, Event, Lock, Thread
from types import SimpleNamespace

from art.megatron.runtime.executor import _GenerationPublisher


def _publisher() -> _GenerationPublisher:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._prepared = {}
    publisher._prepared_order = deque()
    publisher._prepare_order = deque()
    publisher._prepare_condition = Condition(publisher._lock)
    return publisher


def test_snapshot_authorization_order_follows_reservation_not_prepare_completion() -> (
    None
):
    publisher = _publisher()

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
    publisher = _publisher()

    publisher.reserve_snapshot("failed")
    publisher.discard("failed")

    assert not publisher._prepared_order


def test_duplicate_snapshot_reservation_cannot_discard_prior_work() -> None:
    publisher = _publisher()

    publisher.reserve_snapshot("operation")

    try:
        publisher.reserve_snapshot("operation")
    except RuntimeError as error:
        assert "already reserved" in str(error)
    else:
        raise AssertionError("duplicate snapshot reservation succeeded")
    assert tuple(publisher._prepared_order) == ("operation",)


def test_snapshot_prepare_turn_follows_reservation_when_threads_arrive_reversed() -> (
    None
):
    publisher = _publisher()
    publisher.reserve_snapshot("first")
    publisher.reserve_snapshot("second")
    entered = Event()
    order: list[str] = []

    def run(operation_id: str) -> None:
        with publisher.prepare_turn(operation_id):
            order.append(operation_id)
            entered.set()

    second = Thread(target=run, args=("second",))
    second.start()
    assert not entered.wait(0.05)
    first = Thread(target=run, args=("first",))
    first.start()
    first.join(timeout=1)
    second.join(timeout=1)

    assert order == ["first", "second"]
    assert not publisher._prepare_order
