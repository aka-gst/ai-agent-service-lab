from pathlib import Path

import pytest

from agent_lab.memory_agent import SessionStore


@pytest.fixture
def store(tmp_path: Path):
    with SessionStore(tmp_path / "memory.sqlite3") as session_store:
        yield session_store


def test_messages_are_isolated_by_session(store: SessionStore) -> None:
    store.add_message("alice", "user", "Заказ DEMO-1001")
    store.add_message("bob", "user", "Заказ DEMO-1002")

    assert [item.content for item in store.recent_messages("alice")] == [
        "Заказ DEMO-1001"
    ]
    assert [item.content for item in store.recent_messages("bob")] == [
        "Заказ DEMO-1002"
    ]


def test_history_limit_keeps_recent_order(store: SessionStore) -> None:
    for number in range(5):
        store.add_message("demo", "user", f"Сообщение {number}")

    messages = store.recent_messages("demo", limit=2)

    assert [item.content for item in messages] == ["Сообщение 3", "Сообщение 4"]


@pytest.mark.parametrize("session_id", ["", "../secret", "has space", "x" * 65])
def test_invalid_session_id_is_rejected(store: SessionStore, session_id: str) -> None:
    with pytest.raises(ValueError, match="Session ID"):
        store.recent_messages(session_id)


def test_audit_log_round_trip(store: SessionStore) -> None:
    store.add_audit_event("demo", "tool_called", {"name": "lookup_order"})

    events = store.audit_events("demo")

    assert events[0]["event_type"] == "tool_called"
    assert events[0]["details"] == {"name": "lookup_order"}


def test_clear_session_does_not_touch_another_session(store: SessionStore) -> None:
    store.add_message("first", "user", "Удалить")
    store.add_message("second", "user", "Сохранить")

    store.clear_session("first")

    assert store.recent_messages("first") == []
    assert [item.content for item in store.recent_messages("second")] == ["Сохранить"]
