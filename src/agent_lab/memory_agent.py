"""Локальная память сессии и audit log поверх безопасного agent loop."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent_lab.tool_agent import AgentResult, SYSTEM_PROMPT, run_agent_messages


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_DB_PATH = Path("data/private/agent_memory.sqlite3")


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str


class SessionStore:
    """Хранилище сообщений и событий в локальной SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS messages_session_id_id
                ON messages (session_id, id);

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "Session ID должен содержать 1–64 символа: латиницу, цифры, _ или -"
            )
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.validate_session_id(session_id)
        if role not in {"user", "assistant"}:
            raise ValueError(f"Недопустимая роль для памяти: {role}")
        if not content.strip():
            raise ValueError("Пустое сообщение нельзя сохранить")
        self.connection.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self.connection.commit()

    def recent_messages(self, session_id: str, limit: int = 10) -> list[StoredMessage]:
        self.validate_session_id(session_id)
        if limit < 1 or limit > 50:
            raise ValueError("Лимит истории должен быть от 1 до 50")
        rows = self.connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (session_id, limit),
        ).fetchall()
        return [StoredMessage(role=row[0], content=row[1]) for row in rows]

    def add_audit_event(
        self,
        session_id: str,
        event_type: str,
        details: dict[str, object],
    ) -> None:
        self.validate_session_id(session_id)
        self.connection.execute(
            """
            INSERT INTO audit_events (session_id, event_type, details_json)
            VALUES (?, ?, ?)
            """,
            (session_id, event_type, json.dumps(details, ensure_ascii=False)),
        )
        self.connection.commit()

    def audit_events(self, session_id: str) -> list[dict[str, object]]:
        self.validate_session_id(session_id)
        rows = self.connection.execute(
            """
            SELECT event_type, details_json, created_at
            FROM audit_events
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "event_type": row[0],
                "details": json.loads(row[1]),
                "created_at": row[2],
            }
            for row in rows
        ]

    def clear_session(self, session_id: str) -> None:
        self.validate_session_id(session_id)
        self.connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.connection.execute(
            "DELETE FROM audit_events WHERE session_id = ?", (session_id,)
        )
        self.connection.commit()


def run_memory_agent(
    task: str,
    *,
    session_id: str,
    store: SessionStore,
    model: str = "qwen3:8b",
    history_limit: int = 10,
    max_steps: int = 4,
) -> AgentResult:
    """Добавить ограниченную историю к agent loop и сохранить результат."""

    history = store.recent_messages(session_id, limit=history_limit)
    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT} Используй историю только как контекст текущей сессии."
            ),
        },
        *[{"role": item.role, "content": item.content} for item in history],
        {"role": "user", "content": task},
    ]

    result = run_agent_messages(messages, model=model, max_steps=max_steps)
    store.add_message(session_id, "user", task)
    store.add_message(session_id, "assistant", result.answer)
    store.add_audit_event(
        session_id,
        "turn_completed",
        {"steps": result.steps, "history_messages_used": len(history)},
    )
    for event in result.tool_events:
        store.add_audit_event(
            session_id,
            "tool_called",
            {"name": event.name, "arguments": event.arguments, "result": event.result},
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Агент с локальной памятью SQLite")
    parser.add_argument("task", nargs="?", help="Новое сообщение пользователя")
    parser.add_argument("--session", default="demo", help="ID сессии")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--show-history", action="store_true")
    parser.add_argument("--show-audit", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    try:
        with SessionStore(args.db) as store:
            if args.clear:
                store.clear_session(args.session)
                print(f"Сессия очищена: {args.session}")
                return 0
            if args.show_history:
                for message in store.recent_messages(args.session, limit=args.history_limit):
                    print(f"{message.role.upper()}: {message.content}")
                return 0
            if args.show_audit:
                print(json.dumps(store.audit_events(args.session), ensure_ascii=False, indent=2))
                return 0
            if not args.task:
                parser.error("Передайте сообщение или выберите --show-history/--show-audit/--clear")

            result = run_memory_agent(
                args.task,
                session_id=args.session,
                store=store,
                history_limit=args.history_limit,
                max_steps=args.max_steps,
            )
    except (RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    print(f"ANSWER: {result.answer}")
    print(f"TOOLS: {len(result.tool_events)}")
    print(f"STEPS: {result.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
