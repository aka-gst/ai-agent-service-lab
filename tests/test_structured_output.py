import json

import pytest
from pydantic import ValidationError

from agent_lab.structured_output import SupportTicket, build_payload, parse_ticket


def test_build_payload_uses_pydantic_schema() -> None:
    payload = build_payload("Не могу войти")

    assert payload["format"] == SupportTicket.model_json_schema()
    assert payload["options"] == {"temperature": 0}
    assert payload["stream"] is False
    assert payload["think"] is False


def test_parse_valid_ticket() -> None:
    content = json.dumps(
        {
            "category": "access",
            "priority": "high",
            "summary": "Клиент не может войти в аккаунт.",
        }
    )

    ticket = parse_ticket(content)

    assert ticket.category == "access"
    assert ticket.priority == "high"


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"category": "unknown", "priority": "high", "summary": "Ошибка"},
        {"category": "access", "priority": "urgent", "summary": "Ошибка"},
        {"category": "access", "priority": "high"},
        {
            "category": "access",
            "priority": "high",
            "summary": "Ошибка",
            "unexpected": True,
        },
    ],
)
def test_parse_rejects_invalid_ticket(invalid_data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_ticket(json.dumps(invalid_data))
