"""Классификация обращения через structured output Ollama."""

from __future__ import annotations

import argparse
import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SupportTicket(BaseModel):
    """Проверяемая структура результата классификации."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["access", "billing", "technical", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1)


def build_payload(text: str, model: str = "qwen3:8b") -> dict[str, object]:
    """Создать тело запроса к Ollama вместе с JSON Schema."""

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Классифицируй обращение службы поддержки. "
                    "Проблемы входа и доступа относятся к category=access. "
                    "Верни только данные, соответствующие переданной JSON Schema."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "format": SupportTicket.model_json_schema(),
    }


def parse_ticket(content: str) -> SupportTicket:
    """Проверить JSON-строку модели через Pydantic."""

    return SupportTicket.model_validate_json(content)


def classify_ticket(
    text: str,
    *,
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120,
) -> SupportTicket:
    """Отправить обращение в Ollama и вернуть проверенный результат."""

    body = json.dumps(build_payload(text, model), ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            ollama_response = json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama вернула HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(
            "Не удалось подключиться к Ollama. Запустите приложение Ollama "
            "и проверьте http://127.0.0.1:11434/api/version"
        ) from error

    try:
        content = ollama_response["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("В ответе Ollama отсутствует message.content") from error

    if not isinstance(content, str):
        raise RuntimeError("Поле message.content должно быть строкой")

    return parse_ticket(content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Классифицировать обращение через локальную модель Ollama."
    )
    parser.add_argument("text", help="Текст обращения клиента")
    parser.add_argument("--model", default="qwen3:8b", help="Идентификатор модели")
    args = parser.parse_args()

    try:
        ticket = classify_ticket(args.text, model=args.model)
    except (RuntimeError, ValidationError) as error:
        print(f"Ошибка: {error}")
        return 1

    print(ticket.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
