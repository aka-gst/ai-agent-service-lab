"""Безопасный учебный агент с двумя инструментами и лимитом шагов."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError


DEMO_ORDERS = {
    "DEMO-1001": {
        "status": "передан в доставку",
        "amount_rub": 2490,
    },
    "DEMO-1002": {
        "status": "ожидает оплаты",
        "amount_rub": 1750,
    },
}


class LookupOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str


class CalculateOrderTotalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    quantity: int


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Найти статус и сумму демонстрационного заказа по его ID.",
            "parameters": LookupOrderArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "description": "Посчитать стоимость указанного количества единиц демонстрационного заказа.",
            "parameters": CalculateOrderTotalArgs.model_json_schema(),
        },
    },
]


class ToolExecutionError(RuntimeError):
    """Модель запросила неизвестный инструмент или недопустимые аргументы."""


@dataclass(frozen=True)
class ToolEvent:
    name: str
    arguments: dict[str, object]
    result: dict[str, object]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_events: list[ToolEvent]
    steps: int


def lookup_order(order_id: str) -> dict[str, object]:
    """Прочитать заказ только из встроенного демонстрационного набора."""

    order = DEMO_ORDERS.get(order_id.upper())
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id.upper(), **order}


def calculate_order_total(order_id: str, quantity: int) -> dict[str, object]:
    """Посчитать итог по цене из доверенного демонстрационного справочника."""

    if quantity < 1 or quantity > 100:
        raise ToolExecutionError("Количество должно быть от 1 до 100")
    order = DEMO_ORDERS.get(order_id.upper())
    if order is None:
        raise ToolExecutionError(f"Заказ не найден: {order_id}")

    unit_price = Decimal(str(order["amount_rub"]))
    total = unit_price * quantity
    return {
        "order_id": order_id.upper(),
        "quantity": quantity,
        "unit_price_rub": int(unit_price),
        "total_rub": int(total),
    }


def execute_tool(name: str, arguments: object) -> dict[str, object]:
    """Проверить имя и аргументы, затем вызвать функцию из allowlist."""

    if not isinstance(arguments, dict):
        raise ToolExecutionError("Аргументы инструмента должны быть JSON-объектом")

    try:
        if name == "lookup_order":
            args = LookupOrderArgs.model_validate(arguments)
            return lookup_order(args.order_id)
        if name == "calculate_order_total":
            args = CalculateOrderTotalArgs.model_validate(arguments)
            return calculate_order_total(args.order_id, args.quantity)
    except ValidationError as error:
        raise ToolExecutionError(f"Недопустимые аргументы для {name}: {error}") from error

    raise ToolExecutionError(f"Инструмент не разрешён: {name}")


def chat(
    messages: list[dict[str, object]],
    *,
    model: str,
    base_url: str,
    timeout: float,
) -> dict[str, object]:
    """Выполнить один шаг диалога с Ollama."""

    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama вернула HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError("Не удалось подключиться к Ollama на 127.0.0.1:11434") from error

    message = result.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("В ответе Ollama отсутствует объект message")
    return message


def run_agent(
    task: str,
    *,
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120,
    max_steps: int = 4,
) -> AgentResult:
    """Запускать модель и инструменты до ответа или достижения лимита."""

    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "Ты помощник демонстрационной службы заказов. "
                "Используй инструменты для фактов и вычислений, не выдумывай их. "
                "У тебя нет других инструментов. После получения результатов дай "
                "краткий ответ на русском языке."
            ),
        },
        {"role": "user", "content": task},
    ]
    events: list[ToolEvent] = []

    for step in range(1, max_steps + 1):
        assistant_message = chat(
            messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls") or []

        if not isinstance(tool_calls, list):
            raise RuntimeError("Поле tool_calls должно быть массивом")

        if not tool_calls:
            answer = assistant_message.get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Модель завершила цикл без текстового ответа")
            return AgentResult(answer=answer.strip(), tool_events=events, steps=step)

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
                raise ToolExecutionError("Некорректная структура tool call")
            function = tool_call["function"]
            name = function.get("name")
            arguments = function.get("arguments", {})
            if not isinstance(name, str):
                raise ToolExecutionError("В tool call отсутствует имя функции")

            result = execute_tool(name, arguments)
            events.append(ToolEvent(name=name, arguments=arguments, result=result))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    raise RuntimeError(f"Agent loop превысил лимит {max_steps} шагов")


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасный агент службы заказов")
    parser.add_argument("task", help="Задача для агента")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()

    try:
        result = run_agent(args.task, model=args.model, max_steps=args.max_steps)
    except (RuntimeError, ToolExecutionError) as error:
        print(f"Ошибка: {error}")
        return 1

    for event in result.tool_events:
        print(
            "TOOL:",
            json.dumps(
                {"name": event.name, "arguments": event.arguments, "result": event.result},
                ensure_ascii=False,
            ),
        )
    print(f"ANSWER: {result.answer}")
    print(f"STEPS: {result.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
