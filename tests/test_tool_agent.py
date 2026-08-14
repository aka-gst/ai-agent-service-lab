import pytest

from agent_lab.tool_agent import (
    ToolExecutionError,
    calculate_order_total,
    execute_tool,
    lookup_order,
)


def test_lookup_known_order() -> None:
    result = lookup_order("demo-1001")

    assert result == {
        "found": True,
        "order_id": "DEMO-1001",
        "status": "передан в доставку",
        "amount_rub": 2490,
    }


def test_lookup_unknown_order_does_not_invent_data() -> None:
    assert lookup_order("DEMO-9999") == {
        "found": False,
        "order_id": "DEMO-9999",
    }


def test_calculate_order_total_uses_trusted_price() -> None:
    assert calculate_order_total("DEMO-1001", 3) == {
        "order_id": "DEMO-1001",
        "quantity": 3,
        "unit_price_rub": 2490,
        "total_rub": 7470,
    }


def test_calculate_order_total_rejects_unsafe_quantity() -> None:
    with pytest.raises(ToolExecutionError, match="от 1 до 100"):
        calculate_order_total("DEMO-1001", 1000)


def test_execute_tool_rejects_unknown_tool() -> None:
    with pytest.raises(ToolExecutionError, match="не разрешён"):
        execute_tool("run_shell", {"command": "whoami"})


def test_execute_tool_rejects_extra_arguments() -> None:
    with pytest.raises(ToolExecutionError, match="Недопустимые аргументы"):
        execute_tool("lookup_order", {"order_id": "DEMO-1001", "secret": True})
