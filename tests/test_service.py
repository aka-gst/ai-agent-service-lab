from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import agent_lab.service as service
from agent_lab.rag import RagAnswer, SearchResult
from agent_lab.service import Settings, create_app
from agent_lab.structured_output import SupportTicket


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_app(tmp_path: Path):
    settings = Settings(rag_db_path=tmp_path / "rag.sqlite3")
    return create_app(settings)


async def send_request(tmp_path: Path, method: str, url: str, **kwargs):
    transport = ASGITransport(app=make_app(tmp_path))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


async def test_health_does_not_require_ollama(tmp_path: Path) -> None:
    response = await send_request(tmp_path, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service_version": "0.1.0"}


async def test_ready_reports_missing_dependencies(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(service, "ollama_is_ready", lambda _url: False)

    response = await send_request(tmp_path, "GET", "/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "ollama": False,
        "rag_index": False,
    }


async def test_classify_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        service,
        "classify_ticket",
        lambda *_args, **_kwargs: SupportTicket(
            category="access", priority="high", summary="Нет доступа"
        ),
    )

    response = await send_request(
        tmp_path, "POST", "/v1/tickets/classify", json={"text": "Не могу войти"}
    )

    assert response.status_code == 200
    assert response.json()["category"] == "access"


async def test_rag_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        service,
        "answer_question",
        lambda *_args, **_kwargs: (
            RagAnswer(answer="490 рублей", sources=["delivery.md#Стоимость"]),
            [SearchResult("delivery.md#Стоимость", "Текст", 0.9)],
        ),
    )

    response = await send_request(
        tmp_path, "POST", "/v1/rag/ask", json={"question": "Стоимость доставки?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "490 рублей",
        "sources": ["delivery.md#Стоимость"],
        "retrieval": [{"source": "delivery.md#Стоимость", "score": 0.9}],
    }


async def test_request_rejects_extra_fields(tmp_path: Path) -> None:
    response = await send_request(
        tmp_path,
        "POST",
        "/v1/tickets/classify",
        json={"text": "Ошибка", "unexpected": True},
    )

    assert response.status_code == 422
