from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import agent_lab.service as service
from agent_lab.rag import RagAnswer, SearchResult
from agent_lab.marketplace_analytics import AnalyticsAnswer
from agent_lab.marketplace_assistant import MarketplaceChatAnswer
from agent_lab.service import Settings, create_app
from agent_lab.structured_output import SupportTicket


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_app(tmp_path: Path, api_key: str | None = None):
    settings = Settings(
        rag_db_path=tmp_path / "rag.sqlite3",
        marketplace_reports_path=tmp_path,
        marketplace_knowledge_db_path=tmp_path / "marketplace-rag.sqlite3",
        api_key=api_key,
    )
    return create_app(settings)


async def send_request(
    tmp_path: Path, method: str, url: str, *, api_key: str | None = None, **kwargs
):
    transport = ASGITransport(app=make_app(tmp_path, api_key=api_key))
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


async def test_protected_endpoint_rejects_missing_api_key(tmp_path: Path) -> None:
    response = await send_request(
        tmp_path,
        "POST",
        "/v1/rag/ask",
        api_key="expected-secret",
        json={"question": "Стоимость?"},
    )

    assert response.status_code == 401
    assert "expected-secret" not in response.text


async def test_health_does_not_require_configured_api_key(tmp_path: Path) -> None:
    response = await send_request(
        tmp_path, "GET", "/health", api_key="expected-secret"
    )

    assert response.status_code == 200


async def test_marketplace_endpoint_analyzes_report(tmp_path: Path) -> None:
    (tmp_path / "report.csv").write_text(
        "product,ordered,bought,returned\nФутболка,100,60,4\n",
        encoding="utf-8",
    )

    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/ask",
        json={
            "question": "Почему процент выкупа маленький?",
            "report": "report.csv",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"][0]["buyout_rate"] == 60
    assert payload["sources"] == ["report.csv"]
    assert payload["possible_causes"]


async def test_marketplace_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/ask",
        json={"question": "Какой выкуп?", "report": "../../secret.csv"},
    )

    assert response.status_code == 422
    assert "без пути" in response.json()["detail"]


async def test_marketplace_endpoint_does_not_fake_unsupported_analysis(
    tmp_path: Path,
) -> None:
    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/ask",
        json={"question": "Почему упала прибыль?", "report": "report.csv"},
    )

    assert response.status_code == 422
    assert "выкупе и возвратах" in response.json()["detail"]


async def test_marketplace_chat_endpoint(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "report.csv").write_text(
        "product,ordered,bought,returned\nТовар,10,6,1\n", encoding="utf-8"
    )
    analysis = AnalyticsAnswer(
        answer="Есть отклонение.",
        facts=["Выкуп 60%."],
        possible_causes=[],
        missing_data=[],
        metrics=[],
        sources=["report.csv"],
    )
    monkeypatch.setattr(
        service,
        "answer_marketplace_question",
        lambda *_args, **_kwargs: MarketplaceChatAnswer(
            explanation="Причина не доказана.",
            analysis=analysis,
            knowledge_sources=["metrics-guide.md#Процент выкупа"],
        ),
    )

    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/chat",
        json={"question": "Почему низкий выкуп?", "report": "report.csv"},
    )

    assert response.status_code == 200
    assert response.json()["explanation"] == "Причина не доказана."


async def test_marketplace_ui_is_available(tmp_path: Path) -> None:
    response = await send_request(tmp_path, "GET", "/marketplace")

    assert response.status_code == 200
    assert "AI-помощник аналитика" in response.text
    assert "/v1/marketplace/chat-upload" in response.text


async def test_marketplace_upload_is_analyzed_in_memory(tmp_path: Path) -> None:
    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/analyze-upload",
        json={
            "question": "Почему выкуп низкий?",
            "filename": "my-report.csv",
            "csv_text": "product,ordered,bought,returned\nТовар,10,5,1\n",
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"][0]["buyout_rate"] == 50
    assert response.json()["sources"] == ["my-report.csv"]


async def test_marketplace_chat_upload_endpoint(monkeypatch, tmp_path: Path) -> None:
    analysis = AnalyticsAnswer(
        answer="Есть отклонение.",
        facts=["Выкуп 50%."],
        possible_causes=[],
        missing_data=[],
        metrics=[],
        sources=["upload.csv"],
    )
    monkeypatch.setattr(
        service,
        "answer_marketplace_upload_question",
        lambda *_args, **_kwargs: MarketplaceChatAnswer(
            explanation="Модель объяснила расчёт.",
            analysis=analysis,
            knowledge_sources=["guide.md#Выкуп"],
        ),
    )

    response = await send_request(
        tmp_path,
        "POST",
        "/v1/marketplace/chat-upload",
        json={
            "question": "Почему выкуп низкий?",
            "filename": "upload.csv",
            "csv_text": "product,ordered,bought,returned\nТовар,10,5,1\n",
        },
    )

    assert response.status_code == 200
    assert response.json()["explanation"] == "Модель объяснила расчёт."
