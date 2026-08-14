from pathlib import Path

import pytest

import agent_lab.marketplace_assistant as assistant
from agent_lab.marketplace_assistant import (
    MarketplaceExplanation,
    answer_marketplace_question,
    validate_knowledge_sources,
)
from agent_lab.rag import Chunk, RagIndex, SearchResult


def test_chat_combines_calculation_rag_and_explanation(
    monkeypatch, tmp_path: Path
) -> None:
    report = tmp_path / "report.csv"
    report.write_text(
        "product,ordered,bought,returned\nФутболка,100,60,5\n",
        encoding="utf-8",
    )
    db = tmp_path / "knowledge.sqlite3"
    with RagIndex(db) as index:
        index.replace(
            [Chunk("guide.md#Процент выкупа", "Выкуп — отношение покупок к заказам.")],
            [[1.0, 0.0]],
            "test-embedding",
        )
    monkeypatch.setattr(assistant, "embed_texts", lambda *_args, **_kwargs: [[1.0, 0.0]])

    def fake_post(_url, payload):
        prompt = payload["messages"][1]["content"]
        assert "60.0% (60 из 100)" in prompt
        assert "guide.md#Процент выкупа" in prompt
        result = MarketplaceExplanation(
            explanation="Выкуп составляет 60%; точную причину отчёт не показывает.",
            knowledge_sources=["guide.md#Процент выкупа"],
        )
        return {"message": {"content": result.model_dump_json()}}

    monkeypatch.setattr(assistant, "post_json", fake_post)

    answer = answer_marketplace_question("Почему низкий выкуп?", report, db)

    assert answer.analysis.metrics[0].buyout_rate == 60
    assert answer.knowledge_sources == ["guide.md#Процент выкупа"]
    assert "точную причину" in answer.explanation


def test_chat_rejects_hallucinated_knowledge_source() -> None:
    retrieved = [SearchResult("real.md#Раздел", "Текст", 0.9)]

    with pytest.raises(RuntimeError, match="не было в справочнике"):
        validate_knowledge_sources(["invented.md#Раздел"], retrieved)
