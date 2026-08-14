from pathlib import Path

import pytest

import agent_lab.marketplace_assistant as assistant
from agent_lab.marketplace_assistant import (
    MarketplaceExplanation,
    answer_marketplace_comparison_question,
    answer_marketplace_question,
    answer_marketplace_upload_question,
    filter_retrieved_for_analysis,
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


def test_retrieval_filter_keeps_only_relevant_returns_section() -> None:
    retrieved = [
        SearchResult("guide.md#Доля возвратов", "Возвраты", 0.9),
        SearchResult("guide.md#Процент выкупа", "Выкуп", 0.8),
    ]

    filtered = filter_retrieved_for_analysis(retrieved, "returns")

    assert [item.source for item in filtered] == ["guide.md#Доля возвратов"]


def test_retrieval_filter_falls_back_when_expected_section_is_missing() -> None:
    retrieved = [SearchResult("custom.md#Метрика", "Текст", 0.7)]

    assert filter_retrieved_for_analysis(retrieved, "returns") == retrieved


def test_uploaded_chat_keeps_filename_as_data_source(
    monkeypatch, tmp_path: Path
) -> None:
    db = tmp_path / "knowledge.sqlite3"
    with RagIndex(db) as index:
        index.replace(
            [Chunk("guide.md#Выкуп", "Определение")],
            [[1.0]],
            "test",
        )
    monkeypatch.setattr(assistant, "embed_texts", lambda *_args, **_kwargs: [[1.0]])
    monkeypatch.setattr(
        assistant,
        "post_json",
        lambda *_args, **_kwargs: {
            "message": {
                "content": MarketplaceExplanation(
                    explanation="Проверенное объяснение.",
                    knowledge_sources=["guide.md#Выкуп"],
                ).model_dump_json()
            }
        },
    )

    answer = answer_marketplace_upload_question(
        "Почему выкуп низкий?",
        "product,ordered,bought,returned\nТовар,10,5,1\n",
        "browser.csv",
        db,
    )

    assert answer.analysis.sources == ["browser.csv"]
    assert answer.analysis.metrics[0].buyout_rate == 50


def test_comparison_chat_preserves_calculated_change(
    monkeypatch, tmp_path: Path
) -> None:
    db = tmp_path / "knowledge.sqlite3"
    with RagIndex(db) as index:
        index.replace(
            [Chunk("guide.md#Сравнение", "Изменение измеряется в п.п.")],
            [[1.0]],
            "test",
        )
    monkeypatch.setattr(assistant, "embed_texts", lambda *_args, **_kwargs: [[1.0]])
    monkeypatch.setattr(
        assistant,
        "post_json",
        lambda *_args, **_kwargs: {
            "message": {
                "content": MarketplaceExplanation(
                    explanation="Кроссовки снизились на 30 п.п.",
                    knowledge_sources=["guide.md#Сравнение"],
                ).model_dump_json()
            }
        },
    )

    answer = answer_marketplace_comparison_question(
        "Где снизился выкуп?",
        "product,ordered,bought,returned\nКроссовки,50,40,2\n",
        "product,ordered,bought,returned\nКроссовки,50,25,2\n",
        "old.csv",
        "new.csv",
        db,
    )

    assert answer.comparison.metrics[0].change_pp == -30
    assert answer.knowledge_sources == ["guide.md#Сравнение"]
