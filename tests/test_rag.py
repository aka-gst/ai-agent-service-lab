from pathlib import Path

import pytest

from agent_lab.rag import (
    Chunk,
    RagAnswer,
    RagIndex,
    SearchResult,
    chunk_document,
    cosine_similarity,
    validate_sources,
)


def test_chunk_document_uses_headings_as_sources(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Инструкция\n\n## Оплата\n\nСрок — пять дней.\n", encoding="utf-8")

    chunks = chunk_document(document)

    assert chunks == [
        Chunk(source="guide.md#Оплата", content="Документ: Инструкция\nСрок — пять дней.")
    ]


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_index_returns_most_similar_chunk(tmp_path: Path) -> None:
    with RagIndex(tmp_path / "rag.sqlite3") as index:
        index.replace(
            [Chunk("delivery.md#Сроки", "Доставка"), Chunk("returns.md#Срок", "Возврат")],
            [[1.0, 0.0], [0.0, 1.0]],
            "test-model",
        )

        results = index.search([0.9, 0.1], top_k=1)

    assert results[0].source == "delivery.md#Сроки"


def test_validate_sources_rejects_hallucinated_source() -> None:
    answer = RagAnswer(answer="Ответ", sources=["invented.md#Факт"])
    retrieved = [SearchResult("real.md#Факт", "Текст", 0.9)]

    with pytest.raises(RuntimeError, match="не было в контексте"):
        validate_sources(answer, retrieved)


def test_validate_sources_accepts_retrieved_source() -> None:
    answer = RagAnswer(answer="Ответ", sources=["real.md#Факт"])
    retrieved = [SearchResult("real.md#Факт", "Текст", 0.9)]

    validate_sources(answer, retrieved)
