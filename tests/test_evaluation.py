import json
from pathlib import Path

import pytest

from agent_lab.evaluation import EvalCase, build_report, load_cases, score_case
from agent_lab.rag import RagAnswer, SearchResult


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "id": "same",
        "question": "Вопрос",
        "expected_sources": [],
        "required_terms": [],
    }
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(ValueError, match="уникальными"):
        load_cases(path)


def test_score_positive_case() -> None:
    case = EvalCase(
        id="delivery",
        question="Стоимость?",
        expected_sources=["delivery.md#Стоимость"],
        required_terms=["490"],
    )
    answer = RagAnswer(
        answer="Доставка стоит 490 рублей.",
        sources=["delivery.md#Стоимость"],
    )
    retrieved = [SearchResult("delivery.md#Стоимость", "Текст", 0.9)]

    result = score_case(case, answer, retrieved, 1.25)

    assert result.passed is True
    assert result.retrieval_pass is True


def test_score_fails_when_required_term_is_missing() -> None:
    case = EvalCase(
        id="delivery",
        question="Стоимость?",
        expected_sources=["delivery.md#Стоимость"],
        required_terms=["490"],
    )
    answer = RagAnswer(answer="Доставка платная.", sources=["delivery.md#Стоимость"])
    retrieved = [SearchResult("delivery.md#Стоимость", "Текст", 0.9)]

    result = score_case(case, answer, retrieved, 1.0)

    assert result.answer_pass is False
    assert result.passed is False


def test_unknown_question_passes_only_without_sources() -> None:
    case = EvalCase(
        id="unknown",
        question="Гарантия?",
        expected_sources=[],
        required_terms=[],
        expect_no_answer=True,
    )

    result = score_case(case, RagAnswer(answer="Нет данных", sources=[]), [], 0.5)

    assert result.passed is True
    assert result.retrieval_pass is None


def test_build_report_calculates_rates() -> None:
    positive_case = EvalCase(
        id="positive",
        question="Стоимость?",
        expected_sources=["source"],
        required_terms=["490"],
    )
    passed = score_case(
        positive_case,
        RagAnswer(answer="490", sources=["source"]),
        [SearchResult("source", "text", 1.0)],
        1.0,
    )
    failed = score_case(
        positive_case.model_copy(update={"id": "failed"}),
        RagAnswer(answer="не знаю", sources=[]),
        [SearchResult("other", "text", 0.5)],
        3.0,
    )

    report = build_report([passed, failed])

    assert report.pass_rate == pytest.approx(0.5)
    assert report.retrieval_hit_rate == pytest.approx(0.5)
    assert report.average_latency_seconds == pytest.approx(2.0)
