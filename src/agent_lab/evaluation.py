"""Автоматическая evaluation для учебного RAG-помощника."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from agent_lab.rag import DEFAULT_DB_PATH, RagAnswer, SearchResult, answer_question


DEFAULT_CASES_PATH = Path("evals/rag_cases.json")
DEFAULT_REPORT_PATH = Path("artifacts/evals/rag-eval.json")


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    expect_no_answer: bool = False


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    answer: str
    sources: list[str]
    retrieved_sources: list[str]
    source_pass: bool
    answer_pass: bool
    retrieval_pass: bool | None
    passed: bool
    latency_seconds: float = Field(ge=0)
    error: str | None = None


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    retrieval_total: int
    retrieval_passed: int
    retrieval_hit_rate: float
    average_latency_seconds: float
    cases: list[CaseResult]


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation-набор должен быть непустым JSON-массивом")
    cases = [EvalCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("ID evaluation-кейсов должны быть уникальными")
    return cases


def score_case(
    case: EvalCase,
    answer: RagAnswer,
    retrieved: list[SearchResult],
    latency_seconds: float,
) -> CaseResult:
    retrieved_sources = [item.source for item in retrieved]
    if case.expect_no_answer:
        source_pass = answer.sources == []
        answer_pass = answer.sources == []
        retrieval_pass = None
    else:
        expected = set(case.expected_sources)
        source_pass = expected.issubset(answer.sources)
        retrieval_pass = expected.issubset(retrieved_sources)
        normalized_answer = answer.answer.casefold()
        answer_pass = all(term.casefold() in normalized_answer for term in case.required_terms)

    return CaseResult(
        id=case.id,
        question=case.question,
        answer=answer.answer,
        sources=answer.sources,
        retrieved_sources=retrieved_sources,
        source_pass=source_pass,
        answer_pass=answer_pass,
        retrieval_pass=retrieval_pass,
        passed=source_pass and answer_pass and retrieval_pass is not False,
        latency_seconds=round(latency_seconds, 3),
    )


def error_result(case: EvalCase, error: Exception, latency_seconds: float) -> CaseResult:
    return CaseResult(
        id=case.id,
        question=case.question,
        answer="",
        sources=[],
        retrieved_sources=[],
        source_pass=False,
        answer_pass=False,
        retrieval_pass=False if not case.expect_no_answer else None,
        passed=False,
        latency_seconds=round(latency_seconds, 3),
        error=str(error),
    )


def build_report(results: list[CaseResult]) -> EvalReport:
    passed = sum(result.passed for result in results)
    retrieval_results = [
        result.retrieval_pass
        for result in results
        if result.retrieval_pass is not None
    ]
    retrieval_passed = sum(value is True for value in retrieval_results)
    total_latency = sum(result.latency_seconds for result in results)
    return EvalReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results) if results else 0,
        retrieval_total=len(retrieval_results),
        retrieval_passed=retrieval_passed,
        retrieval_hit_rate=(
            retrieval_passed / len(retrieval_results) if retrieval_results else 0
        ),
        average_latency_seconds=total_latency / len(results) if results else 0,
        cases=results,
    )


def run_evaluation(cases: list[EvalCase], db_path: Path) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        started = perf_counter()
        try:
            answer, retrieved = answer_question(case.question, db_path)
            results.append(score_case(case, answer, retrieved, perf_counter() - started))
        except Exception as error:  # evaluation должна записать сбой и продолжить набор
            results.append(error_result(case, error, perf_counter() - started))
    return build_report(results)


def markdown_report(report: EvalReport) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"- Created at: `{report.created_at}`",
        f"- Cases passed: `{report.passed}/{report.total}` ({report.pass_rate:.0%})",
        (
            f"- Retrieval hit rate: `{report.retrieval_passed}/{report.retrieval_total}` "
            f"({report.retrieval_hit_rate:.0%})"
        ),
        f"- Average latency: `{report.average_latency_seconds:.2f}s`",
        "",
        "| Case | Result | Source | Answer | Latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in report.cases:
        lines.append(
            f"| `{result.id}` | {'PASS' if result.passed else 'FAIL'} | "
            f"{'PASS' if result.source_pass else 'FAIL'} | "
            f"{'PASS' if result.answer_pass else 'FAIL'} | "
            f"{result.latency_seconds:.2f}s |"
        )
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation локального RAG")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        cases = load_cases(args.cases)
        report = run_evaluation(cases, args.db)
        json_path, markdown_path = save_report(report, args.report)
    except (RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    for result in report.cases:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.id}")
        if result.error:
            print(f"  ERROR: {result.error}")
    print(f"RESULT: {report.passed}/{report.total} ({report.pass_rate:.0%})")
    print(f"RETRIEVAL: {report.retrieval_hit_rate:.0%}")
    print(f"REPORT: {json_path}")
    print(f"REPORT: {markdown_path}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
