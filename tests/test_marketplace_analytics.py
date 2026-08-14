from pathlib import Path

import pytest

from agent_lab.marketplace_analytics import (
    analyze_low_buyout,
    analyze_low_buyout_text,
    analyze_marketplace_question_text,
    compare_periods_text,
    load_report,
)


def write_report(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "report.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_report_aggregates_products(tmp_path: Path) -> None:
    report = write_report(
        tmp_path,
        "product,ordered,bought,returned\nТовар,10,6,1\nТовар,10,8,0\n",
    )

    metrics = load_report(report)

    assert len(metrics) == 1
    assert metrics[0].ordered == 20
    assert metrics[0].bought == 14
    assert metrics[0].buyout_rate == 70


def test_analysis_separates_facts_from_possible_causes(tmp_path: Path) -> None:
    report = write_report(
        tmp_path,
        "product,ordered,bought,returned\nФутболка,100,55,8\nКроссовки,50,45,1\n",
    )

    answer = analyze_low_buyout(report)

    assert any("Футболка: выкуп 55.0%" in fact for fact in answer.facts)
    assert all("Кроссовки:" not in fact for fact in answer.facts[1:])
    assert answer.possible_causes
    assert answer.missing_data
    assert answer.sources == ["report.csv"]


def test_report_requires_expected_columns(tmp_path: Path) -> None:
    report = write_report(tmp_path, "product,ordered\nТовар,10\n")

    with pytest.raises(ValueError, match="отсутствуют столбцы"):
        load_report(report)


def test_report_rejects_impossible_values(tmp_path: Path) -> None:
    report = write_report(
        tmp_path,
        "product,ordered,bought,returned\nТовар,10,11,0\n",
    )

    with pytest.raises(ValueError, match="bought не может быть больше ordered"):
        load_report(report)


def test_analyze_uploaded_csv_without_saving_file() -> None:
    answer = analyze_low_buyout_text(
        "product,ordered,bought,returned\nТовар,20,10,2\n",
        filename="uploaded.csv",
    )

    assert answer.metrics[0].buyout_rate == 50
    assert answer.sources == ["uploaded.csv"]


def test_return_question_finds_highest_return_rate() -> None:
    answer = analyze_marketplace_question_text(
        "У какого товара больше всего возвратов?",
        "product,ordered,bought,returned\nФутболка,100,80,8\nКроссовки,50,25,5\n",
        filename="returns.csv",
    )

    assert answer.analysis_type == "returns"
    assert "Кроссовки" in answer.answer
    assert next(item for item in answer.metrics if item.product == "Кроссовки").return_rate == 20


def test_unknown_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="выкуп, возвраты и сравнение"):
        analyze_marketplace_question_text(
            "Почему упала прибыль?",
            "product,ordered,bought,returned\nТовар,10,5,1\n",
            filename="report.csv",
        )


def test_return_threshold_counts_only_high_products() -> None:
    answer = analyze_marketplace_question_text(
        "Где высокая доля возвратов?",
        "product,ordered,bought,returned\nФутболка,100,80,8\nКроссовки,50,25,5\n",
        filename="returns.csv",
        high_return_threshold=15,
    )

    assert "Выше порога 15% — 1 товар(ов)" in answer.answer


def test_compare_periods_finds_biggest_buyout_decline() -> None:
    answer = compare_periods_text(
        "product,ordered,bought,returned\nФутболка,100,80,5\nКроссовки,50,40,3\n",
        "product,ordered,bought,returned\nФутболка,100,70,5\nКроссовки,50,25,3\n",
        previous_filename="previous.csv",
        current_filename="current.csv",
    )

    assert "Кроссовки" in answer.answer
    assert next(item for item in answer.metrics if item.product == "Кроссовки").change_pp == -30
    assert answer.sources == ["previous.csv", "current.csv"]
