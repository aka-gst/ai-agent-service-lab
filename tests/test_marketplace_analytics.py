from pathlib import Path

import pytest

from agent_lab.marketplace_analytics import analyze_low_buyout, load_report


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
