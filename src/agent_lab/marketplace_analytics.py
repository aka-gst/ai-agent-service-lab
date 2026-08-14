"""Проверяемая аналитика обезличенных отчётов маркетплейса."""

from __future__ import annotations

import csv
from collections import defaultdict
from io import StringIO
from typing import TextIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductMetric(StrictModel):
    product: str
    ordered: int = Field(ge=0)
    bought: int = Field(ge=0)
    returned: int = Field(ge=0)
    buyout_rate: float = Field(ge=0, le=100)


class AnalyticsAnswer(StrictModel):
    answer: str
    facts: list[str]
    possible_causes: list[str]
    missing_data: list[str]
    metrics: list[ProductMetric]
    sources: list[str]


REQUIRED_COLUMNS = {"product", "ordered", "bought", "returned"}


def _parse_non_negative_int(value: str, *, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"Строка {row_number}: {column} должно быть целым числом"
        ) from error
    if parsed < 0:
        raise ValueError(f"Строка {row_number}: {column} не может быть отрицательным")
    return parsed


def _load_report_stream(report: TextIO) -> list[ProductMetric]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    reader = csv.DictReader(report)
    columns = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"В отчёте отсутствуют столбцы: {', '.join(sorted(missing))}")

    for row_number, row in enumerate(reader, start=2):
        product = (row["product"] or "").strip()
        if not product:
            raise ValueError(f"Строка {row_number}: product не может быть пустым")
        ordered = _parse_non_negative_int(
            row["ordered"], column="ordered", row_number=row_number
        )
        bought = _parse_non_negative_int(
            row["bought"], column="bought", row_number=row_number
        )
        returned = _parse_non_negative_int(
            row["returned"], column="returned", row_number=row_number
        )
        if bought > ordered:
            raise ValueError(f"Строка {row_number}: bought не может быть больше ordered")
        totals[product][0] += ordered
        totals[product][1] += bought
        totals[product][2] += returned

    if not totals:
        raise ValueError("Отчёт не содержит данных")

    return [
        ProductMetric(
            product=product,
            ordered=values[0],
            bought=values[1],
            returned=values[2],
            buyout_rate=round(values[1] / values[0] * 100, 2) if values[0] else 0,
        )
        for product, values in sorted(totals.items())
    ]


def load_report(path: Path) -> list[ProductMetric]:
    """Прочитать CSV и агрегировать строки по товарам."""

    with path.open(encoding="utf-8", newline="") as report:
        return _load_report_stream(report)


def load_report_text(csv_text: str) -> list[ProductMetric]:
    """Прочитать CSV из HTTP-запроса без сохранения файла на диск."""

    if not csv_text.strip():
        raise ValueError("CSV не может быть пустым")
    return _load_report_stream(StringIO(csv_text))


def analyze_metrics(
    metrics: list[ProductMetric], *, source: str, low_threshold: float = 70
) -> AnalyticsAnswer:
    total_ordered = sum(item.ordered for item in metrics)
    total_bought = sum(item.bought for item in metrics)
    overall_rate = round(total_bought / total_ordered * 100, 2) if total_ordered else 0
    low_products = [item for item in metrics if item.buyout_rate < low_threshold]

    facts = [
        f"Общий процент выкупа: {overall_rate}% ({total_bought} из {total_ordered})."
    ]
    facts.extend(
        f"{item.product}: выкуп {item.buyout_rate}% ({item.bought} из {item.ordered})."
        for item in low_products
    )
    possible_causes = [
        "Несоответствие размера или ожиданий покупателя.",
        "Проблемы с карточкой товара, ценой, качеством или доставкой.",
    ] if low_products else []
    missing_data = [
        "Причины отказов и возвратов.",
        "Отзывы, размеры, цены, сроки доставки и изменения карточки товара.",
    ] if low_products else []

    if low_products:
        answer = (
            f"Выкуп ниже порога {low_threshold}% у {len(low_products)} товар(ов). "
            "Отчёт показывает, где возникло отклонение, но не доказывает его причину."
        )
    else:
        answer = f"Товаров с выкупом ниже порога {low_threshold}% в отчёте нет."

    return AnalyticsAnswer(
        answer=answer,
        facts=facts,
        possible_causes=possible_causes,
        missing_data=missing_data,
        metrics=metrics,
        sources=[source],
    )


def analyze_low_buyout(path: Path, *, low_threshold: float = 70) -> AnalyticsAnswer:
    """Объяснить низкий выкуп, не выдавая предположения за доказанные причины."""

    return analyze_metrics(
        load_report(path), source=path.name, low_threshold=low_threshold
    )


def analyze_low_buyout_text(
    csv_text: str, *, filename: str, low_threshold: float = 70
) -> AnalyticsAnswer:
    """Проанализировать загруженный CSV без записи на диск."""

    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Допустимо только имя CSV-файла без пути")
    return analyze_metrics(
        load_report_text(csv_text), source=filename, low_threshold=low_threshold
    )
