"""Проверяемая аналитика обезличенных отчётов маркетплейса."""

from __future__ import annotations

import csv
from collections import defaultdict
from io import StringIO
from typing import Literal, TextIO
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
    return_rate: float = Field(ge=0, le=100)


class AnalyticsAnswer(StrictModel):
    analysis_type: Literal["buyout", "returns"] = "buyout"
    answer: str
    facts: list[str]
    possible_causes: list[str]
    missing_data: list[str]
    metrics: list[ProductMetric]
    sources: list[str]


class PeriodComparisonMetric(StrictModel):
    product: str
    previous_buyout_rate: float = Field(ge=0, le=100)
    current_buyout_rate: float = Field(ge=0, le=100)
    change_pp: float = Field(ge=-100, le=100)


class ComparisonAnswer(StrictModel):
    analysis_type: Literal["comparison"] = "comparison"
    answer: str
    facts: list[str]
    possible_causes: list[str]
    missing_data: list[str]
    metrics: list[PeriodComparisonMetric]
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
        if returned > bought:
            raise ValueError(f"Строка {row_number}: returned не может быть больше bought")
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
            return_rate=round(values[2] / values[1] * 100, 2) if values[1] else 0,
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
        analysis_type="buyout",
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


def analyze_returns(
    metrics: list[ProductMetric], *, source: str, high_threshold: float = 15
) -> AnalyticsAnswer:
    """Найти товары с наибольшей долей возвратов среди выкупленных."""

    ranked = sorted(metrics, key=lambda item: item.return_rate, reverse=True)
    leader = ranked[0]
    high_products = [item for item in ranked if item.return_rate > high_threshold]
    total_bought = sum(item.bought for item in metrics)
    total_returned = sum(item.returned for item in metrics)
    overall_rate = round(total_returned / total_bought * 100, 2) if total_bought else 0
    return AnalyticsAnswer(
        analysis_type="returns",
        answer=(
            f"Максимальная доля возвратов у товара «{leader.product}»: "
            f"{leader.return_rate}%. Выше порога {high_threshold}% — "
            f"{len(high_products)} товар(ов). Отчёт показывает отклонение, но не его причину."
        ),
        facts=[
            f"Общая доля возвратов: {overall_rate}% ({total_returned} из {total_bought} выкупленных).",
            *(
                f"{item.product}: возвраты {item.return_rate}% ({item.returned} из {item.bought})."
                for item in ranked
            ),
        ],
        possible_causes=[
            "Несоответствие качества, размера, комплектации или описания товара.",
            "Повреждение при доставке или ошибочное ожидание покупателя.",
        ],
        missing_data=[
            "Причины возвратов по каждому заказу.",
            "Отзывы, характеристики товара, поставки и сведения о повреждениях.",
        ],
        metrics=metrics,
        sources=[source],
    )


def question_type(question: str) -> Literal["buyout", "returns", "comparison"]:
    normalized = question.casefold()
    if any(
        term in normalized
        for term in ("сравн", "сниз", "измен", "динамик", "период", "недел")
    ):
        return "comparison"
    if any(term in normalized for term in ("возврат", "вернули", "возвращают")):
        return "returns"
    if any(term in normalized for term in ("выкуп", "выкупили", "выкуплен")):
        return "buyout"
    raise ValueError(
        "Я пока умею отвечать только про выкуп, возвраты и сравнение периодов"
    )


def analyze_marketplace_question(
    question: str,
    metrics: list[ProductMetric],
    *,
    source: str,
    low_threshold: float = 70,
    high_return_threshold: float = 15,
) -> AnalyticsAnswer:
    intent = question_type(question)
    if intent == "comparison":
        raise ValueError("Для сравнения периодов нужно загрузить два CSV-отчёта")
    if intent == "returns":
        return analyze_returns(
            metrics, source=source, high_threshold=high_return_threshold
        )
    return analyze_metrics(metrics, source=source, low_threshold=low_threshold)


def analyze_marketplace_question_path(
    question: str,
    path: Path,
    *,
    low_threshold: float = 70,
    high_return_threshold: float = 15,
) -> AnalyticsAnswer:
    return analyze_marketplace_question(
        question,
        load_report(path),
        source=path.name,
        low_threshold=low_threshold,
        high_return_threshold=high_return_threshold,
    )


def analyze_marketplace_question_text(
    question: str,
    csv_text: str,
    *,
    filename: str,
    low_threshold: float = 70,
    high_return_threshold: float = 15,
) -> AnalyticsAnswer:
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Допустимо только имя CSV-файла без пути")
    return analyze_marketplace_question(
        question,
        load_report_text(csv_text),
        source=filename,
        low_threshold=low_threshold,
        high_return_threshold=high_return_threshold,
    )


def compare_periods_text(
    previous_csv_text: str,
    current_csv_text: str,
    *,
    previous_filename: str,
    current_filename: str,
) -> ComparisonAnswer:
    """Сравнить выкуп товаров в двух отчётах по процентным пунктам."""

    for filename in (previous_filename, current_filename):
        if Path(filename).name != filename or not filename.lower().endswith(".csv"):
            raise ValueError("Допустимо только имя CSV-файла без пути")
    previous = {item.product: item for item in load_report_text(previous_csv_text)}
    current = {item.product: item for item in load_report_text(current_csv_text)}
    common = sorted(previous.keys() & current.keys())
    if not common:
        raise ValueError("В отчётах нет общих товаров для сравнения")

    metrics = [
        PeriodComparisonMetric(
            product=product,
            previous_buyout_rate=previous[product].buyout_rate,
            current_buyout_rate=current[product].buyout_rate,
            change_pp=round(
                current[product].buyout_rate - previous[product].buyout_rate, 2
            ),
        )
        for product in common
    ]
    ranked = sorted(metrics, key=lambda item: item.change_pp)
    biggest_decline = ranked[0]
    facts = [
        f"{item.product}: {item.previous_buyout_rate}% → {item.current_buyout_rate}% "
        f"({item.change_pp:+.2f} п.п.)."
        for item in ranked
    ]
    if biggest_decline.change_pp < 0:
        answer = (
            f"Самое сильное снижение у товара «{biggest_decline.product}»: "
            f"{biggest_decline.change_pp:.2f} п.п."
        )
    else:
        answer = "Снижения выкупа среди общих товаров не обнаружено."
    return ComparisonAnswer(
        answer=answer,
        facts=facts,
        possible_causes=[
            "Изменение цены, карточки товара, аудитории или условий доставки.",
            "Сезонность, остатки, размеры или изменение качества поставки.",
        ] if biggest_decline.change_pp < 0 else [],
        missing_data=[
            "Даты периодов, цены, остатки, показы и изменения карточки.",
            "Причины отказов, отзывы и сроки доставки по каждому периоду.",
        ] if biggest_decline.change_pp < 0 else [],
        metrics=metrics,
        sources=[previous_filename, current_filename],
    )
