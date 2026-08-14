"""Контекстный навигатор по демонстрационной структуре кабинета продавца."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PortalPage(StrictModel):
    page_id: str
    title: str
    route: list[str]
    keywords: list[str]
    example_questions: list[str]


class PortalAnswer(StrictModel):
    answer_type: str
    answer: str
    current_page: str | None
    target_page: PortalPage | None
    route: list[str]
    needs_knowledge: bool
    suggested_questions: list[str]
    source: str


PAGES = [
    PortalPage(
        page_id="sales_funnel",
        title="Воронка продаж",
        route=["Аналитика", "Аналитика продавца", "Воронка продаж"],
        keywords=["воронка", "выкуп", "заказ", "конверсия", "ctr"],
        example_questions=[
            "Как считается процент выкупа?",
            "Входят ли в показатель отменённые заказы и заказы в пути?",
            "Почему конверсия могла снизиться?",
        ],
    ),
    PortalPage(
        page_id="stock_analytics",
        title="Аналитика остатков",
        route=["Аналитика", "Аналитика остатков"],
        keywords=["остаток", "остатки", "склад", "запас"],
        example_questions=[
            "Где посмотреть остатки по складам?",
            "Как рассчитывается оборачиваемость?",
        ],
    ),
    PortalPage(
        page_id="weekly_dynamics",
        title="Еженедельная динамика и анализ продаж",
        route=["Аналитика", "Отчёты", "Еженедельная динамика и анализ продаж"],
        keywords=["динамика", "неделя", "недельный", "сравнить период", "продажи"],
        example_questions=[
            "Как сравнить продажи за две недели?",
            "У какого товара сильнее всего снизился выкуп?",
        ],
    ),
    PortalPage(
        page_id="sales_reports",
        title="Отчёты реализации",
        route=["Финансы", "Отчёты реализации"],
        keywords=["номер отчёта", "документ", "реализация", "баланс", "финансы"],
        example_questions=[
            "Где найти отчёт по его номеру?",
            "За какой период сформирован документ?",
        ],
    ),
    PortalPage(
        page_id="nomenclatures",
        title="Отчёт с перечнем номенклатур",
        route=["Аналитика", "Отчёты", "Отчёт с перечнем номенклатур"],
        keywords=["номенклатура", "артикул", "баркод", "размер", "состав"],
        example_questions=[
            "Где найти артикул WB и баркод?",
            "Какие характеристики есть у номенклатуры?",
        ],
    ),
    PortalPage(
        page_id="prices_discounts",
        title="Цены и скидки",
        route=["Товары и цены", "Цены и скидки"],
        keywords=["цена", "скидка", "индекс цен"],
        example_questions=[
            "Где изменить цену товара?",
            "Чем цена продавца отличается от цены со скидкой?",
        ],
    ),
]


def get_page(page_id: str | None) -> PortalPage | None:
    return next((page for page in PAGES if page.page_id == page_id), None)


def find_target_page(question: str) -> PortalPage | None:
    normalized = question.casefold()
    scored = [
        (sum(keyword in normalized for keyword in page.keywords), page)
        for page in PAGES
    ]
    score, page = max(scored, key=lambda item: item[0])
    return page if score else None


def answer_portal_question(
    question: str, *, current_page_id: str | None = None
) -> PortalAnswer:
    """Ответить по контексту экрана, не выдумывая неизвестные формулы."""

    current = get_page(current_page_id)
    normalized = question.casefold()
    navigation_intent = any(
        phrase in normalized for phrase in ("где", "как найти", "куда", "в каком разделе")
    )
    target = find_target_page(question)

    if navigation_intent and target:
        route_text = " → ".join(target.route)
        return PortalAnswer(
            answer_type="navigation",
            answer=f"Откройте: {route_text}.",
            current_page=current.title if current else None,
            target_page=target,
            route=target.route,
            needs_knowledge=False,
            suggested_questions=target.example_questions,
            source="demo portal catalog",
        )

    context = current or target
    if context:
        return PortalAnswer(
            answer_type="knowledge_required",
            answer=(
                f"Вопрос относится к экрану «{context.title}». Для точного ответа "
                "нужно найти официальное определение показателя в базе знаний."
            ),
            current_page=current.title if current else None,
            target_page=context,
            route=context.route,
            needs_knowledge=True,
            suggested_questions=context.example_questions,
            source="demo portal catalog",
        )

    return PortalAnswer(
        answer_type="unsupported",
        answer=(
            "Не удалось определить нужный экран. Уточните название отчёта или показателя."
        ),
        current_page=current.title if current else None,
        target_page=None,
        route=[],
        needs_knowledge=False,
        suggested_questions=current.example_questions if current else [],
        source="demo portal catalog",
    )
