from agent_lab.portal_copilot import answer_portal_question


def test_navigator_finds_stock_analytics() -> None:
    answer = answer_portal_question("Где посмотреть остатки по маркетплейсу?")

    assert answer.answer_type == "navigation"
    assert answer.target_page is not None
    assert answer.target_page.page_id == "stock_analytics"
    assert answer.route == ["Аналитика", "Аналитика остатков"]


def test_current_page_provides_context_for_metric_question() -> None:
    answer = answer_portal_question(
        "Входят ли сюда отменённые заказы?", current_page_id="sales_funnel"
    )

    assert answer.answer_type == "knowledge_required"
    assert answer.current_page == "Воронка продаж"
    assert answer.needs_knowledge is True
    assert "официальное определение" in answer.answer


def test_unknown_question_does_not_invent_route() -> None:
    answer = answer_portal_question("Расскажи что-нибудь интересное")

    assert answer.answer_type == "unsupported"
    assert answer.route == []
    assert answer.target_page is None
