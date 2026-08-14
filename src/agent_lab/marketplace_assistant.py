"""Диалоговый слой над проверяемой аналитикой и RAG-справочником."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_lab.marketplace_analytics import (
    AnalyticsAnswer,
    ComparisonAnswer,
    analyze_marketplace_question_path,
    analyze_marketplace_question_text,
    compare_periods_text,
)
from agent_lab.rag import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    RagIndex,
    SearchResult,
    embed_texts,
    post_json,
)


class MarketplaceExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    knowledge_sources: list[str]


class MarketplaceChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    analysis: AnalyticsAnswer
    knowledge_sources: list[str]


class MarketplaceComparisonChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str
    comparison: ComparisonAnswer
    knowledge_sources: list[str]


def validate_knowledge_sources(
    sources: list[str], retrieved: list[SearchResult]
) -> None:
    allowed = {item.source for item in retrieved}
    unknown = set(sources) - allowed
    if unknown:
        raise RuntimeError(
            f"Модель указала источники, которых не было в справочнике: {unknown}"
        )


def answer_marketplace_question(
    question: str,
    report_path: Path,
    knowledge_db_path: Path,
    *,
    low_threshold: float = 70,
    high_return_threshold: float = 15,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    top_k: int = 2,
) -> MarketplaceChatAnswer:
    """Рассчитать метрики, найти справку и попросить LLM только объяснить факты."""

    analysis = analyze_marketplace_question_path(
        question,
        report_path,
        low_threshold=low_threshold,
        high_return_threshold=high_return_threshold,
    )
    return explain_marketplace_analysis(
        question,
        analysis,
        knowledge_db_path,
        embedding_model=embedding_model,
        chat_model=chat_model,
        base_url=base_url,
        top_k=top_k,
    )


def answer_marketplace_upload_question(
    question: str,
    csv_text: str,
    filename: str,
    knowledge_db_path: Path,
    *,
    low_threshold: float = 70,
    high_return_threshold: float = 15,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    top_k: int = 2,
) -> MarketplaceChatAnswer:
    """Проанализировать загруженный CSV в памяти и объяснить результат."""

    analysis = analyze_marketplace_question_text(
        question,
        csv_text,
        filename=filename,
        low_threshold=low_threshold,
        high_return_threshold=high_return_threshold,
    )
    return explain_marketplace_analysis(
        question,
        analysis,
        knowledge_db_path,
        embedding_model=embedding_model,
        chat_model=chat_model,
        base_url=base_url,
        top_k=top_k,
    )


def explain_marketplace_analysis(
    question: str,
    analysis: AnalyticsAnswer,
    knowledge_db_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    top_k: int = 2,
) -> MarketplaceChatAnswer:
    """Добавить к готовому расчёту RAG-контекст и объяснение модели."""

    query = f"Instruct: найди определение показателя и ограничения анализа.\nQuery: {question}"
    query_embedding = embed_texts(
        [query], model=embedding_model, base_url=base_url
    )[0]
    with RagIndex(knowledge_db_path) as index:
        retrieved = index.search(query_embedding, top_k=top_k)

    facts = "\n".join(f"- {fact}" for fact in analysis.facts)
    hypotheses = "\n".join(f"- {item}" for item in analysis.possible_causes)
    missing = "\n".join(f"- {item}" for item in analysis.missing_data)
    context = "\n\n".join(
        f"SOURCE: {item.source}\n{item.content}" for item in retrieved
    )
    payload = {
        "model": chat_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник аналитика маркетплейса. Объясни результат простым русским "
                    "языком. Не меняй рассчитанные цифры. Не выдавай гипотезы за факты. "
                    "Если данных для причины недостаточно, скажи это. Используй только FACTS "
                    "и KNOWLEDGE. В knowledge_sources указывай только точные значения SOURCE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\nFACTS:\n{facts}\n\n"
                    f"POSSIBLE CAUSES:\n{hypotheses}\n\nMISSING DATA:\n{missing}\n\n"
                    f"KNOWLEDGE:\n{context}"
                ),
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "format": MarketplaceExplanation.model_json_schema(),
    }
    response = post_json(f"{base_url}/api/chat", payload)
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("В ответе Ollama отсутствует message.content") from error
    explanation = MarketplaceExplanation.model_validate_json(content)
    validate_knowledge_sources(explanation.knowledge_sources, retrieved)
    return MarketplaceChatAnswer(
        explanation=explanation.explanation,
        analysis=analysis,
        knowledge_sources=explanation.knowledge_sources,
    )


def answer_marketplace_comparison_question(
    question: str,
    previous_csv_text: str,
    current_csv_text: str,
    previous_filename: str,
    current_filename: str,
    knowledge_db_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    top_k: int = 2,
) -> MarketplaceComparisonChatAnswer:
    """Сравнить периоды кодом и объяснить результат по RAG-справочнику."""

    comparison = compare_periods_text(
        previous_csv_text,
        current_csv_text,
        previous_filename=previous_filename,
        current_filename=current_filename,
    )
    query = (
        "Instruct: найди правила сравнения периодов и ограничения анализа.\n"
        f"Query: {question}"
    )
    query_embedding = embed_texts(
        [query], model=embedding_model, base_url=base_url
    )[0]
    with RagIndex(knowledge_db_path) as index:
        retrieved = index.search(query_embedding, top_k=top_k)

    facts = "\n".join(f"- {fact}" for fact in comparison.facts)
    hypotheses = "\n".join(f"- {item}" for item in comparison.possible_causes)
    missing = "\n".join(f"- {item}" for item in comparison.missing_data)
    context = "\n\n".join(
        f"SOURCE: {item.source}\n{item.content}" for item in retrieved
    )
    payload = {
        "model": chat_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник аналитика маркетплейса. Объясни сравнение периодов "
                    "простым русским языком. Не меняй цифры и направление изменений. "
                    "Не выдавай гипотезы за факты. Используй только FACTS и KNOWLEDGE. "
                    "В knowledge_sources указывай только точные значения SOURCE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\nFACTS:\n{facts}\n\n"
                    f"POSSIBLE CAUSES:\n{hypotheses}\n\nMISSING DATA:\n{missing}\n\n"
                    f"KNOWLEDGE:\n{context}"
                ),
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "format": MarketplaceExplanation.model_json_schema(),
    }
    response = post_json(f"{base_url}/api/chat", payload)
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("В ответе Ollama отсутствует message.content") from error
    explanation = MarketplaceExplanation.model_validate_json(content)
    validate_knowledge_sources(explanation.knowledge_sources, retrieved)
    return MarketplaceComparisonChatAnswer(
        explanation=explanation.explanation,
        comparison=comparison,
        knowledge_sources=explanation.knowledge_sources,
    )
