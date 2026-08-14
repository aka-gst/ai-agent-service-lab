"""FastAPI-обёртка над готовыми компонентами учебного агента."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_lab.marketplace_analytics import AnalyticsAnswer, analyze_low_buyout
from agent_lab.rag import DEFAULT_DB_PATH, answer_question
from agent_lab.structured_output import SupportTicket, classify_ticket


SERVICE_VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    rag_db_path: Path = DEFAULT_DB_PATH
    marketplace_reports_path: Path = Path("data/demo/marketplace")
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            rag_db_path=Path(os.getenv("RAG_DB_PATH", str(cls.rag_db_path))),
            marketplace_reports_path=Path(
                os.getenv(
                    "MARKETPLACE_REPORTS_PATH", str(cls.marketplace_reports_path)
                )
            ),
            api_key=os.getenv("SERVICE_API_KEY") or None,
        )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: str
    service_version: str


class ReadinessResponse(StrictModel):
    status: str
    ollama: bool
    rag_index: bool


class ClassifyRequest(StrictModel):
    text: str = Field(min_length=1, max_length=10_000)


class RagRequest(StrictModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=3, ge=1, le=10)


class RetrievedSource(StrictModel):
    source: str
    score: float


class RagResponse(StrictModel):
    answer: str
    sources: list[str]
    retrieval: list[RetrievedSource]


class MarketplaceQuestionRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2_000)
    report: str = Field(min_length=1, max_length=255)
    low_threshold: float = Field(default=70, ge=0, le=100)


def resolve_report(reports_path: Path, filename: str) -> Path:
    """Разрешить доступ только к CSV непосредственно в папке отчётов."""

    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Допустимо только имя CSV-файла без пути")
    report = reports_path / filename
    if not report.is_file():
        raise ValueError(f"Отчёт не найден: {filename}")
    return report


def is_buyout_question(question: str) -> bool:
    normalized = question.casefold()
    return any(term in normalized for term in ("выкуп", "выкупили", "выкуплен"))


def ollama_is_ready(base_url: str, timeout: float = 2) -> bool:
    try:
        with urlopen(f"{base_url}/api/version", timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("version"), str)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize(api_key: str | None = Depends(api_key_header)) -> None:
        if config.api_key is None:
            return
        if api_key is None or not secrets.compare_digest(api_key, config.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    app = FastAPI(
        title="AI Agent Service Lab",
        version=SERVICE_VERSION,
        description="Локальный учебный API для structured output и RAG.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service_version=SERVICE_VERSION)

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        ollama_ready = ollama_is_ready(config.ollama_base_url)
        rag_ready = config.rag_db_path.is_file()
        status = "ready" if ollama_ready and rag_ready else "degraded"
        return ReadinessResponse(
            status=status,
            ollama=ollama_ready,
            rag_index=rag_ready,
        )

    @app.post(
        "/v1/tickets/classify",
        response_model=SupportTicket,
        dependencies=[Depends(authorize)],
    )
    def classify(request: ClassifyRequest) -> SupportTicket:
        try:
            return classify_ticket(
                request.text,
                model=config.chat_model,
                base_url=config.ollama_base_url,
            )
        except (RuntimeError, ValidationError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post(
        "/v1/rag/ask",
        response_model=RagResponse,
        dependencies=[Depends(authorize)],
    )
    def rag_ask(request: RagRequest) -> RagResponse:
        try:
            answer, retrieved = answer_question(
                request.question,
                config.rag_db_path,
                embedding_model=config.embedding_model,
                chat_model=config.chat_model,
                top_k=request.top_k,
                base_url=config.ollama_base_url,
            )
        except (RuntimeError, ValueError, ValidationError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return RagResponse(
            answer=answer.answer,
            sources=answer.sources,
            retrieval=[
                RetrievedSource(source=item.source, score=item.score) for item in retrieved
            ],
        )

    @app.post(
        "/v1/marketplace/ask",
        response_model=AnalyticsAnswer,
        dependencies=[Depends(authorize)],
    )
    def marketplace_ask(request: MarketplaceQuestionRequest) -> AnalyticsAnswer:
        if not is_buyout_question(request.question):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Пока поддерживаются вопросы только о проценте выкупа. "
                    "Другие показатели добавим отдельными проверяемыми сценариями."
                ),
            )
        try:
            report = resolve_report(config.marketplace_reports_path, request.report)
            return analyze_low_buyout(report, low_threshold=request.low_threshold)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()
