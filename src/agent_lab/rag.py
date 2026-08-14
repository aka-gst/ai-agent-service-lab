"""Минимальный локальный RAG с Ollama, SQLite и проверяемыми источниками."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict


DEFAULT_DOCS_PATH = Path("data/demo/client-docs")
DEFAULT_DB_PATH = Path("data/private/rag.sqlite3")
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_CHAT_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class Chunk:
    source: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    source: str
    content: str
    score: float


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[str]


def chunk_document(path: Path) -> list[Chunk]:
    """Разбить Markdown по заголовкам второго уровня."""

    title = path.stem
    heading = "document"
    body: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            chunks.append(Chunk(source=f"{path.name}#{heading}", content=content))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            body.clear()
            heading = line[3:].strip()
            continue
        if line:
            body.append(line)
    flush()

    return [
        Chunk(source=chunk.source, content=f"Документ: {title}\n{chunk.content}")
        for chunk in chunks
    ]


def load_chunks(docs_path: Path) -> list[Chunk]:
    paths = sorted(docs_path.glob("*.md"))
    if not paths:
        raise ValueError(f"В папке нет Markdown-документов: {docs_path}")
    return [chunk for path in paths for chunk in chunk_document(path)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Векторы должны иметь одинаковую ненулевую размерность")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Нельзя сравнить нулевой вектор")
    return dot / (left_norm * right_norm)


def post_json(url: str, payload: dict[str, object], timeout: float = 120) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama вернула HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError("Не удалось подключиться к Ollama на 127.0.0.1:11434") from error
    if not isinstance(result, dict):
        raise RuntimeError("Ollama вернула неожиданный формат ответа")
    return result


def embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = "http://127.0.0.1:11434",
) -> list[list[float]]:
    result = post_json(f"{base_url}/api/embed", {"model": model, "input": texts})
    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Ollama вернула неправильное число embeddings")
    if not all(isinstance(vector, list) for vector in embeddings):
        raise RuntimeError("Embedding должен быть массивом чисел")
    return embeddings


class RagIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def __enter__(self) -> RagIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def replace(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Число chunks и embeddings должно совпадать")
        with self.connection:
            self.connection.execute("DELETE FROM chunks")
            self.connection.executemany(
                """
                INSERT INTO chunks (source, content, embedding_json, embedding_model)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (chunk.source, chunk.content, json.dumps(vector), model)
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        if top_k < 1 or top_k > 10:
            raise ValueError("top_k должен быть от 1 до 10")
        rows = self.connection.execute(
            "SELECT source, content, embedding_json FROM chunks"
        ).fetchall()
        if not rows:
            raise RuntimeError("RAG-индекс пуст. Сначала выполните команду index")
        results = [
            SearchResult(
                source=row[0],
                content=row[1],
                score=cosine_similarity(query_embedding, json.loads(row[2])),
            )
            for row in rows
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def build_index(
    docs_path: Path,
    db_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
) -> int:
    chunks = load_chunks(docs_path)
    embeddings = embed_texts([chunk.content for chunk in chunks], model=embedding_model)
    with RagIndex(db_path) as index:
        index.replace(chunks, embeddings, embedding_model)
    return len(chunks)


def validate_sources(answer: RagAnswer, retrieved: list[SearchResult]) -> None:
    allowed = {item.source for item in retrieved}
    unknown = set(answer.sources) - allowed
    if unknown:
        raise RuntimeError(f"Модель указала источники, которых не было в контексте: {unknown}")


def answer_question(
    question: str,
    db_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    top_k: int = 3,
    base_url: str = "http://127.0.0.1:11434",
) -> tuple[RagAnswer, list[SearchResult]]:
    query_text = (
        "Instruct: найди фрагменты, содержащие ответ на вопрос.\n"
        f"Query: {question}"
    )
    query_embedding = embed_texts(
        [query_text], model=embedding_model, base_url=base_url
    )[0]
    with RagIndex(db_path) as index:
        retrieved = index.search(query_embedding, top_k=top_k)

    context = "\n\n".join(
        f"SOURCE: {item.source}\n{item.content}" for item in retrieved
    )
    payload = {
        "model": chat_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ответь только по переданному CONTEXT. Не используй внешние знания. "
                    "Если ответа нет, прямо скажи об этом и верни пустой список sources. "
                    "В sources указывай только точные значения после SOURCE."
                ),
            },
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "format": RagAnswer.model_json_schema(),
    }
    response = post_json(f"{base_url}/api/chat", payload)
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("В ответе Ollama отсутствует message.content") from error
    answer = RagAnswer.model_validate_json(content)
    validate_sources(answer, retrieved)
    return answer, retrieved


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный RAG по Markdown-документам")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Построить индекс")
    index_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    index_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    ask_parser = subparsers.add_parser("ask", help="Задать вопрос")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ask_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    try:
        if args.command == "index":
            count = build_index(args.docs, args.db)
            print(f"Проиндексировано фрагментов: {count}")
            return 0

        answer, retrieved = answer_question(args.question, args.db, top_k=args.top_k)
    except (RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    print(f"ANSWER: {answer.answer}")
    print("SOURCES:")
    for source in answer.sources:
        print(f"- {source}")
    print("RETRIEVAL:")
    for item in retrieved:
        print(f"- {item.score:.4f} {item.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
