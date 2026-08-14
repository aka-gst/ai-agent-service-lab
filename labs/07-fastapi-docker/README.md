# Лабораторная работа 7: FastAPI и упаковка

## Цель

Предоставить готовые компоненты как локальный HTTP API с health check, проверяемыми контрактами и воспроизводимой конфигурацией.

## Локальный запуск

```bash
uv run uvicorn agent_lab.service:app --host 127.0.0.1 --port 8000
```

Интерактивная документация будет доступна по адресу `http://127.0.0.1:8000/docs`.

## Эндпоинты

- `GET /health` — работает ли сам FastAPI-процесс;
- `GET /ready` — доступны ли Ollama и RAG-индекс;
- `POST /v1/tickets/classify` — structured output классификатора;
- `POST /v1/rag/ask` — ответ по документам с источниками.

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

RAG-запрос:

```bash
curl -s http://127.0.0.1:8000/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Сколько стоит доставка заказа на 4 000 рублей?"}' \
  | python3 -m json.tool
```

## Конфигурация

Переменные перечислены в `.env.example`. Реальный `.env` не попадает в Git.

## Docker

На macOS контейнер обращается к Ollama хоста через `host.docker.internal`:

```bash
docker compose build
docker compose run --rm api python -m agent_lab.rag index
docker compose up -d
```

SQLite-индекс монтируется из `data/private/` и сохраняется между перезапусками контейнера.

Остановка:

```bash
docker compose down
```

## Тесты

```bash
uv run pytest
```

API-тесты подменяют модель и проверяют HTTP-контракты без сетевых запросов: health, degraded readiness, structured output, RAG и запрет лишних полей.

## Ограничения

- сервис привязан к `127.0.0.1`, пока явно не настроены аутентификация и TLS;
- API не предназначен для публичного интернета;
- Docker CLI требуется установить отдельно;
- `/health` проверяет только процесс, а `/ready` — внешние зависимости;
- изменение документов требует повторного построения RAG-индекса.
