# Лабораторная работа 8: безопасность и передача клиенту

## Цель

Подготовить сервис к контролируемой передаче: доступ, backup, acceptance criteria, rollback и честное описание ограничений.

## API-ключ

Для локальной разработки ключ можно не задавать. Если сервис доступен другим пользователям, создайте секрет вне Git:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Запишите значение в локальный `.env`:

```text
SERVICE_API_KEY=полученное-значение
```

Клиент передаёт ключ заголовком:

```bash
curl http://127.0.0.1:8000/v1/rag/ask \
  -H "X-API-Key: $SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"Каковы сроки доставки?"}'
```

`/health` и `/ready` не требуют ключа, чтобы инфраструктура могла проверять состояние. Рабочие `/v1/*` защищены, когда `SERVICE_API_KEY` непустой.

## Резервная копия

```bash
uv run python -m agent_lab.backup create
```

Архив содержит только `*.sqlite3`, манифест и SHA-256. Он сохраняется в исключённой из Git папке `artifacts/private/backups/`.

## Безопасное восстановление

```bash
uv run python -m agent_lab.backup restore BACKUP.zip \
  --target data/private-restored
```

Программа отказывается писать в непустую папку и блокирует path traversal внутри ZIP. Рабочие базы автоматически не перезаписываются.

## Комплект передачи

- `templates/client-questionnaire.md`;
- `templates/acceptance-criteria.md`;
- `templates/handoff-checklist.md`;
- `templates/incident-rollback.md`;
- `docs/portfolio-case-study.md`.

## Критерии PASS/FAIL

- неправильный или отсутствующий API-ключ даёт HTTP 401 без раскрытия секрета;
- health check остаётся доступным;
- backup восстанавливается с совпадающими контрольными суммами;
- непустая папка назначения и небезопасный ZIP отклоняются;
- все тесты и evaluation проходят;
- инструкция rollback не уничтожает единственную рабочую копию.
