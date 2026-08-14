"""Минимальный read-only клиент официального WB API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ANALYTICS_BASE_URL = "https://seller-analytics-api.wildberries.ru"


class WbApiError(RuntimeError):
    """Безопасная ошибка интеграции без раскрытия токена."""


@dataclass(frozen=True)
class WbApiClient:
    """Клиент методов аналитики WB с минимальными правами."""

    token: str
    base_url: str = ANALYTICS_BASE_URL
    timeout: float = 20
    opener: Callable[..., Any] = urlopen

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("WB_API_TOKEN не настроен")
        if not self.base_url.startswith("https://"):
            raise ValueError("WB API должен использовать HTTPS")

    def sales_funnel(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Получить статистику карточек за период из воронки продаж."""

        return self._request(
            "POST", "/api/analytics/v3/sales-funnel/products", payload
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": self.token,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ai-agent-service-lab/0.1",
            },
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            if error.code in (401, 403):
                raise WbApiError(
                    "WB отклонил токен или у него нет доступа к аналитике"
                ) from error
            if error.code == 429:
                raise WbApiError("Превышен лимит запросов WB API") from error
            raise WbApiError(f"WB API вернул ошибку HTTP {error.code}") from error
        except (OSError, URLError, ValueError) as error:
            raise WbApiError("Не удалось получить корректный ответ WB API") from error

        if not isinstance(result, dict):
            raise WbApiError("WB API вернул неожиданный формат ответа")
        return result
