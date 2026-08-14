import io
import json
from urllib.error import HTTPError

import pytest

from agent_lab.wb_api import WbApiClient, WbApiError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_sales_funnel_uses_authorization_without_exposing_token() -> None:
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(json.dumps({"data": {"products": []}}).encode())

    client = WbApiClient(token="secret", opener=opener)
    result = client.sales_funnel({"period": {"start": "2026-08-01"}})

    request = captured["request"]
    assert request.full_url.endswith("/api/analytics/v3/sales-funnel/products")
    assert request.get_header("Authorization") == "secret"
    assert json.loads(request.data) == {"period": {"start": "2026-08-01"}}
    assert result == {"data": {"products": []}}


def test_client_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="не настроен"):
        WbApiClient(token="")


def test_auth_error_does_not_contain_token() -> None:
    def opener(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    client = WbApiClient(token="secret", opener=opener)
    with pytest.raises(WbApiError) as error:
        client.sales_funnel({})

    assert "secret" not in str(error.value)
    assert "токен" in str(error.value)
