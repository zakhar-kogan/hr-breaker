import httpx
from unittest.mock import patch

from hr_breaker.config import GEMINI_PROVIDER, OPENAI_COMPATIBLE_PROVIDER
from hr_breaker.services.llm_providers import (
    ProviderConnectionStatus,
    STATUS_CONNECTED,
    STATUS_UNKNOWN,
    STATUS_WARNING,
    fetch_provider_catalog,
)


def _json_response(url: str, payload: dict, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", url)
    return httpx.Response(status_code, json=payload, request=request)


def test_fetch_provider_catalog_parses_gemini_models():
    response = _json_response(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/text-embedding-004",
                    "displayName": "Text Embedding 004",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
                },
            ]
        },
    )

    with patch("hr_breaker.services.llm_providers.httpx.get", return_value=response):
        catalog = fetch_provider_catalog(GEMINI_PROVIDER, api_key="gem-key")

    assert catalog.status.state == STATUS_CONNECTED
    assert catalog.status.color == "green"
    assert [option.value for option in catalog.chat_models] == [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-pro",
    ]
    assert [option.value for option in catalog.embedding_models] == [
        "gemini/text-embedding-004"
    ]


def test_fetch_provider_catalog_uses_gemini_api_key_header():
    response = _json_response(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {"models": []},
    )

    with patch("hr_breaker.services.llm_providers.httpx.get", return_value=response) as mocked_get:
        fetch_provider_catalog(GEMINI_PROVIDER, api_key="studio-key")

    kwargs = mocked_get.call_args.kwargs
    assert kwargs["headers"]["x-goog-api-key"] == "studio-key"
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["params"] == {"key": "studio-key"}

def test_fetch_provider_catalog_parses_openai_compatible_models():
    response = _json_response(
        "https://compat.example/v1/models",
        {
            "data": [
                {"id": "meta/llama-3.1-70b-instruct"},
                {"id": "text-embedding-3-small"},
                {"id": "gpt-4.1-mini"},
            ]
        },
    )

    with patch("hr_breaker.services.llm_providers.httpx.get", return_value=response):
        catalog = fetch_provider_catalog(
            OPENAI_COMPATIBLE_PROVIDER,
            api_key="compat-key",
            base_url="https://compat.example/v1/",
        )

    assert catalog.status.state == STATUS_CONNECTED
    assert [option.value for option in catalog.chat_models] == [
        "openai/gpt-4.1-mini",
        "openai/meta/llama-3.1-70b-instruct",
    ]
    assert [option.value for option in catalog.embedding_models] == [
        "text-embedding-3-small"
    ]

def test_fetch_provider_catalog_keeps_raw_embedding_ids_for_openai_compatible_models():
    response = _json_response(
        "https://compat.example/v1/models",
        {
            "data": [
                {"id": "nvidia/embed-qa-4"},
                {"id": "nvidia/nv-embed-v1"},
                {"id": "gpt-5.2"},
            ]
        },
    )

    with patch("hr_breaker.services.llm_providers.httpx.get", return_value=response):
        catalog = fetch_provider_catalog(
            OPENAI_COMPATIBLE_PROVIDER,
            api_key="compat-key",
            base_url="https://compat.example/v1/",
        )

    assert [option.value for option in catalog.embedding_models] == [
        "nvidia/embed-qa-4",
        "nvidia/nv-embed-v1",
    ]
    assert [option.value for option in catalog.chat_models] == ["openai/gpt-5.2"]


def test_fetch_provider_catalog_returns_warning_for_http_errors():
    response = _json_response(
        "https://compat.example/v1/models",
        {"error": {"message": "Bad API key"}},
        status_code=401,
    )

    with patch("hr_breaker.services.llm_providers.httpx.get", return_value=response):
        catalog = fetch_provider_catalog(
            OPENAI_COMPATIBLE_PROVIDER,
            api_key="bad-key",
            base_url="https://compat.example/v1",
        )

    assert catalog.status.state == STATUS_WARNING
    assert catalog.status.color == "orange"
    assert catalog.status.detail == "Bad API key"
    assert catalog.chat_models == ()
    assert catalog.embedding_models == ()


def test_fetch_provider_catalog_returns_unknown_without_api_key():
    catalog = fetch_provider_catalog(GEMINI_PROVIDER, api_key=None)

    assert catalog.status.state == STATUS_UNKNOWN
    assert catalog.status.color == "gray"
    assert catalog.chat_models == ()
    assert catalog.embedding_models == ()


def test_provider_connection_status_color_mapping():
    assert ProviderConnectionStatus(STATUS_UNKNOWN, "waiting").color == "gray"
    assert ProviderConnectionStatus(STATUS_WARNING, "problem").color == "orange"
    assert ProviderConnectionStatus(STATUS_CONNECTED, "ok").color == "green"
