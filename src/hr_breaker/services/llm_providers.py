from dataclasses import dataclass
from typing import Any, Literal

import httpx

from hr_breaker.config import (
    DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    GEMINI_PROVIDER,
    OPENAI_COMPATIBLE_PROVIDER,
    ProviderType,
)

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_MODELS_PATH = "/models"
REQUEST_TIMEOUT_SECONDS = 10.0
CHAT_METHODS = {"generateContent", "streamGenerateContent"}
EMBEDDING_METHODS = {"embedContent", "batchEmbedContents"}
STATUS_UNKNOWN = "unknown"
STATUS_WARNING = "warning"
STATUS_CONNECTED = "connected"
StatusState = Literal["unknown", "warning", "connected"]
_PROVIDER_LABELS: dict[ProviderType, str] = {
    GEMINI_PROVIDER: "Gemini API",
    OPENAI_COMPATIBLE_PROVIDER: "OpenAI-compatible API",
}


@dataclass(frozen=True)
class ModelOption:
    provider: ProviderType
    value: str
    label: str


@dataclass(frozen=True)
class ProviderConnectionStatus:
    state: StatusState
    message: str
    detail: str | None = None

    @property
    def color(self) -> str:
        return {
            STATUS_UNKNOWN: "gray",
            STATUS_WARNING: "orange",
            STATUS_CONNECTED: "green",
        }[self.state]


@dataclass(frozen=True)
class ProviderCatalog:
    provider: ProviderType
    status: ProviderConnectionStatus
    chat_models: tuple[ModelOption, ...]
    embedding_models: tuple[ModelOption, ...]


def get_provider_options() -> tuple[ProviderType, ...]:
    return tuple(_PROVIDER_LABELS)


def get_provider_label(provider: ProviderType) -> str:
    return _PROVIDER_LABELS[provider]


def fetch_provider_catalog(
    provider: ProviderType, api_key: str | None, base_url: str | None = None
) -> ProviderCatalog:
    normalized_api_key = (api_key or "").strip()
    normalized_base_url = _normalize_base_url(provider, base_url)

    if not normalized_api_key:
        return ProviderCatalog(
            provider=provider,
            status=ProviderConnectionStatus(
                STATUS_UNKNOWN,
                "Enter an API key to load models.",
            ),
            chat_models=(),
            embedding_models=(),
        )

    try:
        if provider == GEMINI_PROVIDER:
            payload = _fetch_json(
                GEMINI_MODELS_URL,
                provider=provider,
                api_key=normalized_api_key,
                params={"key": normalized_api_key},
            )
            chat_models, embedding_models = _parse_gemini_models(payload.get("models", []))
        else:
            payload = _fetch_json(
                f"{normalized_base_url}{OPENAI_MODELS_PATH}",
                provider=provider,
                api_key=normalized_api_key,
            )
            chat_models, embedding_models = _parse_openai_models(payload.get("data", []))
    except httpx.HTTPStatusError as exc:
        return ProviderCatalog(
            provider=provider,
            status=ProviderConnectionStatus(
                STATUS_WARNING,
                f"Connection failed ({exc.response.status_code}).",
                detail=_extract_error_detail(exc.response),
            ),
            chat_models=(),
            embedding_models=(),
        )
    except (httpx.RequestError, ValueError) as exc:
        return ProviderCatalog(
            provider=provider,
            status=ProviderConnectionStatus(
                STATUS_WARNING,
                "Connection failed.",
                detail=str(exc),
            ),
            chat_models=(),
            embedding_models=(),
        )

    return ProviderCatalog(
        provider=provider,
        status=ProviderConnectionStatus(
            STATUS_CONNECTED,
            f"Connected · {len(chat_models)} chat / {len(embedding_models)} embedding models",
        ),
        chat_models=tuple(chat_models),
        embedding_models=tuple(embedding_models),
    )


def _fetch_json(
    url: str, *, provider: ProviderType, api_key: str, params: dict[str, Any] | None = None
 ) -> dict[str, Any]:
    headers = {
        "User-Agent": "hr-breaker/llm-provider-settings",
    }
    if api_key:
        if provider == GEMINI_PROVIDER:
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.get(
        url,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Provider response is not a JSON object")
    return payload


def _parse_gemini_models(models: list[dict[str, Any]]) -> tuple[list[ModelOption], list[ModelOption]]:
    chat_models: list[ModelOption] = []
    embedding_models: list[ModelOption] = []

    for model in models:
        raw_name = model.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            continue

        methods = {method for method in model.get("supportedGenerationMethods", []) if isinstance(method, str)}
        label = model.get("displayName") or raw_name.removeprefix("models/")
        option = ModelOption(
            provider=GEMINI_PROVIDER,
            value=f"gemini/{raw_name.removeprefix('models/')}",
            label=str(label),
        )

        if methods & CHAT_METHODS:
            chat_models.append(option)
        if methods & EMBEDDING_METHODS:
            embedding_models.append(option)

    return _sorted_model_options(chat_models), _sorted_model_options(embedding_models)


def _parse_openai_models(models: list[dict[str, Any]]) -> tuple[list[ModelOption], list[ModelOption]]:
    chat_options: list[ModelOption] = []
    embedding_options: list[ModelOption] = []
    fallback_embedding_options: list[ModelOption] = []
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        chat_option = ModelOption(
            provider=OPENAI_COMPATIBLE_PROVIDER,
            value=f"openai/{model_id}",
            label=model_id,
        )
        raw_embedding_option = ModelOption(
            provider=OPENAI_COMPATIBLE_PROVIDER,
            value=model_id,
            label=model_id,
        )
        fallback_embedding_options.append(raw_embedding_option)
        if _looks_like_embedding_model(model_id):
            embedding_options.append(raw_embedding_option)
        else:
            chat_options.append(chat_option)

    chat_models = _sorted_model_options(chat_options)
    embedding_models = _sorted_model_options(embedding_options)
    if not chat_models:
        chat_models = _sorted_model_options([
            ModelOption(
                provider=OPENAI_COMPATIBLE_PROVIDER,
                value=f"openai/{option.value}",
                label=option.label,
            )
            for option in fallback_embedding_options
        ])
    if not embedding_models:
        embedding_models = _sorted_model_options(fallback_embedding_options)

    return chat_models, embedding_models


def _sorted_model_options(options: list[ModelOption]) -> list[ModelOption]:
    deduped = {option.value: option for option in options}
    return sorted(deduped.values(), key=lambda option: option.label.lower())


def _looks_like_embedding_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "embed" in normalized or "embedding" in normalized


def _normalize_base_url(provider: ProviderType, base_url: str | None) -> str:
    if provider == GEMINI_PROVIDER:
        return GEMINI_MODELS_URL

    normalized = (base_url or "").strip() or DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    return normalized.rstrip("/")


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message

    return response.text
