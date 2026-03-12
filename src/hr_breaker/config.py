import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values, load_dotenv, set_key, unset_key
from pydantic import AliasChoices, Field
from pydantic_ai_litellm import LiteLLMModel
from pydantic_settings import BaseSettings

import litellm

from hr_breaker import litellm_patch

load_dotenv()

litellm.suppress_debug_info = True
litellm_patch.apply()

ProviderType = Literal["gemini", "openai_compatible"]
LLMRole = Literal["pro", "flash", "embedding"]

GEMINI_PROVIDER: ProviderType = "gemini"
OPENAI_COMPATIBLE_PROVIDER: ProviderType = "openai_compatible"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PRO_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_FLASH_MODEL = "gemini/gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL = "gemini/text-embedding-004"
ENV_FILE_PATH = Path(__file__).resolve().parents[2] / ".env"
PERSISTED_LLM_FIELD_NAMES = (
    "llm_separate_models",
    "llm_separate_embeddings",
    "llm_shared_provider",
    "llm_shared_api_key",
    "llm_shared_base_url",
    "pro_provider",
    "pro_api_key",
    "pro_base_url",
    "pro_model",
    "flash_provider",
    "flash_api_key",
    "flash_base_url",
    "flash_model",
    "embedding_provider",
    "embedding_api_key",
    "embedding_base_url",
    "embedding_model",
)
LEGACY_LLM_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")



def setup_logging() -> logging.Logger:
    general_level = os.getenv("LOG_LEVEL_GENERAL", "WARNING").upper()
    project_level = os.getenv("LOG_LEVEL", "WARNING").upper()

    logging.basicConfig(
        level=getattr(logging, general_level, logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    project_logger = logging.getLogger("hr_breaker")
    project_logger.setLevel(getattr(logging, project_level, logging.WARNING))
    return project_logger


logger = setup_logging()


@dataclass(frozen=True)
class ResolvedLLMConfig:
    role: LLMRole
    provider: ProviderType
    model_name: str
    api_key: str | None
    api_base: str | None

    @property
    def custom_llm_provider(self) -> str | None:
        return "openai" if self.provider == OPENAI_COMPATIBLE_PROVIDER else None

    def create_model(self) -> LiteLLMModel:
        return LiteLLMModel(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            custom_llm_provider=self.custom_llm_provider,
        )

    def as_embedding_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self.custom_llm_provider:
            kwargs["custom_llm_provider"] = self.custom_llm_provider
            kwargs["encoding_format"] = "float"
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs


class Settings(BaseSettings):
    """Application settings. Reads from env vars (uppercased field names)."""

    # API keys (accepts GOOGLE_API_KEY as fallback for backward compat with Gemini)
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    moonshot_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MOONSHOT_API_KEY"),
    )

    llm_separate_models: bool = False
    llm_separate_embeddings: bool = False
    llm_shared_provider: ProviderType = GEMINI_PROVIDER
    llm_shared_api_key: str | None = None
    llm_shared_base_url: str | None = None

    pro_provider: ProviderType | None = None
    pro_api_key: str | None = None
    pro_base_url: str | None = None
    pro_model: str = DEFAULT_PRO_MODEL

    flash_provider: ProviderType | None = None
    flash_api_key: str | None = None
    flash_base_url: str | None = None
    flash_model: str = DEFAULT_FLASH_MODEL

    reasoning_effort: str = "medium"
    cache_dir: Path = Path(".cache/resumes")
    profile_dir: Path = Path(".cache/profiles")
    output_dir: Path = Path("output")
    max_iterations: int = 5
    pass_threshold: float = 0.7
    fast_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices("fast_mode", "HR_BREAKER_FAST_MODE"),
    )
    sequential: bool = Field(
        default=False,
        validation_alias=AliasChoices("sequential", "HR_BREAKER_SEQ"),
    )
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("debug", "HR_BREAKER_DEBUG"),
    )
    no_shame: bool = Field(
        default=False,
        validation_alias=AliasChoices("no_shame", "HR_BREAKER_NO_SHAME"),
    )

    # Scraper settings
    scraper_httpx_timeout: float = 15.0
    scraper_wayback_timeout: float = 10.0
    scraper_playwright_timeout: int = 30000
    scraper_httpx_max_retries: int = 3
    scraper_wayback_max_age_days: int = 30
    scraper_min_text_length: int = 200

    # Filter thresholds
    filter_hallucination_threshold: float = 0.9
    filter_keyword_threshold: float = 0.25
    filter_llm_threshold: float = 0.7
    filter_vector_threshold: float = 0.4
    filter_ai_generated_threshold: float = 0.4
    filter_translation_threshold: float = 0.95

    # Resume length limits
    resume_max_chars: int = 4500
    resume_max_words: int = 520
    resume_page2_overflow_chars: int = 1000

    # Keyword matcher params
    keyword_tfidf_max_features: int = 200
    keyword_tfidf_cutoff: float = 0.1
    keyword_max_missing_display: int = 10

    # Embedding settings
    embedding_provider: ProviderType | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_output_dimensionality: int = 768

    # Profile archive settings
    profile_retrieval_top_k: int = 4
    profile_source_max_chars: int = 12000
    profile_snippet_max_chars: int = 700

    # Agent limits
    agent_name_extractor_chars: int = 2000

    # Language settings
    default_language: str = "en"

    # Retry settings
    retry_max_attempts: int = 5
    retry_max_wait: float = 60.0

    def model_post_init(self, __context: Any) -> None:
        if self.gemini_api_key and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        if self.moonshot_api_key and "MOONSHOT_API_KEY" not in os.environ:
            os.environ["MOONSHOT_API_KEY"] = self.moonshot_api_key


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stringify_env_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _normalize_string(value)


def _infer_provider_from_model_name(model_name: str | None) -> ProviderType | None:
    if not model_name:
        return None
    if model_name.startswith("gemini/") or model_name.startswith("models/"):
        return GEMINI_PROVIDER
    if model_name.startswith("openai/"):
        return OPENAI_COMPATIBLE_PROVIDER
    return None


def _normalize_model_name(provider: ProviderType, model_name: str, role: LLMRole) -> str:
    normalized = model_name.strip()
    if not normalized:
        return normalized
    if role == "embedding" and provider == OPENAI_COMPATIBLE_PROVIDER:
        return normalized.removeprefix("openai/")
    if normalized.startswith("gemini/") or normalized.startswith("openai/"):
        return normalized
    if provider == GEMINI_PROVIDER:
        if normalized.startswith("models/"):
            normalized = normalized.removeprefix("models/")
        return f"gemini/{normalized}"
    return f"openai/{normalized}"


def use_separate_embedding_settings(settings: Settings | None = None) -> bool:
    resolved = settings or get_settings()
    if resolved.llm_separate_models or resolved.llm_separate_embeddings:
        return True

    shared_provider = resolved.llm_shared_provider or GEMINI_PROVIDER
    shared_api_key = _normalize_string(resolved.llm_shared_api_key)
    shared_base = _normalize_string(resolved.llm_shared_base_url)

    if resolved.embedding_provider and resolved.embedding_provider != shared_provider:
        return True
    if _normalize_string(resolved.embedding_api_key) and _normalize_string(resolved.embedding_api_key) != shared_api_key:
        return True
    if shared_provider == OPENAI_COMPATIBLE_PROVIDER:
        embedding_base = _normalize_string(resolved.embedding_base_url)
        if embedding_base and embedding_base != shared_base:
            return True

    return False


def _resolve_provider(settings: Settings, role: LLMRole) -> ProviderType:
    shared_provider = settings.llm_shared_provider or GEMINI_PROVIDER
    if role == "embedding":
        if not use_separate_embedding_settings(settings):
            return shared_provider
        if settings.embedding_provider:
            return settings.embedding_provider
        return _infer_provider_from_model_name(settings.embedding_model) or shared_provider

    if settings.llm_separate_models:
        explicit_provider = getattr(settings, f"{role}_provider")
        model_name = getattr(settings, f"{role}_model")
        return explicit_provider or _infer_provider_from_model_name(model_name) or shared_provider

    return shared_provider


def _resolve_api_key(settings: Settings, role: LLMRole, provider: ProviderType) -> str | None:
    shared_api_key = _normalize_string(settings.llm_shared_api_key)

    if role == "embedding":
        if use_separate_embedding_settings(settings):
            return (
                _normalize_string(settings.embedding_api_key)
                or (settings.gemini_api_key if provider == GEMINI_PROVIDER else None)
            )
        return shared_api_key or (settings.gemini_api_key if provider == GEMINI_PROVIDER else None)

    if settings.llm_separate_models:
        return (
            _normalize_string(getattr(settings, f"{role}_api_key"))
            or shared_api_key
            or (settings.gemini_api_key if provider == GEMINI_PROVIDER else None)
        )

    return shared_api_key or (settings.gemini_api_key if provider == GEMINI_PROVIDER else None)


def _resolve_api_base(settings: Settings, role: LLMRole, provider: ProviderType) -> str | None:
    if provider == GEMINI_PROVIDER:
        return None

    if role == "embedding":
        candidate = (
            settings.embedding_base_url if use_separate_embedding_settings(settings) else settings.llm_shared_base_url
        )
    elif settings.llm_separate_models:
        candidate = getattr(settings, f"{role}_base_url") or settings.llm_shared_base_url
    else:
        candidate = settings.llm_shared_base_url

    return _normalize_string(candidate)


def clear_settings_cache() -> None:
    get_settings.cache_clear()
    # Also clear agent caches so they pick up new model config.
    # Lazy imports avoid circular dependencies at module load time.
    try:
        from hr_breaker.agents.job_parser import get_job_parser_agent
        from hr_breaker.agents.combined_reviewer import get_combined_reviewer_agent
        from hr_breaker.agents.ai_generated_detector import get_ai_generated_agent
        get_job_parser_agent.cache_clear()
        get_combined_reviewer_agent.cache_clear()
        get_ai_generated_agent.cache_clear()
    except Exception:
        pass


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_resolved_llm_config(role: LLMRole) -> ResolvedLLMConfig:
    settings = get_settings()
    provider = _resolve_provider(settings, role)
    model_name = _normalize_model_name(provider, getattr(settings, f"{role}_model"), role)
    return ResolvedLLMConfig(
        role=role,
        provider=provider,
        model_name=model_name,
        api_key=_resolve_api_key(settings, role, provider),
        api_base=_resolve_api_base(settings, role, provider),
    )


def get_pro_llm_config() -> ResolvedLLMConfig:
    return get_resolved_llm_config("pro")


def get_flash_llm_config() -> ResolvedLLMConfig:
    return get_resolved_llm_config("flash")


def get_embedding_llm_config() -> ResolvedLLMConfig:
    return get_resolved_llm_config("embedding")


def get_pro_model() -> LiteLLMModel:
    return get_pro_llm_config().create_model()


def get_flash_model() -> LiteLLMModel:
    return get_flash_llm_config().create_model()
def get_embedding_dimensions() -> int | None:
    config = get_embedding_llm_config()
    if config.provider == OPENAI_COMPATIBLE_PROVIDER:
        return None
    return get_settings().embedding_output_dimensionality




def get_embedding_request_kwargs() -> dict[str, Any]:
    return get_embedding_llm_config().as_embedding_kwargs()


def get_model_settings() -> dict[str, Any] | None:
    """Get model settings with reasoning effort config."""
    settings = get_settings()
    if settings.reasoning_effort and settings.reasoning_effort != "none":
        return {"reasoning_effort": settings.reasoning_effort}
    return None


PERSISTED_UI_FIELD_NAMES = ("sequential", "debug", "no_shame", "default_language", "max_iterations")

_UI_FIELD_ENV_KEYS: dict[str, str] = {
    "sequential": "HR_BREAKER_SEQ",
    "debug": "HR_BREAKER_DEBUG",
    "no_shame": "HR_BREAKER_NO_SHAME",
    "default_language": "DEFAULT_LANGUAGE",
    "max_iterations": "MAX_ITERATIONS",
}


def persist_ui_settings(values: Mapping[str, Any], *, env_file: Path | None = None) -> None:
    target = env_file or ENV_FILE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    existing_values = dotenv_values(target)

    for field_name, env_key in _UI_FIELD_ENV_KEYS.items():
        if field_name not in values:
            continue
        serialized = _stringify_env_value(values[field_name])
        if serialized is None:
            if env_key in existing_values:
                unset_key(target, env_key, quote_mode="auto")
                existing_values.pop(env_key, None)
            os.environ.pop(env_key, None)
        else:
            set_key(target, env_key, serialized, quote_mode="auto")
            existing_values[env_key] = serialized
            os.environ[env_key] = serialized

    clear_settings_cache()


def persist_llm_settings(
    values: Mapping[str, Any], *, env_file: Path | None = None, remove_legacy: bool = False
) -> None:
    target = env_file or ENV_FILE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    existing_values = dotenv_values(target)

    for field_name in PERSISTED_LLM_FIELD_NAMES:
        if field_name not in values:
            continue

        env_key = field_name.upper()
        serialized = _stringify_env_value(values[field_name])
        if serialized is None:
            if env_key in existing_values:
                unset_key(target, env_key, quote_mode="auto")
                existing_values.pop(env_key, None)
            os.environ.pop(env_key, None)
        else:
            set_key(target, env_key, serialized, quote_mode="auto")
            existing_values[env_key] = serialized
            os.environ[env_key] = serialized

    if remove_legacy:
        for env_key in LEGACY_LLM_ENV_KEYS:
            if env_key in existing_values:
                unset_key(target, env_key, quote_mode="auto")
                existing_values.pop(env_key, None)
            os.environ.pop(env_key, None)
    clear_settings_cache()