from dotenv import dotenv_values
import pytest

from hr_breaker import config


TRACKED_ENV_KEYS = {
    *(field.upper() for field in config.PERSISTED_LLM_FIELD_NAMES),
    *config.LEGACY_LLM_ENV_KEYS,
}


@pytest.fixture(autouse=True)
def isolate_llm_settings(monkeypatch):
    for key in TRACKED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    config.clear_settings_cache()
    yield
    config.clear_settings_cache()


def test_shared_provider_resolution(monkeypatch):
    monkeypatch.setenv("LLM_SHARED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_SHARED_API_KEY", "shared-key")
    monkeypatch.setenv("LLM_SHARED_BASE_URL", "https://compat.example/v1")
    monkeypatch.setenv("PRO_MODEL", "gpt-4.1")
    monkeypatch.setenv("FLASH_MODEL", "meta/llama-3.1-8b-instruct")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    pro = config.get_pro_llm_config()
    flash = config.get_flash_llm_config()
    embedding = config.get_embedding_llm_config()

    assert pro.provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert pro.model_name == "openai/gpt-4.1"
    assert pro.api_key == "shared-key"
    assert pro.api_base == "https://compat.example/v1"

    assert flash.provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert flash.model_name == "openai/meta/llama-3.1-8b-instruct"
    assert flash.api_key == "shared-key"
    assert flash.api_base == "https://compat.example/v1"

    assert embedding.provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert embedding.model_name == "text-embedding-3-small"
    assert embedding.api_key == "shared-key"
    assert embedding.api_base == "https://compat.example/v1"


def test_separate_provider_resolution(monkeypatch):
    monkeypatch.setenv("LLM_SEPARATE_MODELS", "true")
    monkeypatch.setenv("LLM_SHARED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PROVIDER_UNUSED", "ignored")
    monkeypatch.setenv("PRO_PROVIDER", "gemini")
    monkeypatch.setenv("PRO_API_KEY", "gem-key")
    monkeypatch.setenv("PRO_MODEL", "models/gemini-2.5-pro")
    monkeypatch.setenv("FLASH_PROVIDER", "openai_compatible")
    monkeypatch.setenv("FLASH_API_KEY", "flash-key")
    monkeypatch.setenv("FLASH_BASE_URL", "https://flash.example/v1")
    monkeypatch.setenv("FLASH_MODEL", "meta/llama-3.1-70b-instruct")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    pro = config.get_pro_llm_config()
    flash = config.get_flash_llm_config()
    embedding = config.get_embedding_llm_config()

    assert pro.provider == config.GEMINI_PROVIDER
    assert pro.model_name == "gemini/gemini-2.5-pro"
    assert pro.api_key == "gem-key"
    assert pro.api_base is None

    assert flash.provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert flash.model_name == "openai/meta/llama-3.1-70b-instruct"
    assert flash.api_key == "flash-key"
    assert flash.api_base == "https://flash.example/v1"

    assert embedding.provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert embedding.model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert embedding.api_key == "embed-key"
    assert embedding.api_base == "https://embed.example/v1"


def test_embeddings_can_be_separated_without_full_model_split(monkeypatch):
    monkeypatch.setenv("LLM_SHARED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_SHARED_API_KEY", "shared-key")
    monkeypatch.setenv("LLM_SHARED_BASE_URL", "https://compat.example/v1")
    monkeypatch.setenv("PRO_MODEL", "gpt-4.1")
    monkeypatch.setenv("FLASH_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_SEPARATE_EMBEDDINGS", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/embed-qa-4")

    pro = config.get_pro_llm_config()
    flash = config.get_flash_llm_config()
    embedding = config.get_embedding_llm_config()

    assert pro.model_name == "openai/gpt-4.1"
    assert pro.api_base == "https://compat.example/v1"
    assert flash.model_name == "openai/gpt-4.1-mini"
    assert flash.api_base == "https://compat.example/v1"
    assert embedding.model_name == "nvidia/embed-qa-4"
    assert embedding.api_key == "embed-key"
    assert embedding.api_base == "https://embed.example/v1"


def test_legacy_gemini_key_is_migration_input(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")

    pro = config.get_pro_llm_config()
    embedding = config.get_embedding_llm_config()

    assert pro.provider == config.GEMINI_PROVIDER
    assert pro.api_key == "legacy-key"
    assert embedding.api_key == "legacy-key"


def test_persist_llm_settings_writes_expected_env_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=legacy-key\nPRO_MODEL=gemini/old\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")

    config.persist_llm_settings(
        {
            "llm_separate_models": False,
            "llm_separate_embeddings": True,
            "llm_shared_provider": "openai_compatible",
            "llm_shared_api_key": "shared-key",
            "llm_shared_base_url": "https://compat.example/v1",
            "pro_model": "gpt-4.1",
            "flash_model": "gpt-4.1-mini",
            "embedding_provider": "openai_compatible",
            "embedding_api_key": "embed-key",
            "embedding_base_url": "https://embed.example/v1",
            "embedding_model": "text-embedding-3-small",
        },
        env_file=env_file,
        remove_legacy=True,
    )

    values = dotenv_values(env_file)
    assert values["LLM_SEPARATE_MODELS"] == "false"
    assert values["LLM_SEPARATE_EMBEDDINGS"] == "true"
    assert values["LLM_SHARED_PROVIDER"] == "openai_compatible"
    assert values["LLM_SHARED_API_KEY"] == "shared-key"
    assert values["LLM_SHARED_BASE_URL"] == "https://compat.example/v1"
    assert values["PRO_MODEL"] == "gpt-4.1"
    assert values["FLASH_MODEL"] == "gpt-4.1-mini"
    assert values["EMBEDDING_PROVIDER"] == "openai_compatible"
    assert values["EMBEDDING_API_KEY"] == "embed-key"
    assert values["EMBEDDING_BASE_URL"] == "https://embed.example/v1"
    assert values["EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert "GEMINI_API_KEY" not in values

    assert config.get_settings().llm_shared_provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert config.get_settings().llm_shared_api_key == "shared-key"
    assert "GEMINI_API_KEY" not in config.os.environ


def test_persist_llm_settings_clears_cached_settings(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.touch()

    monkeypatch.setenv("LLM_SHARED_PROVIDER", "gemini")
    first = config.get_settings()
    assert first.llm_shared_provider == config.GEMINI_PROVIDER

    config.persist_llm_settings(
        {
            "llm_shared_provider": "openai_compatible",
            "llm_shared_api_key": "fresh-key",
        },
        env_file=env_file,
    )

    second = config.get_settings()
    assert second.llm_shared_provider == config.OPENAI_COMPATIBLE_PROVIDER
    assert second.llm_shared_api_key == "fresh-key"
    assert second is not first


def test_embedding_dimensions_omitted_for_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/embed-qa-4")

    assert config.get_embedding_dimensions() is None


def test_embedding_dimensions_preserved_for_gemini_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-004")

    assert config.get_embedding_dimensions() == config.get_settings().embedding_output_dimensionality

def test_openai_prefix_is_stripped_for_openai_compatible_embeddings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/nvidia/embed-qa-4")

    embedding = config.get_embedding_llm_config()

    assert embedding.model_name == "nvidia/embed-qa-4"


def test_openai_prefix_is_preserved_for_openai_compatible_chat_models(monkeypatch):
    monkeypatch.setenv("LLM_SHARED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_SHARED_API_KEY", "shared-key")
    monkeypatch.setenv("PRO_MODEL", "openai/gpt-5")

    pro = config.get_pro_llm_config()

    assert pro.model_name == "openai/gpt-5"

def test_embedding_request_kwargs_include_openai_provider_context(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/embed-qa-4")

    kwargs = config.get_embedding_request_kwargs()

    assert kwargs["model"] == "nvidia/embed-qa-4"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["encoding_format"] == "float"
    assert kwargs["api_base"] == "https://embed.example/v1"


def test_embedding_request_kwargs_omit_provider_context_for_gemini(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-004")

    kwargs = config.get_embedding_request_kwargs()

    assert kwargs["model"] == "gemini/text-embedding-004"
    assert "custom_llm_provider" not in kwargs
    assert "encoding_format" not in kwargs