"""Sidebar UI — LLM settings, options, and PDF history."""

import subprocess
import sys

import streamlit as st

from hr_breaker.config import (
    OPENAI_COMPATIBLE_PROVIDER,
    PERSISTED_LLM_FIELD_NAMES,
    ProviderType,
    get_embedding_llm_config,
    get_flash_llm_config,
    get_pro_llm_config,
    get_settings,
    persist_llm_settings,
    persist_ui_settings,
    use_separate_embedding_settings,
)
from hr_breaker.models import SUPPORTED_LANGUAGES, Profile, get_language
from hr_breaker.services import (
    PDFStorage,
    ProfileStore,
    fetch_provider_catalog,
    get_provider_label,
    get_provider_options,
)


@st.cache_data(show_spinner=False, ttl=60)
def cached_provider_catalog(
    provider: ProviderType, api_key: str | None, base_url: str | None
):
    return fetch_provider_catalog(provider, api_key, base_url)


# --- LLM persistence helpers ---

def build_llm_persistence_payload() -> dict[str, object]:
    payload = {field: st.session_state.get(field) for field in PERSISTED_LLM_FIELD_NAMES}
    if not st.session_state.get("llm_separate_models") and not st.session_state.get("llm_separate_embeddings"):
        shared_provider = st.session_state.get("llm_shared_provider")
        payload["embedding_provider"] = shared_provider
        payload["embedding_api_key"] = st.session_state.get("llm_shared_api_key")
        payload["embedding_base_url"] = (
            st.session_state.get("llm_shared_base_url")
            if shared_provider == OPENAI_COMPATIBLE_PROVIDER
            else None
        )
    return payload


def persist_llm_sidebar_state() -> None:
    payload = build_llm_persistence_payload()
    has_new_style_key = bool(
        payload.get("llm_shared_api_key")
        or payload.get("pro_api_key")
        or payload.get("flash_api_key")
    )
    persist_llm_settings(payload, remove_legacy=has_new_style_key)
    cached_provider_catalog.clear()


def initialize_ui_sidebar_state() -> None:
    settings = get_settings()
    defaults = {
        "ui_sequential": settings.sequential,
        "ui_debug": settings.debug,
        "ui_no_shame": settings.no_shame,
        "ui_language": settings.default_language,
        "ui_max_iterations": settings.max_iterations,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def persist_ui_sidebar_state() -> None:
    persist_ui_settings({
        "sequential": st.session_state.get("ui_sequential", False),
        "debug": st.session_state.get("ui_debug", False),
        "no_shame": st.session_state.get("ui_no_shame", False),
        "default_language": st.session_state.get("ui_language", "en"),
        "max_iterations": st.session_state.get("ui_max_iterations", 5),
    })


def initialize_llm_sidebar_state() -> None:
    settings = get_settings()
    pro_config = get_pro_llm_config()
    flash_config = get_flash_llm_config()
    embedding_config = get_embedding_llm_config()
    defaults = {
        "llm_separate_models": settings.llm_separate_models,
        "llm_separate_embeddings": use_separate_embedding_settings(settings),
        "llm_shared_provider": settings.llm_shared_provider,
        "llm_shared_api_key": settings.llm_shared_api_key or settings.gemini_api_key or "",
        "llm_shared_base_url": settings.llm_shared_base_url or "",
        "pro_provider": settings.pro_provider or pro_config.provider,
        "pro_api_key": settings.pro_api_key or pro_config.api_key or "",
        "pro_base_url": settings.pro_base_url or pro_config.api_base or "",
        "pro_model": pro_config.model_name,
        "flash_provider": settings.flash_provider or flash_config.provider,
        "flash_api_key": settings.flash_api_key or flash_config.api_key or "",
        "flash_base_url": settings.flash_base_url or flash_config.api_base or "",
        "flash_model": flash_config.model_name,
        "embedding_provider": settings.embedding_provider or embedding_config.provider,
        "embedding_api_key": settings.embedding_api_key or embedding_config.api_key or "",
        "embedding_base_url": settings.embedding_base_url or embedding_config.api_base or "",
        "embedding_model": embedding_config.model_name,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Provider / model UI widgets ---

def render_provider_status(status) -> None:
    st.caption(f"Status: :{status.color}[{status.message}]")
    if status.detail:
        st.caption(f":{status.color}[{status.detail}]")


def render_provider_inputs(section_label: str, field_prefix: str):
    provider_key = f"{field_prefix}_provider"
    api_key_key = f"{field_prefix}_api_key"
    base_url_key = f"{field_prefix}_base_url"
    st.selectbox(
        f"{section_label} provider",
        options=get_provider_options(),
        format_func=get_provider_label,
        key=provider_key,
        on_change=persist_llm_sidebar_state,
    )
    st.text_input(
        f"{section_label} API key",
        type="password",
        key=api_key_key,
        on_change=persist_llm_sidebar_state,
    )

    provider = st.session_state[provider_key]
    base_url = None
    if provider == OPENAI_COMPATIBLE_PROVIDER:
        st.text_input(
            f"{section_label} base URL",
            key=base_url_key,
            help="OpenAI-compatible API base URL. Include the /v1 suffix.",
            on_change=persist_llm_sidebar_state,
        )
        base_url = st.session_state.get(base_url_key)

    if st.button("Check API", key=f"{field_prefix}_check_api", use_container_width=True):
        cached_provider_catalog.clear()

    catalog = cached_provider_catalog(
        provider,
        st.session_state.get(api_key_key),
        base_url,
    )
    render_provider_status(catalog.status)
    return catalog


def render_model_picker(label: str, model_key: str, options) -> bool:
    option_values = [option.value for option in options]
    option_labels = {option.value: option.label for option in options}
    changed = False

    if option_values:
        current_value = st.session_state.get(model_key)
        if current_value not in option_values:
            st.session_state[model_key] = option_values[0]
            current_value = option_values[0]
            changed = True
        st.selectbox(
            label,
            options=option_values,
            index=option_values.index(current_value),
            format_func=lambda value: option_labels.get(value, value),
            key=model_key,
            on_change=persist_llm_sidebar_state,
        )
        return changed

    placeholder = st.session_state.get(model_key) or "No models available"
    st.selectbox(
        label,
        options=[placeholder],
        index=0,
        key=f"{model_key}_disabled",
        disabled=True,
    )
    return changed


# --- Full sidebar render ---

def render_sidebar(
    profiles: list[Profile],
    profile_store: ProfileStore,
    pdf_storage: PDFStorage,
    sync_profile_selection,
    sync_profile_name_draft,
    infer_profile_name_parts,
    _optional_text,
) -> tuple:
    """Render the full sidebar. Returns (active_profile, sequential_mode, debug_mode, no_shame_mode, selected_lang_code, selected_language, max_iterations)."""
    with st.sidebar:
        st.markdown("**Profile**")
        active_profile = None
        active_profile_id = st.session_state.get("active_profile_id")
        for p in profiles:
            if p.id == active_profile_id:
                active_profile = p
                break

        if profiles:
            active_options = [profile.id for profile in profiles]
            if st.session_state.get("active_profile_picker") not in active_options:
                st.session_state["active_profile_picker"] = active_options[0]
            st.selectbox(
                "Active profile",
                options=active_options,
                format_func=lambda profile_id: next(
                    profile.display_name for profile in profiles if profile.id == profile_id
                ),
                key="active_profile_picker",
            )
            st.session_state["active_profile_id"] = st.session_state["active_profile_picker"]
            for p in profiles:
                if p.id == st.session_state["active_profile_id"]:
                    active_profile = p
                    break
        else:
            st.caption("No profiles yet")

        if active_profile is not None:
            active_documents = profile_store.list_documents(active_profile.id)
            selected_count = len(sync_profile_selection(active_profile.id, active_documents))
            st.caption(
                f"{len(active_documents)} docs · {selected_count} selected · updated {active_profile.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )
            st.caption("Profiles act as reusable archive folders for local uploads and notes.")

        sync_profile_name_draft(active_profile)
        with st.expander("Manage profiles", expanded=not profiles):
            st.text_input(
                "Profile name",
                key="profile_name_draft",
                placeholder="Candidate name",
            )
            create_col, rename_col, delete_col = st.columns(3)
            with create_col:
                if st.button("Create profile", key="create_profile", use_container_width=True):
                    draft_name = _optional_text(st.session_state.get("profile_name_draft"))
                    if not draft_name:
                        st.warning("Enter a profile name first.")
                    else:
                        first_name, last_name = infer_profile_name_parts(draft_name)
                        created_profile = profile_store.create_profile(
                            draft_name,
                            first_name=first_name,
                            last_name=last_name,
                        )
                        st.session_state["pending_active_profile_id"] = created_profile.id
                        st.session_state["source_mode"] = "Profile archive"
                        st.session_state.pop("last_result", None)
                        st.rerun()
            with rename_col:
                rename_clicked = st.button(
                    "Rename",
                    key="rename_profile",
                    use_container_width=True,
                    disabled=active_profile is None,
                )
                if rename_clicked and active_profile is not None:
                    draft_name = _optional_text(st.session_state.get("profile_name_draft"))
                    if draft_name:
                        profile_store.rename_profile(active_profile.id, draft_name)
                        st.session_state.pop("last_result", None)
                        st.rerun()
            with delete_col:
                delete_clicked = st.button(
                    "Delete",
                    key="delete_profile",
                    use_container_width=True,
                    disabled=active_profile is None,
                )
                if delete_clicked and active_profile is not None:
                    profile_store.delete_profile(active_profile.id)
                    st.session_state["profile_selected_docs"].pop(active_profile.id, None)
                    st.session_state["profile_seen_doc_ids"].pop(active_profile.id, None)
                    remaining_ids = [profile.id for profile in profiles if profile.id != active_profile.id]
                    st.session_state["pending_active_profile_id"] = remaining_ids[0] if remaining_ids else None
                    if not remaining_ids:
                        st.session_state["source_mode"] = "Direct upload"
                    st.session_state.pop("last_result", None)
                    st.rerun()

        st.divider()
        st.markdown("**LLM Settings**")
        st.checkbox(
            "Separate models",
            key="llm_separate_models",
            help="Configure pro and flash providers independently.",
            on_change=persist_llm_sidebar_state,
        )

        if not st.session_state.llm_separate_models:
            st.checkbox(
                "Separate embeddings",
                key="llm_separate_embeddings",
                help="Use a dedicated embedding provider, API key, and base URL while keeping Pro/Flash shared.",
                on_change=persist_llm_sidebar_state,
            )

        llm_settings_changed = False
        if st.session_state.llm_separate_models:
            st.markdown("**Pro**")
            pro_catalog = render_provider_inputs("Pro", "pro")
            llm_settings_changed |= render_model_picker("Pro model", "pro_model", pro_catalog.chat_models)

            st.divider()
            st.markdown("**Flash**")
            flash_catalog = render_provider_inputs("Flash", "flash")
            llm_settings_changed |= render_model_picker("Flash model", "flash_model", flash_catalog.chat_models)

            st.divider()
            st.markdown("**Embeddings**")
            embedding_catalog = render_provider_inputs("Embedding", "embedding")
            llm_settings_changed |= render_model_picker("Embedding model", "embedding_model", embedding_catalog.embedding_models)
        else:
            shared_catalog = render_provider_inputs("Shared", "llm_shared")
            llm_settings_changed |= render_model_picker("Pro model", "pro_model", shared_catalog.chat_models)
            llm_settings_changed |= render_model_picker("Flash model", "flash_model", shared_catalog.chat_models)

            if st.session_state.llm_separate_embeddings:
                st.divider()
                st.markdown("**Embeddings**")
                embedding_catalog = render_provider_inputs("Embedding", "embedding")
                llm_settings_changed |= render_model_picker("Embedding model", "embedding_model", embedding_catalog.embedding_models)
            else:
                llm_settings_changed |= render_model_picker("Embedding model", "embedding_model", shared_catalog.embedding_models)

        if llm_settings_changed:
            persist_llm_sidebar_state()

        st.divider()
        st.markdown("**Options**")
        sequential_mode = st.checkbox(
            "Sequential",
            key="ui_sequential",
            on_change=persist_ui_sidebar_state,
            help="Run filters sequentially with early exit",
        )
        debug_mode = st.checkbox(
            "Debug",
            key="ui_debug",
            on_change=persist_ui_sidebar_state,
            help="Save each iteration PDF",
        )
        no_shame_mode = st.checkbox(
            "No Shame",
            key="ui_no_shame",
            on_change=persist_ui_sidebar_state,
            help="Lenient mode: allow aggressive content stretching",
        )

        _lang_options = [lang.code for lang in SUPPORTED_LANGUAGES]
        _lang_labels = {lang.code: lang.native_name for lang in SUPPORTED_LANGUAGES}
        selected_lang_code = st.selectbox(
            "Resume language",
            options=_lang_options,
            key="ui_language",
            on_change=persist_ui_sidebar_state,
            format_func=lambda code: _lang_labels[code],
            help="Output language for the final resume. Optimization runs in English, then translates.",
        )
        selected_language = get_language(selected_lang_code)

        max_iterations = st.number_input(
            "Max iterations",
            min_value=1,
            max_value=10,
            key="ui_max_iterations",
            on_change=persist_ui_sidebar_state,
        )

        st.divider()
        settings = get_settings()
        existing_pdfs = pdf_storage.list_all()
        st.markdown(f"**History ({len(existing_pdfs)})**")
        col_open, col_refresh = st.columns(2)
        with col_open:
            if st.button("📂 Open", use_container_width=True, help="Open output folder"):
                folder = str(settings.output_dir.resolve())
                if sys.platform == "darwin":
                    subprocess.run(["open", folder])
                elif sys.platform == "win32":
                    subprocess.run(["explorer", folder])
                else:
                    subprocess.run(["xdg-open", folder])
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True, help="Rescan folder"):
                st.rerun()

        if existing_pdfs:
            for pdf in existing_pdfs[:10]:
                label = f"{pdf.company} • {pdf.job_title}"
                if len(label) > 30:
                    label = label[:27] + "..."
                with open(pdf.path, "rb") as f:
                    st.download_button(
                        label,
                        f.read(),
                        file_name=pdf.path.name,
                        mime="application/pdf",
                        key=str(pdf.timestamp),
                        help=f"{pdf.company} • {pdf.job_title}\n{pdf.timestamp.strftime('%m/%d %H:%M')}",
                        use_container_width=True,
                    )
        else:
            st.caption("No PDFs yet")

    return active_profile, sequential_mode, debug_mode, no_shame_mode, selected_lang_code, selected_language, max_iterations
