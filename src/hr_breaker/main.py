import asyncio

import nest_asyncio
import streamlit as st

nest_asyncio.apply()

# Event loop setup — must happen before any hr_breaker imports that touch asyncio
if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(st.session_state.event_loop)

from hr_breaker.agents import extract_name, parse_job_posting
from hr_breaker.config import (
    get_embedding_llm_config,
    get_flash_llm_config,
    get_pro_llm_config,
    get_settings,
)
from hr_breaker.models import GeneratedPDF, RankedProfileDocument, ResumeSource
from hr_breaker.models.profile import document_needs_extraction
from hr_breaker.orchestration import optimize_for_job
from hr_breaker.runtime_status import emit_runtime_message
from hr_breaker.services import (
    CloudflareBlockedError,
    PDFStorage,
    ProfileStore,
    ResumeCache,
    extraction_worker,
    scrape_job_posting,
)
from hr_breaker.services.pdf_storage import generate_run_id
from hr_breaker.services.profile_retrieval import rank_profile_documents, synthesize_profile_resume_source
from hr_breaker.ui.results_panel import display_filter_results, render_results_panel
from hr_breaker.ui.runtime_log import (
    append_runtime_event,
    capture_runtime_output,
    initialize_runtime_state,
    render_runtime_panel,
    reset_runtime_state,
)
from hr_breaker.ui.sidebar import (
    cached_provider_catalog,
    initialize_llm_sidebar_state,
    initialize_ui_sidebar_state,
    render_sidebar,
)
from hr_breaker.ui.source_panel import (
    _optional_text,
    combine_instructions,
    get_active_profile,
    infer_profile_name_parts,
    initialize_profile_state,
    render_direct_resume_panel,
    render_profile_panel,
    sync_profile_name_draft,
    sync_profile_selection,
)

# Initialize services
cache = ResumeCache()
pdf_storage = PDFStorage()
profile_store = ProfileStore()

st.set_page_config(page_title="HR-Breaker", page_icon="*", layout="wide")


def run_async(coro):
    loop = st.session_state.event_loop
    return loop.run_until_complete(coro)


@st.cache_data(show_spinner=False)
def cached_scrape_job(url: str) -> str:
    return scrape_job_posting(url)


@st.cache_data(show_spinner=False)
def cached_extract_name(content: str) -> tuple[str | None, str | None]:
    return run_async(extract_name(content))


@st.cache_data(show_spinner=False)
def cached_parse_job(text: str):
    return run_async(parse_job_posting(text))


# --- App initialization ---

initialize_llm_sidebar_state()
initialize_ui_sidebar_state()
initialize_runtime_state()
settings = get_settings()
profiles = profile_store.list_profiles()
initialize_profile_state(profiles)

# --- Sidebar ---

active_profile, sequential_mode, debug_mode, no_shame_mode, selected_lang_code, selected_language, max_iterations = render_sidebar(
    profiles=profiles,
    profile_store=profile_store,
    pdf_storage=pdf_storage,
    sync_profile_selection=sync_profile_selection,
    sync_profile_name_draft=sync_profile_name_draft,
    infer_profile_name_parts=infer_profile_name_parts,
    _optional_text=_optional_text,
)

# --- Main content ---

st.markdown("### HR-Breaker")

# Restore cached resume if available and not explicitly cleared
if (
    "source_resume" not in st.session_state
    and not st.session_state.get("resume_cleared")
    and cache.list_all()
):
    cached_resumes = cache.list_all()
    if cached_resumes:
        st.session_state["source_resume"] = cached_resumes[-1]
        if cached_resumes[-1].instructions:
            st.session_state["user_instructions"] = cached_resumes[-1].instructions

job_text = st.session_state.get("job_text", "")
has_job = bool(job_text)
job_header = "**Job Posting ✓**" if has_job else "**Job Posting**"

col_resume, col_job = st.columns(2)
active_profile = get_active_profile(profiles)
direct_source = st.session_state.get("source_resume")
active_profile_documents = []
selected_profile_documents = []

with col_resume:
    st.markdown("**Source mode**")
    if profiles:
        st.radio(
            "Source mode",
            ["Profile archive", "Direct upload"],
            horizontal=True,
            key="source_mode",
            help="Profiles are reusable archive folders. Direct upload keeps the one-off workflow.",
            label_visibility="collapsed",
        )
    else:
        st.session_state["source_mode"] = "Direct upload"

    if st.session_state["source_mode"] == "Profile archive":
        active_profile, active_profile_documents, selected_profile_documents = render_profile_panel(
            active_profile, profile_store, extraction_worker
        )
    else:
        direct_source = render_direct_resume_panel(cache, cached_extract_name)

with col_job:
    st.markdown(job_header)
    if has_job:
        preview = (
            job_text[:80].replace("\n", " ") + "..."
            if len(job_text) > 80
            else job_text.replace("\n", " ")
        )
        c1, c2 = st.columns([5, 1.4])
        with c1:
            st.success(f"✓ {preview}")
        with c2:
            if st.button("Change", key="clear_job", use_container_width=True):
                st.session_state.pop("job_text", None)
                st.session_state.pop("last_job_url", None)
                st.session_state.pop("last_result", None)
                st.rerun()
        with st.expander("Preview", expanded=False):
            st.text(job_text)
    else:
        job_input_method = st.radio(
            "Job input method",
            ["URL", "Paste"],
            horizontal=True,
            key="job_method",
            label_visibility="collapsed",
        )

        if job_input_method == "URL":
            job_url = st.text_input(
                "Job URL", label_visibility="collapsed", placeholder="https://..."
            )
            if job_url and job_url != st.session_state.get("last_job_url"):
                st.session_state["last_job_url"] = job_url
                with st.spinner("Fetching..."):
                    try:
                        job_text = cached_scrape_job(job_url)
                        st.session_state["job_text"] = job_text
                        st.session_state.pop("scrape_failed_url", None)
                        st.rerun()
                    except CloudflareBlockedError:
                        st.session_state["scrape_failed_url"] = job_url
                        st.warning("Bot protection. Copy & paste instead.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            if st.session_state.get("scrape_failed_url"):
                st.markdown(f"[Open in browser]({st.session_state['scrape_failed_url']})")
        else:
            pasted_job = st.text_area(
                "Paste job",
                height=100,
                label_visibility="collapsed",
                placeholder="Paste job posting...",
            )
            if pasted_job:
                st.session_state["job_text"] = pasted_job
                st.session_state.pop("scrape_failed_url", None)
                st.rerun()

    if "user_instructions" not in st.session_state:
        st.session_state["user_instructions"] = ""
    instructions_expanded = bool(st.session_state.get("user_instructions"))
    with st.expander("Instructions (optional)", expanded=instructions_expanded):
        user_instructions = st.text_area(
            "Instructions (optional)",
            placeholder="E.g. Focus on Python and AWS experience, add my Kubernetes certification...",
            help="Instructions for the optimizer: extra experience, style preferences, emphasis areas.",
            key="user_instructions",
            label_visibility="collapsed",
        )

# --- Optimize button ---

is_profile_mode = st.session_state["source_mode"] == "Profile archive"
has_source = bool(selected_profile_documents) if is_profile_mode else direct_source is not None
is_running = st.session_state.get("optimization_running", False)
can_optimize = has_source and has_job and not is_running
btn_help = None
if is_profile_mode and active_profile is None:
    btn_help = "Need active profile"
elif not has_source:
    btn_help = "Need selected profile documents" if is_profile_mode else "Need resume"
elif not has_job:
    btn_help = "Need job posting"
elif is_running:
    btn_help = "Optimization in progress"
clicked = st.button(
    "🚀 Optimize",
    disabled=not can_optimize,
    use_container_width=True,
    help=btn_help,
)
runtime_panel = st.empty()
render_runtime_panel(runtime_panel)

# Drain background extraction events into runtime panel on every rerun
_drained_events = extraction_worker.drain_events()
if _drained_events:
    for _event in _drained_events:
        append_runtime_event(_event)
    render_runtime_panel(runtime_panel)

# --- Optimization flow ---

if clicked:
    st.session_state["run_id"] = generate_run_id()
    session_instructions = _optional_text(user_instructions)
    optimizer_instructions = combine_instructions(
        active_profile.instructions if is_profile_mode and active_profile else None,
        session_instructions,
    )
    source = direct_source
    preflight = None

    if not is_profile_mode:
        if source is None:
            raise ValueError("No resume loaded")
        if session_instructions != source.instructions:
            source = source.model_copy(update={"instructions": session_instructions})
            cache.put(source)
            st.session_state["source_resume"] = source

    reset_runtime_state()
    render_runtime_panel(runtime_panel)
    st.session_state["optimization_running"] = True
    error_occurred = None

    try:
        with capture_runtime_output(runtime_panel):
            pro_config = get_pro_llm_config()
            flash_config = get_flash_llm_config()
            embedding_config = get_embedding_llm_config()
            emit_runtime_message(
                f"Pro model: {pro_config.model_name} | endpoint: {pro_config.api_base or 'provider default'}"
            )
            emit_runtime_message(
                f"Flash model: {flash_config.model_name} | endpoint: {flash_config.api_base or 'provider default'}"
            )
            emit_runtime_message(
                f"Embedding model: {embedding_config.model_name} | endpoint: {embedding_config.api_base or 'provider default'}"
            )
            emit_runtime_message("Parsing job posting...")
            with st.spinner("Parsing job posting..."):
                job = cached_parse_job(job_text)
            emit_runtime_message(f"Parsed job posting for {job.title} at {job.company}")

            debug_dir = None
            if debug_mode:
                debug_dir = pdf_storage.generate_debug_dir(
                    job.company, job.title, run_id=st.session_state["run_id"]
                )
                emit_runtime_message(f"Debug output enabled: {debug_dir}")

            iteration_results = []
            ranked_documents: list[RankedProfileDocument] = []

            with st.status("Optimizing resume...", expanded=True) as status_container:
                if is_profile_mode:
                    if active_profile is None:
                        raise ValueError("No active profile selected")
                    selected_count = len(selected_profile_documents)
                    status_container.write(
                        f"Profile preflight: {selected_count} selected archive documents"
                    )
                    emit_runtime_message(
                        f"Profile preflight: {selected_count} selected archive documents"
                    )
                    _n_with = sum(1 for d in selected_profile_documents if not document_needs_extraction(d))
                    _n_total = len(selected_profile_documents)

                    status_container.update(label="Ranking archive evidence...")
                    emit_runtime_message("Ranking archive evidence...")
                    ranked_documents = run_async(
                        rank_profile_documents(selected_profile_documents, job)
                    )

                    if _n_with > 0:
                        _pending_note = f", {_n_total - _n_with} pending" if _n_with < _n_total else ""
                        emit_runtime_message(f"Synthesis: extraction path — {_n_with}/{_n_total} docs{_pending_note}")
                    else:
                        emit_runtime_message("Synthesis: whole-doc fallback")

                    status_container.update(label="Synthesizing profile source...")
                    source = synthesize_profile_resume_source(
                        active_profile,
                        selected_profile_documents,
                        ranked_documents,
                    )
                    if optimizer_instructions != source.instructions:
                        source = source.model_copy(update={"instructions": optimizer_instructions})
                    status_container.write(f"Synthesized source length: {len(source.content)} chars")
                    emit_runtime_message(f"Synthesized profile source: {len(source.content)} chars")
                    if _n_with == 0:
                        for match in ranked_documents:
                            emit_runtime_message(
                                f"  Included {match.document.title} [{match.document.kind}] score={match.score:.2f}"
                            )
                    preflight = {
                        "selected_count": len(selected_profile_documents),
                        "ranked_documents": ranked_documents,
                        "synthesized_chars": len(source.content),
                        "profile_id": active_profile.id,
                    }
                elif source is None:
                    raise ValueError("No resume loaded")

                def on_iteration(i, opt, val):
                    iteration_results.append((i, opt, val))
                    status_container.update(label=f"Iteration {i + 1}/{max_iterations}")
                    status_container.write(f"Iteration {i + 1} complete")
                    emit_runtime_message(f"Iteration {i + 1}/{max_iterations} complete")

                    if debug_mode and debug_dir:
                        if opt.html:
                            (debug_dir / f"iteration_{i + 1}.html").write_text(opt.html, encoding="utf-8")
                        if opt.pdf_bytes:
                            (debug_dir / f"iteration_{i + 1}.pdf").write_bytes(opt.pdf_bytes)

                target_lang = selected_language if selected_language.code != "en" else None
                status_container.update(label=f"Generating iteration 1/{max_iterations}...")
                emit_runtime_message(f"Starting optimizer loop ({max_iterations} iterations max)")

                optimized, validation, job = run_async(
                    optimize_for_job(
                        source,
                        job_text,
                        max_iterations=max_iterations,
                        on_iteration=on_iteration,
                        job=job,
                        parallel=not sequential_mode,
                        no_shame=no_shame_mode,
                        user_instructions=optimizer_instructions,
                        language=target_lang,
                    )
                )
                status_container.update(label="Optimization complete", state="complete")
                emit_runtime_message("Optimization complete")

            pdf_path = None
            if optimized and optimized.pdf_bytes:
                pdf_path = pdf_storage.generate_path(
                    source.first_name, source.last_name, job.company, job.title,
                    lang_code=selected_lang_code,
                    run_id=st.session_state["run_id"],
                )
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(optimized.pdf_bytes)

                pdf_record = GeneratedPDF(
                    path=pdf_path,
                    source_checksum=source.checksum,
                    company=job.company,
                    job_title=job.title,
                    first_name=source.first_name,
                    last_name=source.last_name,
                )
                pdf_storage.save_record(pdf_record)
                emit_runtime_message(f"Saved PDF: {pdf_path}")

            st.session_state["last_result"] = {
                "optimized": optimized,
                "validation": validation,
                "job": job,
                "iterations": iteration_results,
                "pdf_path": pdf_path,
                "debug_dir": debug_dir,
                "source": source,
                "preflight": preflight,
            }
    except Exception as e:
        error_occurred = e
        append_runtime_event(f"{type(e).__name__}: {e}", kind="error")
        render_runtime_panel(runtime_panel)
    finally:
        st.session_state["optimization_running"] = False

    if error_occurred:
        st.error(f"Optimization failed: {error_occurred}")
    else:
        st.rerun()

# --- Results ---

if "last_result" in st.session_state:
    render_results_panel(st.session_state["last_result"], run_async, pdf_storage)

# --- Auto-refresh while background extraction is active ---

if extraction_worker.any_active():
    import time
    time.sleep(1.5)
    st.rerun()
