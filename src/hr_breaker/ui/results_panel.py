"""Results panel UI — filter results, PDF actions, translation, and resume preview."""

import subprocess
import sys

import streamlit as st

from hr_breaker.models import ValidationResult


def display_filter_results(validation: ValidationResult) -> None:
    for result in validation.results:
        icon = "[OK]" if result.passed else "[X]"
        with st.expander(
            f"{icon} {result.filter_name} - Score: {result.score:.2f}/{result.threshold:.2f}"
        ):
            if result.issues:
                st.write("**Issues:**")
                for issue in result.issues:
                    st.write(f"- {issue}")
            if result.suggestions:
                st.write("**Suggestions:**")
                for suggestion in result.suggestions:
                    st.write(f"- {suggestion}")


def render_results_panel(result: dict, run_async, pdf_storage) -> None:
    from hr_breaker.config import get_settings
    settings = get_settings()

    optimized = result["optimized"]
    validation = result["validation"]
    job = result["job"]
    iterations = result["iterations"]
    pdf_path = result["pdf_path"]
    debug_dir = result["debug_dir"]
    source = result.get("source")
    preflight = result.get("preflight")

    st.markdown("---")
    st.markdown(f"### Result: {job.title} at {job.company}")

    if preflight:
        ranked_documents = preflight.get("ranked_documents", [])
        st.info(
            f"Profile preflight: {preflight['selected_count']} selected docs · "
            f"{len(ranked_documents)} retrieved docs · {preflight['synthesized_chars']} synthesized chars"
        )
        if ranked_documents:
            with st.expander("Retrieved archive evidence", expanded=False):
                for match in ranked_documents:
                    st.write(
                        f"- {match.document.title} [{match.document.kind}] · score {match.score:.2f}"
                    )
                    if match.snippet:
                        st.caption(match.snippet)

    if validation.passed:
        st.success("All filters passed!")
    else:
        passed = [r.filter_name for r in validation.results if r.passed]
        failed = [r.filter_name for r in validation.results if not r.passed]
        st.warning(
            f"Max iterations ({len(passed)}/{len(validation.results)} passed). Failed: {', '.join(failed)}"
        )

    if debug_dir:
        st.info(f"Debug output: {debug_dir}")

    if pdf_path:
        st.success(f"PDF saved: {pdf_path}")
        if st.button("📂 Open Output Folder", use_container_width=True):
            folder = str(pdf_path.parent.resolve())
            if sys.platform == "darwin":
                subprocess.run(["open", folder])
            elif sys.platform == "win32":
                subprocess.run(["explorer", folder])
            else:
                subprocess.run(["xdg-open", folder])
    elif optimized:
        st.error("Failed to render PDF")

    if optimized:
        with st.expander("Resume Content", expanded=False):
            if optimized.html:
                st.code(optimized.html, language="html")
            elif optimized.data:
                st.json(optimized.data.model_dump())

    for i, opt, val in iterations:
        with st.expander(f"Iteration {i + 1}", expanded=False):
            if opt.changes:
                st.write("**Changes:**")
                for change in opt.changes:
                    st.write(f"- {change}")
            display_filter_results(val)

    if st.button("Clear Result", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.rerun()
