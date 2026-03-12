"""Runtime log panel — captures LLM events and log records into a Streamlit UI panel."""

import logging
from contextlib import contextmanager
from datetime import datetime

import streamlit as st

from hr_breaker.runtime_status import EventKind, RuntimeEvent, runtime_event_sink

MAX_RUNTIME_LINES = 120


def initialize_runtime_state() -> None:
    defaults = {
        "runtime_lines": [],
        "runtime_prompt_tokens": 0,
        "runtime_completion_tokens": 0,
        "runtime_total_tokens": 0,
        "runtime_requests": 0,
        "runtime_errors": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_runtime_state() -> None:
    st.session_state["runtime_lines"] = []
    st.session_state["runtime_prompt_tokens"] = 0
    st.session_state["runtime_completion_tokens"] = 0
    st.session_state["runtime_total_tokens"] = 0
    st.session_state["runtime_requests"] = 0
    st.session_state["runtime_errors"] = 0


def append_runtime_event(event: RuntimeEvent | str, kind: EventKind = "info") -> None:
    initialize_runtime_state()
    runtime_event = (
        event if isinstance(event, RuntimeEvent) else RuntimeEvent(kind=kind, message=str(event))
    )
    prefix = {
        "info": "INFO",
        "warning": "WARN",
        "error": "ERROR",
        "usage": "TOKENS",
    }[runtime_event.kind]
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state["runtime_lines"] = (
        st.session_state["runtime_lines"] + [f"{ts} [{prefix}] {runtime_event.message}"]
    )[-MAX_RUNTIME_LINES:]
    st.session_state["runtime_prompt_tokens"] += runtime_event.prompt_tokens
    st.session_state["runtime_completion_tokens"] += runtime_event.completion_tokens
    st.session_state["runtime_total_tokens"] += runtime_event.total_tokens
    st.session_state["runtime_requests"] += runtime_event.requests
    if runtime_event.kind == "error":
        st.session_state["runtime_errors"] += 1


def render_runtime_panel(panel_placeholder) -> None:
    initialize_runtime_state()
    summary = (
        f"Requests: {st.session_state['runtime_requests']} · "
        f"Prompt tokens: {st.session_state['runtime_prompt_tokens']} · "
        f"Completion tokens: {st.session_state['runtime_completion_tokens']} · "
        f"Total tokens: {st.session_state['runtime_total_tokens']} · "
        f"Errors: {st.session_state['runtime_errors']}"
    )
    with panel_placeholder.container():
        st.markdown("**Runtime log**")
        st.caption(summary)
        with st.container(height=220, border=True):
            st.code(
                "\n".join(st.session_state["runtime_lines"])
                or "No activity yet. Request logs, token usage, and errors appear here.",
                language="text",
            )


class RuntimeLogHandler(logging.Handler):
    def __init__(self, panel_placeholder):
        super().__init__()
        self.panel_placeholder = panel_placeholder

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("hr_breaker.runtime"):
            return
        if record.name.startswith("httpx") and record.levelno < logging.WARNING:
            return
        kind = (
            "error" if record.levelno >= logging.ERROR
            else "warning" if record.levelno >= logging.WARNING
            else "info"
        )
        append_runtime_event(record.getMessage(), kind=kind)
        render_runtime_panel(self.panel_placeholder)


@contextmanager
def capture_runtime_output(panel_placeholder):
    from hr_breaker.config import logger as hr_logger  # avoid circular at module level
    handler = RuntimeLogHandler(panel_placeholder)
    handler.setLevel(logging.INFO)
    logger_names = ("hr_breaker", "litellm")
    prior_levels = {}
    for logger_name in logger_names:
        lg = logging.getLogger(logger_name)
        prior_levels[logger_name] = lg.level
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    try:
        with runtime_event_sink(
            lambda event: (append_runtime_event(event), render_runtime_panel(panel_placeholder))
        ):
            yield
    finally:
        for logger_name in logger_names:
            lg = logging.getLogger(logger_name)
            lg.removeHandler(handler)
            lg.setLevel(prior_levels[logger_name])
