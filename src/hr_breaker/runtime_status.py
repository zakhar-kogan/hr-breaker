import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import logging

EventKind = Literal["info", "warning", "error", "usage"]
logger = logging.getLogger("hr_breaker.runtime")


@dataclass(frozen=True)
class RuntimeEvent:
    kind: EventKind
    message: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


# Thread-local sink so background threads don't interfere with the main thread's sink
_local = threading.local()


def _get_sink() -> Callable[[RuntimeEvent], None] | None:
    return getattr(_local, "event_sink", None)


@contextmanager
def runtime_event_sink(sink: Callable[[RuntimeEvent], None]) -> Iterator[None]:
    previous = _get_sink()
    _local.event_sink = sink
    try:
        yield
    finally:
        _local.event_sink = previous


def emit_runtime_event(event: RuntimeEvent) -> None:
    if event.kind == "error":
        logger.error(event.message)
    elif event.kind == "warning":
        logger.warning(event.message)
    else:
        logger.info(event.message)

    sink = _get_sink()
    if sink is not None:
        sink(event)


def emit_runtime_message(message: str, kind: EventKind = "info") -> None:
    emit_runtime_event(RuntimeEvent(kind=kind, message=message))


def emit_usage_event(operation: str, result: Any, model_name: str | None = None) -> None:
    usage_accessor = getattr(result, "usage", None)
    usage = usage_accessor() if callable(usage_accessor) else usage_accessor
    if usage is None:
        return

    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    )
    requests = int(getattr(usage, "requests", 0) or 0)
    model_suffix = f" [{model_name}]" if model_name else ""
    emit_runtime_event(
        RuntimeEvent(
            kind="usage",
            message=(
                f"{operation}{model_suffix}: requests={requests}, prompt={prompt_tokens}, "
                f"completion={completion_tokens}, total={total_tokens}"
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            requests=requests,
        )
    )
