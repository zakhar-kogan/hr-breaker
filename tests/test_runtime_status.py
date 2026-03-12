from hr_breaker.runtime_status import RuntimeEvent, emit_runtime_event, emit_usage_event, runtime_event_sink


class DummyUsage:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int, requests: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.requests = requests


class DummyResult:
    def __init__(self, usage: DummyUsage):
        self._usage = usage

    def usage(self):
        return self._usage


def test_emit_usage_event_sends_tokens_to_sink():
    events: list[RuntimeEvent] = []
    result = DummyResult(DummyUsage(input_tokens=120, output_tokens=45, total_tokens=165, requests=1))

    with runtime_event_sink(events.append):
        emit_usage_event("job_parser", result, model_name="openai/gpt-4.1-mini")

    assert events == [
        RuntimeEvent(
            kind="usage",
            message=(
                "job_parser [openai/gpt-4.1-mini]: requests=1, prompt=120, "
                "completion=45, total=165"
            ),
            prompt_tokens=120,
            completion_tokens=45,
            total_tokens=165,
            requests=1,
        )
    ]


def test_emit_runtime_event_sends_message_to_sink():
    events: list[RuntimeEvent] = []

    with runtime_event_sink(events.append):
        emit_runtime_event(RuntimeEvent(kind="error", message="Authentication failed"))

    assert events == [RuntimeEvent(kind="error", message="Authentication failed")]
