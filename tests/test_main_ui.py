from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from hr_breaker import config
from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource, ValidationResult, FilterResult
from hr_breaker.services.profile_store import ProfileStore


async def fake_parse_job_posting(text: str) -> JobPosting:
    return JobPosting(
        title="Engineer",
        company="Acme",
        requirements=["Python"],
        keywords=["python"],
        description=text,
    )


async def fake_optimize_for_job(
    source,
    job_text,
    max_iterations=None,
    on_iteration=None,
    job=None,
    parallel=False,
    no_shame=False,
    user_instructions=None,
    language=None,
    on_translation_status=None,
):
    optimized = OptimizedResume(
        html="<div>Optimized</div>",
        pdf_text="Optimized",
        pdf_bytes=b"%PDF-1.4 test",
        source_checksum=source.checksum,
    )
    validation = ValidationResult(
        results=[
            FilterResult(
                filter_name="KeywordMatcher",
                passed=True,
                score=1.0,
                threshold=0.25,
                issues=[],
                suggestions=[],
            )
        ]
    )
    if on_iteration is not None:
        on_iteration(0, optimized, validation)
    return optimized, validation, job or await fake_parse_job_posting(job_text)


@pytest.fixture
def isolated_app_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    profile_dir = tmp_path / "profiles"
    output_dir = tmp_path / "output"
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PROFILE_DIR", str(profile_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    config.clear_settings_cache()
    yield {"cache_dir": cache_dir, "profile_dir": profile_dir, "output_dir": output_dir}
    config.clear_settings_cache()


@pytest.mark.parametrize("source_mode", ["Direct upload"])
def test_optimize_flow_does_not_hit_cache_replay_error(isolated_app_dirs, source_mode):
    source = ResumeSource(content="Jane Doe\nPython developer")

    with (
        patch("hr_breaker.agents.parse_job_posting", new=fake_parse_job_posting),
        patch("hr_breaker.orchestration.optimize_for_job", new=fake_optimize_for_job),
    ):
        at = AppTest.from_file("src/hr_breaker/main.py")
        at.session_state["source_resume"] = source
        at.session_state["job_text"] = "Build APIs in Python"
        at.session_state["source_mode"] = source_mode
        at.run(timeout=60)

        optimize_button = next(button for button in at.button if button.label == "🚀 Optimize")
        optimize_button.click().run(timeout=60)

        runtime_output = at.code[0].value
        error_messages = [element.value for element in at.error]

    assert "CacheReplayClosureError" not in runtime_output
    assert not any("CacheReplayClosureError" in message for message in error_messages)
    assert "Parsing job posting..." in runtime_output


def test_profile_ui_can_create_switch_and_show_folder_mode(isolated_app_dirs):
    at = AppTest.from_file("src/hr_breaker/main.py")
    at.run(timeout=60)

    new_profile_input = next(widget for widget in at.text_input if widget.label == "Profile name")
    create_button = next(button for button in at.button if button.label == "Create profile")

    new_profile_input.set_value("Jane Doe")
    create_button.click().run(timeout=60)

    active_profile = next(widget for widget in at.selectbox if widget.label == "Active profile")
    assert active_profile.value == "jane_doe"

    new_profile_input = next(widget for widget in at.text_input if widget.label == "Profile name")
    new_profile_input.set_value("Alex Smith")
    create_button = next(button for button in at.button if button.label == "Create profile")
    create_button.click().run(timeout=60)

    active_profile = next(widget for widget in at.selectbox if widget.label == "Active profile")
    assert active_profile.value == "alex_smith"

    active_profile.set_value("jane_doe").run(timeout=60)
    assert at.session_state["active_profile_id"] == "jane_doe"

    ingest_mode = next(widget for widget in at.radio if widget.label == "Add to profile")
    ingest_mode.set_value("Folder").run(timeout=60)
    assert any(button.label == "Import folder" for button in at.button)
    assert any(widget.label == "Separate embeddings" for widget in at.checkbox)

def test_profile_optimize_flow_uses_selected_archive_documents(isolated_app_dirs):
    store = ProfileStore(root_dir=isolated_app_dirs["profile_dir"])
    profile = store.create_profile("Jane Doe", first_name="Jane", last_name="Doe")
    store.add_note(
        profile.id,
        title="Relevant Project",
        content_text="Built Python retrieval systems for LLM ranking and semantic search.",
    )
    store.add_note(
        profile.id,
        title="Hackathon Win",
        content_text="Won first place for an AI hiring assistant with resume ranking.",
    )

    captured_sources: list[ResumeSource] = []

    async def fake_optimize_with_capture(*args, **kwargs):
        source = args[0]
        captured_sources.append(source)
        return await fake_optimize_for_job(*args, **kwargs)

    with (
        patch("hr_breaker.agents.parse_job_posting", new=fake_parse_job_posting),
        patch("hr_breaker.orchestration.optimize_for_job", new=fake_optimize_with_capture),
        patch("hr_breaker.services.profile_retrieval._vector_scores", return_value=[None, None]),
    ):
        at = AppTest.from_file("src/hr_breaker/main.py")
        at.session_state["job_text"] = "Build Python retrieval and ranking systems for hiring."
        at.run(timeout=60)

        use_all_documents = next(
            widget for widget in at.checkbox if widget.label == "Use all documents"
        )
        use_all_documents.set_value(False).run(timeout=60)
        optimize_button = next(button for button in at.button if button.label == "🚀 Optimize")
        assert optimize_button.disabled

        use_all_documents = next(
            widget for widget in at.checkbox if widget.label == "Use all documents"
        )
        use_all_documents.set_value(True).run(timeout=60)
        optimize_button = next(button for button in at.button if button.label == "🚀 Optimize")
        assert not optimize_button.disabled

        optimize_button.click().run(timeout=60)

    assert captured_sources
    assert "Hackathon Win" in captured_sources[0].content
    info_messages = [element.value for element in at.info]
    assert any("Profile preflight" in message for message in info_messages)
    assert "Profile preflight" in at.code[0].value
