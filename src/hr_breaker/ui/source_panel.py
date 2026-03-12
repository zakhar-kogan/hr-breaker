"""Source panel UI — direct resume upload and profile archive panel."""

import time

import streamlit as st

from hr_breaker.models import Profile, ProfileDocument, ResumeSource
from hr_breaker.services.pdf_parser import load_resume_content_from_upload


# --- Profile state helpers ---

def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def infer_profile_name_parts(display_name: str) -> tuple[str | None, str | None]:
    parts = [part for part in display_name.strip().split() if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def initialize_profile_state(profiles: list[Profile]) -> None:
    defaults = {
        "active_profile_id": None,
        "active_profile_picker": None,
        "pending_active_profile_id": None,
        "profile_name_draft": "",
        "profile_name_draft_profile_id": None,
        "profile_selected_docs": {},
        "profile_seen_doc_ids": {},
        "profile_uploader_key": 0,
        "profile_folder_uploader_key": 0,
        "source_mode": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else value

    profile_ids = [profile.id for profile in profiles]
    pending_profile_id = st.session_state.get("pending_active_profile_id")
    if pending_profile_id in profile_ids:
        st.session_state["active_profile_id"] = pending_profile_id
        st.session_state["active_profile_picker"] = pending_profile_id
        st.session_state["pending_active_profile_id"] = None

    active_profile_id = st.session_state.get("active_profile_id")
    if active_profile_id not in profile_ids:
        active_profile_id = profile_ids[0] if profile_ids else None
        st.session_state["active_profile_id"] = active_profile_id
    if st.session_state.get("active_profile_picker") not in profile_ids:
        st.session_state["active_profile_picker"] = active_profile_id

    if profiles and st.session_state.get("source_mode") not in {"Profile archive", "Direct upload"}:
        st.session_state["source_mode"] = "Profile archive"
    if not profiles:
        st.session_state["source_mode"] = "Direct upload"


def get_active_profile(profiles: list[Profile]) -> Profile | None:
    active_profile_id = st.session_state.get("active_profile_id")
    for profile in profiles:
        if profile.id == active_profile_id:
            return profile
    return None


def sync_profile_name_draft(active_profile: Profile | None) -> None:
    active_profile_id = active_profile.id if active_profile else None
    if st.session_state.get("profile_name_draft_profile_id") != active_profile_id:
        st.session_state["profile_name_draft"] = active_profile.display_name if active_profile else ""
        st.session_state["profile_name_draft_profile_id"] = active_profile_id


def _profile_checkbox_key(profile_id: str, document_id: str) -> str:
    return f"profile_doc_selected::{profile_id}::{document_id}"


def set_profile_selection(
    profile_id: str,
    documents: list[ProfileDocument],
    selected_ids: set[str],
    *,
    update_widgets: bool = True,
) -> None:
    st.session_state["profile_selected_docs"][profile_id] = sorted(selected_ids)
    st.session_state["profile_seen_doc_ids"][profile_id] = [document.id for document in documents]
    if update_widgets:
        for document in documents:
            st.session_state[_profile_checkbox_key(profile_id, document.id)] = document.id in selected_ids


def sync_profile_selection(profile_id: str, documents: list[ProfileDocument]) -> set[str]:
    selection_map = st.session_state["profile_selected_docs"]
    seen_map = st.session_state["profile_seen_doc_ids"]
    document_ids = [document.id for document in documents]
    current_selected = set(selection_map.get(profile_id, [])) & set(document_ids)
    seen_ids = set(seen_map.get(profile_id, []))

    if profile_id not in selection_map:
        current_selected = {
            document.id for document in documents if document.included_by_default
        }
    else:
        current_selected |= {
            document.id
            for document in documents
            if document.id not in seen_ids and document.included_by_default
        }

    set_profile_selection(profile_id, documents, current_selected)
    return current_selected


def queue_profile_master_selection(profile_id: str) -> None:
    st.session_state[f"profile_master_dirty::{profile_id}"] = True


def combine_instructions(profile_instructions: str | None, user_instructions: str | None) -> str | None:
    parts = []
    profile_text = _optional_text(profile_instructions)
    user_text = _optional_text(user_instructions)
    if profile_text:
        parts.append(f"Profile instructions:\n{profile_text}")
    if user_text:
        parts.append(f"Session instructions:\n{user_text}")
    if not parts:
        return None
    return "\n\n".join(parts)


# --- Resume panels ---

def render_direct_resume_panel(cache, cached_extract_name) -> ResumeSource | None:
    has_resume = "source_resume" in st.session_state
    resume_header = "**Resume ✓**" if has_resume else "**Resume**"
    st.markdown(resume_header)

    if has_resume:
        source = st.session_state["source_resume"]
        name = f"{source.first_name or ''} {source.last_name or ''}".strip() or "Unknown"
        info_col, action_col = st.columns([5, 1.4])
        with info_col:
            st.success(f"✓ {name}")
        with action_col:
            if st.button("Change", key="clear_resume", use_container_width=True):
                st.session_state.pop("source_resume", None)
                st.session_state.pop("last_result", None)
                st.session_state["resume_uploader_key"] = (
                    st.session_state.get("resume_uploader_key", 0) + 1
                )
                st.session_state["resume_cleared"] = True
                st.rerun()
        with st.expander("Preview", expanded=False):
            st.text(source.content)
        return source

    resume_method = st.radio(
        "Resume input method",
        ["Upload", "Paste"],
        horizontal=True,
        key="resume_method",
        label_visibility="collapsed",
    )

    resume_content = None
    if resume_method == "Upload":
        uploader_key = f"resume_uploader_{st.session_state.get('resume_uploader_key', 0)}"
        uploaded_file = st.file_uploader(
            "Upload (.tex, .md, .txt, .pdf)",
            type=["tex", "md", "txt", "pdf"],
            label_visibility="collapsed",
            key=uploader_key,
        )
        if uploaded_file:
            resume_content = load_resume_content_from_upload(
                uploaded_file.name, uploaded_file.read()
            )
    else:
        pasted_resume = st.text_area(
            "Paste resume",
            height=100,
            label_visibility="collapsed",
            placeholder="Paste resume text...",
        )
        if pasted_resume:
            resume_content = pasted_resume

    if resume_content:
        with st.spinner("Extracting name..."):
            first_name, last_name = cached_extract_name(resume_content)
        source = ResumeSource(
            content=resume_content,
            first_name=first_name,
            last_name=last_name,
        )
        cache.put(source)
        st.session_state["source_resume"] = source
        st.session_state.pop("resume_cleared", None)
        st.rerun()

    return None


def render_profile_panel(
    active_profile: Profile | None,
    profile_store,
    extraction_worker,
) -> tuple[Profile | None, list[ProfileDocument], list[ProfileDocument]]:
    if active_profile is None:
        st.info("Create a profile in the sidebar to build a reusable archive.")
        return None, [], []

    refreshed_profile = profile_store.get_profile(active_profile.id) or active_profile
    documents = profile_store.list_documents(refreshed_profile.id)
    selected_ids = sync_profile_selection(refreshed_profile.id, documents)

    with st.expander(
        f"Profile archive ({len(documents)} docs)",
        expanded=False,
    ):
        first_name_key = f"profile_first_name::{refreshed_profile.id}"
        last_name_key = f"profile_last_name::{refreshed_profile.id}"
        instructions_key = f"profile_instructions::{refreshed_profile.id}"
        if first_name_key not in st.session_state:
            st.session_state[first_name_key] = refreshed_profile.first_name or ""
        if last_name_key not in st.session_state:
            st.session_state[last_name_key] = refreshed_profile.last_name or ""
        if instructions_key not in st.session_state:
            st.session_state[instructions_key] = refreshed_profile.instructions or ""

        with st.expander("Profile details", expanded=False):
            st.text_input("First name", key=first_name_key)
            st.text_input("Last name", key=last_name_key)
            st.text_area(
                "Profile instructions",
                key=instructions_key,
                height=100,
                placeholder="Persistent guidance for this candidate profile...",
            )
            if st.button(
                "Save profile details",
                key=f"save_profile_details::{refreshed_profile.id}",
                use_container_width=True,
            ):
                profile_store.update_profile_details(
                    refreshed_profile.id,
                    first_name=_optional_text(st.session_state[first_name_key]),
                    last_name=_optional_text(st.session_state[last_name_key]),
                    instructions=_optional_text(st.session_state[instructions_key]),
                )
                st.session_state.pop("last_result", None)
                st.rerun()

        supports_directory_upload = tuple(int(part) for part in st.__version__.split(".")[:2]) >= (1, 49)
        ingest_options = ["Files", "Folder", "Note"] if supports_directory_upload else ["Files", "Note"]
        ingest_mode_key = f"profile_ingest_mode::{refreshed_profile.id}"
        if ingest_mode_key not in st.session_state or st.session_state[ingest_mode_key] not in ingest_options:
            st.session_state[ingest_mode_key] = ingest_options[0]
        st.radio(
            "Add to profile",
            ingest_options,
            horizontal=True,
            key=ingest_mode_key,
            label_visibility="collapsed",
        )

        ingest_mode = st.session_state[ingest_mode_key]
        if ingest_mode == "Files":
            uploaded_files = st.file_uploader(
                "Add profile documents (.pdf, .tex, .md, .txt)",
                type=["pdf", "tex", "md", "txt"],
                accept_multiple_files=True,
                key=f"profile_uploader_{st.session_state['profile_uploader_key']}",
            )
            if st.button(
                "Add files to profile",
                key=f"add_profile_files::{refreshed_profile.id}",
                use_container_width=True,
            ):
                if not uploaded_files:
                    st.warning("Select at least one file to add to the profile archive.")
                else:
                    new_doc_ids = []
                    for uploaded_file in uploaded_files:
                        doc = profile_store.add_upload(
                            refreshed_profile.id,
                            filename=uploaded_file.name,
                            data=uploaded_file.read(),
                            mime_type=uploaded_file.type or None,
                        )
                        new_doc_ids.append(doc.id)
                    if new_doc_ids:
                        extraction_worker.submit(refreshed_profile.id, new_doc_ids)
                    st.session_state["profile_uploader_key"] += 1
                    st.session_state.pop("last_result", None)
                    st.rerun()
        elif ingest_mode == "Folder":
            folder_files = st.file_uploader(
                "Import a folder into this profile",
                type=["pdf", "tex", "md", "txt"],
                accept_multiple_files="directory",
                key=f"profile_folder_uploader_{st.session_state['profile_folder_uploader_key']}",
                help="Choose a local folder and index all supported files into the current profile.",
            )
            if st.button(
                "Import folder",
                key=f"import_profile_folder::{refreshed_profile.id}",
                use_container_width=True,
            ):
                if not folder_files:
                    st.warning("Choose a folder with supported files first.")
                else:
                    new_doc_ids = []
                    for uploaded_file in folder_files:
                        doc = profile_store.add_upload(
                            refreshed_profile.id,
                            filename=uploaded_file.name,
                            data=uploaded_file.read(),
                            mime_type=uploaded_file.type or None,
                        )
                        new_doc_ids.append(doc.id)
                    if new_doc_ids:
                        extraction_worker.submit(refreshed_profile.id, new_doc_ids)
                    st.session_state["profile_folder_uploader_key"] += 1
                    st.session_state.pop("last_result", None)
                    st.rerun()
        else:
            note_version_key = f"profile_note_version::{refreshed_profile.id}"
            if note_version_key not in st.session_state:
                st.session_state[note_version_key] = 0
            note_title_key = f"profile_note_title::{refreshed_profile.id}::{st.session_state[note_version_key]}"
            note_content_key = f"profile_note_content::{refreshed_profile.id}::{st.session_state[note_version_key]}"
            st.text_input("Note title", key=note_title_key, placeholder="Hackathon, paper, award...")
            st.text_area(
                "Add archive note",
                key=note_content_key,
                height=120,
                placeholder="Paste supporting facts, papers, hackathon results, awards, or other details...",
            )
            if st.button(
                "Add note",
                key=f"add_profile_note::{refreshed_profile.id}",
                use_container_width=True,
            ):
                note_title = _optional_text(st.session_state.get(note_title_key))
                note_content = _optional_text(st.session_state.get(note_content_key))
                if not note_title or not note_content:
                    st.warning("Both note title and note content are required.")
                else:
                    doc = profile_store.add_note(
                        refreshed_profile.id,
                        title=note_title,
                        content_text=note_content,
                    )
                    extraction_worker.submit(refreshed_profile.id, [doc.id])
                    st.session_state[note_version_key] += 1
                    st.session_state.pop("last_result", None)
                    st.rerun()

        if not documents:
            st.caption("No archive documents yet. Add files, import a folder, or save a note above.")
            return refreshed_profile, documents, []

        def _needs_extraction(doc) -> bool:
            raw = doc.metadata.get("extraction")
            if not raw:
                return True
            return "personal_info" not in raw

        pending_extraction = [d for d in documents if _needs_extraction(d)]
        if pending_extraction:
            _btn_col, _rebtn_col = st.columns([3, 1])
            with _btn_col:
                if st.button(
                    f"Extract facts ({len(pending_extraction)} pending)",
                    key=f"backfill_extractions::{refreshed_profile.id}",
                    use_container_width=True,
                    help="Run structured extraction on documents that haven't been processed yet.",
                ):
                    extraction_worker.submit(refreshed_profile.id, [d.id for d in pending_extraction])
                    st.rerun()
            with _rebtn_col:
                if st.button(
                    "Re-extract all",
                    key=f"reextract_all::{refreshed_profile.id}",
                    use_container_width=True,
                    help="Force re-extraction of all documents (use after extraction prompts change).",
                ):
                    extraction_worker.submit(refreshed_profile.id, [d.id for d in documents])
                    st.rerun()
        else:
            if st.button(
                "Re-extract all",
                key=f"reextract_all::{refreshed_profile.id}",
                use_container_width=True,
                help="Force re-extraction of all documents (use after extraction prompts change).",
            ):
                extraction_worker.submit(refreshed_profile.id, [d.id for d in documents])
                st.rerun()

        with st.expander(
            f"Archive documents ({len(selected_ids)}/{len(documents)} selected)",
            expanded=False,
        ):
            master_key = f"profile_master_select::{refreshed_profile.id}"
            master_dirty_key = f"profile_master_dirty::{refreshed_profile.id}"
            all_selected = len(selected_ids) == len(documents)
            if not st.session_state.get(master_dirty_key):
                st.session_state[master_key] = all_selected
            st.checkbox(
                "Use all documents",
                key=master_key,
                help="Checked means every archive document is available for synthesis and optimization.",
                on_change=queue_profile_master_selection,
                args=(refreshed_profile.id,),
            )
            if st.session_state.get(master_dirty_key):
                selected_ids = {document.id for document in documents} if st.session_state[master_key] else set()
                set_profile_selection(refreshed_profile.id, documents, selected_ids)
                st.session_state[master_dirty_key] = False

            st.caption(f"{len(selected_ids)}/{len(documents)} documents selected")
            updated_selected_ids: set[str] = set()
            with st.container(height=380, border=True):
                for document in documents:
                    checkbox_key = _profile_checkbox_key(refreshed_profile.id, document.id)
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = document.id in selected_ids
                    checked = st.checkbox(
                        f"{document.title} [{document.kind}]",
                        key=checkbox_key,
                        help=document.preview_text or document.source_name,
                    )
                    st.caption(
                        f"{document.source_name} · {document.timestamp.strftime('%Y-%m-%d %H:%M')}"
                    )
                    preview = document.preview_text.replace("\n", " ")
                    if preview:
                        st.caption(preview)
                    _confirm_time_key = f"confirm_remove_at::{document.id}"
                    _confirm_at = st.session_state.get(_confirm_time_key, 0)
                    _confirming = (time.time() - _confirm_at) < 15
                    if _confirming:
                        if st.button("Confirm delete?", key=f"remove_doc::{document.id}", type="primary"):
                            profile_store.remove_document(refreshed_profile.id, document.id)
                            st.session_state.pop(_confirm_time_key, None)
                            st.rerun()
                    else:
                        if st.button("Remove", key=f"remove_doc::{document.id}", help=f"Remove '{document.title}' from profile"):
                            st.session_state[_confirm_time_key] = time.time()
                            st.rerun()
                    st.divider()
                    if checked:
                        updated_selected_ids.add(document.id)

    set_profile_selection(
        refreshed_profile.id,
        documents,
        updated_selected_ids,
        update_widgets=False,
    )
    selected_documents = [
        document for document in documents if document.id in updated_selected_ids
    ]
    return refreshed_profile, documents, selected_documents
