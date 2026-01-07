import streamlit as st

from agentic_rag.ui.utils import state as ui_state


def _reset_state() -> None:
    st.session_state.clear()
    ui_state.init_session_state()


def test_parse_file_tags_single() -> None:
    _reset_state()
    ui_state.add_document("id1", "doc1.pdf")

    query, doc_ids = ui_state.parse_file_tags("What is in @doc1.pdf")

    assert query == "What is in"
    assert doc_ids == ["id1"]


def test_parse_file_tags_multiple() -> None:
    _reset_state()
    ui_state.add_document("id1", "doc1.pdf")
    ui_state.add_document("id2", "notes.md")

    query, doc_ids = ui_state.parse_file_tags("Compare @doc1.pdf and @notes.md")

    assert query == "Compare and"
    assert doc_ids == ["id1", "id2"]
