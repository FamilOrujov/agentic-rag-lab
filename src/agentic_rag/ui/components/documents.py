"""Document upload and management component."""
from __future__ import annotations

import streamlit as st
from agentic_rag.ui.utils.state import add_document
from agentic_rag.ui.utils.api_client import get_client


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    mime_types = {
        "pdf": "application/pdf",
        "txt": "text/plain",
        "md": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "html": "text/html",
        "htm": "text/html",
    }
    return mime_types.get(ext, "application/octet-stream")


def render_documents() -> None:
    """Render the document management section."""
    
    st.markdown("### 📁 Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, MD, or DOCX files",
        key="file_uploader",
    )
    
    # Upload button
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"{len(uploaded_files)} file(s) selected")
        with col2:
            if st.button("📤 Upload All", use_container_width=True):
                upload_files(uploaded_files)
    
    st.divider()
    
    # Uploaded documents list
    if st.session_state.uploaded_docs:
        st.markdown("**Uploaded Documents:**")
        
        # Select all / none
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_doc_ids = [
                    d["doc_id"] for d in st.session_state.uploaded_docs
                ]
                st.rerun()
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.selected_doc_ids = []
                st.rerun()
        
        # Document cards
        for doc in st.session_state.uploaded_docs:
            render_document_card(doc)
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem; opacity: 0.6;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <p>No documents uploaded yet.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_document_card(doc: dict) -> None:
    """Render a single document card."""
    doc_id = doc["doc_id"]
    filename = doc["filename"]
    
    is_selected = doc_id in st.session_state.selected_doc_ids
    
    col1, col2, col3 = st.columns([0.5, 3, 1])
    
    with col1:
        # Checkbox for selection
        if st.checkbox(
            "",
            value=is_selected,
            key=f"doc_select_{doc_id}",
            label_visibility="collapsed",
        ):
            if doc_id not in st.session_state.selected_doc_ids:
                st.session_state.selected_doc_ids.append(doc_id)
        else:
            if doc_id in st.session_state.selected_doc_ids:
                st.session_state.selected_doc_ids.remove(doc_id)
    
    with col2:
        # Document name with tag hint
        tag_name = f"@{filename}"
        st.markdown(f"📄 **{filename}**")
        st.caption(f"Use `{tag_name}` in chat")
    
    with col3:
        st.caption(f"`{doc_id[:8]}...`")


def upload_files(files) -> None:
    """Upload files to the backend."""
    if not st.session_state.backend_status:
        st.error("❌ Backend is offline")
        return
    
    client = get_client()
    
    # Prepare files for upload
    files_data = []
    for f in files:
        content = f.read()
        mime_type = get_mime_type(f.name)
        files_data.append((f.name, content, mime_type))
    
    with st.spinner(f"Uploading {len(files)} file(s)..."):
        result = client.upload_documents(files_data)
    
    if result and "error" not in result:
        # Add documents to state
        for doc in result.get("documents", []):
            add_document(doc["doc_id"], doc["filename"])
        
        summary = result.get("summary", {})
        st.success(
            f"✅ Uploaded {summary.get('documents_ingested', 0)} documents, "
            f"{summary.get('total_chunks_added', 0)} chunks created"
        )
        st.rerun()
    else:
        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")

