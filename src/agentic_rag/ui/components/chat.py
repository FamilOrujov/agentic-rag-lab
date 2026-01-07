"""Chat interface component."""
from __future__ import annotations

import json
import streamlit as st
from agentic_rag.ui.utils.state import (
    add_message,
    get_session_id,
    parse_file_tags,
    Message,
)
from agentic_rag.ui.utils.api_client import get_client


def render_doc_mentions() -> None:
    """Inject JS to show @mention suggestions for uploaded docs."""
    doc_names = [d["filename"] for d in st.session_state.uploaded_docs]
    docs_json = json.dumps(doc_names)
    st.html(
        f"""
        <script>
        (function() {{
            const docList = {docs_json};
            window.__docMentionList = Array.isArray(docList) ? docList : [];

            function ensurePopup(wrapper) {{
                if (!wrapper) return null;
                let popup = wrapper.querySelector(".doc-mention-popup");
                if (!popup) {{
                    popup = document.createElement("div");
                    popup.className = "doc-mention-popup";
                    popup.style.display = "none";
                    wrapper.appendChild(popup);
                }}
                return popup;
            }}

            function findMention(value, cursor) {{
                if (cursor === null || cursor === undefined) {{
                    cursor = value.length;
                }}
                const upto = value.slice(0, cursor);
                const atIndex = upto.lastIndexOf("@");
                if (atIndex === -1) return null;
                if (atIndex > 0 && !/\\s/.test(upto[atIndex - 1])) return null;
                const query = upto.slice(atIndex + 1);
                if (/[\\s]/.test(query)) return null;
                return {{ start: atIndex, end: cursor, query }};
            }}

            function renderSuggestions(input) {{
                const wrapper = input.closest('div[data-testid="stChatInput"]');
                const popup = ensurePopup(wrapper);
                if (!popup) return;

                const mention = findMention(input.value || "", input.selectionStart);
                const list = window.__docMentionList || [];
                if (!mention || list.length === 0) {{
                    popup.style.display = "none";
                    popup.innerHTML = "";
                    return;
                }}

                const query = mention.query.toLowerCase();
                const matches = list.filter((name) =>
                    name.toLowerCase().includes(query)
                );

                popup.innerHTML = "";
                if (!matches.length) {{
                    const empty = document.createElement("div");
                    empty.className = "doc-mention-empty";
                    empty.textContent = "No matching documents";
                    popup.appendChild(empty);
                    popup.style.display = "block";
                    return;
                }}

                matches.slice(0, 8).forEach((name) => {{
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "doc-mention-item";
                    btn.textContent = name;
                    btn.addEventListener("click", () => {{
                        const before = input.value.slice(0, mention.start);
                        const after = input.value.slice(mention.end);
                        const insert = "@" + name + " ";
                        const nextValue = before + insert + after;
                        input.value = nextValue;
                        const caret = before.length + insert.length;
                        input.setSelectionRange(caret, caret);
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.focus();
                        popup.style.display = "none";
                    }});
                    popup.appendChild(btn);
                }});

                popup.style.display = "block";
            }}

            function bindInput(input) {{
                if (input.dataset.docMentionBound === "1") return;
                input.dataset.docMentionBound = "1";

                input.addEventListener("input", () => renderSuggestions(input));
                input.addEventListener("click", () => renderSuggestions(input));
                input.addEventListener("keydown", (event) => {{
                    if (event.key === "Escape") {{
                        const wrapper = input.closest('div[data-testid="stChatInput"]');
                        const popup = wrapper ? wrapper.querySelector(".doc-mention-popup") : null;
                        if (popup) {{
                            popup.style.display = "none";
                        }}
                    }}
                }});
            }}

            function tick() {{
                const input = document.querySelector('div[data-testid="stChatInput"] textarea');
                if (input) {{
                    bindInput(input);
                }}
            }}

            if (!window.__docMentionInterval) {{
                window.__docMentionInterval = setInterval(tick, 400);
            }} else {{
                tick();
            }}

            if (!window.__docMentionClickBound) {{
                window.__docMentionClickBound = true;
                document.addEventListener("click", (event) => {{
                    const popup = document.querySelector(".doc-mention-popup");
                    const input = document.querySelector('div[data-testid="stChatInput"] textarea');
                    if (!popup || !input) return;
                    if (popup.contains(event.target) || input.contains(event.target)) return;
                    popup.style.display = "none";
                }});
            }}
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_message(msg: Message, idx: int) -> None:
    """Render a single chat message."""
    is_user = msg.role == "user"
    
    with st.chat_message(msg.role, avatar="👤" if is_user else "🤖"):
        # Message content
        st.markdown(msg.content)
        
        # Metadata for assistant messages
        if not is_user:
            cols = st.columns([1, 1, 2])
            with cols[0]:
                if msg.route:
                    route_class = "route-direct" if msg.route == "direct" else "route-retrieve"
                    route_icon = "💬" if msg.route == "direct" else "📚"
                    st.caption(f"{route_icon} {msg.route}")
            with cols[1]:
                if msg.memory_enabled:
                    st.caption("🧠 memory")
            
            # Citations
            if msg.citations:
                with st.expander(f"📎 {len(msg.citations)} sources", expanded=False):
                    for cite in msg.citations:
                        st.markdown(
                            f"**[{cite.get('source_id', 'S?')}]** "
                            f"`{cite.get('chunk_id', '')[:20]}...`"
                        )


def render_chat_input() -> None:
    """Render the chat input area with memory toggle."""
    pass  # Memory toggle moved to sidebar for better UX


def process_message(user_input: str) -> None:
    """Process user message and get response."""
    if not user_input.strip():
        return
    
    # Check backend
    if not st.session_state.backend_status:
        st.error("❌ Backend is offline. Please start the FastAPI server.")
        return
    
    # Parse @file tags
    clean_query, tagged_doc_ids = parse_file_tags(user_input)
    
    # Combine tagged docs with selected docs
    doc_ids = list(set(tagged_doc_ids + st.session_state.selected_doc_ids))
    
    # Use clean query if tags were found, otherwise original
    query = clean_query if clean_query else user_input
    
    # Add user message
    add_message("user", user_input)
    
    # Get response
    client = get_client()
    
    with st.spinner(""):
        response = client.ask_agentic(
            query=query,
            k=st.session_state.k_value,
            doc_ids=doc_ids if doc_ids else None,
            session_id=get_session_id(),
            provider=st.session_state.model_provider,
            model=st.session_state.model_name,
            api_key=st.session_state.api_key if st.session_state.api_key else None,
            temperature=st.session_state.temperature,
        )
    
    # Add assistant response
    add_message(
        "assistant",
        response.get("answer", "Sorry, I couldn't generate a response."),
        route=response.get("route"),
        memory_enabled=response.get("memory_enabled", False),
        citations=response.get("citations"),
    )


def render_chat() -> None:
    """Render the main chat interface."""
    
    # Chat header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.8rem; font-weight: 600; margin: 0;">
                💬 Agentic RAG Chat
            </h1>
            <p style="margin-top: 0.5rem;">
                Ask questions about your documents or have a conversation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            # Empty state
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                    <p>Start a conversation or upload documents to begin.</p>
                    <p style="font-size: 0.85rem;">
                        Tip: Use <code>@filename</code> to query specific documents.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Render all messages
            for idx, msg in enumerate(st.session_state.messages):
                render_message(msg, idx)
    
    # File tags hint if documents uploaded
    if st.session_state.uploaded_docs:
        doc_names = [d["filename"] for d in st.session_state.uploaded_docs[:3]]
        hint = ", ".join([f"@{n}" for n in doc_names])
        if len(st.session_state.uploaded_docs) > 3:
            hint += "..."
        st.caption(f"💡 Tag files in your message: {hint}")
    
    # Chat input
    render_doc_mentions()
    if user_input := st.chat_input(
        "Ask anything...",
        key="chat_input",
    ):
        process_message(user_input)
        st.rerun()
