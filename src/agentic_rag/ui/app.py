"""
Agentic RAG - Streamlit UI

A modern chat interface for the Agentic RAG system with:
- Dark/Light theme toggle
- Multi-provider LLM support
- Conversation memory
- Document upload and @file tagging
- ChatGPT-style interface

Run: streamlit run src/agentic_rag/ui/app.py
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

# Import components
from agentic_rag.ui.utils.state import init_session_state
from agentic_rag.ui.components.sidebar import render_sidebar
from agentic_rag.ui.components.chat import render_chat
from agentic_rag.ui.components.documents import render_documents


def load_css() -> None:
    """Load custom CSS styles."""
    css_path = Path(__file__).parent / "styles" / "main.css"
    if css_path.exists():
        with open(css_path) as f:
            css = f.read()
    else:
        css = ""
    
    # Dark theme CSS (only theme)
    theme_css = """
        .stApp {
            background-color: #121212;
            color: #f5f5f5;
        }

        div[data-testid="stAppViewContainer"], section.main {
            background-color: #121212;
        }
        
        /* Force all text to be visible */
        .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, 
        .stApp h4, .stApp h5, .stApp h6, .stApp div, .stApp li {
            color: #f5f5f5 !important;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
            border-right: 1px solid #404040;
        }
        
        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #f5f5f5 !important;
        }
        
        .stChatMessage {
            background-color: #1e1e1e;
            border: 1px solid #404040;
        }
        
        .stChatMessage p, .stChatMessage span {
            color: #f5f5f5 !important;
        }
        
        .stChatInputContainer {
            background-color: #1e1e1e;
            border-top: 1px solid #404040;
        }
        
        .stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
            background-color: #2a2a2a !important;
            color: #f5f5f5 !important;
            border-color: #404040 !important;
        }
        
        .stButton button {
            background-color: #2a2a2a;
            color: #f5f5f5 !important;
            border-color: #404040;
        }
        
        .stButton button:hover {
            background-color: #3a3a3a;
            border-color: #10a37f;
        }
        
        .stExpander {
            background-color: #1e1e1e;
            border: 1px solid #404040;
        }
        
        .stExpander p, .stExpander span {
            color: #f5f5f5 !important;
        }
        
        .stDivider {
            border-color: #404040;
        }
        
        .stCaption, .stCaption p {
            color: #b0b0b0 !important;
        }

        .sidebar-status-line {
            color: #f5f5f5 !important;
        }

        .sidebar-status-model {
            color: #e0e0e0 !important;
        }

        .sidebar-status-model code {
            color: #10a37f !important;
        }
        
        div[data-testid="stFileUploader"] {
            background-color: #1e1e1e;
            border: 1px dashed #505050;
            border-radius: 12px;
        }
        
        div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] span {
            color: #e0e0e0 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #b0b0b0 !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
        }
        
        /* Chat input styling */
        .stChatInput {
            border-color: #404040 !important;
        }
        
        .stChatInput:focus-within {
            border-color: #10a37f !important;
        }
        
        /* Markdown content */
        .stMarkdown, .stMarkdown p {
            color: #f5f5f5 !important;
        }
        
        /* Code blocks */
        code {
            background-color: #2a2a2a !important;
            color: #10a37f !important;
        }

        /* Doc mention popup */
        .doc-mention-popup {
            background-color: #1e1e1e;
            border: 1px solid #404040;
            color: #f5f5f5;
        }

        .doc-mention-item {
            color: #f5f5f5;
        }

        .doc-mention-item:hover {
            background-color: #2a2a2a;
        }

        .doc-mention-empty {
            color: #b0b0b0;
        }
        """
    
    # Combine CSS
    full_css = f"""
    <style>
    {css}
    {theme_css}
    
    /* Hide Streamlit branding (keep header for sidebar toggle) */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: visible;}}
    header[data-testid="stHeader"] {{
        background: #121212 !important;
        border-bottom: 1px solid #2a2a2a !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }}
    header[data-testid="stHeader"]::before {{
        background: #121212 !important;
    }}
    header [data-testid="stToolbar"] {{
        background: transparent !important;
    }}
    header [data-testid="stDecoration"] {{
        display: none;
    }}
    
    /* Keep sidebar collapse control visible */
    header [data-testid="stSidebarCollapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #1a1a1a;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: #3d3d3d;
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #4d4d4d;
    }}
    
    /* Chat message animations */
    .stChatMessage {{
        animation: fadeIn 0.3s ease-out;
        border-radius: 12px;
        margin-bottom: 1rem;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Toggle styling */
    .stCheckbox label {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* File uploader hover effect */
    div[data-testid="stFileUploader"]:hover {{
        border-color: #10a37f;
    }}
    
    /* Primary button accent */
    .stButton button[kind="primary"] {{
        background-color: #10a37f !important;
        border-color: #10a37f !important;
        color: white !important;
    }}
    
    .stButton button[kind="primary"]:hover {{
        background-color: #0d8a6a !important;
    }}
    
    /* Fix chat input - align with main content */
    div[data-testid="stChatInput"] {{
        position: relative !important;
        padding: 16px 0 20px !important;
        background: #121212 !important;
        z-index: 1000 !important;
        pointer-events: auto !important;
        overflow: visible !important;
    }}
    
    div[data-testid="stChatInput"] > div {{
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    div[data-testid="stChatInput"] textarea {{
        pointer-events: auto !important;
        background-color: #2a2a2a !important;
        color: #f5f5f5 !important;
        border: 1px solid #505050 !important;
        border-radius: 24px !important;
        padding: 12px 48px 12px 16px !important;
        width: 100% !important;
    }}
    
    div[data-testid="stChatInput"] textarea:focus {{
        border-color: #10a37f !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
    }}
    
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #888888 !important;
    }}

    .doc-mention-popup {{
        position: absolute;
        left: 16px;
        right: 16px;
        bottom: calc(100% + 8px);
        max-height: 220px;
        overflow-y: auto;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        z-index: 1001;
    }}

    .doc-mention-item {{
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 0.95rem;
    }}

    .doc-mention-empty {{
        padding: 8px 12px;
        font-size: 0.9rem;
    }}
    
    /* Add padding to chat content so it's not hidden by the chat input */
    .stChatMessageContainer {{
        padding-bottom: 100px !important;
    }}
    </style>
    """
    
    st.markdown(full_css, unsafe_allow_html=True)


def main() -> None:
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="Agentic RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    init_session_state()
    
    # Load CSS
    load_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area with tabs
    tab_chat, tab_docs = st.tabs(["💬 Chat", "📁 Documents"])
    
    with tab_chat:
        render_chat()
    
    with tab_docs:
        render_documents()


if __name__ == "__main__":
    main()
