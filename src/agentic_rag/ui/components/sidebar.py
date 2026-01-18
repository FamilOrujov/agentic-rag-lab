"""Sidebar component with model settings."""
from __future__ import annotations

import html
import streamlit as st
from agentic_rag.ui.utils.state import reset_chat
from agentic_rag.ui.utils.api_client import get_client


# Default fallback models for each provider
DEFAULT_OLLAMA_MODELS = ["gemma3:4b", "llama3.2:latest", "mistral:latest", "phi3:latest"]

MODEL_PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "models": [],  # Populated dynamically from Ollama API
        "requires_key": False,
    },
}


def _get_ollama_models() -> list[str]:
    """Fetch Ollama models, using cache to avoid repeated API calls."""
    if "ollama_models_cache" not in st.session_state:
        client = get_client()
        models = client.list_ollama_models()
        st.session_state.ollama_models_cache = models if models else DEFAULT_OLLAMA_MODELS
    return st.session_state.ollama_models_cache


def _refresh_ollama_models() -> list[str]:
    """Force refresh of Ollama models cache."""
    client = get_client()
    models = client.list_ollama_models()
    st.session_state.ollama_models_cache = models if models else DEFAULT_OLLAMA_MODELS
    return st.session_state.ollama_models_cache


def render_sidebar() -> None:
    """Render the settings sidebar."""
    with st.sidebar:
        # Header
        st.markdown("## ⚡ Agentic RAG")
        
        st.divider()
        
        # Backend status
        client = get_client()
        status = client.health_check()
        st.session_state.backend_status = status
        
        status_color = "🟢" if status else "🔴"
        status_text = "Backend Online" if status else "Backend Offline"
        selected_model = html.escape(str(st.session_state.get("model_name") or "N/A"))
        model_html = f'<div class="sidebar-status-model">Model: <code>{selected_model}</code></div>'

        st.markdown(
            f"""
            <div class="sidebar-status">
                <div class="sidebar-status-line">{status_color} <strong>{status_text}</strong></div>
                {model_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.divider()
        
        # Model Provider Selection
        st.markdown("### 🤖 Model Settings")
        
        provider = st.selectbox(
            "Provider",
            options=list(MODEL_PROVIDERS.keys()),
            format_func=lambda x: MODEL_PROVIDERS[x]["name"],
            key="model_provider",
        )
        
        provider_info = MODEL_PROVIDERS[provider]
        
        # Model selection - use dynamic list for Ollama
        if provider == "ollama":
            ollama_models = _get_ollama_models()
            st.selectbox(
                "Model",
                options=ollama_models,
                key="model_name",
            )
        else:
            st.selectbox(
                "Model",
                options=provider_info["models"],
                key="model_name",
            )
        
        # API Key input for non-Ollama providers
        if provider_info["requires_key"]:
            st.text_input(
                "API Key",
                type="password",
                placeholder=f"Enter your {provider_info['name']} API key",
                key="api_key",
            )
            if not st.session_state.api_key:
                st.warning(f"⚠️ API key required for {provider_info['name']}")
        
        # Ollama controls
        if provider == "ollama":
            if st.button("🔄 Refresh", help="Refresh available models", use_container_width=True):
                _refresh_ollama_models()
                st.rerun()
        
        st.divider()
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                key="temperature",
                help="Higher = more creative, Lower = more focused",
            )
            
            st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="top_p",
                help="Nucleus sampling threshold",
            )
            
            st.slider(
                "Retrieved Chunks (k)",
                min_value=1,
                max_value=20,
                step=1,
                key="k_value",
                help="Number of document chunks to retrieve",
            )
        
        st.divider()
        
        # Session controls
        st.markdown("### 💬 Session")
        
        # Memory toggle - prominently placed
        memory_on = st.toggle(
            "🧠 Enable Memory",
            value=st.session_state.memory_enabled,
            key="memory_toggle_sidebar",
            help="Remember conversation history across messages",
        )
        # Sync with session state
        if memory_on != st.session_state.memory_enabled:
            st.session_state.memory_enabled = memory_on
        
        # Session info when memory is enabled
        if st.session_state.memory_enabled:
            st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                reset_chat()
                st.rerun()
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                reset_chat()
                st.rerun()
        
        st.divider()
        
        # Footer
        st.markdown(
            """
            <div style="text-align: center; opacity: 0.6; font-size: 0.8rem;">
                Built with ❤️ using LangGraph
            </div>
            """,
            unsafe_allow_html=True,
        )
