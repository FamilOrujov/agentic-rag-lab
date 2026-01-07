"""API client for FastAPI backend communication."""
from __future__ import annotations

import requests
from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIClient:
    """Client for the Agentic RAG FastAPI backend."""
    
    base_url: str = "http://127.0.0.1:8000"
    timeout: int = 300  # 5 minutes for slow models
    
    def health_check(self) -> bool:
        """Check if the backend is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def get_meta(self) -> dict[str, Any] | None:
        """Get backend metadata."""
        try:
            resp = requests.get(f"{self.base_url}/meta", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None
    
    def upload_documents(self, files: list[tuple[str, bytes, str]]) -> dict[str, Any] | None:
        """
        Upload documents to the backend.
        
        Args:
            files: List of (filename, content, mime_type) tuples
        
        Returns:
            Upload response or None on error
        """
        try:
            files_data = [
                ("files", (name, content, mime_type))
                for name, content, mime_type in files
            ]
            resp = requests.post(
                f"{self.base_url}/documents/upload",
                files=files_data,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}
    
    def ask(
        self,
        query: str,
        k: int = 6,
        doc_ids: list[str] | None = None,
        max_context_chars: int = 12000,
    ) -> dict[str, Any]:
        """
        Classic RAG query with citations.
        """
        payload = {
            "query": query,
            "k": k,
            "max_context_chars": max_context_chars,
        }
        if doc_ids:
            payload["doc_ids"] = doc_ids
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/ask",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text, "answer": "Error communicating with backend."}
        except Exception as e:
            return {"error": str(e), "answer": f"Connection error: {e}"}
    
    def ask_agentic(
        self,
        query: str,
        k: int = 6,
        doc_ids: list[str] | None = None,
        max_context_chars: int = 12000,
        session_id: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Agentic RAG query with optional memory and provider selection.
        """
        payload = {
            "query": query,
            "k": k,
            "max_context_chars": max_context_chars,
            "provider": provider,
            "temperature": temperature,
        }
        if doc_ids:
            payload["doc_ids"] = doc_ids
        if session_id:
            payload["session_id"] = session_id
        if model:
            payload["model"] = model
        if api_key:
            payload["api_key"] = api_key
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/ask_agentic",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text, "answer": "Error communicating with backend."}
        except Exception as e:
            return {"error": str(e), "answer": f"Connection error: {e}"}
    
    def retrieve(
        self,
        query: str,
        k: int = 6,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve chunks without generating an answer.
        """
        payload = {"query": query, "k": k}
        if doc_ids:
            payload["doc_ids"] = doc_ids
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/retrieve",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}
    
    def warmup_ollama(self, model: str = "gemma3:4b") -> bool:
        """
        Warm up Ollama by making a simple request.
        """
        try:
            # Make a simple query to load the model
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": "hi", "stream": False},
                timeout=60,
            )
            return resp.status_code == 200
        except Exception:
            return False
    
    def list_ollama_models(self) -> list[str]:
        """
        Fetch list of installed Ollama models via the Ollama API.
        
        Returns:
            List of model names (e.g., ["gemma3:4b", "llama3.2:latest"])
        """
        try:
            resp = requests.get(
                "http://localhost:11434/api/tags",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []


# Singleton client
_client: APIClient | None = None


def get_client(base_url: str = "http://127.0.0.1:8000") -> APIClient:
    """Get or create the API client."""
    global _client
    if _client is None or _client.base_url != base_url:
        _client = APIClient(base_url=base_url)
    return _client

