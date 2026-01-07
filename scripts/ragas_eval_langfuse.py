"""
Ragas Evaluation Script with Langfuse Integration.

This script pulls traces from Langfuse by tag, evaluates them using Ragas metrics
(Faithfulness, Context Precision, optionally Response Relevancy), and pushes
the evaluation scores back to Langfuse for analysis.

Usage:
    python scripts/ragas_eval_langfuse.py --tag ragas_eval --limit 20

Requirements:
    - Langfuse credentials in environment (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)
    - Ollama running locally with the configured models
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Using deprecated ragas.metrics imports because ragas.metrics.collections
# only supports InstructorLLM, not LangChain models (ChatOllama).
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

from langfuse.api.client import FernLangfuse

from agentic_rag.config.config import settings


@dataclass(frozen=True, slots=True)
class EvalRow:
    """A single evaluation row extracted from a Langfuse trace."""

    trace_id: str
    user_input: str
    response: str
    retrieved_contexts: list[str]


def normalize_to_str_list(value: Any) -> list[str]:
    """
    Normalize any value to a list of strings for Ragas compatibility.

    Handles: None, str, dict with 'text'/'page_content' keys,
    list/tuple/set, bytes, and other iterables.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        # Common patterns: {"text": "..."} or {"page_content": "..."}
        if "text" in value:
            return [str(value["text"])]
        if "page_content" in value:
            return [str(value["page_content"])]
        return [str(value)]

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    if isinstance(value, (bytes, bytearray)):
        return [value.decode(errors="replace")]

    # For other iterables, convert directly to avoid unintended iteration
    if isinstance(value, Iterable):
        return [str(value)]

    return [str(value)]


def create_api_client() -> FernLangfuse:
    """
    Create the low-level Langfuse API client for trace fetching.

    The FernLangfuse client provides direct REST API access for listing
    and retrieving traces, which is not available on the high-level Langfuse class.
    """
    if not settings.LANGFUSE_BASE_URL:
        raise RuntimeError("LANGFUSE_BASE_URL is not configured")

    public_key = (
        settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
        if settings.LANGFUSE_PUBLIC_KEY
        else None
    )
    secret_key = (
        settings.LANGFUSE_SECRET_KEY.get_secret_value()
        if settings.LANGFUSE_SECRET_KEY
        else None
    )

    if not public_key or not secret_key:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured"
        )

    return FernLangfuse(
        base_url=settings.LANGFUSE_BASE_URL,
        username=public_key,
        password=secret_key,
    )


def extract_eval_row(trace: Any) -> EvalRow:
    """
    Extract evaluation fields from a Langfuse trace.

    Ragas expects: user_input, response, retrieved_contexts.
    Supports multiple input/output key formats for flexibility.
    """
    trace_id = getattr(trace, "id", "") or ""
    inp = getattr(trace, "input", None) or {}
    out = getattr(trace, "output", None) or {}

    # Support multiple key naming conventions
    user_input = (
        inp.get("user_input")
        or inp.get("query")
        or inp.get("question")
        or ""
    )

    response = out.get("response") or out.get("answer") or ""

    retrieved_contexts = (
        out.get("retrieved_contexts")
        or out.get("retrieved_context")
        or out.get("contexts")
        or []
    )

    return EvalRow(
        trace_id=str(trace_id),
        user_input=str(user_input),
        response=str(response),
        retrieved_contexts=normalize_to_str_list(retrieved_contexts),
    )


def fetch_traces(client: FernLangfuse, tag: str, limit: int) -> list[EvalRow]:
    """
    Fetch traces from Langfuse by tag and extract evaluation rows.

    Returns only rows with valid user_input, response, and retrieved_contexts.
    """
    # List traces matching the tag
    traces_response = client.trace.list(tags=[tag], limit=limit)
    traces = traces_response.data or []

    if not traces:
        return []

    rows: list[EvalRow] = []
    for trace in traces:
        # Fetch full trace details (list may return partial objects)
        full_trace = client.trace.get(trace.id)
        row = extract_eval_row(full_trace)

        # Skip rows missing required fields
        if row.user_input and row.response and row.retrieved_contexts:
            rows.append(row)

    return rows


def create_metrics(with_relevancy: bool) -> list:
    """Create the list of Ragas metrics to evaluate."""
    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithoutReference(),
    ]
    if with_relevancy:
        metrics.append(ResponseRelevancy())
    return metrics


def run_evaluation(
    rows: list[EvalRow],
    metrics: list,
    with_relevancy: bool,
) -> EvaluationResult:
    """
    Run Ragas evaluation on the extracted rows.

    Uses local Ollama models for LLM and embeddings.
    """
    # Convert to Ragas dataset format
    dataset = Dataset.from_list([
        {
            "trace_id": row.trace_id,
            "user_input": row.user_input,
            "response": row.response,
            "retrieved_contexts": row.retrieved_contexts,
        }
        for row in rows
    ])

    # Configure local Ollama LLM
    llm = ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0,
    )

    # Optional embeddings for ResponseRelevancy metric
    embeddings = None
    if with_relevancy:
        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBED_MODEL,
        )

    # RAM-safe config for local LLM inference
    run_config = RunConfig(max_workers=1, timeout=180)

    result: EvaluationResult = evaluate(  # type: ignore[assignment]
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    return result


def create_langfuse_client():
    """
    Create a Langfuse client with explicit credentials from settings.
    
    This avoids relying on environment variables being exported in the shell,
    since pydantic-settings already loads them from .env.
    """
    from langfuse import Langfuse

    if not settings.LANGFUSE_BASE_URL:
        return None

    public_key = (
        settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
        if settings.LANGFUSE_PUBLIC_KEY
        else None
    )
    secret_key = (
        settings.LANGFUSE_SECRET_KEY.get_secret_value()
        if settings.LANGFUSE_SECRET_KEY
        else None
    )

    if not public_key or not secret_key:
        return None

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=settings.LANGFUSE_BASE_URL,
    )


def push_scores_to_langfuse(
    rows: list[EvalRow],
    result: EvaluationResult,
    metrics: list,
) -> None:
    """
    Push evaluation scores back to Langfuse.

    Creates scores on each trace using explicit credentials from settings.
    """
    lf = create_langfuse_client()
    if lf is None:
        print("Warning: Langfuse client unavailable, scores not pushed")
        return

    df = result.to_pandas()
    # Ragas preserves dataset row order
    df["trace_id"] = [row.trace_id for row in rows]

    for _, df_row in df.iterrows():
        trace_id = str(df_row["trace_id"])

        for metric in metrics:
            name = metric.name
            value = df_row.get(name)

            if value is None:
                continue

            lf.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                data_type="NUMERIC",
                score_id=f"{trace_id}-{name}",  # Prevents duplicate scores
                comment="ragas batch eval",
            )

    lf.flush()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Langfuse traces using Ragas metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag to filter traces for evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of traces to evaluate",
    )
    parser.add_argument(
        "--with-relevancy",
        action="store_true",
        help="Include ResponseRelevancy metric (requires embeddings)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create API client for fetching traces
    try:
        api_client = create_api_client()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    # Fetch traces by tag
    print(f"Fetching traces with tag '{args.tag}' (limit: {args.limit})...")
    rows = fetch_traces(api_client, args.tag, args.limit)

    if not rows:
        print("No evaluable traces found (need user_input, response, and retrieved_contexts)")
        return 0

    print(f"Found {len(rows)} evaluable traces")

    # Setup metrics
    metrics = create_metrics(args.with_relevancy)
    metric_names = ", ".join(m.name for m in metrics)
    print(f"Running evaluation with metrics: {metric_names}")

    # Run evaluation
    result = run_evaluation(rows, metrics, args.with_relevancy)

    # Push scores back to Langfuse
    push_scores_to_langfuse(rows, result, metrics)

    print(f"Evaluated {len(rows)} traces and pushed scores to Langfuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
