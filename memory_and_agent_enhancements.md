# Memory and Agent Prompt Enhancements

This document explains how conversation memory works in this project and what changed in the agent system prompts to improve behavior for both document related and unrelated questions.

## Do the agents really have memory

Yes, but only when PostgreSQL is configured and the request includes a session id. The memory is implemented with LangGraph checkpointing and stores the graph state between requests. That state includes the message history and other fields in `AgentState`. It is not a vector memory and it does not replace the document retrieval system.

### What is stored

- Conversation history in the `messages` channel. This is appended on each turn and persisted when checkpointing is enabled.
- Other state fields such as `route`, `answer`, and `citations` from prior runs. These are stored for continuity but are not explicitly used by the retrieval step today.

### What is not stored

- Document embeddings or retrieval indexes. Those live in Chroma.
- Any external knowledge outside your uploaded documents.
- Long term memory beyond the checkpoint tables in Postgres.

## How memory works end to end

1. The UI generates a `session_id` and stores it in Streamlit session state at startup. See `src/agentic_rag/ui/utils/state.py`.
2. When memory is enabled in the UI sidebar, the session id is included in each request to the API. See `src/agentic_rag/ui/components/sidebar.py` and `src/agentic_rag/ui/utils/api_client.py`.
3. The API endpoint `/chat/ask_agentic` checks whether `session_id` is present. See `src/agentic_rag/api/routes/chat.py`.
4. If a session id exists, the API attempts to build a Postgres checkpointer from `POSTGRES_DSN`. See `src/agentic_rag/db/checkpoint.py`.
5. If the checkpointer is available, the graph is compiled with checkpointing and `thread_id` is set to the session id. See `src/agentic_rag/agents/graph.py`.
6. LangGraph persists state to Postgres so the next request with the same session id sees the prior state.

### How to enable memory in practice

1. Make sure PostgreSQL is running and reachable from the API.
2. Set `POSTGRES_DSN` in `.env` or via environment variables. Example format: `postgresql://user:password@host:5432/dbname`.
3. Start the API. On startup it will attempt to create the LangGraph checkpoint tables.
4. In the UI, enable memory and keep the same session id across turns.
5. Confirm `memory_enabled` is true in the API response or UI message metadata.

To reset memory, start a new session in the UI. This generates a new session id and uses a new checkpoint thread.

### When memory is disabled

Memory is disabled if any of the following is true:

- `session_id` is missing from the request.
- `POSTGRES_DSN` is not set.
- PostgreSQL is unreachable or the connection fails.

In those cases the API falls back to a stateless graph and returns `memory_enabled: false`.

### Where the tables come from

The app startup calls `setup_checkpointer_tables()` in `src/agentic_rag/api/main.py`. This creates the LangGraph checkpoint tables if PostgreSQL is configured. If it fails, the app logs a warning and continues without memory.

## Practical implications for behavior

- Direct mode responses can use the conversation history because `messages` are included in the LLM call for direct route answers.
- Retrieval mode still uses the current query and the retrieved sources only. It does not yet incorporate previous turns into the retrieval or the answer prompt. Memory here is for continuity and future extensibility, not for retrieval context.

## Agent prompt enhancements

Your request was to make the agents more professional for off topic questions and more capable of synthesis for document related questions. Two prompts were updated.

### Direct response mode prompt

File: `src/agentic_rag/agents/nodes.py`

What changed:

- The system prompt now enforces a professional boundary for off topic questions.
- It explicitly tells the model to describe the assistant and its capabilities when asked who it is.
- It requires varied phrasing across similar refusals while keeping the same meaning.
- It keeps responses concise and prohibits questions at the end.

Effect:

- If a user asks for general help unrelated to the documents, the assistant responds politely, explains it is document focused, and states what it can do instead.
- If the user asks about the assistant itself or the conversation, the response stays grounded in the conversation history.

### Retrieval mode prompt

File: `src/agentic_rag/rag/answering.py`

What changed:

- The prompt now requires synthesis and analysis when the question calls for it, while staying grounded in the sources.
- It enforces citation on every factual claim and prohibits external knowledge.
- It allows analysis but requires it to be clearly labeled as Analysis and tied back to sources.
- It keeps a direct answer first and prevents the model from asking questions.

Effect:

- The assistant can answer higher level tasks like summaries, comparisons, critiques, or recommendations using only the uploaded documents.
- The output makes it easier to separate facts from analysis without losing citations.

## How to validate the behavior

1. Enable memory in the UI and keep the same session id.
2. Ask a direct, off topic question like "Who are you" or "Help me build a web app".
3. Confirm the response is professional, does not ask questions, and explains that the assistant is document focused.
4. Ask a document based question that requires synthesis, such as a summary or comparison across two files.
5. Confirm the answer includes citations and labels any analysis explicitly.

## Summary of changes made

- Updated direct mode prompt in `src/agentic_rag/agents/nodes.py` for professional boundaries and consistent off topic handling.
- Updated retrieval mode prompt in `src/agentic_rag/rag/answering.py` to support deeper synthesis and clearer analysis while remaining fully grounded in sources.
