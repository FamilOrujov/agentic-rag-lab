<h1 align="center">Agentic RAG</h1>

<h3 align="center">Intelligent Document Question Answering System with Agentic Retrieval and Memory</h3>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.127+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-1.0+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-1.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"></a>
  <a href="https://www.trychroma.com"><img src="https://img.shields.io/badge/ChromaDB-1.4+-FF6F61?style=for-the-badge&logoColor=white" alt="ChromaDB"></a>
</p>

<p align="center">
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.52+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-2.12+-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic"></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-17-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"></a>
</p>

<p align="center">
  <a href="https://langfuse.com"><img src="https://img.shields.io/badge/Langfuse-Observability-4F46E5?style=for-the-badge&logoColor=white" alt="Langfuse"></a>
  <a href="https://docs.ragas.io/"><img src="https://img.shields.io/badge/RAGAS-Evaluation-FF6B6B?style=for-the-badge&logoColor=white" alt="RAGAS"></a>
  <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/Pytest-8.3+-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest"></a>
  <a href="https://docker.com"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"></a>
</p>

<p align="center">
  <em>Agentic RAG is a production ready document question answering system that I built to demonstrate how retrieval augmented generation can be elevated through intelligent routing and conversation memory. The system autonomously decides whether to retrieve documents or respond directly, while maintaining conversational context across interactions through PostgreSQL backed checkpointing.</em>
</p>

---

## Table of Contents

1. [Overview](#1-overview)
   - 1.1 [The Business Problem](#11-the-business-problem)
   - 1.2 [The Technical Solution](#12-the-technical-solution)
   - 1.3 [Key Capabilities](#13-key-capabilities)
   - 1.4 [Why I Built This](#14-why-i-built-this)
2. [Project Structure](#2-project-structure)
   - 2.1 [Directory Layout](#21-directory-layout)
   - 2.2 [Core Components](#22-core-components)
3. [System Architecture](#3-system-architecture)
   - 3.1 [Pipeline Overview](#31-pipeline-overview)
   - 3.2 [Data Flow](#32-data-flow)
   - 3.3 [Component Interactions](#33-component-interactions)
   - 3.4 [Design Decisions](#34-design-decisions)
4. [Features](#4-features)
   - 4.1 [Intelligent Query Routing](#41-intelligent-query-routing)
   - 4.2 [Conversation Memory](#42-conversation-memory)
   - 4.3 [Document Ingestion Pipeline](#43-document-ingestion-pipeline)
   - 4.4 [Streamlit Dashboard](#44-streamlit-dashboard)
   - 4.5 [Langfuse Observability](#45-langfuse-observability)
   - 4.6 [RAGAS Evaluation](#46-ragas-evaluation)
   - 4.7 [Docker Containerization](#47-docker-containerization)
5. [Quick Start](#5-quick-start)
   - 5.1 [Prerequisites](#51-prerequisites)
   - 5.2 [Installation with uv](#52-installation-with-uv)
   - 5.3 [Installation with pip](#53-installation-with-pip)
   - 5.4 [Environment Configuration](#54-environment-configuration)
   - 5.5 [Running the API Server](#55-running-the-api-server)
   - 5.6 [Launching the UI](#56-launching-the-ui)
   - 5.7 [Docker Deployment](#57-docker-deployment)
6. [Usage Guide](#6-usage-guide)
   - 6.1 [Uploading Documents](#61-uploading-documents)
   - 6.2 [Querying Documents](#62-querying-documents)
   - 6.3 [Using Session Memory](#63-using-session-memory)
   - 6.4 [Running RAGAS Evaluation](#64-running-ragas-evaluation)
7. [API Reference](#7-api-reference)
   - 7.1 [Document Management](#71-document-management)
   - 7.2 [Chat Endpoints](#72-chat-endpoints)
   - 7.3 [Health and Metadata](#73-health-and-metadata)
8. [Configuration](#8-configuration)
   - 8.1 [Environment Variables](#81-environment-variables)
   - 8.2 [LLM Configuration](#82-llm-configuration)
   - 8.3 [Storage Settings](#83-storage-settings)
9. [Testing](#9-testing)
10. [License](#10-license)

---

## 1. Overview

### 1.1 The Business Problem

Organizations accumulate vast collections of documents including reports, policies, technical specifications, and research papers. Extracting actionable insights from these materials requires either time consuming manual review or sophisticated search capabilities that traditional keyword based systems cannot provide. Users need a way to query documents using natural language and receive accurate, cited answers without reading through entire documents.

### 1.2 The Technical Solution

Agentic RAG addresses these challenges by implementing a state graph architecture using LangGraph where an AI agent autonomously determines the optimal response strategy for each query. I designed the system to combine vector similarity search over document embeddings with a routing mechanism that distinguishes between queries requiring document retrieval and those that can be answered conversationally. Unlike traditional RAG systems that always retrieve, this agentic approach reduces latency and improves relevance by skipping retrieval when unnecessary.

### 1.3 Key Capabilities

The system provides four primary capabilities that work together to deliver an intelligent document assistant. First, the routing layer examines each incoming query and classifies it as either requiring document retrieval or suitable for direct response, eliminating unnecessary vector searches for simple conversational exchanges. Second, the retrieval pipeline uses ChromaDB to perform semantic similarity search across chunked documents, returning the most relevant passages with confidence scores. Third, the answering component synthesizes retrieved context into coherent responses with inline citations that trace back to source documents. Fourth, the memory system persists conversation state in PostgreSQL, enabling multi turn dialogues where the assistant recalls prior exchanges within a session.

### 1.4 Why I Built This

I created this project because I observed a gap between simple RAG demonstrations and production ready systems. Most tutorials show retrieval happening on every query, which wastes resources when users ask simple questions like greetings or clarifications. I wanted to build something that thinks before it retrieves, much like a human expert who knows when to consult reference materials and when to respond from memory.

The choice of LangGraph came from my need for a framework that makes agent state management explicit and debuggable. Traditional chain based approaches hide state transitions, making it difficult to understand why an agent took a particular path. With LangGraph, I can visualize the graph, inspect state at each node, and checkpoint conversations reliably.

I opted for Ollama as the LLM backend because I believe in keeping inference local whenever possible. This eliminates API costs, removes external dependencies, and ensures data never leaves the deployment environment. For teams requiring cloud models, the architecture supports extension, but the default configuration runs entirely on local hardware.

The integration with Langfuse and RAGAS reflects my commitment to measurable quality. Building an AI system without observability is like flying blind. By capturing traces and computing evaluation metrics, I can quantify improvements and detect regressions as the system evolves.

---

## 2. Project Structure

### 2.1 Directory Layout

I organized the repository following a modular architecture that separates concerns across distinct packages. This structure evolved from my experience maintaining AI systems where tangled dependencies became maintenance nightmares.

```
agentic-rag/
├── src/
│   └── agentic_rag/
│       ├── agents/          # LangGraph state machine implementation
│       ├── api/             # FastAPI application and route handlers
│       ├── config/          # Pydantic settings management
│       ├── db/              # PostgreSQL checkpoint persistence
│       ├── ops/             # Langfuse observability integration
│       ├── rag/             # Retrieval, ingestion, and LLM utilities
│       └── ui/              # Streamlit dashboard components
├── scripts/                 # Utility scripts including RAGAS evaluation
├── tests/                   # Pytest test suite
├── data/                    # Document upload storage
├── Dockerfile.api           # API service container definition
├── Dockerfile.ui            # UI service container definition
├── docker-compose.yml       # Multi service orchestration
├── pyproject.toml           # Project metadata and dependencies
└── requirements.txt         # Pip compatible dependency list
```

### 2.2 Core Components

I divided the codebase into seven functional modules, each with a single responsibility. The `agents` package defines the LangGraph state machine including node functions for routing, retrieval, answering, and finalization. I kept this separate from the RAG logic so the graph structure can be modified without touching retrieval code.

The `api` package exposes RESTful endpoints through FastAPI. I chose FastAPI for its async support, automatic OpenAPI documentation, and Pydantic integration that validates requests at the boundary.

The `config` package centralizes environment variable management. By using Pydantic Settings, I get type validation and default values in one place, avoiding scattered configuration logic.

The `db` package manages PostgreSQL connection pooling and LangGraph checkpoint persistence. I implemented lazy connection opening so the application starts even when PostgreSQL is unavailable, falling back to stateless mode gracefully.

The `ops` package integrates with Langfuse. I encapsulated all observability concerns here so tracing can be disabled or swapped without modifying business logic.

The `rag` package contains the core retrieval augmented generation logic. This includes document extraction, chunking, embedding, vector storage, and answer synthesis. I separated these concerns into focused modules that can be tested independently.

The `ui` package provides the Streamlit interface. I kept UI components modular with separate files for sidebar, chat, and document management, making it easier to modify individual sections.

---

## 3. System Architecture

### 3.1 Pipeline Overview

The system processes queries through a multi stage pipeline orchestrated by LangGraph. When a request arrives, the routing node examines the query content and determines whether document retrieval is required. I designed the router to use a dedicated LLM call that returns a single classification token, keeping the decision fast and deterministic.

Queries identified as greetings, meta questions about the chat, or general conversation bypass retrieval entirely and proceed directly to the answer node. Queries requiring document context trigger the retrieval node, which searches ChromaDB for semantically similar chunks.

The answer node generates a response using either retrieved context or conversation history depending on the route taken. I structured the prompts to enforce citation requirements so every factual claim traces back to source documents.

Finally, the finalize node performs safety checks. I added this step because I noticed LLMs sometimes end responses with questions, which felt inappropriate for an assistant that should provide answers. The finalize node strips trailing questions and ensures declarative endings.

### 3.2 Data Flow

Document ingestion begins when files are uploaded through the API or UI. I built the extraction layer to process PDF, DOCX, TXT, and HTML files using format specific extractors. PDF processing uses PyMuPDF because it preserves page structure and handles complex layouts better than alternatives I tested.

The splitter component segments text into overlapping chunks. I chose recursive character splitting with overlap because it handles paragraph boundaries gracefully and ensures relevant context is not split across chunk boundaries.

Each chunk receives metadata including document identifiers, source names, and timestamps. I designed the metadata schema to support filtering by document during retrieval, enabling users to scope queries to specific files.

Query processing flows through these stages: API receives the request, LangGraph invokes the routing node, retrieval fetches relevant chunks if needed, the answering component generates a response with citations, and finalization cleans the output before returning to the client.

### 3.3 Component Interactions

The FastAPI application initializes at startup by creating PostgreSQL checkpoint tables if configured and establishing the Langfuse client connection. I implemented lifespan management using FastAPI's context manager pattern, ensuring clean shutdown of database pools and trace buffers.

Request handlers instantiate the appropriate LangGraph variant. For single turn queries, I use a cached stateless graph. For session aware conversations, I create a checkpointed graph with the PostgreSQL saver. This dual mode approach means users without PostgreSQL still get full functionality except memory.

The graph nodes communicate through a typed state dictionary. I defined this using TypedDict so type checkers catch mismatches between what nodes produce and what they consume. State accumulates the query, route decisions, retrieved sources, and final answer as it flows through the graph.

### 3.4 Design Decisions

I made several architectural choices that deserve explanation.

The routing mechanism uses a separate LLM call rather than parsing the first response token. I tried the token parsing approach but found it unreliable across different models. A dedicated routing call with explicit instructions produces consistent classifications.

I implemented the checkpointer with lazy connection opening because mandatory database connections at startup would break development workflows where PostgreSQL is not always running. The fallback to stateless mode ensures the system remains useful even in degraded configurations.

For the answer synthesis prompt, I enforced inline citations using source identifiers like S1 and S2 rather than footnotes. Inline citations make it immediately clear which statements come from which sources without requiring readers to scroll.

I chose to run RAGAS evaluation using local Ollama models rather than cloud APIs. This keeps the evaluation pipeline consistent with production inference and avoids API costs during iterative prompt tuning.

---

## 4. Features

### 4.1 Intelligent Query Routing

The routing mechanism uses a dedicated LLM call to classify incoming queries as either requiring document retrieval or suitable for direct response. I designed the router prompt to be explicit about edge cases: greetings go direct, document questions go to retrieval, meta questions about the conversation go direct.

This approach reduces unnecessary vector searches for conversational exchanges, improving response latency and reducing resource consumption. In my testing, roughly 20 to 30 percent of queries in typical usage patterns can skip retrieval entirely.

### 4.2 Conversation Memory

Multi turn conversations are enabled through PostgreSQL backed checkpointing using LangGraph's checkpoint saver abstraction. When a session identifier accompanies a request, I persist the message history and state between invocations.

I implemented this because single turn RAG feels unnatural. Users expect to ask follow up questions without restating context. The checkpoint system stores the complete conversation state, allowing questions like "tell me more about that" to resolve correctly.

The checkpoint tables are automatically created during application startup. I added migration logic that checks for existing tables before attempting creation, making the system idempotent across restarts.

### 4.3 Document Ingestion Pipeline

I built the ingestion system to support PDF, DOCX, TXT, and HTML files through format specific extractors. Each extractor is designed to preserve as much structure as possible while producing clean text.

PDF processing uses PyMuPDF to extract text by page. I chose this library after testing several alternatives because it handles scanned documents and complex layouts more reliably.

DOCX handling leverages python-docx for paragraph extraction. HTML content is processed through BeautifulSoup to strip markup and retain meaningful text.

The recursive character text splitter segments extracted content into chunks with configurable size and overlap. I tuned the default values through experimentation to balance retrieval precision against context completeness.

### 4.4 Streamlit Dashboard

I designed the user interface to provide a chat experience modeled after modern AI assistants. The sidebar contains model selection, session management, and document upload controls. The chat panel displays the conversation with markdown rendering for formatted responses.

Users can mention specific documents using an @ tagging pattern. I implemented this because broad retrieval across all documents sometimes returns irrelevant results when users have uploaded diverse materials. Targeted retrieval improves precision.

The dark theme styling reflects my preference for comfortable extended use. All CSS is injected through Streamlit's markdown mechanism, keeping the styling customizable without external files.

### 4.5 Langfuse Observability

I integrated distributed tracing to capture the complete lifecycle of each request. Traces include input queries, retrieved contexts, model outputs, and latency metrics. This visibility is essential for debugging production issues and understanding system behavior.

The callback handler propagates trace context through LangGraph node invocations, enabling end to end visibility. I structured traces with consistent naming so they can be filtered and aggregated in the Langfuse dashboard.

Traces tagged for RAGAS evaluation include structured input and output payloads matching the schema expected by the evaluation pipeline. This eliminates transformation steps between tracing and evaluation.

### 4.6 RAGAS Evaluation

I built the evaluation script to fetch traces from Langfuse by tag and score them using RAGAS metrics. The metrics include Faithfulness, which measures whether answers are grounded in retrieved context, and Context Precision, which measures whether retrieved chunks are relevant to the question.

Evaluation uses local Ollama models for both LLM judgments and embeddings, keeping the process entirely offline. Computed scores are pushed back to Langfuse and attached to the original traces for analysis.

I added this capability because qualitative review does not scale. With automated metrics, I can compare prompt variations, model upgrades, and retrieval configurations quantitatively.

### 4.7 Docker Containerization

The application deploys as a three container stack comprising the API server, Streamlit UI, and PostgreSQL database. I configured Docker Compose to orchestrate health checks and startup ordering so dependencies initialize before dependent services.

Volume mounts persist ChromaDB embeddings and uploaded documents across container restarts. I separated concerns into independent Dockerfiles for API and UI so they can be deployed to different hosts if needed.

Environment variables inject configuration without requiring image rebuilds. This follows twelve factor app principles and enables the same images to run across development, staging, and production.

---

## 5. Quick Start

### 5.1 Prerequisites

Before installation, ensure the following software is available on your system.

Python 3.13 or later is required for dependency compatibility. The project uses modern Python features and type annotations that require recent interpreter versions.

Ollama must be running locally to provide LLM and embedding model inference. Install Ollama following the official documentation for your operating system.

Docker Desktop or Docker Engine with Compose plugin is needed if you plan to use containerized deployment. This is optional for local development.

PostgreSQL is optional but required for conversation memory functionality. Without it, the system operates in stateless mode where each request is independent.

Pull the required Ollama models before starting the application.

```bash
ollama pull gemma3:4b
ollama pull mxbai-embed-large:latest
```

### 5.2 Installation with uv

I recommend uv for dependency management because it provides fast resolution and consistent environments. Install uv following the official documentation, then clone and sync.

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Sync dependencies (creates .venv automatically)
uv sync
```

The sync command creates a virtual environment in `.venv` and installs all dependencies from the lock file. This ensures reproducible installations across machines.

To activate the environment manually (optional, uv run handles this automatically):

```bash
source .venv/bin/activate
```

Common uv commands for this project:

```bash
# Run any command in the virtual environment
uv run python --version

# Add a new dependency
uv add package-name

# Update the lock file after changing pyproject.toml
uv lock

# Sync after pulling new changes
uv sync
```

### 5.3 Installation with pip

For environments where uv is not available, pip installation works through the requirements file.

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

The requirements file pins versions for reproducibility. If you need to update dependencies, modify pyproject.toml and regenerate requirements.

### 5.4 Environment Configuration

Copy the example environment file and configure the required variables.

```bash
cp .env.example .env
```

Edit `.env` with your preferred text editor. At minimum, configure the Ollama connection:

```bash
# Core Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3:4b
EMBED_MODEL=mxbai-embed-large:latest

# Storage paths
CHROMA_DIR=./.chroma
CHROMA_COLLECTION=agentic_rag
UPLOAD_DIR=./data/uploads
```

For conversation memory, add PostgreSQL credentials:

```bash
POSTGRES_DSN=postgresql://agentic:agentic@localhost:5433/agentic_rag
```

For observability, add Langfuse credentials:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk_your_public_key
LANGFUSE_SECRET_KEY=sk_your_secret_key
LANGFUSE_BASE_URL=http://localhost:3000
```

### 5.5 Running the API Server

Start the FastAPI server using uvicorn. With uv:

```bash
uv run uvicorn agentic_rag.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

With pip installation:

```bash
python -m uvicorn agentic_rag.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto restart on code changes, useful during development. Remove it for production deployments.

The API becomes available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

### 5.6 Launching the UI

Start the Streamlit dashboard in a separate terminal. With uv:

```bash
uv run streamlit run src/agentic_rag/ui/app.py --server.port 8501
```

With pip installation:

```bash
python -m streamlit run src/agentic_rag/ui/app.py --server.port 8501
```

The interface opens automatically at http://localhost:8501. If it does not open, navigate there manually in your browser.

### 5.7 Docker Deployment

For production or isolated development, Docker Compose brings up all services with a single command.

```bash
# Build and start all services
docker compose up --build

# Run in detached mode (background)
docker compose up --build -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Stop and remove volumes (clears all data)
docker compose down -v
```

The compose configuration exposes the API on port 8000 and the UI on port 8501. PostgreSQL runs internally on port 5432 with external access mapped to port 5433 for debugging.

To run only specific services:

```bash
# Start only PostgreSQL (for local development with memory)
docker compose up postgres -d

# Start API without UI
docker compose up api -d
```

---

## 6. Usage Guide

### 6.1 Uploading Documents

Documents can be uploaded through the UI or API.

Through the UI, navigate to the Documents tab and use the file uploader. Multiple files can be selected simultaneously. Supported formats include PDF, DOCX, TXT, and HTML.

Through the API, send a multipart POST request:

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

The response includes document identifiers that can be used to filter queries:

```json
{
  "summary": {
    "files_received": 2,
    "documents_ingested": 2,
    "total_chunks_added": 47
  },
  "documents": [
    {"doc_id": "abc123", "filename": "document1.pdf", "chunks_added": 25},
    {"doc_id": "def456", "filename": "document2.docx", "chunks_added": 22}
  ]
}
```

### 6.2 Querying Documents

The chat interface in the UI provides the simplest way to query. Type your question and the system automatically retrieves relevant context and generates an answer with citations.

For programmatic access, use the agentic endpoint:

```bash
curl -X POST http://localhost:8000/chat/ask_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings in the research?",
    "k": 6
  }'
```

To scope queries to specific documents, include document identifiers:

```bash
curl -X POST http://localhost:8000/chat/ask_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the conclusions",
    "doc_ids": ["abc123"]
  }'
```

### 6.3 Using Session Memory

To enable conversation memory, include a session identifier with your requests:

```bash
# First message in conversation
curl -X POST http://localhost:8000/chat/ask_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the document say about pricing?",
    "session_id": "user-session-001"
  }'

# Follow up message (system remembers context)
curl -X POST http://localhost:8000/chat/ask_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me more about that",
    "session_id": "user-session-001"
  }'
```

The session identifier can be any unique string. I recommend including user and session components for multi user deployments.

Memory requires PostgreSQL to be configured. Check the response for `memory_enabled: true` to confirm memory is active.

### 6.4 Running RAGAS Evaluation

After collecting traces in Langfuse, run the evaluation script to compute quality metrics.

```bash
# Evaluate traces with a specific tag
uv run python scripts/ragas_eval_langfuse.py --tag ragas_eval --limit 20

# Include response relevancy metric (requires embeddings)
uv run python scripts/ragas_eval_langfuse.py --tag ragas_eval --limit 20 --with-relevancy
```

With pip installation:

```bash
python scripts/ragas_eval_langfuse.py --tag ragas_eval --limit 20
```

The script fetches traces, extracts the required fields, computes metrics locally using Ollama, and pushes scores back to Langfuse.

---

## 7. API Reference

### 7.1 Document Management

**Upload Documents**

```
POST /documents/upload
Content-Type: multipart/form-data
```

Request body contains one or more files as form data.

Response:

```json
{
  "summary": {
    "files_received": 1,
    "documents_ingested": 1,
    "total_file_bytes": 102400,
    "total_extracted_chars": 45000,
    "total_chunks_added": 23,
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "documents": [
    {
      "doc_id": "uuid-string",
      "filename": "example.pdf",
      "stored_path": "./data/uploads/uuid__example.pdf",
      "chunks_added": 23,
      "stats": {
        "file_bytes": 102400,
        "extracted_units": 12,
        "extracted_chars": 45000,
        "avg_chunk_chars": 980
      }
    }
  ]
}
```

### 7.2 Chat Endpoints

**Retrieve Chunks**

```
POST /chat/retrieve
Content-Type: application/json
```

Request:

```json
{
  "query": "What are the key findings?",
  "k": 6,
  "doc_ids": null
}
```

Response returns matched chunks with scores.

**Ask with Citations**

```
POST /chat/ask
Content-Type: application/json
```

Request:

```json
{
  "query": "Summarize the main conclusions",
  "k": 6,
  "max_context_chars": 12000,
  "doc_ids": null
}
```

Response includes answer text and citation metadata.

**Agentic Ask**

```
POST /chat/ask_agentic
Content-Type: application/json
```

Request:

```json
{
  "query": "Hello, what can you help me with?",
  "session_id": "optional-session-id",
  "k": 6,
  "max_context_chars": 12000,
  "doc_ids": null,
  "provider": "ollama",
  "model": null,
  "temperature": 0.0
}
```

Response:

```json
{
  "answer": "Generated response text",
  "route": "direct",
  "session_id": "optional-session-id",
  "memory_enabled": true
}
```

### 7.3 Health and Metadata

**Health Check**

```
GET /health
```

Returns 200 OK when the service is running.

**Metadata**

```
GET /meta
```

Returns non secret configuration for debugging:

```json
{
  "llm_model": "gemma3:4b",
  "embed_model": "mxbai-embed-large:latest",
  "chroma_path": "./.chroma",
  "postgres_configured": true
}
```

---

## 8. Configuration

### 8.1 Environment Variables

All configuration flows through environment variables, optionally sourced from a `.env` file. I designed the settings module to validate and provide type safe access to configuration values.

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_BASE_URL | Ollama server endpoint | http://localhost:11434 |
| LLM_MODEL | Default chat model name | gemma3:4b |
| EMBED_MODEL | Embedding model name | mxbai-embed-large:latest |
| CHROMA_DIR | ChromaDB persistence path | ./.chroma |
| CHROMA_COLLECTION | Collection name | agentic_rag |
| DATA_DIR | Base data directory | ./data |
| UPLOAD_DIR | Uploaded file storage | ./data/uploads |
| POSTGRES_DSN | PostgreSQL connection string | None |
| LANGFUSE_PUBLIC_KEY | Langfuse public key | None |
| LANGFUSE_SECRET_KEY | Langfuse secret key | None |
| LANGFUSE_BASE_URL | Langfuse server URL | None |
| LANGFUSE_ENABLED | Enable tracing | true |

### 8.2 LLM Configuration

I designed the system to use Ollama as the LLM provider, keeping all inference local and avoiding external API dependencies. The default model is Gemma 3 4B which balances capability and resource requirements. For machines with more memory, larger models can be configured by changing the LLM_MODEL variable.

Embeddings use the mxbai-embed-large model. I selected this after testing several options because it provides strong semantic similarity performance while remaining efficient enough for interactive use.

### 8.3 Storage Settings

ChromaDB stores document embeddings in a local directory that persists across restarts. The collection name determines the namespace for vectors. I separated these concerns so multiple projects can share a ChromaDB instance if needed.

Uploaded files are stored in the upload directory with unique identifiers prepended to prevent filename collisions. The storage path is included in document metadata so files can be retrieved or deleted later.

---

## 9. Testing

The test suite uses pytest with coverage reporting. Run tests from the repository root.

With uv:

```bash
uv run pytest tests/ -v
```

With pip:

```bash
python -m pytest tests/ -v
```

For coverage reporting:

```bash
uv run pytest tests/ -v --cov=src/agentic_rag --cov-report=term-missing
```

Tests cover core functionality including metadata filter construction and retrieval logic. I structured tests to run without external dependencies so they execute quickly during development.

---

## 10. License

This project is provided for educational and research purposes. See the LICENSE file for terms of use.
