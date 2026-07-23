# QEDS-GPT - Asynchronous Hybrid RAG Academic Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)
![Redis](https://img.shields.io/badge/Queue-Redis-dc382d.svg)
![RQ](https://img.shields.io/badge/Worker-RQ-orange.svg)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq-black.svg)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue.svg)

QEDS-GPT is a production-style Retrieval-Augmented Generation application for Quantitative Economics and Data Science notes. It combines a Streamlit frontend, FastAPI backend, Redis-backed RQ job queue, background worker, ChromaDB, hybrid retrieval, Groq LLM generation, and optional Pydantic Logfire observability.

The application is designed to answer academic questions from indexed notes while still supporting conversational follow-ups and general-knowledge fallback when the answer is outside the note corpus.

---

## Project Overview

QEDS-GPT lets users ask natural-language questions about Economics, Statistics, Mathematics, Data Science, Machine Learning, and related semester-note content.

For academic queries, the system retrieves relevant chunks from a persisted ChromaDB database, combines semantic search with BM25 keyword retrieval, fuses the rankings with Reciprocal Rank Fusion, builds a grounded prompt, and sends the final prompt to Groq.

The current milestone uses asynchronous job processing:

- Streamlit submits the user question to FastAPI.
- FastAPI queues the work and immediately returns a `job_id`.
- An RQ worker processes the RAG pipeline in the background.
- Streamlit polls the result endpoint and renders the answer when complete.

This keeps the frontend responsive and allows multiple questions to be queued and processed concurrently by one or more workers.

---

## Architecture

The project started as a Streamlit-only synchronous RAG app. That version worked well for local demos, but each question was processed inside the Streamlit request cycle. A slow retrieval or LLM call could block the UI and made concurrent usage harder to reason about.

The current architecture separates user interaction from backend execution.

```text
Previous milestone
------------------
User -> Streamlit -> Classifier -> Retriever -> Groq -> Streamlit response

Current milestone
-----------------
User -> Streamlit -> FastAPI -> Redis Queue -> RQ Worker -> Retriever -> Groq
                                                    |
Streamlit <- polls /result/{job_id} <- FastAPI <----+
```

Why asynchronous processing was introduced:

- It prevents long LLM calls from blocking the Streamlit frontend.
- It gives every request a stable `job_id` and status lifecycle.
- It allows multiple user questions to be queued at the same time.
- It makes worker scaling possible without redesigning the RAG logic.
- It separates API endpoints, business logic, retrieval, model calls, and UI responsibilities.

---

## Key Features

- Hybrid retrieval using dense vector search plus BM25 keyword search.
- Reciprocal Rank Fusion to merge rankings without comparing incompatible scores.
- Persisted ChromaDB vector database for academic note chunks.
- Remote HuggingFace embedding support to avoid heavy local `torch` installs in Docker.
- Groq LLM integration for classification and answer generation.
- Query classification for academic, greeting, meta, general, and prompt-injection categories.
- General-knowledge fallback without false citations when notes are not used.
- Conversation memory through Streamlit session state.
- Async job lifecycle: `queued`, `running`, `completed`, `error`.
- Redis + RQ background worker architecture.
- Health endpoints for API, queue, and worker visibility.
- Optional Pydantic Logfire instrumentation for API and worker spans.
- Split Dockerfiles and dependency files to keep service images smaller.

---

## Architecture Diagram

```text
                                 +----------------------+
                                 |        User          |
                                 +----------+-----------+
                                            |
                                            v
                                 +----------------------+
                                 |  Streamlit Frontend  |
                                 |  src/streamlit_app.py|
                                 +----------+-----------+
                                            |
                                  POST /api/v1/chat
                                            |
                                            v
                                 +----------------------+
                                 |   FastAPI Backend    |
                                 |   src/api/main.py    |
                                 +----------+-----------+
                                            |
                                      enqueue job
                                            |
                                            v
                                 +----------------------+
                                 |  Redis + RQ Queue    |
                                 |  qeds-rag-jobs       |
                                 +----------+-----------+
                                            |
                                      consume job
                                            |
                                            v
                                 +----------------------+
                                 |  Background Worker   |
                                 |  src/worker.py       |
                                 +----------+-----------+
                                            |
                   +------------------------+------------------------+
                   |                         |                       |
                   v                         v                       v
          +----------------+        +----------------+       +----------------+
          | Classifier     |        | Hybrid         |       | Groq LLM       |
          | classifier.py  |        | Retriever      |       | llm.py         |
          +----------------+        | retriever.py   |       +----------------+
                                    +-------+--------+
                                            |
                           +----------------+----------------+
                           |                                 |
                           v                                 v
                 +-------------------+             +-------------------+
                 | ChromaDB Dense    |             | BM25 Sparse       |
                 | Vector Search     |             | Keyword Search    |
                 +-------------------+             +-------------------+
                           \                                 /
                            \                               /
                             v                             v
                         +-------------------------------+
                         | Reciprocal Rank Fusion        |
                         +-------------------------------+
```

---

## Request Flow

1. The user enters a question in the Streamlit chat input.
2. Streamlit reads recent conversation context from `ChatManager`.
3. Streamlit sends `message`, `semester_filter`, and `history` to `POST /api/v1/chat`.
4. FastAPI validates the request with Pydantic models.
5. FastAPI enqueues `worker.process_chat_job` into Redis using RQ.
6. FastAPI returns immediately with a `job_id`.
7. Streamlit polls `GET /api/v1/result/{job_id}` every second.
8. The worker picks up the job and marks it as running.
9. The worker uses `RAGService` to classify the query.
10. Non-academic requests are answered directly by Groq.
11. Academic requests retrieve relevant chunks using hybrid retrieval.
12. Retrieved chunks are cleaned and inserted into the academic prompt.
13. Groq generates the final answer.
14. The worker returns answer metadata, sources, retrieved chunks, and category.
15. FastAPI exposes the completed result through the result endpoint.
16. Streamlit renders the answer and source panels only when retrieval was used.
17. Streamlit updates session memory for follow-up questions.

---

## Project Structure

```text
qeds-gpt-refactored-hf/
|
|-- chroma_db/
|   |-- chroma.sqlite3
|   `-- <chroma-index-files>/
|
|-- src/
|   |-- api/
|   |   |-- routers/
|   |   |   |-- chat.py
|   |   |   |-- health.py
|   |   |   `-- version.py
|   |   |-- __init__.py
|   |   |-- dependencies.py
|   |   |-- main.py
|   |   `-- models.py
|   |
|   |-- __init__.py
|   |-- api_client.py
|   |-- classifier.py
|   |-- config.py
|   |-- embeddings.py
|   |-- job_queue.py
|   |-- llm.py
|   |-- memory.py
|   |-- observability.py
|   |-- prompts.py
|   |-- retriever.py
|   |-- services.py
|   |-- streamlit_app.py
|   |-- text_processor.py
|   |-- ui.py
|   `-- worker.py
|
|-- .dockerignore
|-- .env.example
|-- .gitattributes
|-- .gitignore
|-- Dockerfile
|-- Dockerfile.api
|-- Dockerfile.streamlit
|-- Dockerfile.worker
|-- docker-compose.yml
|-- PROJECT_EXPLANATION.txt
|-- README.md
|-- requirements-api.txt
|-- requirements-local.txt
|-- requirements-streamlit.txt
|-- requirements-worker.txt
`-- requirements.txt
```

---

## Project Structure Details

| File | Responsibility |
|---|---|
| `src/streamlit_app.py` | Streamlit entrypoint. Renders the UI, submits jobs to FastAPI, polls job results, and updates chat memory. |
| `src/api_client.py` | Small HTTP client used by Streamlit to call `POST /chat` and `GET /result/{job_id}`. |
| `src/api/main.py` | FastAPI application factory area. Registers routers, CORS, and Logfire instrumentation. |
| `src/api/models.py` | Pydantic request and response schemas for chat, result, health, queue, worker, and version endpoints. |
| `src/api/routers/chat.py` | Chat API endpoints. Enqueues work and exposes job results. Does not run LLM inference directly. |
| `src/api/routers/health.py` | Health checks for API, Redis queue, and worker status. |
| `src/api/routers/version.py` | Version endpoint for deployment sanity checks. |
| `src/job_queue.py` | Redis/RQ connection helpers and queue configuration. |
| `src/worker.py` | Background job executor. Runs classification, retrieval, prompt construction, Groq calls, and result packaging. |
| `src/services.py` | Business logic layer that coordinates classifier, retriever, prompts, memory context, and LLM calls. |
| `src/retriever.py` | Hybrid retrieval implementation using ChromaDB dense search, BM25, and Reciprocal Rank Fusion. |
| `src/embeddings.py` | Embedding adapter. Defaults to remote HuggingFace inference to keep Docker images lightweight. |
| `src/llm.py` | Groq client wrapper and model-call utilities. |
| `src/classifier.py` | Query category detection. |
| `src/prompts.py` | Prompt templates for academic answers, classification, summaries, and meta/general responses. |
| `src/memory.py` | Streamlit session-state memory manager. |
| `src/ui.py` | Streamlit rendering helpers for header, sidebar, chat history, sources, and retrieved chunks. |
| `src/text_processor.py` | Cleaning and formatting utilities for retrieved document text and metadata. |
| `src/config.py` | Central constants for model names, retrieval settings, Chroma path, collection name, and memory limits. |
| `src/observability.py` | Optional Pydantic Logfire setup and safe tracing helpers. |
| `docker-compose.yml` | Local multi-service orchestration for Redis, FastAPI, worker, and Streamlit. |
| `Dockerfile.api` | Lightweight API container. |
| `Dockerfile.worker` | Worker container with RAG dependencies and ChromaDB data. |
| `Dockerfile.streamlit` | Lightweight Streamlit frontend container. |
| `requirements.txt` | Minimal frontend dependencies for Streamlit Community Cloud. |
| `requirements-*.txt` | Split dependencies per service to avoid installing heavy packages where they are not needed. |
| `.env.example` | Safe template for local environment variables. |
| `.gitignore` | Keeps secrets, virtual environments, caches, logs, and local artifacts out of Git. |
| `.gitattributes` | Tracks `chroma_db/**` with Git LFS. |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Queue broker | Redis |
| Job system | RQ |
| Worker runtime | Python worker process |
| Vector database | ChromaDB |
| Dense embeddings | HuggingFace embedding endpoint or compatible backend |
| Sparse retrieval | BM25 via `rank-bm25` |
| Fusion | Reciprocal Rank Fusion |
| LLM | Groq |
| Observability | Standard Python logging plus optional Pydantic Logfire |
| Local orchestration | Docker Compose |
| Large files | Git LFS for `chroma_db/` |

---

## Environment Variables

| Variable | Required | Used By | Description |
|---|---:|---|---|
| `GROQ_API_KEY` | Yes | API, worker, local app | Groq API key for LLM calls. |
| `REDIS_URL` | Yes for async backend | API, worker | Redis connection string. Defaults locally to `redis://localhost:6379/0`. |
| `QEDS_API_URL` | Yes when frontend is separate | Streamlit | Base URL for FastAPI, for example `https://your-api.example.com/api/v1`. |
| `HF_TOKEN` | Recommended | Worker | HuggingFace token for remote embedding calls. |
| `EMBEDDING_BACKEND` | No | Worker | Use `remote` to avoid local heavy ML dependencies. |
| `LOGFIRE_API_KEY` | Optional | API, worker | Pydantic Logfire write token. |
| `LOGFIRE_TOKEN` | Optional | API, worker | Alternate Logfire token variable. |
| `LOGFIRE_ENABLED` | Optional | API, worker | Enables Logfire configuration. |
| `LOGFIRE_SEND_TO_LOGFIRE` | Optional | API, worker | Usually `if-token-present` locally. |
| `APP_ENV` | Optional | API, worker | Environment label such as `local`, `staging`, or `production`. |
| `CHROMA_DB_PATH` | Optional | Worker/RAG | Custom ChromaDB path. Defaults to the project `chroma_db/`. |

Never commit `.env` or `.streamlit/secrets.toml`.

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/apooorv19/QEDS-RAG-Project.git
cd QEDS-RAG-Project
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Local Dependencies

```bash
pip install -r requirements-local.txt
```

### 4. Configure Secrets

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then set your real values in `.env`:

```text
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
LOGFIRE_API_KEY=your_logfire_write_token
LOGFIRE_ENABLED=true
LOGFIRE_SEND_TO_LOGFIRE=if-token-present
APP_ENV=local
```

---

## Running Locally Without Full Docker Compose

Use three terminals.

Terminal 1: Redis

```bash
docker run -p 6379:6379 redis:7-alpine
```

Terminal 2: FastAPI

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Terminal 3: Worker

```bash
python src/worker.py
```

Terminal 4: Streamlit

```bash
streamlit run src/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Running With Docker Compose

The recommended local production-style run is:

```bash
docker compose up --build
```

This starts:

- Redis on `localhost:6379`
- FastAPI on `localhost:8000`
- Streamlit on `localhost:8501`
- One RQ worker connected to the Redis queue

Useful health URLs:

```text
http://localhost:8000/api/v1/health
http://localhost:8000/api/v1/health/queue
http://localhost:8000/api/v1/health/worker
http://localhost:8000/api/v1/version
```

To scale workers locally:

```bash
docker compose up --build --scale worker=3
```

---

## Troubleshooting

### Streamlit Shows API Unavailable

Confirm FastAPI is running:

```text
http://localhost:8000/api/v1/health
```

Confirm `QEDS_API_URL` points to the backend:

```text
QEDS_API_URL=http://localhost:8000/api/v1
```

### Jobs Stay Queued

Start the worker:

```bash
python src/worker.py
```

Then check:

```text
http://localhost:8000/api/v1/health/worker
```

### Redis Connection Fails

Start Redis locally:

```bash
docker run -p 6379:6379 redis:7-alpine
```

Or set `REDIS_URL` to a managed Redis URL.

### Logfire Shows No Records

Check that API and worker processes can read the token:

```text
LOGFIRE_API_KEY=your_logfire_write_token
LOGFIRE_ENABLED=true
LOGFIRE_SEND_TO_LOGFIRE=if-token-present
```

Logfire spans are emitted by the API and worker, not by Redis itself.

### ChromaDB Is Not Found

Confirm `chroma_db/` exists at the project root. If deployed elsewhere, set:

```text
CHROMA_DB_PATH=/path/to/chroma_db
```

### Docker Images Become Too Large

Use the split Dockerfiles and split requirements files. The worker is the only service that needs RAG dependencies. The Streamlit image should only install frontend dependencies, and the API image should only install API/queue dependencies.

The project avoids local `torch` and `sentence-transformers` in Docker by using the remote embedding path by default.

---

## Security Notes

- Keep `GROQ_API_KEY`, `HF_TOKEN`, and `LOGFIRE_API_KEY` out of Git.
- Use hosted secret managers for cloud deployments.
- Do not record full prompts, retrieved documents, or private user conversations in logs.
- Logfire instrumentation is configured to avoid capturing request bodies.
- Source citations are displayed only when retrieval was used.
- Prompt-injection categories are detected before retrieval/generation.
- For a public app, consider adding authentication and rate limiting.

---

## Future Improvements

- Add automated tests and CI.
- Add document ingestion and re-indexing scripts.
- Add retrieval evaluation metrics.
- Add optional reranking.
- Add streaming answer updates from the backend.
- Add API authentication.
- Add per-user persistent chat history.
- Add queue dashboards and alerting.
- Add deployment manifests for Render, Koyeb, or Fly.io.

---

## Author

**Apurva Mishra**  
IMSc Quantitative Economics and Data Science  
Birla Institute of Technology, Mesra

GitHub: [apooorv19](https://github.com/apooorv19)  
LinkedIn: [Apurva Mishra](https://www.linkedin.com/in/apooorv/)
