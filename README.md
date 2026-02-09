# llm-cancer-risk-assessor

Clinical decision support agent using NICE NG12 guidelines with Google Vertex AI (Gemini). Accepts a patient ID, retrieves their data, searches the NG12 PDF via RAG, and returns a risk assessment with citations (Urgent Referral / Urgent Investigation / Routine). Also provides a conversational interface for guideline Q&A with citations.

## Prerequisites

- **Python 3.11+**
- **uv** — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **Google Cloud** — Project with Vertex AI API enabled
- **Service account** — JSON key with Vertex AI User (or equivalent) role

## Setup

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Configure environment**
   - Copy `.env.local.example` to `.env.local`
   - Set **`GOOGLE_APPLICATION_CREDENTIALS`** to the path of your service account JSON (use a full path, or path relative to the project root).
   - Set **`GOOGLE_CLOUD_PROJECT`** and **`GOOGLE_CLOUD_LOCATION`** (e.g. `us-central1`).
   - For **local development**, set `DATA_DIR=data` and `VECTOR_DB_PATH=data/vector_db` so the app uses the project’s `data/` folder.
   - **`VERTEX_AI_MODEL_NAME`** — Use a [valid Vertex model ID](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions), e.g. `gemini-2.5-flash-lite`.
   - **`VERTEX_AI_EMBEDDING_MODEL`** — Embedding model for ingestion and search (default `text-embedding-004`).
   - **`VECTOR_DB_COLLECTION`** — Chroma collection name for NG12 chunks (default `ng12_guidelines`).
   - Optional: **`API_PORT`** (default `8000`), **`PATIENTS_JSON_PATH`** if you want to point to a different patients file.

3. **Build the vector index**
   ```bash
   uv run python -m src.ingestion
   ```
   This downloads the NG12 PDF (if missing), parses it, creates embeddings with Vertex AI (`text-embedding-004`), and stores them in ChromaDB under `data/vector_db`. Without this step, `/assess` will fail when the agent tries to search guidelines.

## Run

- **API**
  ```bash
  uv run uvicorn src.main:app --reload --port 8000
  ```
- **UI**
  ```bash
  uv run streamlit run streamlit_app.py
  ```
  The Streamlit app uses the API at `http://localhost:8000` by default.

## Docker

Launches both the API (port 8000) and Streamlit UI (port 8501) in a single container.

```bash
docker build -t cancer-risk-assessor .
```

Then run (pick your shell):

**Mac / Linux:**
```bash
docker run -p 8000:8000 -p 8501:8501 --env-file .env.local -v "$(pwd)/data:/app/data" -v "$(pwd)/your-service-account-key.json:/app/credentials.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json cancer-risk-assessor
```

**Windows (Git Bash):**
```bash
MSYS_NO_PATHCONV=1 docker run -p 8000:8000 -p 8501:8501 --env-file .env.local -v "$(pwd)/data:/app/data" -v "$(pwd)/your-service-account-key.json:/app/credentials.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json cancer-risk-assessor
```

**Windows (PowerShell):**
```powershell
docker run -p 8000:8000 -p 8501:8501 --env-file .env.local -v "${PWD}/data:/app/data" -v "${PWD}/your-service-account-key.json:/app/credentials.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json cancer-risk-assessor
```

Replace `your-service-account-key.json` with the filename of your GCP service account key. 

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

## Project layout

| Path | Description |
|------|-------------|
| `src/main.py` | FastAPI app entry point |
| `src/routes.py` | API routes: `/`, `/health`, `/patients`, `/assess` |
| `src/agent.py` | Gemini agent for risk assessment (function calling, system prompt, response parsing) |
| `src/chat.py` | Gemini chat agent for NG12 Q&A (RAG over guidelines, session history) |
| `src/tools.py` | Tools: `get_patient_data`, `search_clinical_guidelines`, `get_all_patients` |
| `src/ingestion.py` | PDF download → parse → embed → ChromaDB (run as `python -m src.ingestion`) |
| `src/config.py` | Settings from env / `.env.local` |
| `src/schemas.py` | Pydantic models for assessment and chat (`AssessmentRequest`, `AssessmentResponse`, `ChatRequest`, `ChatResponse`, etc.) |
| `data/` | PDF, vector DB, and mock patients (see below) |
| `data/patients.json` | Mock patient records (configurable via `PATIENTS_JSON_PATH`) |
| `data/vector_db/` | ChromaDB index (created by ingestion) |
| `streamlit_app.py` | Streamlit UI: patient list, assess, view results, and guideline chat |
| `PROMPTS.md` | Prompt engineering overview for assessment and chat agents |

## API

- **GET /** — Service info  
- **GET /health** — Health check  
- **GET /patients** — List patient IDs (and count)  
- **POST /assess** — Body: `{"patient_id": "PT-101"}` → assessment + reasoning + citations  
- **POST /chat** — Body: `{"session_id": "abc", "message": "question", "top_k": 5}` → NG12‑grounded answer + citations  
- **GET /chat/{session_id}/history** — Conversation history for a given chat session  
- **DELETE /chat/{session_id}`** — Clear conversation history for a given chat session  