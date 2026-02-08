# llm-cancer-risk-assessor

Clinical decision support agent using NICE NG12 guidelines with Google Vertex AI (Gemini). Accepts a patient ID, retrieves their data, searches the NG12 PDF via RAG, and returns a risk assessment with citations (Urgent Referral / Urgent Investigation / Routine).

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

3. **Build the vector index (required for assessments)**
   ```bash
   python -m src.ingestion
   ```
   This downloads the NG12 PDF (if missing), parses it, creates embeddings with Vertex AI (`text-embedding-004`), and stores them in ChromaDB under `data/vector_db`. Without this step, `/assess` will fail when the agent tries to search guidelines.

## Run

- **API**
  ```bash
  uvicorn src.main:app --reload --port 8000
  ```
- **UI**
  ```bash
  streamlit run streamlit_app.py
  ```
  The Streamlit app uses the API at `http://localhost:8000` by default.

## Docker

```bash
docker build -t cancer-risk-assessor .
docker run -p 8000:8000 \
  -v /path/to/your-key.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_CLOUD_LOCATION=us-central1 \
  -v "$(pwd)/data:/app/data" \
  cancer-risk-assessor
```

## Project layout

| Path | Description |
|------|-------------|
| `src/main.py` | FastAPI app entry point |
| `src/routes.py` | API routes: `/`, `/health`, `/patients`, `/assess` |
| `src/agent.py` | Gemini agent (function calling, system prompt, response parsing) |
| `src/tools.py` | Tools: `get_patient_data`, `search_clinical_guidelines`, `get_all_patients` |
| `src/ingestion.py` | PDF download → parse → embed → ChromaDB (run as `python -m src.ingestion`) |
| `src/config.py` | Settings from env / `.env.local` |
| `src/schemas.py` | Pydantic models: `AssessmentRequest`, `AssessmentResponse`, `Citation` |
| `data/` | PDF, vector DB, and mock patients (see below) |
| `data/patients.json` | Mock patient records (configurable via `PATIENTS_JSON_PATH`) |
| `data/vector_db/` | ChromaDB index (created by ingestion) |
| `streamlit_app.py` | Streamlit UI: patient list, assess, view results |
| `architecture.md` | System design and data flow |
| `PROMPTS.md` | System prompt design and rationale |

## API

- **GET /** — Service info  
- **GET /health** — Health check  
- **GET /patients** — List patient IDs  
- **POST /assess** — Body: `{"patient_id": "PT-101"}` → assessment + reasoning + citations  