# llm-cancer-risk-assessor

Clinical decision support agent using NICE NG12 guidelines with Google Vertex AI (Gemini 1.5).

## Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- Service account key (JSON) with Vertex AI User role
- uv

## Setup

1. **Clone and install**
   ```bash
   uv sync
   ```

2. **Configure environment**
   - Copy `.env.local.example` to `.env.local`
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your service account JSON
   - Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (e.g. `us-central1`)

3. **Optional: build vector index**  
   Run the ingestion pipeline (or use `test.ipynb`) to download the NG12 PDF, embed it, and fill `data/vector_db`. The API can run without this; RAG will be empty until ingestion is done.

## Run

- **API**
  ```bash
  uvicorn src.main:app --reload --port 8000
  ```
- **UI**
  ```bash
  streamlit run streamlit_app.py
  ```
  Uses API at `http://localhost:8000` by default.

## Docker

```bash
docker build -t cancer-risk-assessor .
docker run -p 8000:8000 \
  -v /path/to/key.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_CLOUD_LOCATION=us-central1 \
  -v $(pwd)/data:/app/data \
  cancer-risk-assessor
```

## Project layout

- `src/main.py` — FastAPI app
- `src/routes.py` — Endpoints (e.g. `/assess`, `/patients`)
- `src/agent.py` — Assessment agent (Gemini)
- `src/tools.py` — Patient data + guideline search
- `src/ingestion.py` — PDF → chunks → embeddings → ChromaDB
- `src/config.py` — Settings from env / `.env.local`
- `patients.json` — Mock patient data
- `streamlit_app.py` — Simple UI for Patient ID and results