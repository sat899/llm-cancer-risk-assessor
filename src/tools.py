"""Tool functions for patient data retrieval and RAG operations.

This module provides function-calling tools that the Gemini agent uses to:
- Retrieve patient data from the simulated JSON database
- Search the ChromaDB vector store for relevant NG12 clinical guidelines
- List available patients

These are the "hands" of the agent — it cannot access data directly,
only through these tool functions.
"""

import json
import logging
from typing import Dict, List, Any

from src.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (initialised on first use)
# ---------------------------------------------------------------------------
_embedding_model = None
_chroma_collection = None


def _get_embedding_model():
    """Get or initialise the Vertex AI embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from vertexai import init as vertex_init
        from vertexai.language_models import TextEmbeddingModel

        vertex_init(
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION,
        )
        _embedding_model = TextEmbeddingModel.from_pretrained(
            config.VERTEX_AI_EMBEDDING_MODEL
        )
        logger.info(
            "Initialised embedding model: %s", config.VERTEX_AI_EMBEDDING_MODEL
        )
    return _embedding_model


def _get_chroma_collection():
    """Get or initialise the ChromaDB collection (singleton)."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        db_path = str(config.VECTOR_DB_PATH)
        client = chromadb.PersistentClient(path=db_path)
        _chroma_collection = client.get_collection(config.VECTOR_DB_COLLECTION)
        logger.info(
            "Connected to ChromaDB collection '%s' at %s",
            config.VECTOR_DB_COLLECTION,
            db_path,
        )
    return _chroma_collection


# ---------------------------------------------------------------------------
# Patient helpers
# ---------------------------------------------------------------------------

def _load_patients() -> List[Dict[str, Any]]:
    """Load all patient records from the JSON file."""
    path = config.PATIENTS_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(f"Patients file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public tool functions (exposed to the Gemini agent via function calling)
# ---------------------------------------------------------------------------

def get_patient_data(patient_id: str) -> Dict[str, Any]:
    """Retrieve structured patient data from the simulated database.

    The agent calls this tool first to obtain the patient's demographics,
    smoking history, symptoms, and symptom duration.

    Args:
        patient_id: Unique patient identifier (e.g. "PT-101").

    Returns:
        Dictionary with keys: patient_id, name, age, gender,
        smoking_history, symptoms, symptom_duration_days.

    Raises:
        FileNotFoundError: If patients.json is missing.
        ValueError: If patient_id does not exist.
    """
    patients = _load_patients()
    for patient in patients:
        if patient["patient_id"] == patient_id:
            logger.info("Retrieved data for patient %s", patient_id)
            return patient
    raise ValueError(f"Patient not found: {patient_id}")


def search_clinical_guidelines(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the vector store for relevant sections of the NG12 guidelines.

    The agent calls this tool after obtaining patient data.  It formulates a
    query from the patient's symptoms and characteristics, and this function
    returns the most relevant guideline passages with page numbers.

    Args:
        query: Natural-language search query (e.g. "hemoptysis urgent referral
               criteria for male smoker over 40").
        top_k: Number of top results to return (default 5).

    Returns:
        List of dicts with keys: content, page_number, section, relevance_score.

    Raises:
        RuntimeError: If the vector store is not available.
    """
    try:
        model = _get_embedding_model()
        collection = _get_chroma_collection()

        # Embed the query with the same model used during ingestion
        query_embedding = model.get_embeddings([query])[0].values

        # Semantic search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        guidelines: List[Dict[str, Any]] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        for i, doc in enumerate(documents):
            distance = distances[i] if i < len(distances) else 0.0
            # ChromaDB L2 distance → similarity score in [0, 1]
            relevance_score = 1.0 / (1.0 + distance)

            # Use the first meaningful line as the section title
            section = _first_meaningful_line(doc)

            guidelines.append(
                {
                    "content": doc[:2000],  # cap length for API response
                    "page_number": metadatas[i].get("page", 0),
                    "section": section,
                    "relevance_score": round(relevance_score, 3),
                }
            )

        logger.info("Guideline search returned %d results for: %s", len(guidelines), query[:80])
        return guidelines

    except Exception as e:
        logger.error("Error searching clinical guidelines: %s", e)
        raise RuntimeError(f"Failed to search guidelines: {e}") from e


def get_all_patients() -> List[str]:
    """Return a list of all available patient IDs.

    Used by the /patients endpoint so the UI can offer a dropdown.
    """
    patients = _load_patients()
    return [p["patient_id"] for p in patients]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_meaningful_line(text: str, max_len: int = 100) -> str:
    """Extract the first non-trivial line from a text block."""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            return stripped[:max_len]
    return "NG12 Guideline Section"
