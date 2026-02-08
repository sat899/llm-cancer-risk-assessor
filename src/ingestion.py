"""PDF ingestion and vector store creation for clinical guidelines.

This module handles the complete ingestion pipeline:
1. Download the NICE NG12 "Suspected cancer: recognition and referral" PDF
2. Parse the PDF into page-aware text chunks (max 20 000 chars each)
3. Create embeddings via Vertex AI (text-embedding-004)
4. Store embeddings + text + metadata in a ChromaDB persistent collection

Run standalone:
    python -m src.ingestion
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pymupdf
import requests

from src.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NG12_PDF_URL = (
    "https://www.nice.org.uk/guidance/ng12/resources/"
    "suspected-cancer-recognition-and-referral-pdf-1837268071621"
)
CHUNK_MAX_CHARS = 20_000  # matches the chunking strategy from test.ipynb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, start_page: int, end_page: int) -> Dict[str, Any]:
    """Create a chunk dictionary with section title extracted from text."""
    return {
        "page_number": start_page,
        "section": _extract_section_title(text),
        "content": text,
        "metadata": {"start_page": start_page, "end_page": end_page},
    }


def _extract_section_title(text: str) -> str:
    """Return the first non-trivial line of *text* as a section title."""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            return stripped[:100]
    return "Unknown Section"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def download_ng12_pdf(output_dir: Path) -> Path:
    """Download the NICE NG12 PDF if not already present.

    Args:
        output_dir: Directory to save the PDF in.

    Returns:
        Path to the (possibly already-existing) PDF file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "ng12.pdf"

    if pdf_path.exists():
        logger.info("PDF already exists at %s — skipping download", pdf_path)
        return pdf_path

    logger.info("Downloading NG12 PDF from NICE website …")
    response = requests.get(NG12_PDF_URL, timeout=120)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    logger.info("Downloaded %d bytes -> %s", len(response.content), pdf_path)
    return pdf_path


def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Parse the NG12 PDF into text chunks with page metadata.

    Pages are concatenated until the running text exceeds CHUNK_MAX_CHARS,
    at which point a new chunk starts.  This keeps related content together
    while staying within embedding-model token limits.

    Args:
        pdf_path: Path to the NG12 PDF file.

    Returns:
        List of chunk dicts with keys: page_number, section, content, metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    chunks: List[Dict[str, Any]] = []
    current_text = ""
    current_start_page: Optional[int] = None

    for page_num in range(1, len(doc) + 1):
        page_text = doc[page_num - 1].get_text()

        # If adding this page would exceed the limit, flush current chunk
        if current_text and len(current_text) + len(page_text) > CHUNK_MAX_CHARS:
            chunks.append(_make_chunk(current_text, current_start_page, page_num - 1))
            current_text = page_text
            current_start_page = page_num
        else:
            if not current_text:
                current_start_page = page_num
            current_text += ("\n" + page_text) if current_text else page_text

    # Flush the final chunk
    if current_text:
        chunks.append(_make_chunk(current_text, current_start_page, len(doc)))

    doc.close()
    logger.info("Parsed PDF into %d chunks", len(chunks))
    return chunks


def create_embeddings(text_chunks: List[Dict[str, Any]]) -> List[List[float]]:
    """Create Vertex AI embeddings for each text chunk.

    Uses the model specified by ``config.VERTEX_AI_EMBEDDING_MODEL``
    (default: ``text-embedding-004``, 768-dim).

    Args:
        text_chunks: Chunk dicts produced by :func:`parse_pdf`.

    Returns:
        Parallel list of embedding vectors (list of floats).
    """
    from vertexai import init as vertex_init
    from vertexai.language_models import TextEmbeddingModel

    vertex_init(
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )
    model = TextEmbeddingModel.from_pretrained(config.VERTEX_AI_EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in text_chunks]
    logger.info(
        "Creating embeddings for %d chunks with %s …",
        len(texts),
        config.VERTEX_AI_EMBEDDING_MODEL,
    )

    embeddings_response = model.get_embeddings(texts)
    embeddings = [emb.values for emb in embeddings_response]
    logger.info(
        "Created %d embeddings (dimension=%d)", len(embeddings), len(embeddings[0])
    )
    return embeddings


def build_vector_index(
    text_chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    vector_db_path: Path,
) -> None:
    """Build (or rebuild) the ChromaDB collection from chunks + embeddings.

    If a collection with the configured name already exists it is deleted
    first so the index is always consistent with the current PDF content.

    Args:
        text_chunks: Chunk dicts from :func:`parse_pdf`.
        embeddings: Parallel embedding vectors from :func:`create_embeddings`.
        vector_db_path: Directory for the persistent ChromaDB store.
    """
    import chromadb

    vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_path))

    collection_name = config.VECTOR_DB_COLLECTION

    # Drop existing collection so re-ingestion is idempotent
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(collection_name)

    ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    metadatas = [
        {"page": chunk["page_number"], "section": chunk["section"]}
        for chunk in text_chunks
    ]
    documents = [chunk["content"] for chunk in text_chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(
        "Stored %d chunks in ChromaDB collection '%s' at %s",
        len(text_chunks),
        collection_name,
        vector_db_path,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def ingest_ng12_guidelines(
    pdf_url: Optional[str] = None,
    output_dir: Optional[Path] = None,
    vector_db_path: Optional[Path] = None,
) -> Path:
    """Run the full ingestion pipeline: download -> parse -> embed -> index.

    Args:
        pdf_url: (unused, reserved for future override)
        output_dir: Where to store the PDF (default: ``config.DATA_DIR``).
        vector_db_path: Where to persist ChromaDB (default: ``config.VECTOR_DB_PATH``).

    Returns:
        Path to the populated vector database directory.
    """
    if output_dir is None:
        output_dir = config.DATA_DIR
    if vector_db_path is None:
        vector_db_path = config.VECTOR_DB_PATH

    # 1. Download
    pdf_path = download_ng12_pdf(output_dir)

    # 2. Parse
    chunks = parse_pdf(pdf_path)

    # 3. Embed
    embeddings = create_embeddings(chunks)

    # 4. Index
    build_vector_index(chunks, embeddings, vector_db_path)

    logger.info("Ingestion complete — vector DB at %s", vector_db_path)
    return vector_db_path


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.ingestion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ingest_ng12_guidelines()
