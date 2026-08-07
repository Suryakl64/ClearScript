"""
ChromaDB vector store for report RAG.

Indexes structured findings + explanations as embeddings so the chat
engine can retrieve relevant findings for user questions.

Uses sentence-transformers/all-MiniLM-L6-v2 for local embeddings.
"""
import hashlib
import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_chroma_client = None
_embedding_fn = None
COLLECTION_NAME = "clearscript_reports"


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils import embedding_functions

        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Index structured findings (Phase 4 enhancement)
# ---------------------------------------------------------------------------

def index_findings(report_id: str, findings: list[dict]) -> dict:
    """
    Index structured findings into ChromaDB for RAG retrieval.

    Each finding becomes one document with rich text for embedding:
        "Hemoglobin (Hb): 9.2 g/dL [LOW]. Reference range: 13.0-17.0.
         Explanation: Your hemoglobin is below normal..."

    Parameters
    ----------
    report_id : str
        Unique report identifier.
    findings : list[dict]
        Structured findings from Phase 2 (unified schema).

    Returns
    -------
    dict
        { "report_id": str, "findings_indexed": int, "status": str }
    """
    if not findings:
        return {"report_id": report_id, "findings_indexed": 0, "status": "empty"}

    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for i, f in enumerate(findings):
        test = f.get("test", "Unknown")
        full_name = f.get("full_name") or test
        value = f.get("value", "N/A")
        unit = f.get("unit", "")
        flag = f.get("flag", "UNKNOWN")
        category = f.get("category", "unknown")
        explanation = f.get("explanation", "")

        # Build range string
        low = f.get("range_low")
        high = f.get("range_high")
        if low is not None and high is not None:
            ref = f"{low}-{high}"
        elif high is not None:
            ref = f"< {high}"
        elif low is not None:
            ref = f"> {low}"
        else:
            ref = "not specified"

        # Rich text for embedding
        doc_text = (
            f"{full_name} ({test}): {value} {unit} [{flag}]. "
            f"Reference range: {ref} {unit}. "
            f"Category: {category}."
        )
        if explanation:
            doc_text += f" {explanation}"

        chunk_id = f"{report_id}_finding_{i}"
        ids.append(chunk_id)
        documents.append(doc_text)
        metadatas.append({
            "report_id": report_id,
            "test": test,
            "full_name": full_name,
            "value": str(value),
            "unit": unit,
            "flag": flag,
            "category": category,
            "finding_index": i,
            "type": "finding",
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info("Indexed %d findings for report %s", len(ids), report_id)
    return {
        "report_id": report_id,
        "findings_indexed": len(ids),
        "status": "indexed",
    }


# ---------------------------------------------------------------------------
# Index raw text (legacy / Phase 2 compatibility)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks if chunks else [text]


def index_report(
    report_id: str,
    text: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Index raw report text into ChromaDB (legacy)."""
    collection = get_collection()
    chunks = _chunk_text(text)
    meta = metadata or {}

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{report_id}_chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "report_id": report_id,
            "chunk_index": i,
            "filename": meta.get("filename", ""),
            "type": "raw_text",
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "report_id": report_id,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_report(report_id: str, question: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant findings/chunks for a question within a specific report.

    Returns a list of matching documents with metadata and similarity scores.
    """
    collection = get_collection()

    try:
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"report_id": report_id},
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    chunks = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "relevance": round(1 - dist, 3),  # cosine: 1 = identical
            })

    return chunks


def generate_report_id(filename: str = "") -> str:
    """Generate a stable report ID from filename + random suffix."""
    base = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"report_{base}_{uuid.uuid4().hex[:6]}"
