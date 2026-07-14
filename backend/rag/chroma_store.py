"""
ChromaDB vector store for report RAG.
Uses sentence-transformers for local embeddings.
"""
import hashlib
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_DIR, EMBEDDING_MODEL

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
    """Index a report's text into ChromaDB for RAG."""
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
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "report_id": report_id,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


def query_report(report_id: str, question: str, top_k: int = 4) -> list[dict]:
    """Retrieve relevant chunks for a question within a specific report."""
    collection = get_collection()

    try:
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"report_id": report_id},
        )
    except Exception:
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
            })

    return chunks


def generate_report_id(filename: str) -> str:
    """Generate a stable report ID from filename + random suffix."""
    base = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"report_{base}_{uuid.uuid4().hex[:6]}"
