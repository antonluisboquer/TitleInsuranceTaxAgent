"""Local proof-of-concept for the ChromaDB-backed semantic document search
described in the "Agentic RM System" production spec (Part 7, "Data &
Knowledge Engineering" -> ChromaDB `loan_documents` collection), and flagged
as backlog item T1 in LG-discOrch's architecture doc: "search_documents /
ChromaDB -- new infrastructure, not an extension of existing code."

This is deliberately NOT what title_insurance_tax_agent.tools.search_documents
does today. That function does exact bucket-name/keyword matching over
document TYPE (e.g. "find documents filed under the 'Title Report' bucket").
This module does semantic search over document CONTENT -- e.g. "does this
title commitment mention a chain of title" -- which bucket/keyword matching
cannot answer, because bucket matching has no visibility into what's
actually written inside a document; it only knows what folder it's filed
under. See `examples/run_semantic_search_demo.py` for a concrete side-by-side
comparison proving this.

Uses ChromaDB's default local embedding function (all-MiniLM-L6-v2, ONNX,
runs fully offline after a one-time ~80MB model download) so this can be
exercised with NO OpenAI API key and no per-query network/cost. The
production spec calls for OpenAI `text-embedding-3-small` instead -- swap it
in via `embedding_function=chromadb.utils.embedding_functions
.OpenAIEmbeddingFunction(api_key=..., model_name="text-embedding-3-small")`
when this graduates from POC to production. See "Where this needs to go"
in the demo script / README for the full gap list (real document text
ingestion, metadata schema completeness, persistence, cost).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when chromadb isn't installed
    chromadb = None  # type: ignore
    CHROMADB_AVAILABLE = False

# Spec's collection name (agentic_rm_overview_spec.pdf, Part 7.1):
# "Documents (loan_documents)".
COLLECTION_NAME = "loan_documents"

# A naive word-count chunker. Production should chunk by token count
# (tiktoken, to match the embedding model's tokenizer) and should respect
# page boundaries from real OCR/extraction output, not just word counts.
DEFAULT_CHUNK_SIZE_WORDS = 150
DEFAULT_CHUNK_OVERLAP_WORDS = 30


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Split `text` into overlapping word-count windows."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


class LoanDocumentIndex:
    """Thin wrapper around a Chroma collection scoped to the spec's
    `loan_documents` collection + metadata shape
    ({loan_guid, doc_type, filename, page_num, upload_date}).

    Defaults to an in-memory (`EphemeralClient`) collection, since this is a
    POC, not a production deployment. Pass `client=chromadb.PersistentClient(
    path=...)` to persist across runs.
    """

    def __init__(self, client: Optional[Any] = None, collection_name: str = COLLECTION_NAME):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run `pip install chromadb` to use "
                "LoanDocumentIndex (this is an optional dependency -- the rest "
                "of this repo's tools/tests don't need it)."
            )
        self._client = client or chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)

    def ingest_document(
        self,
        loan_guid: str,
        doc_type: str,
        filename: str,
        text: str,
        upload_date: Optional[str] = None,
    ) -> int:
        """Chunk, embed, and store one document's text. Returns the number
        of chunks ingested (0 if `text` is empty)."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{loan_guid}:{filename}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "loan_guid": loan_guid,
                "doc_type": doc_type,
                "filename": filename,
                "page_num": i,
                "upload_date": upload_date or "",
            }
            for i in range(len(chunks))
        ]
        self._collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def search(
        self,
        query: str,
        loan_guid: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search scoped to one loan (and optionally one doc type).
        Returns hits ordered by relevance (lowest distance first), each as
        {"text", "metadata", "distance"}."""
        where: Dict[str, Any] = (
            {"$and": [{"loan_guid": loan_guid}, {"doc_type": doc_type}]}
            if doc_type
            else {"loan_guid": loan_guid}
        )
        result = self._collection.query(query_texts=[query], n_results=n_results, where=where)
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
