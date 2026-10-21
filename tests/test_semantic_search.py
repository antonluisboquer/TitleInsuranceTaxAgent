"""Tests for the ChromaDB semantic-search POC (core/semantic_search.py).

chromadb is an optional dependency (not in requirements.txt -- it's only
needed for this POC/demo, not for the deterministic tool functions the rest
of this repo is built on). These tests skip automatically if it isn't
installed. `pip install chromadb` to run them.
"""
import pytest

pytest.importorskip("chromadb")

from core.semantic_search import LoanDocumentIndex, chunk_text  # noqa: E402

LOAN_A = "loan-a"
LOAN_B = "loan-b"


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_is_a_single_chunk():
    chunks = chunk_text("just a few words here", chunk_size=150, overlap=30)
    assert len(chunks) == 1
    assert chunks[0] == "just a few words here"


def test_chunk_text_overlaps_long_text():
    words = [f"word{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), chunk_size=150, overlap=30)
    assert len(chunks) > 1
    # Consecutive chunks should share the overlap words.
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-30:] == second_chunk_words[:30]


def test_ingest_and_search_finds_relevant_chunk():
    index = LoanDocumentIndex()
    index.ingest_document(
        LOAN_A, "title_report", "title_commitment.pdf",
        "This title commitment includes a full chain of title back to 1985 with no gaps.",
    )
    index.ingest_document(
        LOAN_A, "tax_summary", "tax_summary.pdf",
        "Annual property taxes for 2026 are $4,148.29, due in two installments.",
    )
    hits = index.search("does the title commitment mention chain of title", loan_guid=LOAN_A, n_results=1)
    assert len(hits) == 1
    assert "chain of title" in hits[0]["text"].lower()
    assert hits[0]["metadata"]["doc_type"] == "title_report"


def test_search_is_scoped_to_loan_guid():
    index = LoanDocumentIndex()
    index.ingest_document(LOAN_A, "title_report", "tc.pdf", "Loan A's title commitment has a chain of title gap.")
    index.ingest_document(LOAN_B, "title_report", "tc.pdf", "Loan B's title commitment is clean, nothing outstanding.")

    hits_a = index.search("title commitment", loan_guid=LOAN_A, n_results=5)
    assert all(h["metadata"]["loan_guid"] == LOAN_A for h in hits_a)
    assert any("gap" in h["text"].lower() for h in hits_a)

    hits_b = index.search("title commitment", loan_guid=LOAN_B, n_results=5)
    assert all(h["metadata"]["loan_guid"] == LOAN_B for h in hits_b)
    assert all("gap" not in h["text"].lower() for h in hits_b)


def test_search_can_filter_by_doc_type():
    index = LoanDocumentIndex()
    index.ingest_document(LOAN_A, "title_report", "tc.pdf", "Title commitment schedule B exceptions and requirements.")
    index.ingest_document(LOAN_A, "tax_summary", "tax.pdf", "Property tax bill for the current year.")

    hits = index.search("document", loan_guid=LOAN_A, n_results=5, doc_type="tax_summary")
    assert hits
    assert all(h["metadata"]["doc_type"] == "tax_summary" for h in hits)


def test_ingesting_empty_text_adds_nothing():
    index = LoanDocumentIndex()
    chunk_count = index.ingest_document(LOAN_A, "title_report", "empty.pdf", "")
    assert chunk_count == 0
