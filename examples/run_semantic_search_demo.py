"""Proof-of-concept: does this repo actually need ChromaDB semantic search,
or is the existing bucket/keyword `search_documents` MVP good enough?

Background: the "Agentic RM System" production spec calls for a ChromaDB
`loan_documents` collection (semantic search over document content) as a
shared tool. LG-discOrch's architecture doc flags this as missing
infrastructure everywhere (item T1) and explicitly asks: "priority high IF
the Title/Insurance/Taxes Agent genuinely needs free-text querying across a
loan's docs; otherwise reconsider whether bucket-based lookup is
sufficient." This script answers that question with a concrete experiment
instead of guessing.

The experiment: two Title Commitments for two different loans, one that
actually includes chain-of-title language and one that doesn't -- both
filed under the exact same "Title Report" eFolder bucket (which is all
bucket/keyword matching can see). We ask the same free-text question of
both search implementations: "does this title commitment include a chain
of title".

Run:
    pip install chromadb   # one-time, downloads a small local embedding model
    python3 examples/run_semantic_search_demo.py

No OpenAI API key or network calls are required at query time -- ChromaDB's
default embedding function (all-MiniLM-L6-v2) runs fully offline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_search import LoanDocumentIndex
from title_insurance_tax_agent.tools import search_documents

LOAN_WITH_CHAIN = "loan-aaa-has-chain-of-title"
LOAN_WITHOUT_CHAIN = "loan-bbb-missing-chain-of-title"

TITLE_COMMITMENT_WITH_CHAIN = """
PRELIMINARY TITLE COMMITMENT
Commitment Number: 26-915969-AR
Issuing Agent: Capital Title of Texas, LLC
Property: 1013 Waterford Dr, Ellis County, Texas

SCHEDULE B-1 -- REQUIREMENTS
This Company will require the following prior to issuing its policy:
A full chain of title for the subject property has been examined, tracing
record ownership back through the following conveyances: Warranty Deed
recorded 1998 from Original Grantor to First Owner; Warranty Deed recorded
2006 from First Owner to Second Owner; Deed of Trust recorded 2006; Warranty
Deed recorded 2019 from Second Owner to Current Seller. No gaps in the chain
of title were identified. Current vesting is confirmed in Current Seller.
"""

TITLE_COMMITMENT_WITHOUT_CHAIN = """
PRELIMINARY TITLE COMMITMENT
Commitment Number: 26-440219-BQ
Issuing Agent: Metro National Title
Property: 4471 Larkspur Ln, Davis County, Utah

SCHEDULE B-1 -- REQUIREMENTS
This Company will require the following prior to issuing its policy:
Payoff of the existing deed of trust recorded against the property. Payment
of all outstanding taxes. A survey meeting current ALTA standards. Removal
of the mechanic's lien recorded against the property in favor of ABC
Contracting. Current vesting is confirmed in Current Seller, record owner
per the most recently recorded deed.
"""


def run_bucket_search(query: str, loan_docs: dict) -> list:
    """The repo's existing MVP: exact bucket-name/keyword matching."""
    efolder_index = {"Title Report": [loan_docs]}
    return search_documents(query, loan_guid="n/a", efolder_index=efolder_index)


def main():
    print("=== Semantic search POC: is ChromaDB actually needed here? ===\n")

    query = "does this title commitment include a chain of title"

    print("--- Loan A: title commitment DOES include chain-of-title language ---")
    bucket_hits_a = run_bucket_search(
        query, {"doc_id": "A1", "filename": "Title Commitment.pdf", "text": TITLE_COMMITMENT_WITH_CHAIN}
    )
    print(f"Bucket/keyword search_documents(): {len(bucket_hits_a)} hit(s) "
          f"(found the document, but can't tell us *whether it has chain of title*)")

    print("\n--- Loan B: title commitment does NOT include chain-of-title language ---")
    bucket_hits_b = run_bucket_search(
        query, {"doc_id": "B1", "filename": "Title Commitment.pdf", "text": TITLE_COMMITMENT_WITHOUT_CHAIN}
    )
    print(f"Bucket/keyword search_documents(): {len(bucket_hits_b)} hit(s) "
          f"(SAME result shape as Loan A -- bucket matching can't distinguish them)")

    print("\n--- Now the same question via ChromaDB semantic search ---")
    index = LoanDocumentIndex()
    index.ingest_document(LOAN_WITH_CHAIN, "title_report", "Title Commitment.pdf", TITLE_COMMITMENT_WITH_CHAIN)
    index.ingest_document(LOAN_WITHOUT_CHAIN, "title_report", "Title Commitment.pdf", TITLE_COMMITMENT_WITHOUT_CHAIN)

    for loan_guid, label in [(LOAN_WITH_CHAIN, "Loan A (has chain of title)"),
                              (LOAN_WITHOUT_CHAIN, "Loan B (missing chain of title)")]:
        hits = index.search(query, loan_guid=loan_guid, n_results=1)
        top = hits[0]
        mentions_chain = "chain of title" in top["text"].lower()
        print(f"\n{label}:")
        print(f"  top chunk (distance={top['distance']:.3f}): "
              f"{'...' + top['text'][:160].strip() + '...'}")
        print(f"  -> retrieved chunk literally mentions 'chain of title': {mentions_chain}")

    print(
        "\n=== Conclusion ===\n"
        "Bucket/keyword matching finds the right DOCUMENT (both loans return "
        "a hit) but is blind to document CONTENT -- it would report both "
        "loans identically. Semantic search over content at least surfaces "
        "the passage an LLM (or a dumb substring check) would need to read "
        "to answer the actual question. For THIS specific question (chain "
        "of title present y/n), a plain substring check on the full doc "
        "text ('chain of title' in text.lower()) would also work and is far "
        "cheaper -- so this is really only worth building for genuinely "
        "fuzzy/paraphrased free-text questions where keyword matching misses "
        "synonyms, not for yes/no presence checks on a known phrase.\n"
        "See README.md 'Where the ChromaDB piece needs to go' for the full "
        "gap list to take this from POC to production."
    )


if __name__ == "__main__":
    main()
