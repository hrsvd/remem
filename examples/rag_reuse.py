"""End-to-end RAG work-reuse example using the explicit check() + remember() API.

Flow:
    Query -> embed -> client.check()
        RESPONSE_REUSED  -> return cached response       (no LLM call, no vector-DB call)
        RETRIEVAL_REUSED -> cached docs + call LLM only  (no vector-DB call)
        MISS             -> vector-DB + LLM + remember() (full pipeline)

Remem is NOT RAG-specific.  The same pattern works for agents, tool calling,
SQL generation, or any expensive AI step.

Run it with:
    python examples/rag_reuse.py
"""

from remem import Client, ExecutionContext, InMemoryStorage, ReuseDecision, ReusePolicy

# ---------------------------------------------------------------------------
# Fake AI stack — replace these with your real embedding model, vector DB, LLM
# ---------------------------------------------------------------------------

# Hard-coded latent vectors over three topics: [leave, refund, general].
# Your real embedding model (OpenAI, Cohere, etc.) produces these automatically.
_FAKE_EMBEDDINGS = {
    "What is our company's vacation policy?": [1.00, 0.00, 0.20],
    "How many paid leaves do employees receive?": [
        0.99,
        0.00,
        0.25,
    ],  # near-identical -> RESPONSE_REUSED
    "What is our refund policy?": [0.00, 1.00, 0.20],  # different topic -> MISS
    "Tell me about time off and holiday entitlement": [
        0.85,
        0.00,
        0.55,
    ],  # related -> RETRIEVAL_REUSED
}

# Fake knowledge base: what your vector DB would return for each topic
_FAKE_VECTOR_DB = {
    "leave": ["hr_handbook#pto", "hr_handbook#leave_policy"],
    "refund": ["policy_docs#returns", "policy_docs#refund_window"],
}

# Fake LLM answers per document set (different answers for different topics)
_FAKE_LLM_ANSWERS = {
    "leave": "Employees receive 20 paid leave days per year.",
    "refund": "Refunds are processed within 14 business days of return approval.",
}


def embed(text: str) -> list[float]:
    return _FAKE_EMBEDDINGS.get(text, [0.0, 0.0, 1.0])


def search_vector_db(query: str) -> list[str]:
    print(f"   [vector-db] searching: {query!r}")
    topic = (
        "leave"
        if "leave" in query.lower()
        or "vacation" in query.lower()
        or "pto" in query.lower()
        else "refund"
    )
    return _FAKE_VECTOR_DB.get(topic, ["generic_doc#1"])


def call_llm(query: str, documents: list[str]) -> str:
    print(f"   [llm] generating answer using {len(documents)} docs: {documents}")
    topic = "leave" if any("pto" in d or "leave" in d for d in documents) else "refund"
    return _FAKE_LLM_ANSWERS.get(topic, "I don't have information on that.")


# ---------------------------------------------------------------------------
# The RAG function that uses Remem's explicit check() + remember() API
# ---------------------------------------------------------------------------


def rag_with_remem(query: str, client: Client, context: ExecutionContext) -> str:
    embedding = embed(query)

    # Step 1: check remem before touching any expensive resource
    outcome = client.check(embedding, context=context)

    if outcome.decision == ReuseDecision.RESPONSE_REUSED:
        # Full cache hit — skip the entire pipeline
        print(
            f"   [remem] RESPONSE_REUSED (sim={outcome.similarity_score:.2f}) — skipped vector-DB and LLM"
        )
        return outcome.result

    if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
        # Partial hit — skip the vector-DB search, just call the LLM
        print(
            f"   [remem] RETRIEVAL_REUSED (sim={outcome.similarity_score:.2f}) — skipped vector-DB, calling LLM with cached docs"
        )
        response = call_llm(query, outcome.references)
        client.remember(embedding, response, outcome.references, context=context)
        return response

    # MISS — run the full pipeline and store the result
    print("   [remem] MISS — running full pipeline")
    docs = search_vector_db(query)
    response = call_llm(query, docs)
    client.remember(embedding, response, references=docs, context=context)
    return response


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    client = Client(
        storage_backend=InMemoryStorage(),  # swap for Client() to persist to disk
        policy=ReusePolicy(retrieval_threshold=0.80, response_threshold=0.95),
    )
    context = ExecutionContext(namespace="hr-bot", kb_version="2024.1", model="gpt-4o")

    queries = [
        ("What is our company's vacation policy?", "cold start -> full pipeline"),
        (
            "How many paid leaves do employees receive?",
            "near-identical -> full response reused",
        ),
        ("What is our refund policy?", "different topic -> full pipeline"),
        (
            "Tell me about time off and holiday entitlement",
            "related -> cached docs, fresh LLM",
        ),
    ]

    for query, label in queries:
        print(f"\n>>> [{label}]")
        print(f"    Query: {query!r}")
        result = rag_with_remem(query, client, context)
        print(f"    Answer: {result}")

    print("\n--- Metrics ---")
    print(client.metrics.snapshot())


if __name__ == "__main__":
    main()
