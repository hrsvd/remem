# Remem — Quick Start

Get from zero to a working integration in under five minutes.

---

## Prerequisites

- Python 3.10 or later
- pip

---

## 1. Clone (optional) and Install

If you just want the library:

```bash
pip install remem-ai
```

To also run the bundled examples and tests, clone the repo first:

```bash
git clone https://github.com/harshvardhansingh7/remem.git
cd remem
pip install -e ".[dev]"
```

---

## 2. Verify the Installation

```bash
python -c "import remem; print(remem.__version__)"
```

Expected output: `0.6.0`

---

## 3. No Environment Variables Required

Remem does not read any environment variables. All configuration is in code. By default it stores data in `remem_store.json` in the current working directory.

---

## 4. Your First Integration

```python
from remem import Client, ExecutionContext, ExecutionResult, ReuseDecision

# Create a client — persists to remem_store.json by default
client = Client()

# Scope the cache to your application, KB version, and model
context = ExecutionContext(
    namespace="my-app",
    kb_version="v1",
    model="gpt-4o",
)

def handle_query(user_query: str) -> str:
    embedding = my_embedding_model.embed(user_query)   # your embedding model
    outcome   = client.check(embedding, context=context)

    if outcome.decision == ReuseDecision.RESPONSE_REUSED:
        return outcome.result                          # cached — skip everything

    if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
        answer = my_llm.generate(user_query, outcome.references)  # skip vector DB
        client.remember(embedding, answer, outcome.references, context=context)
        return answer

    # MISS — run the full pipeline
    docs   = my_vector_db.search(embedding)
    answer = my_llm.generate(user_query, docs)
    client.remember(embedding, answer, references=docs, context=context)
    return answer
```

Prefer a single-callback style? Use `get_or_compute` instead:

```python
from remem import Client, ExecutionResult

client = Client()

outcome = client.get_or_compute(
    query_embedding=my_embedding_model.embed(user_query),
    compute_callback=lambda: ExecutionResult(
        response=my_llm.generate(user_query, my_vector_db.search(user_query)),
        references=my_vector_db.last_doc_ids,
    ),
)

return outcome.result
```

---

## 5. Run the Bundled Examples

These require a source install (`pip install -e ".[dev]"`).

```bash
# End-to-end RAG demo — shows MISS, RESPONSE_REUSED, and RETRIEVAL_REUSED
python examples/rag_reuse.py

# Durable persistence demo — simulates a process restart
python examples/persistent_storage.py
```

Both use fake embeddings and a fake LLM so they run with no external dependencies.

---

## 6. Confirm Everything is Working

Run the test suite:

```bash
pytest
```

All tests should pass. Then run the RAG example and look for output like:

```
>>> [cold start -> full pipeline]
    Query: 'What is our company's vacation policy?'
   [remem] MISS — running full pipeline
   [vector-db] searching: ...
   [llm] generating answer using 2 docs: ...
    Answer: Employees receive 20 paid leave days per year.

>>> [near-identical -> full response reused]
    Query: 'How many paid leaves do employees receive?'
   [remem] RESPONSE_REUSED (sim=0.99) — skipped vector-DB and LLM
    Answer: Employees receive 20 paid leave days per year.
```

A `RESPONSE_REUSED` decision on the second query confirms Remem is working.

---

## What to Read Next

- [Getting Started Guide](getting-started.md) — full API reference, advanced configuration, common patterns, and FAQ
- [examples/rag_reuse.py](../examples/rag_reuse.py) — annotated end-to-end RAG integration
- [examples/persistent_storage.py](../examples/persistent_storage.py) — durable JsonStorage walkthrough
