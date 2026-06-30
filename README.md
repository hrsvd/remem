# Remem

> **Remember expensive AI work. Reuse it intelligently.**

<p align="center">
  <strong>An AI Work Reuse Engine for RAG Pipelines, AI Agents, and LLM Applications.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="Version" src="https://img.shields.io/badge/Version-v0.6.0--alpha-orange">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
  <img alt="PyPI" src="https://img.shields.io/badge/PyPI-remem--ai-blue">
</p>

---

## Overview

Modern AI applications repeatedly execute expensive operations — vector database searches, LLM calls, tool executions, reranking passes — for requests that are often semantically identical or very similar.

Remem sits between your application and its expensive AI operations. It observes what has already been computed, and for each new request it determines the highest level of previous work that can be safely reused — based on **semantic similarity**, not exact key matching.

```
User asks: "What is our company's vacation policy?"
User asks: "How many paid leaves do employees get?"   <-- Remem reuses the first answer
User asks: "What are the PTO rules?"                 <-- Remem reuses the first answer
```

---

## Why Not Just Use Redis?

Redis answers: *"Have I seen this exact key before?"*

Remem answers: *"Have I already done similar expensive AI work before?"*

| | Redis | Remem |
|---|---|---|
| Matching strategy | Exact key | Semantic similarity (embeddings) |
| Scope | Key-value lookup | Full AI pipeline reuse |
| Deployment | Separate service | Python library — import and go |

---

## How Remem Works

When a new request arrives, Remem computes its similarity against previously stored executions. It then makes a three-way decision:

```
Incoming Request
      │
      ▼
Generate Embedding  (your embedding model)
      │
      ▼
client.check(embedding)
      │
      ├──► RESPONSE_REUSED   ──► Return cached LLM response immediately
      │    (similarity ≥ 0.95)     Skip vector DB and LLM entirely
      │
      ├──► RETRIEVAL_REUSED  ──► Use cached document IDs, call LLM only
      │    (similarity ≥ 0.80)     Skip the vector DB search
      │
      └──► MISS              ──► Run your full pipeline
           (similarity < 0.80)     Store result with client.remember()
```

Before computing similarity, Remem first filters stored records by metadata compatibility — namespace, knowledge-base version, model, and prompt version. This means updating your knowledge base or switching LLM models automatically invalidates old results with no manual cache-busting.

### The Three Decisions

| Decision | Condition | What to do |
|---|---|---|
| `RESPONSE_REUSED` | Similarity ≥ `response_threshold` (default `0.95`) | Return `outcome.result` directly. Skip everything. |
| `RETRIEVAL_REUSED` | Similarity ≥ `retrieval_threshold` (default `0.80`) | Use `outcome.references` (cached doc IDs). Call your LLM. Store the new response. |
| `MISS` | Similarity < `retrieval_threshold` | Run your full pipeline. Store with `client.remember()`. |

---

## Features

**Implemented in v0.6.0-alpha:**

- **Three-level reuse decision engine** — full response reuse, partial retrieval reuse, or miss
- **Semantic similarity matching** — cosine similarity over embedding vectors; no exact-match required
- **Metadata-aware filtering** — namespace, KB version, prompt version, and model constraints applied before similarity computation
- **Configurable reuse policy** — tune thresholds and metadata constraints via `ReusePolicy`
- **Durable persistence** — `JsonStorage` writes to disk with atomic rename cycles; survives process restarts
- **In-memory storage** — `InMemoryStorage` for tests, notebooks, and short-lived jobs
- **Custom storage backends** — implement `StorageInterface` to plug in Redis, Postgres, S3, or any system
- **Built-in telemetry** — `MetricsCollector` tracks hit rate, reuse breakdown, and average similarity
- **Pip-installable** — `pip install remem-ai`
- **Single runtime dependency** — only `numpy`

**Not yet implemented (planned):**

- HTTP server / REST API
- Language SDKs (Java, Rust)
- Distributed mode
- Rust-accelerated similarity search

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Client                           │
│   check() · remember() · get_or_compute() · store()     │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │       ReuseEngine       │
          │  check() · get_or_compute()  │
          └──┬──────────┬───────────┘
             │          │
   ┌──────────▼──┐  ┌───▼──────────────┐
   │  Metadata   │  │   Similarity     │
   │  Matcher    │  │   Engine         │
   │             │  │  (cosine sim)    │
   └──────────┬──┘  └───┬──────────────┘
              │          │
              └────┬─────┘
                   │
         ┌─────────▼──────────┐
         │   StorageInterface  │
         ├─────────────────────┤
         │  JsonStorage        │  ← durable, atomic writes
         │  InMemoryStorage    │  ← volatile, in-process
         │  [Custom]           │  ← implement StorageInterface
         └─────────────────────┘
```

### Component Responsibilities

| Component | File | Role |
|---|---|---|
| `Client` | `remem/client.py` | Public façade; all user-facing methods |
| `ReuseEngine` | `remem/reuse/engine.py` | Orchestrates metadata filtering + similarity + decision |
| `MetadataMatcher` | `remem/reuse/matcher.py` | Filters stored records by `ReusePolicy` constraints |
| `SimilarityEngine` | `remem/similarity/engine.py` | Cosine similarity search over all compatible candidates |
| `ReusePolicy` | `remem/reuse/policy.py` | Configurable thresholds and metadata constraints |
| `JsonStorage` | `remem/storage/json_storage.py` | File-backed persistence; atomic write cycle |
| `InMemoryStorage` | `remem/storage/memory_storage.py` | In-process dict; no disk I/O |
| `Serializer` | `remem/storage/serializer.py` | JSON serialization of `ExecutionRecord` |
| `MetricsCollector` | `remem/metrics/collector.py` | Accumulates telemetry; produces `MetricsSnapshot` |

### Data Models

**`ExecutionRecord`** — the unit stored in the persistence layer:

```
id           UUID
embedding    list[float]         # the query embedding
response     Any                 # LLM output (string, dict, DataFrame, ...)
references   list[str]           # document / chunk IDs from vector DB
context      ExecutionContext    # metadata for filtering
hit_count    int                 # how many times this record was reused
created_at   datetime
```

**`ExecutionContext`** — metadata scoping for a request:

```
namespace       str              # e.g. "support-bot", "tenant:acme"
kb_version      str              # knowledge-base version string
prompt_version  str              # prompt template version
model           str | None       # LLM identifier, e.g. "gpt-4o"
metadata        dict[str, Any]   # custom key-value pairs
```

---

## Folder Structure

```
remem/
├── docs/
│   ├── getting-started.md     # Full API reference and integration guide
│   └── QuickStart.md          # Minimal fast-start guide
├── examples/
│   ├── rag_reuse.py           # End-to-end RAG demo (check + remember)
│   └── persistent_storage.py  # Durable JsonStorage demo
├── remem/
│   ├── __init__.py            # Package exports and __version__
│   ├── client.py              # Client — public façade
│   ├── models/
│   │   ├── execution_context.py
│   │   ├── execution_record.py
│   │   ├── execution_result.py
│   │   └── retrieval_entry.py
│   ├── reuse/
│   │   ├── engine.py          # ReuseEngine — core decision logic
│   │   ├── matcher.py         # MetadataMatcher — policy-based filtering
│   │   └── policy.py          # ReusePolicy — thresholds and constraints
│   ├── similarity/
│   │   ├── engine.py          # SimilarityEngine — cosine similarity search
│   │   └── metrics.py         # cosine_similarity(), dot_product(), vector_norm()
│   ├── storage/
│   │   ├── storage.py         # StorageInterface — abstract base class
│   │   ├── json_storage.py    # JsonStorage — durable, atomic file persistence
│   │   ├── memory_storage.py  # InMemoryStorage — in-process volatile store
│   │   ├── serializer.py      # Serializer — JSON round-trip for ExecutionRecord
│   │   ├── snapshot.py        # StorageSnapshot — immutable persistence payload
│   │   └── exceptions.py      # StorageException, PersistenceException, etc.
│   └── metrics/
│       ├── collector.py       # MetricsCollector
│       ├── events.py          # MetricEvent enum
│       └── snapshot.py        # MetricsSnapshot — immutable telemetry payload
├── tests/
│   └── test_persistence.py
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Installation

**Requirements:** Python 3.10 or later.

```bash
pip install remem-ai
```

Remem has a single runtime dependency (`numpy`) which is installed automatically.

### From Source

```bash
git clone https://github.com/harshvardhansingh7/remem.git
cd remem
pip install -e ".[dev]"    # editable install with pytest and ruff
```

---

## Quickstart

### One-liner — `get_or_compute`

The simplest integration. Pass your embedding and a callback that runs your pipeline. Remem decides whether to call it.

```python
from remem import Client, ExecutionResult

client = Client()   # durable JSON persistence to remem_store.json by default

def my_pipeline():
    docs = search_vector_db(query)
    answer = call_llm(query, docs)
    return ExecutionResult(response=answer, references=docs)

outcome = client.get_or_compute(
    query_embedding=embed(query),
    compute_callback=my_pipeline,
)

print(outcome.decision)   # RESPONSE_REUSED | RETRIEVAL_REUSED | MISS
print(outcome.result)     # the answer — cached or freshly computed
```

### Explicit — `check` + `remember`

For finer control (e.g. skipping only the vector-DB call on `RETRIEVAL_REUSED`):

```python
from remem import Client, ExecutionContext, ReuseDecision

client  = Client()
context = ExecutionContext(namespace="hr-bot", kb_version="2024.1", model="gpt-4o")

embedding = embed(user_query)
outcome   = client.check(embedding, context=context)

if outcome.decision == ReuseDecision.RESPONSE_REUSED:
    return outcome.result                          # skip everything

if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
    answer = call_llm(user_query, outcome.references)   # skip vector DB
    client.remember(embedding, answer, outcome.references, context=context)
    return answer

# MISS: run the full pipeline
docs   = search_vector_db(embedding)
answer = call_llm(user_query, docs)
client.remember(embedding, answer, references=docs, context=context)
return answer
```

### In-memory (for tests and notebooks)

```python
from remem import Client, InMemoryStorage

client = Client(storage_backend=InMemoryStorage())   # nothing written to disk
```

---

## Configuration

### `ReusePolicy` — similarity thresholds and metadata constraints

```python
from remem import Client, ReusePolicy

client = Client(
    policy=ReusePolicy(
        retrieval_threshold=0.80,       # min similarity to reuse cached documents
        response_threshold=0.95,        # min similarity to return the cached response
        require_same_namespace=True,    # reject cross-namespace candidates
        require_same_kb_version=True,   # reject candidates from outdated KB
        require_same_prompt_version=True,
        require_same_model=True,        # reject responses from a different model
    )
)
```

### `ExecutionContext` — metadata scoping

```python
from remem import ExecutionContext

context = ExecutionContext(
    namespace="support-bot",
    kb_version="2024-Q4",
    prompt_version="v3",
    model="gpt-4o",
)
```

Remem filters stored records by context compatibility before computing similarity. Updating `kb_version` automatically excludes stale entries — no manual invalidation needed.

### Environment Variables

Remem does not require or read any environment variables. All configuration is code-based.

---

## Storage Backends

| Backend | Import | Persistence | Best for |
|---|---|---|---|
| `JsonStorage` (default) | `from remem import JsonStorage` | Durable — survives restarts | Production, any long-running app |
| `InMemoryStorage` | `from remem import InMemoryStorage` | Volatile — lost on exit | Tests, notebooks, short jobs |
| Custom | Subclass `StorageInterface` | Your choice | Redis, Postgres, S3, custom DBs |

`JsonStorage` uses an atomic write cycle (`write to .tmp` → `os.replace`) so a crash mid-save never corrupts the file. On Windows it includes retry logic to handle transient `PermissionError` from antivirus or file indexers.

```python
# Custom path
from remem import Client, JsonStorage
client = Client(storage_backend=JsonStorage(filepath="/var/data/remem_cache.json"))
```

### Implementing a Custom Backend

```python
from remem import StorageInterface

class MyRedisStorage(StorageInterface):
    def put(self, record): ...
    def get(self, entry_id): ...
    def delete(self, entry_id): ...
    def update(self, record): ...
    def all(self): ...
    def flush(self): ...
    def load(self): ...
    def increment_hit(self, entry_id): ...
```

---

## API Overview

All operations go through `Client`.

| Method | Signature | Description |
|---|---|---|
| `check` | `(embedding, context=None) → ReuseOutcome` | Check reuse without running your pipeline |
| `remember` | `(embedding, response, references=None, context=None) → None` | Store a pipeline result |
| `get_or_compute` | `(embedding, callback, context=None) → ReuseOutcome` | Check + optional compute + store, all in one |
| `store` | `(record: ExecutionRecord) → None` | Store a fully constructed record directly |
| `delete` | `(entry_id: UUID) → bool` | Remove a record by ID |
| `all` | `() → list[ExecutionRecord]` | Return all stored records |
| `flush_storage` | `() → None` | Clear all records |
| `save_snapshot` | `() → None` | Force-write in-memory state to disk (JsonStorage) |
| `load_snapshot` | `() → None` | Reload records from disk (JsonStorage) |

### `ReuseOutcome` fields

```python
outcome.decision          # ReuseDecision — RESPONSE_REUSED | RETRIEVAL_REUSED | MISS
outcome.result            # cached response (RESPONSE_REUSED) or None
outcome.references        # list[str] — cached document IDs (if any)
outcome.similarity_score  # float — cosine similarity to best match (0.0 for MISS)
outcome.reason            # str — human-readable explanation
outcome.matched_record_id # UUID | None — ID of the matched record
```

---

## Metrics

Remem automatically tracks telemetry. Access it at any time:

```python
print(client.metrics.snapshot())
```

```
========== Metrics ==========
Requests:           100
Hits:                73
Misses:              27
Response Reused:     51
Retrieval Reused:    22
Hit Rate:          73.0%
Average Similarity: 0.941
```

---

## Running the Examples

```bash
# Clone and install
git clone https://github.com/harshvardhansingh7/remem.git
cd remem
pip install -e ".[dev]"

# End-to-end RAG demo — shows all three reuse decisions
python examples/rag_reuse.py

# Durable persistence demo — simulates a process restart
python examples/persistent_storage.py
```

---

## Development Setup

```bash
git clone https://github.com/harshvardhansingh7/remem.git
cd remem
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check remem/

# Format
ruff format remem/
```

Tests live in `tests/test_persistence.py` and cover serialization round-trips, atomic persistence cycles, and error handling for corrupted snapshots.

---

## Important Workflows

### Multi-tenant isolation

Each tenant gets its own namespace — results from one tenant are never served to another:

```python
context = ExecutionContext(
    namespace=f"tenant:{tenant_id}",
    kb_version=tenant.kb_version,
    model="gpt-4o",
)
outcome = client.check(embed(user_query), context=context)
```

### Agent tool-execution reuse

```python
for step in agent_plan:
    embedding = embed(step.description)
    outcome   = client.check(embedding, context)

    if outcome.decision == ReuseDecision.RESPONSE_REUSED:
        tool_result = outcome.result          # skip execution entirely
    else:
        tool_result = execute_tool(step)
        client.remember(embedding, tool_result, context=context)

    agent_state.update(tool_result)
```

### Cache warm-up

Pre-populate known queries so the first real users hit the cache:

```python
context = ExecutionContext(namespace="support", kb_version="2024-Q4")

for question, answer, docs in KNOWN_QA_PAIRS:
    client.remember(embed(question), answer, references=docs, context=context)
```

---

## Troubleshooting

**`CorruptedSnapshotException` on startup**
The JSON store file is malformed. Delete `remem_store.json` (or your custom path) to start fresh. This should not happen under normal operation — `JsonStorage` uses atomic writes to prevent mid-save corruption.

**Unexpected `MISS` decisions**
Check that the `ExecutionContext` fields (`namespace`, `kb_version`, `model`) match exactly between the `check()` call and the stored records. A mismatch causes the record to be filtered out before similarity is computed.

**All decisions are `MISS` even for identical queries**
Verify that you are passing the same embedding model output for similar queries. Remem compares embedding vectors — if your embedding model is non-deterministic or you have changed models between runs, similarity scores will not reflect semantic similarity.

**Similarity scores seem low**
The default `retrieval_threshold` is `0.80` and `response_threshold` is `0.95`. These thresholds are calibrated for well-trained text embedding models. If you are using low-dimensional or custom embeddings, lower the thresholds via `ReusePolicy`.

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.1.0-alpha | Similarity engine, in-memory storage, unit tests | Done |
| v0.2.0 | Public Remem API (`find_similar`, storage) | Done |
| v0.3.0 | Execution reuse engine (`get_or_compute`) | Done |
| v0.4.0 | Metadata-aware policy matching (`ExecutionContext`, `ReusePolicy`) | Done |
| v0.5.0 | Persistence layer (durable `JsonStorage`) | Done |
| v0.6.0 | Performance optimization, profiling, benchmarking | **Current** |
| v0.7.0 | Language SDKs (Python packaging polish, Java, Rust) | Planned |
| v1.0.0 | Production-ready AI Work Reuse Engine | Planned |

---

## Contributing

Contributions are welcome. As the project is in early alpha, architecture discussions and design feedback are especially valuable.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

## Author

**Harshvardhan Singh** — building Remem as an open-source AI infrastructure project to explore distributed systems, storage engines, and high-performance backend engineering.

If this project is useful to you, consider giving it a star and following its progress.
