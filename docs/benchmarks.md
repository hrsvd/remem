# Benchmarks

Benchmarking is an important part of Remem's development. The goal is to measure the effectiveness of **semantic reuse** — not simply raw lookup speed.

## What Remem Aims to Optimize

- Cache hit rate
- Semantic reuse rate (retrieval reuse vs. full response reuse)
- Average similarity of served results
- Latency reduction vs. running the full pipeline
- Token savings
- API cost reduction
- Memory usage

## Release Validation in 1.1.0

Remem does not publish unverified latency or throughput claims. The `1.1.0`
suite instead verifies the work avoided by each ANN optimization:

- ANN queries use candidate-ID lookup and do not call `storage.all()` in the
  built-in query path.
- Incremental inserts, updates, deletes, and repeated reloads expose rebuild
  counters proving normal mutations avoid a full rebuild.
- Valid persistent namespace indexes expose load counters proving startup did
  not re-add every vector.
- Strict namespace filtering avoids searching unrelated namespace partitions.
- Structured filtering may expand native search conservatively, but storage
  fetches only policy-eligible IDs.

These are operation-count and lifecycle invariants, not wall-clock benchmark
results. Inspect `client.ann_index_stats` for index load/rebuild telemetry.

## Planned Comparative Benchmarks

### Similarity Search

Compares similarity-search strategies as the store grows:

- Exact cosine similarity
- Optional HNSW candidate retrieval plus exact cosine reranking
- HNSW construction, incremental mutation, persistence, and filtered retrieval

Measured on:

- Recall
- Query latency
- Memory consumption

### Storage

Compares supported storage backends:

- Read latency
- Write latency
- Disk usage

### End-to-End

Measures complete LLM workflows rather than individual components:

**Example workloads:**

- RAG pipelines
- Chat applications
- AI agents

**Measured on:**

- Response time
- Cost savings
- Retrieval reuse rate
- Generation reuse rate

## Benchmark Environment

Future benchmark reports will document:

- Hardware specifications
- Python version
- Dataset size
- Embedding model
- LLM model
- Operating system

This ensures results are reproducible and comparable across releases.

## Current Status

The correctness and operation-count instrumentation above ships in `1.1.0`.
A reproducible wall-clock benchmark harness and published hardware-specific
reports remain planned; see the [Roadmap](roadmap.md). Until then, no fixed
latency, throughput, memory, or recall improvement is claimed.
