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

## Real-world evaluation

The post-1.1 benchmark work now includes measured Banking77, PAWS-Wiki, SQuAD,
and BEIR SciFact evidence. The [real-world report](../benchmarks/reports/report.md) covers
quality, thresholds, scale through 8,000 records, persistence, failures, and
simulated work savings.

The evidence classifies Remem as **suitable for internal testing**. HNSW
materially improves query overhead, but unsafe response reuse and threshold
instability block a production-ready claim. Results are hardware-specific, not
package-level performance guarantees.

## Comparative benchmark coverage

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

## Benchmark environment

Benchmark reports document:

- Hardware specifications
- Python version
- Dataset size
- Embedding model
- LLM model
- Operating system

This ensures results are reproducible and comparable across releases.

## Current status

The correctness and operation-count instrumentation above ships in `1.1.0`.
The repository's post-release benchmark work publishes small reviewed summary
artifacts while raw data, observations, embeddings, and indexes stay out of
Git. Results remain specific to the recorded hardware, model, datasets, and
configuration; they are not part of the `1.1.0` package release.

## Reproducible Evaluation Framework

The repository now contains a separate [real-world evaluation harness](../benchmarks/README.md)
covering labeled reuse quality, threshold sweeps, exact-key and no-reuse
baselines, exact-versus-HNSW diagnostics, persistence, mutations, resource use,
and configurable cost estimates. Generated datasets and results remain ignored
until reviewed; the main documentation continues to avoid unsupported
performance claims.

A 2026-07-19 follow-up compares threshold-only decisions with dependency-light
multi-signal policy checks on the same held-out workloads. The checks can
downgrade a response hit to retrieval reuse when intent, entities, values,
time, negation, direction, format, freshness, or score ambiguity differs. The
results show improved response-tier precision on difficult workloads, but also
show that lexical signals alone cannot prove response equivalence; applications
still need labeled calibration and explicit dependency metadata.
