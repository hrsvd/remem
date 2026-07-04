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

## Planned Benchmarks

### Similarity Search

Compares similarity-search strategies as the store grows:

- Exact cosine similarity (current default)
- Approximate Nearest Neighbor (ANN) search
- HNSW indexing

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

Benchmark infrastructure is under active development. Performance reports will be published here as the project matures — see the [Roadmap](roadmap.md) for timing.