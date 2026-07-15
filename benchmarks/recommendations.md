# Evidence-ranked recommendations

This document is populated from completed benchmark results. It intentionally
does not start the next product roadmap.

## P0 — correctness or safety blockers

- Pending held-out real-data results.

## P1 — production-readiness blockers

- Add supported stage-level latency and candidate diagnostics; the 1.1 public
  metrics cannot separate storage, policy filtering, ANN discovery, and exact
  reranking time.
- Validate concurrency and multi-process behavior only after defining an
  ownership contract beyond the current single-process guarantee.

## P2 — scale improvements

- Pending exact/HNSW crossover, memory, construction, and reload results.

## P3 — developer experience

- Consider a stable export format for per-decision telemetry after the
  benchmark result schema has been exercised on all selected datasets.
