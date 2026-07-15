# Benchmark plan

## Questions

1. Does Remem choose the safe tier—response, retrieval, or miss—on labeled
   paraphrases, hard negatives, support intents, and shared-document QA?
2. Do the default `0.95`/`0.80` thresholds prioritize precision sufficiently?
3. How much repeated retrieval and generation is avoided relative to no reuse
   and normalized exact-key caching?
4. When does HNSW outperform exhaustive cosine, and what candidate or decision
   recall does it lose as `ef_search` and candidate count change?
5. How do construction, mutation, persistence, reload, memory, and disk costs
   change with record count?
6. Do namespace and version constraints prevent otherwise identical requests
   from crossing compatibility boundaries?

## Evaluation protocol

The dataset-specific labels described in [README.md](README.md) form ground
truth. Validation workloads choose three configurations: highest precision,
best F1 subject to a precision floor, and highest recall. Those fixed
configurations are then evaluated once on held-out test workloads. Default
thresholds are always included and are not modified by the benchmark.

Every search comparison uses identical seed records, queries, embeddings,
eligibility policy, repetitions, warmups, and hardware. HNSW results are
compared with exact cosine for candidate Recall@K, reranked top-1 agreement,
decision agreement, false-positive deltas, latency, throughput, RSS, and disk.

Scale runs use progressively larger prefixes chosen before timing. Dataset
download and embedding generation occur before index construction. Each run
records its actual size; unsupported target sizes are never extrapolated into
measured claims.

## Decision accounting

- True-positive reuse: a response or retrieval decision matches its labeled
  response/retrieval group.
- False-positive reuse: any reused artifact belongs to an incompatible group,
  including a response served when only retrieval was valid.
- True-negative miss: a non-reusable or isolated query receives no artifact.
- False-negative miss: a labeled reusable query is recomputed.
- Retrieval instead of safe response is tier under-reuse; response instead of
  retrieval is unsafe over-reuse.

Precision is the primary safety measure. F1 and savings never override an
unsafe-reuse finding.

## Reliability cases

Tiny fixtures and product tests cover empty indexes, duplicate records,
dimension errors, missing IDs, deletion, corruption recovery, and namespace
partitioning. Real workloads add repeated-run variance and controlled
namespace, KB, prompt, and model isolation. Multi-process tests are excluded
because 1.1 explicitly supports a single owner per persistent path.
