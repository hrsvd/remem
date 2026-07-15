# Evidence-ranked recommendations

This document is populated from completed benchmark results. It intentionally
does not start the next product roadmap.

## P0 — correctness or safety blockers

- Keep full-response reuse opt-in until deployments validate representative
  labeled traffic. PAWS held-out unsafe response reuse was 70.05%; no tested
  threshold produced a safe operating point.
- Do not adopt one global threshold. Defaults produced validation unsafe-hit
  rates of 9.52% (Banking77), 68.21% (PAWS), and 46.40% (SQuAD).
- Treat matched-record identity and response-equivalence safety as acceptance
  criteria. PAWS exact/HNSW tiers agreed 100% while top-1 agreed only 93.9%.
- Require a minimum support count or confidence interval for threshold claims.
  Banking77 validation had 11 conservative hits; held-out precision was 92.31%.

## P1 — production-readiness blockers

- Define and test concurrency and multi-process ownership. Current evidence is
  single-process and cannot establish persistence or tenant safety under writers.
- Add deployment observability for requested/resolved mode, fallback reason,
  audited unsafe labels, ANN recall samples, index loads/rebuilds, and tier hits.
- Expose stage-level timing only if it accurately separates storage, filtering,
  ANN discovery, and reranking; current timings are public end-to-end overhead.

## P2 — scale improvements

- Tune `candidate_count` and `ef_search` together. `ef_search=50` was the
  1k-record knee (99.61% recall), but 8k recall was 97.15% at candidate count 10.
- Reduce index memory/build cost before much larger stores. The 8k index added
  74.80 MiB RSS, took 4.97 s to build, and persisted 13.52 MiB.
- Pursue ANN rather than exact scans for scale: exact p95 reached 233.18 ms at
  8k while HNSW remained 4.39 ms, subject to a deployment recall target.

## P3 — developer experience

- Document per-tier threshold calibration with minimum support counts. Safe
  settings in this suite often reduced recall below 1%.
- Provide a supported audit export for matched ID, tier, score, compatibility
  context, and mode resolution; these fields diagnosed the important failures.
- Keep offline, revision-pinned commands so model network probes cannot drift
  a cached benchmark snapshot.
