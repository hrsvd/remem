# Remem real-world evaluation

This directory contains the reproducible benchmark harness for reuse quality,
retrieval quality, search performance, scalability, persistence, and estimated
work savings. Benchmark code is not included in the `remem-ai` wheel and does
not change Remem's runtime defaults.

The completed 2026-07-15 evaluation is published in the
[real-world benchmark report](reports/report.md), with compact
[JSON](reports/summary.json), [CSV](reports/summary.csv), charts, and
[failure examples](reports/failure-examples.json).

The benchmark asks whether each query should receive full-response reuse,
retrieval-only reuse, or a complete miss. A fast decision is not credited when
the matched record belongs to the wrong ground-truth response or retrieval
group.

## Selected datasets

| Dataset | Version and source | License | Benchmark role |
|---|---|---|---|
| PAWS-Wiki | `labeled_final`, [Google Research](https://github.com/google-research-datasets/paws) | Free use for any purpose; attribution requested | Human-labeled paraphrases and high-overlap non-paraphrases for response safety |
| Banking77 | Official train/test files, [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) | CC BY 4.0 | Fine-grained customer-support intents for retrieval reuse and semantic collisions |
| SQuAD | 1.1, [Stanford](https://rajpurkar.github.io/SQuAD-explorer/) | CC BY-SA 4.0 | Shared source passages for retrieval reuse and answer spans for response equivalence |
| BEIR SciFact | BEIR archive, [UKP-TUDA](https://github.com/beir-cellar/beir) | CC BY-NC 2.0 | Scientific claims sharing cited evidence, with no assumed response equivalence |

Download receipts record source URLs, byte sizes, and SHA-256 checksums. Raw and
processed datasets are ignored by Git. PAWS-QQP is deliberately excluded
because its upstream project cannot redistribute the underlying Quora text.

## Ground truth and splits

- PAWS label `1` permits response reuse between the labeled pair; label `0`
  requires a miss. Official development data is validation and official test
  data remains held out.
- Banking77 queries sharing an intent may reuse the same FAQ/retrieval result,
  but are not assumed to have identical final answers. The official test split
  is deterministically divided by text hash into validation and held-out test.
- SQuAD questions sharing a source passage may reuse retrieval. Full-response
  reuse additionally requires the normalized annotated answer text to match.
  Development passages are split by context hash, preventing passage leakage.
- SciFact claims with a positive qrel to the same document may reuse retrieval;
  distinct claims never receive response-equivalent labels. Official train is
  validation and official test remains held out.
- Exact duplicates and controlled namespace, KB-version, prompt-version, and
  model changes exercise response hits and isolation failures.

Validation results select thresholds. Only the held-out test workloads should
be used for final claims. The embedding score is never used as ground truth.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[ann,benchmark,dev]"
```

The real-data default is
`sentence-transformers/all-MiniLM-L6-v2` at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. It produces 384-dimensional,
normalized embeddings on CPU. Embeddings are batched and cached beneath the
ignored results directory. The hash encoder in `smoke-exact.json` is only a
dependency-free framework fixture and must not be reported as model-quality
evidence.

## Commands

Smoke test without network access:

```bash
python -m benchmarks.run --config benchmarks/configs/smoke-exact.json
python -m benchmarks.report benchmarks/results/smoke-exact.json --output benchmarks/results/smoke-report.md
```

Download and preprocess validation data:

```bash
python -m benchmarks.datasets.download banking77
python -m benchmarks.datasets.download paws_wiki
python -m benchmarks.datasets.download squad_v1
python -m benchmarks.datasets.download beir_scifact
python -m benchmarks.datasets.preprocess banking77 --split validation
python -m benchmarks.datasets.preprocess paws_wiki --split validation
python -m benchmarks.datasets.preprocess squad_v1 --split validation
python -m benchmarks.datasets.preprocess beir_scifact --split validation
```

Create held-out workloads by replacing `validation` with `test`. Use `--limit`
for a representative local subset. Copy
`benchmarks/configs/real-quality-template.json`, change its workload, run name,
search mode, and ANN configuration, then execute:

```bash
python -m benchmarks.run --config benchmarks/configs/my-run.json
python -m benchmarks.report benchmarks/results/*.json --output benchmarks/results/report.md
```

The configuration's threshold arrays drive a sweep without tuning on the test
split. For exact-versus-HNSW comparisons, keep the workload, embeddings,
thresholds, warmups, and repeats identical while changing only `search_mode`
and `ann` parameters. Use `search_mode: "auto"` to record resolution and the
fallback reason.

The checked-in `*-validation-*`, `*-test-selected-*`, and `paws-scale-*`
configurations reproduce the reported threshold, held-out, ANN sensitivity,
and scale experiments after the ignored datasets have been downloaded and
preprocessed. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after the
pinned model snapshot is cached to prevent network drift.

The `*-similarity-only-*` and `*-multi-signal-*` configs are paired held-out
comparisons. They keep data, embeddings, thresholds, and search mode fixed; the
similarity-only profile disables the new policy checks, while the multi-signal
profile applies the validation-selected `0.10` response score margin. This
margin is an experiment setting, not a new runtime default.

## Outputs and metrics

Each run produces JSON plus an observation CSV. JSON includes:

- TP/FP/TN/FN reuse confusion, precision, recall, F1, unsafe reuse, and
  response/retrieval precision;
- p50/p95/p99 query latency and throughput from `perf_counter_ns`;
- Recall@K, Precision@K, MRR, nDCG@K, ANN candidate recall relative to exact,
  and final top-1 agreement;
- cold index construction, optional persistent reload, mutation latency,
  process RSS delta, and index bytes;
- no-reuse and metadata-equivalent normalized exact-key baselines;
- model/retrieval calls avoided and configurable simulated token/cost savings;
- machine, Python, dependency, model, threshold, seed, and ANN configuration.

Warmups precede measured repetitions. Download and embedding time are excluded
from query latency. RSS is process-wide; simulated costs are labeled estimates.
Candidate and ranking diagnostics use Remem's internal similarity hooks and are
identified as such, while all reported reuse decisions come from public
`Client.check()` calls.

## Known measurement gaps

Remem 1.1 does not expose separate timers for storage lookup, HNSW discovery,
exact reranking, or policy filtering. The harness therefore reports public
end-to-end overhead and direct operation timings without pretending the
residual is a precise stage breakdown. Multi-process ownership, networked
storage, concurrent writers, and GPU embeddings are outside the current local
single-process product contract and require separate future experiments.

See [benchmark-plan.md](benchmark-plan.md) for the evaluation protocol and
[recommendations.md](recommendations.md) for evidence-ranked follow-up work.
