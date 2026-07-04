# Troubleshooting

## Errors

**`CorruptedSnapshotException` on startup**
The JSON store file is malformed. Delete `remem_store.json` (or your custom path) to start fresh. This should not happen under normal operation — `JsonStorage` uses atomic writes to prevent mid-save corruption.

## Unexpected Behavior

**Unexpected `MISS` decisions**
Check that `ExecutionContext` fields (`namespace`, `kb_version`, `model`, `prompt_version`) match exactly between the `check()` call and the stored records. A mismatch causes the record to be filtered out before similarity is computed.

**All decisions are `MISS`, even for identical queries**
Verify you're passing output from the *same* embedding model for similar queries. Remem compares raw embedding vectors — if your embedding model is non-deterministic, or you've changed models between runs, similarity scores won't reflect true semantic similarity.

**Similarity scores seem low**
The defaults are `retrieval_threshold=0.80` and `response_threshold=0.95`, calibrated for well-trained text embedding models. If you're using low-dimensional or custom embeddings, lower the thresholds via `ReusePolicy` — see [API Reference](api.md#reusepolicy).

## Common Mistakes

**Using a different embedding model when querying vs. when storing**
Remem compares raw vector values directly. If records were stored using one embedding model and queries use another, similarity scores become meaningless. Always use the same model throughout.

**Not passing `context` consistently**
If `context` is passed to `remember()` but omitted on `check()` (or vice versa), the metadata filter won't match and you'll always get `MISS`. Use the same `ExecutionContext` values in both calls.

**Expecting cache hits before any data has been stored**
Remem can't reuse what it hasn't seen. The first time a query is processed, it is always a `MISS`. Hits only appear after similar queries have been stored via `remember()`.

**Treating every `MISS` as an error**
`MISS` is the expected outcome for any genuinely new request. Optimize for a reasonable hit rate, not 100% — a 100% hit rate would indicate over-caching.

**Mutating the knowledge base without bumping `kb_version`**
If documents are updated but `kb_version` stays the same, Remem may serve stale cached responses. Always bump `kb_version` after a knowledge-base update.

**Forgetting to call `remember()` after `MISS` or `RETRIEVAL_REUSED`**
`check()` does not observe your pipeline automatically — you must call `remember()` explicitly for the store to grow. If you'd rather this happen automatically, use `get_or_compute()` instead.

---

If none of the above resolves your issue, please open an issue with a minimal reproduction — see [CONTRIBUTING.md](../CONTRIBUTING.md).