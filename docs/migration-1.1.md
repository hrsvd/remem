# Migrating from 1.0.0 to 1.1.0

Remem `1.1.0` keeps the `1.0.0` record model, JSON storage format, reuse
thresholds, and exact cosine behavior compatible. No data migration is required.

## Upgrade

For exact cosine with the dependency-light base package:

```bash
python -m pip install --upgrade remem-ai
```

To enable optional HNSW candidate retrieval:

```bash
python -m pip install --upgrade "remem-ai[ann]"
```

Existing `remem_store.json` files load directly. Back up production data as you
would for any package upgrade, but do not rewrite the JSON schema.

## Search modes

`auto` is the new default. It selects HNSW when USearch is installed and exact
cosine otherwise. The resolution is inspectable:

```python
from remem import Client

client = Client(search_mode="auto")
print(client.resolved_search_mode)
print(client.search_fallback_reason)
```

Use `search_mode="exact_cosine"` for deterministic exhaustive discovery or
`search_mode="hnsw_cosine"` to require USearch. Forced HNSW mode raises a clear
installation error instead of silently falling back.

The old `similarity_backend="exact"|"hnsw"` argument remains temporarily
available and emits `DeprecationWarning`. Migrate new code to `search_mode`.

## ANN result semantics

HNSW discovers candidate IDs. Remem then fetches only those records and
recalculates exact cosine similarity before sorting and applying reuse
thresholds. Public similarity scores therefore keep the `-1.0` to `1.0` cosine
semantics from `1.0.0`, while approximate discovery does not promise exhaustive
nearest-neighbor recall.

## Incremental and persistent indexes

Client-mediated stores, updates, deletes, namespace moves, and clears keep ANN
state synchronized incrementally. Storage remains authoritative and mutation
failures roll back storage before deterministic recovery.

Persistence is opt-in:

```python
from remem import AnnConfig, Client

client = Client(
    search_mode="hnsw_cosine",
    ann_config=AnnConfig(persistence_path=".remem/records.usearch"),
)
```

Native indexes are derived caches. Startup validates versioned metadata,
configuration, dimensions, record mappings, storage identity, native size, and
checksums. Invalid artifacts rebuild automatically. Give one process ownership
of a configured path; multi-process index writers are not supported.

## Namespace and metadata behavior

HNSW indexes are partitioned by `ExecutionContext.namespace`. The default
policy searches only the matching namespace. `kb_version`, `prompt_version`,
and `model` constraints are applied before storage candidate lookup, with
conservative adaptive ANN expansion when compatible records are sparse.

Arbitrary `ExecutionContext.metadata` remains descriptive. It is not compared
or indexed because `ReusePolicy` does not define key, operator, normalization,
or missing-value semantics for custom dictionaries.

## Verification after upgrade

```python
import remem

assert remem.__version__ == "1.1.0"
```

Run an exact-mode query against an existing record store first. If enabling
ANN persistence, inspect `client.ann_index_stats` and
`client.ann_persistence_recovery_reason` after the initial build and the next
restart. A clean restart reports zero rebuilds and one load per non-empty
namespace partition.
