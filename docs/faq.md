# Frequently Asked Questions (FAQ)

## What problem does Remem solve?

Modern LLM applications repeatedly perform expensive operations such as retrieval, reranking, and generation—even when similar requests have already been processed.

Remem reduces latency and inference cost by identifying semantically similar requests and intelligently reusing previous executions where appropriate.

---

## How is Remem different from Redis?

Redis is a traditional key-value cache.

A cache hit only occurs when the exact key already exists.

Remem performs semantic lookup instead of exact-key lookup.

Instead of asking:

> "Have I seen this exact request before?"

Remem asks:

> "Have I already completed similar work that can be safely reused?"

This makes Remem particularly useful for LLM applications where semantically equivalent requests rarely produce identical cache keys.

---

## Is Remem a vector database?

No.

Remem may use vector search internally, but it is **not** intended to replace vector databases.

Its primary responsibility is deciding whether previous LLM executions can be safely reused.

Vector search is only one component of that decision process.

---

## Does Remem replace Retrieval-Augmented Generation (RAG)?

No.

Remem complements RAG systems.

It sits alongside existing LLM pipelines and determines whether expensive operations—such as retrieval or generation—can be reused instead of executed again.

---

## Does Remem replace Redis?

Not necessarily.

Redis and Remem solve different problems.

Many applications may benefit from using both:

* Redis for exact-key caching.
* Remem for semantic execution reuse.

---

## Which LLM providers are supported?

Remem is designed to be model-agnostic.

Any provider capable of generating embeddings and running LLM inference can be integrated.

Support for additional providers will continue to expand.

---

## Which storage backends are supported?

The storage layer is intentionally modular.

Current and planned storage options include:

* Local persistence
* SQLite
* PostgreSQL
* Redis
* Custom storage implementations

---

## Can I use Remem in production?

The project is actively evolving.

Production readiness depends on the specific release version.

Always refer to the latest GitHub Release notes before deploying Remem in production environments.

---

## How does Remem decide whether something can be reused?

Reuse decisions are based on multiple factors, including:

* Semantic similarity
* Metadata compatibility
* Execution policies
* User-defined constraints

A high similarity score alone does not guarantee reuse.

---

## Does Remem reuse the entire LLM response?

Not always.

Depending on compatibility, Remem may reuse:

* Retrieval results
* Intermediate pipeline outputs
* Final generated responses

The reuse strategy depends on project configuration and policy decisions.

---

## Is Remem open source?

Yes.

Remem is released under the Apache 2.0 License and welcomes community contributions.

Please see the CONTRIBUTING guide before opening pull requests.

---

## How can I contribute?

You can contribute by:

* Reporting bugs
* Suggesting new features
* Improving documentation
* Adding examples
* Implementing new storage backends
* Improving similarity algorithms
* Optimizing performance

Every contribution—large or small—is appreciated.

---

## Where is the project heading?

The long-term vision for Remem includes:

* Approximate Nearest Neighbor (ANN) search
* Distributed semantic caching
* Advanced policy engines
* Cloud-native deployments
* Multi-tenant support
* AI agent memory optimization
* Research-driven execution reuse

See the project roadmap for upcoming milestones.
