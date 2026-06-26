# Remem

> **Remember expensive AI work. Reuse it intelligently.**

<p align="center">
  <strong>An AI Work Reuse Engine for Retrieval-Augmented Generation (RAG), AI Agents, and LLM Applications.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue.svg">
  <img alt="Status" src="https://img.shields.io/badge/Status-v0.1.0--alpha-orange">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
</p>

---

# Why Remem?

Modern AI applications repeatedly execute expensive operations for requests that are often semantically similar.

A typical AI workflow may involve:

* Generating embeddings
* Searching vector databases
* Retrieving knowledge chunks
* Reranking retrieved documents
* Constructing prompts
* Calling Large Language Models (LLMs)
* Executing tools or SQL queries
* Running multi-step agent workflows

Although user queries may be phrased differently, they frequently require nearly identical computation.

For example:

```text
"What is our company's vacation policy?"

"What are the PTO rules?"

"How many paid leaves do employees receive?"
```

Most AI systems execute the entire pipeline independently for each request, even when much of the previous work could be safely reused.

This leads to:

* Higher inference costs
* Increased latency
* Unnecessary vector searches
* Repeated reranking
* Duplicate LLM calls
* Repeated tool execution
* Wasted infrastructure resources

Remem exists to eliminate redundant AI work.

Instead of treating every request as completely new, Remem remembers previously executed work and determines whether any part of it can be safely reused.

---

# Vision

Remem is an **AI Work Reuse Engine**.

Rather than replacing your existing infrastructure (Redis, Postgres, Pinecone, Qdrant, Weaviate, Milvus, pgvector, or custom retrieval systems), Remem integrates alongside your application and continuously observes expensive AI operations.

When a new request arrives, Remem analyzes previously completed work and determines the highest level of computation that can be safely reused.

Depending on the request, this may include:

* Previously generated LLM responses
* Retrieved knowledge chunks
* Reranked search results
* Tool outputs
* SQL query results
* Agent execution artifacts
* Other reusable intermediate computations

If no reusable work exists, the application executes normally, and Remem learns from the new execution for future requests.

The long-term vision is to become a lightweight infrastructure component that AI engineers can integrate into existing RAG systems, AI agents, and LLM applications with minimal changes.

---

# The Problem

A typical Retrieval-Augmented Generation (RAG) pipeline looks like this:

```text
                 User Query
                      │
                      ▼
            Generate Embedding
                      │
                      ▼
          Search Vector Database
                      │
                      ▼
            Retrieve Knowledge
                      │
                      ▼
                Rerank Results
                      │
                      ▼
            Construct Prompt
                      │
                      ▼
                 Call the LLM
                      │
                      ▼
                  Final Answer
```

Even when multiple users ask nearly identical questions, this entire pipeline is often executed repeatedly.

As applications scale, this repeated computation becomes one of the largest contributors to latency and infrastructure cost.

---

# How Remem Works

Instead of assuming every request requires a full execution, Remem attempts to reuse previous work whenever it is safe to do so.

```text
                 Incoming Request
                        │
                        ▼
              Generate Embedding
                        │
                        ▼
               Semantic Similarity
                        │
                        ▼
            AI Work Reuse Decision
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
     ▼                  ▼                  ▼
Reuse Response   Reuse Retrieval     Execute Pipeline
     │                  │                  │
     │                  ▼                  ▼
     │           Skip Retrieval      Store New Execution
     │             & Reranking             │
     └──────────────────┴──────────────────┘
                        │
                        ▼
                 Return Result
```

Rather than acting as a traditional cache, Remem behaves like a decision engine that selects the highest level of reusable computation for each request.

---

# What Can Remem Reuse?

Depending on the request and available metadata, Remem may reuse:

* Entire LLM responses
* Retrieved knowledge chunks
* Reranked search results
* Prompt construction artifacts
* Tool execution results
* SQL query results
* Agent execution artifacts
* Future AI workflow outputs

Not every request can safely reuse every artifact.

For example:

* If the knowledge base has changed, Remem may skip response reuse but still reuse retrieval results.
* If the retrieval results have changed, Remem executes a fresh retrieval.
* If no previous work is reusable, the application executes normally.

Rather than guaranteeing that every request avoids an LLM call, Remem always chooses the **highest level of reusable work that preserves correctness**.

In some situations, this completely eliminates another LLM invocation.

In others, it may reuse only the retrieval stage while generating a fresh response.

This adaptive approach minimizes latency, reduces infrastructure cost, and maintains response quality without requiring changes to an application's existing retrieval pipeline.


# Design Philosophy

Remem follows a few simple principles.

## Correctness before optimization

Always build the correct solution before making it faster.

---

## Measure before optimizing

Every optimization should be supported by benchmarks.

---

## Keep public APIs simple

Internal architecture may evolve.

The public API should remain stable.

---

## Observe rather than replace

Remem should integrate with existing AI stacks instead of forcing engineers to adopt new databases or retrieval systems.

---

# Current Status

Current Version:

```
v0.1.0-alpha
```

Implemented:

- ✅ RetrievalEntry model
- ✅ Cosine similarity implementation
- ✅ Similarity engine
- ✅ In-memory storage
- ✅ Storage abstraction
- ✅ Unit tests
- ✅ Example application

Not yet implemented:

- Execution engine
- Work reuse API
- Persistence
- HTTP server
- SDKs
- Metrics
- Distributed mode
- Rust acceleration

---

# Current Architecture

```
Application
      │
      ▼
Similarity Engine
      │
      ▼
In-Memory Storage
      │
      ▼
Retrieval Entries
```

Current implementation focuses on building the core foundations before introducing more advanced features.

---

# Repository Structure

```
remem/

├── docs/
├── server/
│   ├── main.py
│   ├── tests/
│   └── remem/
│       ├── models/
│       ├── similarity/
│       └── storage/
├── examples/
├── benchmarks/
└── README.md
```

---

# Running the Demo

Clone the repository:

```bash
git clone https://github.com/<your-username>/remem.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python server/main.py
```

---

# Running Tests

```
pytest
```

---

# Development Roadmap

## v0.1.0-alpha

- RetrievalEntry
- Similarity Engine
- In-memory Storage
- Unit Tests

---

## v0.2.0

Public Remem API

```
Remem()

↓

find_similar()

↓

Storage

↓

Similarity
```

---

## v0.3.0

Execution Engine

Prevent duplicate retrieval work while requests are executing simultaneously.

---

## v0.4.0

Work Reuse API

```
get_or_compute()
```

The application supplies a callback.

Remem determines whether previous work can be reused.

---

## v0.5.0

Persistence Layer

Support durable storage instead of in-memory only.

---

## v0.6.0

Performance Optimization

- Faster similarity search
- Profiling
- Benchmarking
- Memory optimization

---

## v0.7.0

SDKs

- Python
- Java
- Rust

---

## v1.0.0

Production-ready AI Work Reuse Engine

---

# Long-Term Goals

Remem aims to support:

- Retrieval reuse
- Agent memory reuse
- Prompt context reuse
- SQL query reuse
- Tool execution reuse
- Knowledge versioning
- Distributed deployments
- High-performance storage engine
- Rust acceleration

---

# Why Not Just Use Redis?

Redis is an excellent key-value cache.

Remem solves a different problem.

Redis answers:

> "Have I seen this exact key before?"

Remem aims to answer:

> "Have I already performed similar expensive AI work before?"

Instead of exact key matching, Remem focuses on semantic similarity and work reuse.

---

# Learning Goals

Remem is also a personal engineering journey.

This project is being built from first principles to deeply understand:

- Distributed systems
- Storage engines
- Database internals
- Concurrency
- Memory management
- Networking
- Performance optimization
- AI infrastructure
- Open-source engineering

The goal is not simply to build another cache.

The goal is to build a useful infrastructure project while understanding every layer involved.

---

# Contributing

Contributions are welcome.

As the project is still in its early stages, architecture discussions and feedback are especially valuable.

Please read `CONTRIBUTING.md` before opening issues or pull requests.

---

# License

Licensed under the Apache License 2.0.

See the `LICENSE` file for details.

---

# Project Status

⚠️ **Early Alpha**

The project is under active development.

Breaking API changes are expected until the first stable release.

---

## Author

**Harshvardhan Singh**

Building Remem as an open-source AI infrastructure project to explore distributed systems, storage engines, and high-performance backend engineering.

If this project interests you, consider giving it a ⭐ and following its progress.