# Roadmap

This roadmap outlines the long-term vision for Remem. The project is developed incrementally, with each release building toward a production-grade AI infrastructure library.

## Release Overview

| Version | Focus | Status |
|---|---|---|
| v1.0.0-beta | Production-ready AI work reuse engine | **Current release** |
| v1.0.0 | Stable release with production hardening | Planned |
| v1.1.0 | Approximate Nearest Neighbor (ANN) search | Planned |
| v1.2.0 | Distributed semantic cache | Planned |
| v1.3.0 | Additional storage backends (Redis, Postgres, S3) | Planned |
| v1.4.0 | Advanced policy engine with ML-based thresholds | Planned |
| v2.0.0 | Multi-tenant support and cloud-native deployment | Planned |

## Version 1.x

**Completed**

- Semantic similarity engine
- Reuse engine (full / partial / miss decisions)
- Metadata policy engine
- Durable, atomic persistent storage
- Built-in observability and metrics
- Python SDK
- PyPI distribution

**Planned**

*Performance*
- Approximate Nearest Neighbor (ANN) search
- HNSW indexing
- Faster retrieval and memory optimizations

*Storage*
- SQLite backend
- PostgreSQL backend
- Redis backend

*Developer experience*
- Improved configuration ergonomics
- CLI support
- Richer examples and documentation

## Version 2.x — Production Deployments

- Distributed cache
- Remote storage
- Multi-process synchronization
- Async API
- Streaming support
- Horizontal scalability

## Version 3.x — AI Infrastructure Platform

- Cloud deployment
- Enterprise features
- Dashboard and observability UI
- Formal benchmark suite
- Advanced policy engine
- Plugin ecosystem

## Research Directions

Longer-term research interests that may shape future versions:

- Intelligent execution reuse beyond request/response caching
- Adaptive semantic caching (self-tuning thresholds)
- Agent memory optimization
- Cost-aware LLM routing
- Retrieval optimization strategies

---

Roadmaps evolve over time. Priorities may change based on community feedback and research findings. Have a feature request? Open an issue — see [CONTRIBUTING.md](../CONTRIBUTING.md).