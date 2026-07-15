from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from benchmarks.io import read_json, write_json


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Dependency-free deterministic smoke-test encoder, not a quality model."""

    name = "deterministic-token-hash-v1"

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm:
            vector /= norm
        return vector.tolist()


class SentenceTransformerProvider:
    def __init__(self, model: str, revision: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install benchmark dependencies with: pip install -e '.[benchmark]'"
            ) from exc
        self.name = f"{model}@{revision}"
        self._model = SentenceTransformer(model, revision=revision, device="cpu")
        self.batch_size = batch_size
        self.dimension = int(self._model.get_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        values = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= self.batch_size * 2,
        )
        return np.asarray(values, dtype=np.float32).tolist()


class CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache_path: str | Path) -> None:
        self.provider = provider
        self.name = provider.name
        self.dimension = provider.dimension
        self.path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        if self.path.exists():
            raw = read_json(self.path)
            if raw.get("provider") == self.name:
                self._cache = raw.get("vectors", {})

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        missing_texts = [
            text for text, key in zip(texts, keys) if key not in self._cache
        ]
        if missing_texts:
            vectors = self.provider.encode(missing_texts)
            missing_keys = [key for key in keys if key not in self._cache]
            self._cache.update(zip(missing_keys, vectors))
            write_json(
                self.path,
                {
                    "provider": self.name,
                    "dimension": self.dimension,
                    "vectors": self._cache,
                },
            )
        return [self._cache[key] for key in keys]


def provider_from_config(
    config: dict[str, object], cache_dir: str | Path
) -> EmbeddingProvider:
    kind = str(config.get("provider", "hash"))
    if kind == "hash":
        provider: EmbeddingProvider = HashEmbeddingProvider(
            int(str(config.get("dimension", 128)))
        )
    elif kind == "sentence_transformers":
        provider = SentenceTransformerProvider(
            model=str(config["model"]),
            revision=str(config["revision"]),
            batch_size=int(str(config.get("batch_size", 64))),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {kind}")
    safe_name = hashlib.sha256(provider.name.encode("utf-8")).hexdigest()[:16]
    return CachedEmbeddingProvider(provider, Path(cache_dir) / f"{safe_name}.json")
