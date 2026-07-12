from collections.abc import Sequence

import numpy as np


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the scalar dot product of two vectors."""
    return float(np.dot(v1, v2))


def vector_norm(vector: np.ndarray) -> float:
    """Returns the Euclidean (L2) norm of a vector."""
    return float(np.linalg.norm(vector))


def cosine_similarity(
    vector1: Sequence[float],
    vector2: Sequence[float],
) -> float:
    """Returns cosine similarity in the range [-1, 1]."""

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensions.")

    v1 = np.asarray(vector1, dtype=np.float64)
    v2 = np.asarray(vector2, dtype=np.float64)

    if v1.size == 0:
        raise ValueError("Vectors cannot be empty.")

    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product(v1, v2) / (norm1 * norm2)
