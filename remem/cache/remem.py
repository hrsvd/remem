from remem.similarity.engine import SimilarityEngine


class Remem:
    """Public entry point into the Remem engine."""

    def __init__(self):
        self._similarity_engine = SimilarityEngine()

    @property
    def similarity(self):
        return self._similarity_engine