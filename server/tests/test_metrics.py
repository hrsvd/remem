import unittest
import numpy as np
from remem.similarity.metrics import cosine_similarity, dot_product, vector_norm


class TestMetrics(unittest.TestCase):

    def test_identical_vectors(self):
        """Identical vectors yield a similarity of 1.0."""
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_opposite_vectors(self):
        """Opposite vectors yield a similarity of -1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), -1.0)

    def test_orthogonal_vectors(self):
        """Perpendicular vectors yield a similarity of 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)

    def test_random_vectors(self):
        """Random vector bounds verification."""
        v1 = [0.54, 0.23, 0.91]
        v2 = [0.12, 0.77, 0.44]
        sim = cosine_similarity(v1, v2)
        self.assertTrue(-1.0 <= sim <= 1.0)

    def test_different_length_vectors(self):
        """Mismatched coordinate dimensions raise value errors."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0]
        with self.assertRaises(ValueError):
            cosine_similarity(v1, v2)

    def test_empty_vectors(self):
        """Empty lists raise value errors."""
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_zero_vector(self):
        """Zero divisions bypass safely returning 0.0."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)

    def test_vector_norm_calculation(self):
        """Vector magnitude calculations manually checked."""
        v = [3.0, 4.0]  # A 3-4-5 geometric right triangle yields 5.0 magnitude
        self.assertAlmostEqual(vector_norm(np.array(v)), 5.0)

    def test_dot_product_calculation(self):
        """Dot product scalar results manually checked."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        self.assertEqual(dot_product(v1, v2), 11.0)


if __name__ == "__main__":
    unittest.main()