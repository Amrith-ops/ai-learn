import unittest

from udemy_learn.rag.retrieval import (
    calculate_cosine_similarity,
    rank_documents_by_similarity,
)


class RetrievalTests(unittest.TestCase):
    def test_cosine_similarity_is_one_for_identical_vectors(self) -> None:
        self.assertAlmostEqual(calculate_cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_rank_documents_orders_most_similar_first(self) -> None:
        ranked = rank_documents_by_similarity(
            [1.0, 0.0],
            {
                "unrelated": [0.0, 1.0],
                "related": [1.0, 0.0],
            },
        )

        self.assertEqual([document_id for document_id, _ in ranked], ["related", "unrelated"])

    def test_cosine_similarity_rejects_different_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            calculate_cosine_similarity([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
