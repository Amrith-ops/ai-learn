import unittest
from unittest.mock import patch

from udemy_learn.rag.retrieval import rank_documents_by_similarity, run_retrieval_workflow
from udemy_learn.rag.retrieval_tools import calculate_cosine_similarity


class RetrievalTests(unittest.TestCase):
    def test_cosine_similarity_is_one_for_identical_vectors(self) -> None:
        self.assertAlmostEqual(calculate_cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_rank_documents_orders_most_similar_first(self) -> None:
        ranked = rank_documents_by_similarity(
            [1.0, 0.0],
            {
                "unrelated": {"embedding": [0.0, 1.0], "content": "Unrelated"},
                "related": {"embedding": [1.0, 0.0], "content": "Relevant"},
            },
        )

        self.assertEqual([document_id for document_id, _ in ranked], ["related", "unrelated"])

    def test_cosine_similarity_rejects_different_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            calculate_cosine_similarity([1.0], [1.0, 2.0])

    @patch("udemy_learn.rag.retrieval.rank_documents_by_similarity")
    @patch("udemy_learn.rag.retrieval.load_documents_and_create_embeddings")
    @patch("udemy_learn.rag.retrieval.user_query_embedding_create")
    def test_workflow_returns_best_document_content(
        self,
        create_query_embedding,
        load_documents,
        rank_documents,
    ) -> None:
        documents = {
            "python.txt": {
                "embedding": [1.0, 0.0],
                "content": "Python supports automation.",
            },
            "weather.txt": {
                "embedding": [0.0, 1.0],
                "content": "Weather uses atmospheric data.",
            },
        }
        create_query_embedding.return_value = [1.0, 0.0]
        load_documents.return_value = documents
        rank_documents.return_value = [("python.txt", 0.82), ("weather.txt", 0.14)]

        result = run_retrieval_workflow("data", "What supports automation?")

        create_query_embedding.assert_called_once_with("What supports automation?")
        load_documents.assert_called_once_with("data")
        rank_documents.assert_called_once_with([1.0, 0.0], documents)
        self.assertEqual(
            result,
            ("python.txt", "Python supports automation.", 0.82),
        )

    @patch("udemy_learn.rag.retrieval.load_documents_and_create_embeddings")
    @patch("udemy_learn.rag.retrieval.user_query_embedding_create")
    def test_workflow_returns_none_when_no_documents_exist(
        self,
        create_query_embedding,
        load_documents,
    ) -> None:
        load_documents.return_value = {}

        result = run_retrieval_workflow("empty-data", "question")

        self.assertIsNone(result)
        create_query_embedding.assert_called_once_with("question")


if __name__ == "__main__":
    unittest.main()
