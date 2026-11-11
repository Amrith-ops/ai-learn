import json
import math
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen

from udemy_learn.rag.embeddings import get_embedding_model_response


class EmbeddingModelTests(unittest.TestCase):
    def test_nomic_embed_text_returns_valid_embeddings(self) -> None:
        payload = {
            "model": "nomic-embed-text",
            "input": [
                "Python is a programming language.",
                "The weather forecast predicts rain.",
                "Moon"
            ],
        }
        request = Request(
            "http://localhost:11434/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except URLError as error:
            self.fail(
                "Could not reach Ollama at http://localhost:11434. "
                "Start Ollama and make sure nomic-embed-text is installed "
                f"(ollama pull nomic-embed-text). Details: {error}"
            )

        embeddings = result.get("embeddings")
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(payload["input"]), len(embeddings))
        # print(f"\nEmbedding response type: {type(embeddings).__name__}")
        print(f"Number of input texts: {len(embeddings)}")

        for index, embedding in enumerate(embeddings):
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)
            self.assertTrue(all(math.isfinite(value) for value in embedding))
            print(f"Vector {index + 1} length: {len(embedding)}")
            self.assertAlmostEqual(
                math.sqrt(sum(value * value for value in embedding)),
                1.0,
                places=4,
            )

        #print("Full embeddings (one vector per input text):")
        #print(json.dumps(embeddings, indent=2))
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))
        self.assertTrue(all(len(embedding) == len(embeddings[0]) for embedding in embeddings))
        self.assertNotEqual(embeddings[0], embeddings[1])

        single_embedding = get_embedding_model_response("Testing the project embedding helper.")
        self.assertEqual(len(single_embedding), len(embeddings[0]))


if __name__ == "__main__":
    unittest.main()
