import os
import tempfile
import unittest
from pathlib import Path

from udemy_learn.tools.files import build_retriever_rag


class BuildRetrieverRagTests(unittest.TestCase):
    def test_finds_matching_content_outside_project_directory(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                result = build_retriever_rag("ATMOSPHERIC")
            finally:
                os.chdir(original_directory)

        self.assertIn("Found in weather.txt:", result)
        self.assertIn("Weather forecasting predicts future atmospheric conditions.", result)

    def test_returns_message_when_no_content_matches(self) -> None:
        result = build_retriever_rag("term-that-is-not-in-the-data")

        self.assertEqual("No relevant content found.", result)


if __name__ == "__main__":
    unittest.main()
