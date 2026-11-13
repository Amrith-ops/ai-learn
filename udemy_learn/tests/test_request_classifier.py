import unittest
from unittest.mock import patch

from udemy_learn.request_classifier import classify_request


class RequestClassifierTests(unittest.TestCase):
    def test_returns_single_step_classification(self) -> None:
        with patch(
            "udemy_learn.request_classifier.generate_assistant_reply",
            return_value="single-step\n",
        ) as generate_reply:
            result = classify_request("What time is it?", object(), "test-model")

        self.assertEqual(result, "single-step")
        self.assertEqual(
            generate_reply.call_args.kwargs["messages"],
            [{"role": "user", "content": "What time is it?"}],
        )

    def test_returns_multi_step_classification_case_insensitively(self) -> None:
        with patch(
            "udemy_learn.request_classifier.generate_assistant_reply",
            return_value="Multi-Step",
        ):
            result = classify_request(
                "Get the time, then roll a die",
                object(),
                "test-model",
            )

        self.assertEqual(result, "multi-step")

    def test_rejects_unexpected_classifier_response(self) -> None:
        with patch(
            "udemy_learn.request_classifier.generate_assistant_reply",
            return_value="This needs several tools.",
        ):
            with self.assertRaisesRegex(ValueError, "must return exactly"):
                classify_request("Do a task", object(), "test-model")

    def test_rejects_empty_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_input"):
            classify_request(" ", object(), "test-model")


if __name__ == "__main__":
    unittest.main()
