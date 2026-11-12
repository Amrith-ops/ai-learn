import os
import unittest
from unittest.mock import patch

from udemy_learn.config.config import load_config_env


class ConfigTests(unittest.TestCase):
    def test_enable_personality_parses_false_as_false(self) -> None:
        with patch.dict(os.environ, {"ENABLE_PERSONALITY": "false"}):
            self.assertFalse(load_config_env().enable_personality)

    def test_enable_personality_accepts_true_values(self) -> None:
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"ENABLE_PERSONALITY": value},
            ):
                self.assertTrue(load_config_env().enable_personality)


if __name__ == "__main__":
    unittest.main()
