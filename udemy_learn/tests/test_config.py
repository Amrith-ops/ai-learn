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

    def test_workflow_flags_are_booleans_and_default_to_false(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_MANUAL_WORKFLOW": "",
                "TOOL_DISCOVERY_MANUAL": "",
            },
        ):
            settings = load_config_env()

        self.assertIs(settings.enable_manual_workflow, False)
        self.assertIs(settings.tool_discovery_manual, False)

    def test_workflow_flags_parse_false_and_true_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_MANUAL_WORKFLOW": "false",
                "TOOL_DISCOVERY_MANUAL": "yes",
            },
        ):
            settings = load_config_env()

        self.assertIs(settings.enable_manual_workflow, False)
        self.assertIs(settings.tool_discovery_manual, True)


if __name__ == "__main__":
    unittest.main()
