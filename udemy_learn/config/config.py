from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    enable_personality: bool = False,
    enable_manual_workflow: bool = False,
    tool_discovery_manual: bool = False


def load_config_env() -> Settings:
    """Load settings only when the application starts, not when modules are imported."""
    load_dotenv()
    return Settings(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY"),
        model=os.getenv("MODEL", "gemini-3.5-flash-lite"),
        enable_personality=os.getenv("ENABLE_PERSONALITY", "").strip().casefold()
        in {"1", "true", "yes", "on"},
        enable_manual_workflow=os.getenv("ENABLE_MANUAL_WORKFLOW", "").strip().casefold()
    )