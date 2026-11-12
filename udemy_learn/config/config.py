from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    enable_personality: bool = False
    enable_manual_workflow: bool = False
    tool_discovery_manual: bool = False


def _environment_flag(name: str) -> bool:
    """Parse a boolean environment setting using the project's accepted values."""
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def load_config_env() -> Settings:
    """Load settings only when the application starts, not when modules are imported."""
    load_dotenv()
    return Settings(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY"),
        model=os.getenv("MODEL", "gemini-3.5-flash-lite"),
        enable_personality=_environment_flag("ENABLE_PERSONALITY"),
        enable_manual_workflow=_environment_flag("ENABLE_MANUAL_WORKFLOW"),
        tool_discovery_manual=_environment_flag("TOOL_DISCOVERY_MANUAL"),
    )