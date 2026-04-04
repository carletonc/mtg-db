"""
Centralised configuration — loads from .env, validates, exposes typed values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Walk up until we find .env (works from any sub-directory)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required env var: {key}. "
            f"Copy .env.example → .env and fill it in."
        )
    return val


# --- Postgres / Neon ---
DATABASE_URL: str = _require("DATABASE_URL")

# --- MTGJSON ---
MTGJSON_BASE_URL: str = os.getenv("MTGJSON_BASE_URL", "https://mtgjson.com/api/v5")
ATOMIC_CARDS_GZ_URL: str = f"{MTGJSON_BASE_URL}/AtomicCards.json.gz"
ATOMIC_CARDS_SHA256_URL: str = f"{MTGJSON_BASE_URL}/AtomicCards.json.gz.sha256"

# --- OpenAI (optional for foundation pipeline, required for embeddings) ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
