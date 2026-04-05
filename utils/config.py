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

# --- MTG Comprehensive Rules ---
MTG_RULES_URL: str = os.getenv(
    "MTG_RULES_URL",
    "https://media.wizards.com/2026/downloads/MagicCompRules%2020260227.txt",
)

# --- Embedding backend ---
# Options: "openai", "huggingface"
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "huggingface")

# OpenAI settings (used when EMBEDDING_BACKEND=openai)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# HuggingFace settings (used when EMBEDDING_BACKEND=huggingface)
HF_EMBED_MODEL: str = os.getenv(
    "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Resolved at runtime — the active model name for DB tracking
EMBED_MODEL: str = (
    OPENAI_EMBED_MODEL if EMBEDDING_BACKEND == "openai" else HF_EMBED_MODEL
)
