"""
Pluggable embedding backend.

Supports:
  - "huggingface" (default) — runs locally via sentence-transformers, free
  - "openai"                — calls the OpenAI API, requires OPENAI_API_KEY

The backend is selected by the EMBEDDING_BACKEND env var.  Both backends
expose the same interface: get_embeddings(texts) → list[list[float]] and
get_dimension() → int.
"""

from utils.config import (
    EMBEDDING_BACKEND,
    HF_EMBED_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
)

# --------------------------------------------------------------------------- #
#  Lazy singletons — heavy imports happen only when first called               #
# --------------------------------------------------------------------------- #
_hf_model = None
_openai_client = None


def _get_hf_model():
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer

        _hf_model = SentenceTransformer(HF_EMBED_MODEL)
        print(f"  ✓ Loaded HuggingFace model: {HF_EMBED_MODEL}")
    return _hf_model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai."
            )
        from openai import OpenAI

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured backend."""
    if EMBEDDING_BACKEND == "openai":
        client = _get_openai_client()
        resp = client.embeddings.create(input=texts, model=OPENAI_EMBED_MODEL)
        return [item.embedding for item in resp.data]

    # Default: huggingface
    model = _get_hf_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def get_dimension() -> int:
    """Return the embedding vector dimension for the active model."""
    if EMBEDDING_BACKEND == "openai":
        # Known dimensions for common OpenAI models
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(OPENAI_EMBED_MODEL, 1536)

    model = _get_hf_model()
    return model.get_sentence_embedding_dimension()
