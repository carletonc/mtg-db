"""
Download AtomicCards.json.gz from MTGJson with SHA-256 verification.
"""

import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from utils.config import ATOMIC_CARDS_GZ_URL, ATOMIC_CARDS_SHA256_URL

_HEADERS = {"User-Agent": "mtg-rag-pipeline/1.0"}


def _fetch_bytes(url: str) -> bytes:
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_remote_sha256() -> str:
    """Fetch the expected SHA-256 from MTGJson."""
    raw = _fetch_bytes(ATOMIC_CARDS_SHA256_URL).decode().strip()
    # Format is typically just the hex digest, or "hash  filename"
    return raw.split()[0]


def download_and_parse() -> tuple[dict, dict]:
    """
    Download AtomicCards.json.gz, verify integrity, decompress, parse.

    Returns
    -------
    (meta, data) where
        meta  = {"date": "...", "version": "..."}
        data  = {"CardName": [face_dict, ...], ...}
    """
    print(f"⬇  Downloading {ATOMIC_CARDS_GZ_URL} ...")
    gz_bytes = _fetch_bytes(ATOMIC_CARDS_GZ_URL)

    # SHA-256 of the .gz file
    actual_sha = hashlib.sha256(gz_bytes).hexdigest()
    expected_sha = fetch_remote_sha256()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA-256 mismatch!\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual_sha}"
        )
    print(f"✓  SHA-256 verified: {actual_sha[:16]}…")

    # Decompress and parse
    json_bytes = gzip.decompress(gz_bytes)
    payload = json.loads(json_bytes)
    meta = payload["meta"]
    data = payload["data"]
    print(f"✓  Parsed {len(data):,} card names (version {meta['version']})")
    return meta, data
