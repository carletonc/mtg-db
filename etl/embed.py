"""
Downstream pipeline: generate embeddings for card faces and upsert into
the card_embeddings table for pgvector similarity search.

This reads from the `cards` foundation table and writes to `card_embeddings`.
It only processes cards whose content_hash has changed since the last
embedding run (tracked via updated_at comparison).

Usage:
    python -m etl.embed                  # embed all un-embedded / stale cards
    python -m etl.embed --full-refresh   # re-embed everything
"""

import argparse
import json
import time
from typing import Any

import psycopg2.extras

from db.connection import get_connection, get_cursor
from utils.config import OPENAI_API_KEY


def _build_chunk_text(row: dict) -> str:
    """
    Build a single text chunk for embedding from a card face row.
    Format designed to be information-dense for semantic search.
    """
    parts = []

    # Card identity
    display_name = row["face_name"] or row["name"]
    parts.append(f"Name: {display_name}")

    if row["type_line"]:
        parts.append(f"Type: {row['type_line']}")

    if row["mana_cost"]:
        parts.append(f"Mana Cost: {row['mana_cost']}")

    if row["colors"]:
        parts.append(f"Colors: {', '.join(row['colors'])}")

    # Stats
    stat_parts = []
    if row["power"] is not None:
        stat_parts.append(f"Power: {row['power']}")
    if row["toughness"] is not None:
        stat_parts.append(f"Toughness: {row['toughness']}")
    if row["loyalty"] is not None:
        stat_parts.append(f"Loyalty: {row['loyalty']}")
    if row["defense"] is not None:
        stat_parts.append(f"Defense: {row['defense']}")
    if stat_parts:
        parts.append(" | ".join(stat_parts))

    # Oracle text (the most important part for semantic search)
    if row["oracle_text"]:
        parts.append(f"Rules: {row['oracle_text']}")

    if row["keywords"]:
        parts.append(f"Keywords: {', '.join(row['keywords'])}")

    # Layout context for multi-faced
    if row["layout"] not in ("normal",):
        parts.append(f"Layout: {row['layout']}")
        if row["side"]:
            parts.append(f"Side: {row['side']}")

    return "\n".join(parts)


def _get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Call OpenAI embeddings API. Batches handled by the API (max 2048 inputs)."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is required for the embedding pipeline.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def run(*, full_refresh: bool = False, batch_size: int = 200):
    """Generate and upsert embeddings for cards that need them."""
    print("━" * 60)
    print("  MTG RAG — Embedding Pipeline")
    print("━" * 60)

    model = "text-embedding-3-small"

    with get_connection() as conn:
        with get_cursor(conn, dict_cursor=True) as cur:
            # Fetch cards that need embedding
            if full_refresh:
                print("\n[1/3] Full refresh — fetching all cards…")
                cur.execute("SELECT * FROM cards ORDER BY scryfall_oracle_id, face_index")
            else:
                print("\n[1/3] Fetching cards updated since last embedding…")
                cur.execute("""
                    SELECT c.*
                    FROM cards c
                    LEFT JOIN card_embeddings ce
                        ON c.scryfall_oracle_id = ce.scryfall_oracle_id
                       AND c.face_index = ce.face_index
                       AND ce.chunk_index = 0
                    WHERE ce.scryfall_oracle_id IS NULL
                       OR c.updated_at > ce.updated_at
                    ORDER BY c.scryfall_oracle_id, c.face_index
                """)

            cards = cur.fetchall()
            print(f"       {len(cards):,} cards to embed")

            if not cards:
                print("\n✓ All embeddings up to date.")
                return

            # Build chunks
            print("\n[2/3] Building text chunks…")
            chunks = []
            for card in cards:
                text = _build_chunk_text(card)
                chunks.append({
                    "scryfall_oracle_id": card["scryfall_oracle_id"],
                    "face_index": card["face_index"],
                    "chunk_index": 0,
                    "chunk_text": text,
                })

            # Embed in batches
            print(f"\n[3/3] Generating embeddings ({len(chunks):,} chunks, batch={batch_size})…")
            t0 = time.time()
            upsert_sql = """
                INSERT INTO card_embeddings
                    (scryfall_oracle_id, face_index, chunk_index, chunk_text, embedding, model)
                VALUES
                    (%(scryfall_oracle_id)s, %(face_index)s, %(chunk_index)s,
                     %(chunk_text)s, %(embedding)s, %(model)s)
                ON CONFLICT (scryfall_oracle_id, face_index, chunk_index)
                DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding  = EXCLUDED.embedding,
                    model      = EXCLUDED.model,
                    updated_at = now()
            """

            total_embedded = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c["chunk_text"] for c in batch]
                embeddings = _get_embeddings(texts, model=model)

                rows = []
                for chunk, emb in zip(batch, embeddings):
                    chunk["embedding"] = emb
                    chunk["model"] = model
                    rows.append(chunk)

                psycopg2.extras.execute_batch(cur, upsert_sql, rows, page_size=batch_size)
                total_embedded += len(batch)
                print(f"       {total_embedded:,} / {len(chunks):,} embedded…")

            elapsed = time.time() - t0

    print("\n" + "━" * 60)
    print("  Embedding Pipeline Complete")
    print("━" * 60)
    print(f"  Cards embedded : {total_embedded:,}")
    print(f"  Model          : {model}")
    print(f"  Time           : {elapsed:.1f}s")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="MTG RAG embedding pipeline")
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Re-embed all cards, not just changed ones",
    )
    args = parser.parse_args()
    run(full_refresh=args.full_refresh)


if __name__ == "__main__":
    main()
