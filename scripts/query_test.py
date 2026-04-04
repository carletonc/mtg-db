#!/usr/bin/env python3
"""
Quick smoke test: query the cards table and optionally run a vector
similarity search to verify both pipelines worked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection, get_cursor


def show_stats():
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT count(*) FROM cards")
            card_count = cur.fetchone()[0]

            cur.execute(
                "SELECT count(DISTINCT scryfall_oracle_id) FROM cards"
            )
            unique_cards = cur.fetchone()[0]

            cur.execute(
                "SELECT count(*) FROM cards WHERE face_index > 0"
            )
            extra_faces = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM card_embeddings")
            emb_count = cur.fetchone()[0]

            cur.execute(
                "SELECT layout, count(*) FROM cards GROUP BY layout ORDER BY count(*) DESC"
            )
            layouts = cur.fetchall()

    print("━" * 50)
    print("  Database Stats")
    print("━" * 50)
    print(f"  Total card rows       : {card_count:,}")
    print(f"  Unique oracle IDs     : {unique_cards:,}")
    print(f"  Multi-face rows       : {extra_faces:,}")
    print(f"  Embeddings            : {emb_count:,}")
    print()
    print("  Layout distribution:")
    for layout, cnt in layouts:
        print(f"    {layout:25s} {cnt:>6,}")
    print("━" * 50)


def sample_search(query: str = "destroy target creature"):
    """Run a basic vector similarity search if embeddings exist."""
    try:
        from openai import OpenAI
        from utils.config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            print("\n(Skipping vector search — no OPENAI_API_KEY)")
            return

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(input=[query], model="text-embedding-3-small")
        qvec = resp.data[0].embedding

        with get_connection() as conn:
            with get_cursor(conn) as cur:
                cur.execute("SELECT count(*) FROM card_embeddings")
                if cur.fetchone()[0] == 0:
                    print("\n(Skipping vector search — no embeddings yet)")
                    return

                cur.execute(
                    """
                    SELECT ce.chunk_text,
                           1 - (ce.embedding <=> %s::vector) AS similarity
                    FROM card_embeddings ce
                    ORDER BY ce.embedding <=> %s::vector
                    LIMIT 5
                    """,
                    (qvec, qvec),
                )
                results = cur.fetchall()

        print(f"\n  Vector search: \"{query}\"")
        print("  " + "─" * 46)
        for chunk_text, sim in results:
            first_line = chunk_text.split("\n")[0]
            print(f"  {sim:.4f}  {first_line}")

    except ImportError:
        print("\n(Skipping vector search — openai package not installed)")


if __name__ == "__main__":
    show_stats()
    sample_search()
