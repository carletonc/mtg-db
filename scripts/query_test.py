#!/usr/bin/env python3
"""
Test suite for the MTG RAG pipeline.

Validates:
  1. Foundation table stats (insert / upsert integrity)
  2. Exact card name lookup
  3. Card text phrase search (SQL ILIKE)
  4. Vector similarity — card names
  5. Vector similarity — card text
  6. Vector similarity — rules
  7. Embedding source coverage

Usage:
    python scripts/query_test.py              # run all tests
    python scripts/query_test.py --verbose    # show extended output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection, get_cursor

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _vector_search(cur, query_text: str, source: str, limit: int = 5):
    """Run a vector similarity search against a specific source."""
    from utils.embeddings import get_embeddings

    qvec = get_embeddings([query_text])[0]
    cur.execute(
        """
        SELECT e.id, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM embeddings e
        WHERE e.source = %s AND e.embedding IS NOT NULL
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (qvec, source, qvec, limit),
    )
    return cur.fetchall()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_foundation_stats(cur, verbose: bool):
    """1. Check cards table integrity."""
    print("\n  1. Foundation Table Stats")
    print("  " + "─" * 46)

    cur.execute("SELECT count(*) FROM cards")
    total = cur.fetchone()[0]

    cur.execute("SELECT count(DISTINCT scryfall_oracle_id) FROM cards")
    unique_oracles = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM cards WHERE face_index > 0")
    extra_faces = cur.fetchone()[0]

    cur.execute(
        "SELECT count(*) FROM cards WHERE content_hash IS NULL OR content_hash = ''"
    )
    missing_hash = cur.fetchone()[0]

    # PK uniqueness (should always be 0 by constraint, but verify)
    cur.execute("""
        SELECT scryfall_oracle_id, face_index, count(*)
        FROM cards
        GROUP BY scryfall_oracle_id, face_index
        HAVING count(*) > 1
    """)
    pk_dupes = cur.fetchall()

    print(f"     Total rows:           {total:>8,}")
    print(f"     Unique oracle IDs:    {unique_oracles:>8,}")
    print(f"     Multi-face rows:      {extra_faces:>8,}")
    print(f"     Missing content_hash: {missing_hash:>8,}")
    print(f"     PK duplicates:        {len(pk_dupes):>8}")

    ok = total > 30000 and missing_hash == 0 and len(pk_dupes) == 0
    print(f"  {PASS if ok else FAIL}  Foundation table {'OK' if ok else 'HAS ISSUES'}")
    return ok


def test_sync_log(cur, verbose: bool):
    """1b. Check sync_log has at least one successful run."""
    cur.execute(
        "SELECT count(*) FROM sync_log WHERE status = 'success'"
    )
    successes = cur.fetchone()[0]
    if successes > 0:
        cur.execute(
            "SELECT source_version, total_cards, inserted, updated, unchanged "
            "FROM sync_log WHERE status = 'success' ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        print(f"     Last sync: v{row[0]} — "
              f"{row[1]:,} total, {row[2]:,} ins, "
              f"{row[3]:,} upd, {row[4]:,} unch")
    ok = successes > 0
    print(f"  {PASS if ok else WARN}  Sync log: {successes} successful run(s)")
    return ok


def test_exact_name_lookup(cur, verbose: bool):
    """2. Retrieve a card by exact name."""
    print("\n  2. Exact Name Lookup")
    print("  " + "─" * 46)

    test_names = ["Lightning Bolt", "Counterspell", "Sol Ring", "Black Lotus"]
    found = 0
    for name in test_names:
        cur.execute("SELECT name, type_line, oracle_text FROM cards WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            found += 1
            if verbose:
                print(f"     {PASS} {row[0]:20s} | {row[1]}")
        else:
            print(f"     {FAIL} '{name}' not found")

    ok = found == len(test_names)
    print(f"  {PASS if ok else FAIL}  Found {found}/{len(test_names)} test cards by exact name")
    return ok


def test_text_phrase_search(cur, verbose: bool):
    """3. Search card text using SQL ILIKE (not vector)."""
    print("\n  3. Text Phrase Search (SQL)")
    print("  " + "─" * 46)

    phrases = {
        "destroy target creature": 20,
        "draw a card": 50,
        "can't be blocked": 10,
        "flying": 50,
        "lose all abilities": 3,
    }
    all_ok = True
    for phrase, min_expected in phrases.items():
        cur.execute(
            "SELECT count(*) FROM cards WHERE oracle_text ILIKE %s",
            (f"%{phrase}%",),
        )
        count = cur.fetchone()[0]
        ok = count >= min_expected
        if not ok:
            all_ok = False
        if verbose or not ok:
            print(f"     {'✓' if ok else '✗'} \"{phrase}\": {count:,} cards "
                  f"(expected ≥{min_expected})")

    print(f"  {PASS if all_ok else FAIL}  Text phrase search "
          f"{'all passed' if all_ok else 'some failed'}")
    return all_ok


def test_embedding_coverage(cur, verbose: bool):
    """4. Check that embeddings exist for each source."""
    print("\n  4. Embedding Coverage")
    print("  " + "─" * 46)

    cur.execute(
        "SELECT source, count(*) FROM embeddings GROUP BY source ORDER BY source"
    )
    sources = {row[0]: row[1] for row in cur.fetchall()}

    expected_sources = {
        "card_name": 30000,
        "card_text": 30000,
        "rules": 500,
    }

    all_ok = True
    for source, min_count in expected_sources.items():
        count = sources.get(source, 0)
        ok = count >= min_count
        if not ok:
            all_ok = False
        print(f"     {'✓' if ok else '✗'} {source:15s} {count:>7,} embeddings "
              f"(expected ≥{min_count:,})")

    print(f"  {PASS if all_ok else FAIL}  Embedding coverage "
          f"{'sufficient' if all_ok else 'incomplete'}")
    return all_ok


def test_vector_search_card_name(cur, verbose: bool):
    """5. Vector similarity search on card names."""
    print("\n  5. Vector Search — Card Names")
    print("  " + "─" * 46)

    cur.execute("SELECT count(*) FROM embeddings WHERE source = 'card_name'")
    if cur.fetchone()[0] == 0:
        print(f"  {WARN}  No card_name embeddings — skipping")
        return None

    queries = {
        "Lightning Bolt": "Lightning Bolt",      # exact should rank #1
        "Counterspell": "Counterspell",
        "Wrath of God": "Wrath of God",
    }

    all_ok = True
    for query, expected_top in queries.items():
        results = _vector_search(cur, query, "card_name", limit=5)
        top_name = results[0][1] if results else "(none)"
        ok = top_name == expected_top
        if not ok:
            all_ok = False
        if verbose:
            print(f"     Query: \"{query}\"")
            for _, name, sim in results:
                marker = " ←" if name == expected_top else ""
                print(f"       {sim:.4f}  {name}{marker}")
        else:
            print(f"     {'✓' if ok else '✗'} \"{query}\" → top: \"{top_name}\" "
                  f"({'match' if ok else 'MISMATCH'})")

    print(f"  {PASS if all_ok else FAIL}  Name vector search "
          f"{'all top-1 correct' if all_ok else 'some mismatches'}")
    return all_ok


def test_vector_search_card_text(cur, verbose: bool):
    """6. Vector similarity search on card text."""
    print("\n  6. Vector Search — Card Text")
    print("  " + "─" * 46)

    cur.execute("SELECT count(*) FROM embeddings WHERE source = 'card_text'")
    if cur.fetchone()[0] == 0:
        print(f"  {WARN}  No card_text embeddings — skipping")
        return None

    queries = [
        "destroy all creatures",
        "counter target spell",
        "can't be blocked",
        "deals 3 damage to any target",
    ]

    for query in queries:
        results = _vector_search(cur, query, "card_text", limit=5)
        print(f"     Query: \"{query}\"")
        for _, chunk, sim in results:
            # Show first line (card name) from the chunk
            first_line = chunk.split("\n")[0][:50]
            print(f"       {sim:.4f}  {first_line}")

    print(f"  {PASS}  Card text vector search returned results")
    return True


def test_vector_search_rules(cur, verbose: bool):
    """7. Vector similarity search on rules text."""
    print("\n  7. Vector Search — Rules")
    print("  " + "─" * 46)

    cur.execute("SELECT count(*) FROM embeddings WHERE source = 'rules'")
    if cur.fetchone()[0] == 0:
        print(f"  {WARN}  No rules embeddings — skipping")
        return None

    queries = [
        "what can block a creature with flying",
        "what happens when a creature is destroyed",
        "how does trample damage work",
        "what does hexproof protect against",
    ]

    for query in queries:
        results = _vector_search(cur, query, "rules", limit=3)
        print(f"     Query: \"{query}\"")
        for _, chunk, sim in results:
            first_line = chunk.split("\n")[0][:60]
            print(f"       {sim:.4f}  {first_line}")

    print(f"  {PASS}  Rules vector search returned results")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run(verbose: bool = False):
    print("━" * 50)
    print("  MTG RAG — Test Suite")
    print("━" * 50)

    results = {}

    with get_connection() as conn:
        with get_cursor(conn, dict_cursor=True) as cur:
            results["foundation"] = test_foundation_stats(cur, verbose)
            results["sync_log"] = test_sync_log(cur, verbose)
            results["exact_name"] = test_exact_name_lookup(cur, verbose)
            results["text_phrase"] = test_text_phrase_search(cur, verbose)
            results["embedding_coverage"] = test_embedding_coverage(cur, verbose)
            results["vec_card_name"] = test_vector_search_card_name(cur, verbose)
            results["vec_card_text"] = test_vector_search_card_text(cur, verbose)
            results["vec_rules"] = test_vector_search_rules(cur, verbose)

    print("\n" + "━" * 50)
    print("  Summary")
    print("━" * 50)
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)

    for name, result in results.items():
        status = PASS if result is True else (WARN if result is None else FAIL)
        label = "PASS" if result is True else ("SKIP" if result is None else "FAIL")
        print(f"  {status} {name:25s} {label}")

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped / {total} total")
    print("━" * 50)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    success = run(verbose=args.verbose)
    sys.exit(0 if success else 1)
