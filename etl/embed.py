"""
Embedding pipeline — generates vectors for any registered source and
upserts them into the unified `embeddings` table.

Sources are pluggable.  Each source is a callable that yields
(id, chunk_index, chunk_text) tuples.  The pipeline batches these, calls
the configured embedding backend (HuggingFace or OpenAI), and upserts.

Change detection
────────────────
The pipeline uses **md5(chunk_text)** to decide whether to re-embed a row.
This means ANY change to the final text triggers a re-embed:
  - Raw card data changed (oracle errata, new rulings)  → re-embed ✓
  - Preprocessing logic changed (symbol expansions)     → re-embed ✓
  - Embedding model changed (different model name)      → re-embed ✓

This replaces the old timestamp-based comparison.  The pipeline builds
the preprocessed text for every row on each run, hashes it, and only
embeds rows whose hash differs from the stored chunk_hash.

Built-in sources:
  - "card_name" — card names only (for name-based similarity search)
  - "card_text" — preprocessed oracle text (for rules text similarity)
  - "rules"    — MTG Comprehensive Rules (downloaded + parsed on the fly)

Usage:
    python -m etl.embed                      # embed all sources (incremental)
    python -m etl.embed --source card_name   # only card name embeddings
    python -m etl.embed --source card_text   # only card text embeddings
    python -m etl.embed --source rules       # only rules embeddings
    python -m etl.embed --full-refresh       # re-embed everything
"""

import argparse
import hashlib
import time

import psycopg2.extras

from db.connection import get_connection, get_cursor
from utils.config import EMBED_MODEL
from utils.embeddings import get_embeddings
from utils.mtg_text import (
    preprocess_loyalty,
    preprocess_mana_cost,
    preprocess_oracle_text,
    preprocess_power_toughness,
)


def _md5(text: str) -> str:
    """Stable md5 hex digest of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#  Source: card_name
# ═══════════════════════════════════════════════════════════════════════════════


def card_name_chunks(cur, *, full_refresh: bool = False):
    """
    Yield (id, chunk_index, chunk_text) for card names.

    The embedded text is just the card name.  This enables fast vector
    similarity search when a user asks for a card by name (fuzzy matching,
    misspellings, partial names).

    Card names are NOT preprocessed — the tokenizer handles fantasy words
    fine, and users search by the actual card name.

    ID format: "{oracle_id}:{face_index}" (same card face key used everywhere).
    """
    cur.execute(
        "SELECT scryfall_oracle_id, face_index, name FROM cards"
        " ORDER BY scryfall_oracle_id, face_index"
    )

    for row in cur:
        doc_id = f"{row['scryfall_oracle_id']}:{row['face_index']}"
        yield (doc_id, 0, row["name"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Source: card_text
# ═══════════════════════════════════════════════════════════════════════════════


def _build_card_text_chunk(row: dict) -> str:
    """
    Build a preprocessed text chunk for a card face.

    Raw columns (oracle_text, mana_cost, power, toughness, loyalty) are
    expanded from MTG notation to natural language before embedding.  This
    dramatically improves retrieval quality (0.18 → 0.58 avg cosine sim
    in benchmark tests).

    The raw data stays untouched in the cards table — preprocessing is
    embedding-time only.
    """
    parts = []

    #display_name = row["face_name"] or row["name"]
    #parts.append(display_name)
    #parts.append(row["type_line"])

    #if row["mana_cost"]:
    #    parts.append(f"Mana Cost: {preprocess_mana_cost(row['mana_cost'])}")

    # Power/toughness with variable expansion
    #if row["power"] is not None:
    #    pt = preprocess_power_toughness(row["power"], row["toughness"])
    #    parts.append(pt)

    # Loyalty (planeswalkers)
    #if row["loyalty"] is not None:
    #    parts.append(preprocess_loyalty(row["loyalty"]))

    # Defense (battles)
    #if row["defense"] is not None:
    #    parts.append(f"Defense: {row['defense']}")

    # Oracle text — the big one: full symbol expansion + self-ref replacement
    if row["oracle_text"]:
        clean = preprocess_oracle_text(
            row["oracle_text"],
            card_name=row["name"],
        )
        parts.append(clean)

    #if row["keywords"]:
    #    parts.append(f"Keywords: {', '.join(row['keywords'])}")

    # ── Future: taxonomy tagging ──────────────────────────────────
    # If we decide to tag cards with taxonomy keys at embedding time,
    # this is where it would go.  Example:
    #
    #   from utils.mtg_taxonomy import TAXONOMY  # (already stdlib-only)
    #   tags = classify_card(row)  # match oracle_hints patterns
    #   if tags:
    #       parts.append(f"Tags: {', '.join(tags)}")
    #
    # For now, the taxonomy is decoupled — used only at query time
    # in the retrieval repo, not baked into document embeddings.
    # ──────────────────────────────────────────────────────────────

    return "\n".join(parts)


def card_text_chunks(cur, *, full_refresh: bool = False):
    """
    Yield (id, chunk_index, chunk_text) for preprocessed card oracle text.
    """
    cur.execute(
        "SELECT * FROM cards ORDER BY scryfall_oracle_id, face_index"
    )

    for card in cur:
        doc_id = f"{card['scryfall_oracle_id']}:{card['face_index']}"
        text = _build_card_text_chunk(card)
        yield (doc_id, 0, text)


# ═══════════════════════════════════════════════════════════════════════════════
#  Source: rules
# ═══════════════════════════════════════════════════════════════════════════════


def rules_chunks(cur, *, full_refresh: bool = False):
    """
    Yield (id, chunk_index, chunk_text) for MTG Comprehensive Rules sections.

    Parsing strategy
    ────────────────
    The official rules .txt has this structure:

        1. Table of Contents
        2. Rules body (sections 1-9, numbered rules like 100.1, 702.17a)
        3. Glossary (term + definition blocks, separated by blank lines)
        4. Credits

    We split it into:
      - Numbered rules → one chunk per rule (e.g., "702.17" = Reach)
      - Glossary terms → one chunk per term

    This produces ~2,900 rule chunks + ~725 glossary chunks.

    Rules text contains minimal MTG notation ({T}, {E}, etc. appear in
    a few rules), so we preprocess it the same way as card text.
    """
    import re
    from urllib.request import Request, urlopen

    from utils.config import MTG_RULES_URL

    print("  ⬇  Downloading MTG Comprehensive Rules…")
    req = Request(MTG_RULES_URL, headers={"User-Agent": "mtg-rag-pipeline/1.0"})
    with urlopen(req, timeout=60) as resp:
        raw_text = resp.read().decode("utf-8-sig").replace("\r\n", "\n")

    # ── Locate document sections ──────────────────────────────────
    glossary_marker = "\nGlossary\n"
    credits_marker = "\nCredits\n"

    glossary_start = raw_text.rfind(glossary_marker)
    credits_start = raw_text.rfind(credits_marker)

    body_start_marker = "\n1. Game Concepts\n"
    toc_entries = [
        m.start()
        for m in re.finditer(re.escape(body_start_marker), raw_text)
    ]
    body_start = (
        toc_entries[1]
        if len(toc_entries) > 1
        else toc_entries[0]
        if toc_entries
        else 0
    )

    if glossary_start == -1 or glossary_start < body_start:
        rules_body = raw_text[body_start:]
        glossary_body = ""
    else:
        rules_body = raw_text[body_start:glossary_start]
        if credits_start > glossary_start:
            glossary_body = raw_text[
                glossary_start + len(glossary_marker) : credits_start
            ]
        else:
            glossary_body = raw_text[glossary_start + len(glossary_marker) :]

    # ── Parse numbered rules ──────────────────────────────────────
    rule_pattern = re.compile(r"^(\d{3}(?:\.\d+[a-z]?)?)\.?\s", re.MULTILINE)
    matches = list(rule_pattern.finditer(rules_body))

    rule_chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(rules_body)
        )
        section_text = rules_body[start:end].strip()
        if len(section_text) < 20:
            continue
        rule_num = match.group(1)
        # Preprocess rules text (some rules reference {T}, {E}, etc.)
        clean_text = preprocess_oracle_text(section_text)
        rule_chunks.append((f"rule:{rule_num}", clean_text))

    # ── Parse glossary terms ──────────────────────────────────────
    glossary_entries = []
    if glossary_body.strip():
        blocks = re.split(r"\n\n+", glossary_body.strip())
        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            lines = block.split("\n", 1)
            term = lines[0].strip()
            clean_block = preprocess_oracle_text(block)
            glossary_entries.append((f"glossary:{term}", clean_block))

    print(
        f"  ✓ Parsed {len(rule_chunks):,} rules + "
        f"{len(glossary_entries):,} glossary entries"
    )

    all_chunks = rule_chunks + glossary_entries
    for doc_id, text in all_chunks:
        yield (doc_id, 0, text)


# ═══════════════════════════════════════════════════════════════════════════════
#  Source registry
# ═══════════════════════════════════════════════════════════════════════════════

# ── Future: taxonomy reference chunks ─────────────────────────────
# The taxonomy (utils/mtg_taxonomy.py) can optionally be embedded as
# its own source.  Each node's description becomes a searchable chunk
# so the agent can discover "card advantage" → sub-concepts via
# vector search.  To enable:
#
#   def taxonomy_chunks(cur, *, full_refresh=False):
#       from utils.mtg_taxonomy import TAXONOMY, _walk
#       for node in _walk(TAXONOMY):
#           yield (f"taxonomy:{node['key']}", 0, node["description"])
#
#   SOURCES["taxonomy"] = taxonomy_chunks
#
# Then: python -m etl.embed --source taxonomy
# ──────────────────────────────────────────────────────────────────

SOURCES = {
    "card_name": card_name_chunks,
    "card_text": card_text_chunks,
    "rules": rules_chunks,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════════


def embed_source(
    source_name: str,
    *,
    full_refresh: bool = False,
    batch_size: int = 10,
):
    """
    Embed a single source and upsert into the embeddings table.

    Change detection uses md5(chunk_text + model):
      - If the preprocessed text changed → re-embed
      - If the embedding model changed → re-embed
      - If neither changed → skip (even if raw card data was re-synced)
    """
    source_fn = SOURCES[source_name]

    with get_connection() as conn:
        with get_cursor(conn, dict_cursor=True) as cur:
            print(f"\n  [{source_name}] Collecting and preprocessing chunks…")
            all_chunks = list(source_fn(cur, full_refresh=full_refresh))
            print(f"  [{source_name}] {len(all_chunks):,} total chunks")

            if not all_chunks:
                print(f"  [{source_name}] No data found.")
                return 0

            # ── Compute hashes for change detection ───────────────────
            # Hash includes the chunk text AND model name, so model
            # changes also trigger re-embedding.
            chunk_hashes = {}
            for doc_id, chunk_idx, text in all_chunks:
                chunk_hashes[(doc_id, chunk_idx)] = _md5(text + "|" + EMBED_MODEL)

            if not full_refresh:
                # Fetch existing hashes from DB
                cur.execute(
                    "SELECT id, chunk_index, chunk_hash "
                    "FROM embeddings WHERE source = %s",
                    (source_name,),
                )
                existing = {(r["id"], r["chunk_index"]): r["chunk_hash"] for r in cur}

                # Filter to only changed chunks
                chunks_to_embed = [
                    (doc_id, chunk_idx, text)
                    for doc_id, chunk_idx, text in all_chunks
                    if chunk_hashes[(doc_id, chunk_idx)] != existing.get((doc_id, chunk_idx), "")
                ]

                skipped = len(all_chunks) - len(chunks_to_embed)
                print(f"  [{source_name}] {skipped:,} unchanged, {len(chunks_to_embed):,} to embed")
            else:
                chunks_to_embed = all_chunks

            if not chunks_to_embed:
                print(f"  [{source_name}] All embeddings up to date.")
                return 0

            upsert_sql = """
                INSERT INTO embeddings
                    (id, source, chunk_index, chunk_text, chunk_hash,
                     embedding, model)
                VALUES
                    (%(id)s, %(source)s, %(chunk_index)s,
                     %(chunk_text)s, %(chunk_hash)s,
                     %(embedding)s, %(model)s)
                ON CONFLICT (id, source, chunk_index)
                DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    chunk_hash = EXCLUDED.chunk_hash,
                    embedding  = EXCLUDED.embedding,
                    model      = EXCLUDED.model,
                    updated_at = now()
            """

            total = 0
            for i in range(0, len(chunks_to_embed), batch_size):
                batch = chunks_to_embed[i : i + batch_size]
                texts = [c[2] for c in batch]
                vectors = get_embeddings(texts)

                rows = []
                for (doc_id, chunk_idx, text), vec in zip(batch, vectors):
                    rows.append(
                        {
                            "id": doc_id,
                            "source": source_name,
                            "chunk_index": chunk_idx,
                            "chunk_text": text,
                            "chunk_hash": chunk_hashes[(doc_id, chunk_idx)],
                            "embedding": vec,
                            "model": EMBED_MODEL,
                        }
                    )

                psycopg2.extras.execute_batch(
                    cur, upsert_sql, rows, page_size=batch_size
                )
                total += len(batch)
                print(
                    f"  [{source_name}] {total:,} / {len(chunks_to_embed):,} embedded…"
                )

    return total


def run(
    *,
    sources: list[str] | None = None,
    full_refresh: bool = False,
    batch_size: int = 200,
):
    """Run the embedding pipeline for one or more sources."""
    print("━" * 60)
    print("  MTG RAG — Embedding Pipeline")
    print("━" * 60)
    print(f"  Backend : {EMBED_MODEL}")
    print(f"  Detect  : md5(chunk_text + model)")

    targets = sources or list(SOURCES.keys())
    t0 = time.time()
    grand_total = 0

    for name in targets:
        if name not in SOURCES:
            print(f"  ⚠ Unknown source '{name}', skipping.")
            continue
        grand_total += embed_source(
            name, full_refresh=full_refresh, batch_size=batch_size
        )

    elapsed = time.time() - t0
    print("\n" + "━" * 60)
    print("  Embedding Pipeline Complete")
    print("━" * 60)
    print(f"  Sources  : {', '.join(targets)}")
    print(f"  Embedded : {grand_total:,} chunks")
    print(f"  Model    : {EMBED_MODEL}")
    print(f"  Time     : {elapsed:.1f}s")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="MTG RAG embedding pipeline")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help=(
            "Embed a specific source (card_name, card_text, rules). "
            "Can be repeated. Default: all."
        ),
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-embed all chunks, not just changed ones",
    )
    args = parser.parse_args()
    run(sources=args.sources, full_refresh=args.full_refresh)


if __name__ == "__main__":
    main()
