"""
End-to-end pipeline: ETL + embedding for all sources.

Usage:
    python -m etl.pipeline                # full run: ETL → embed all sources
    python -m etl.pipeline --etl-only     # foundation ETL only (no embeddings)
    python -m etl.pipeline --embed-only   # embeddings only (skip ETL)
    python -m etl.pipeline --init         # create tables only
    python -m etl.pipeline --full-refresh # re-embed everything
"""

import argparse
import sys
import time

from db.schema import create_all_tables
from etl.download import download_and_parse, fetch_remote_sha256
from etl.normalize import normalize_cards
from etl.upsert import upsert_cards


def run_etl() -> dict:
    """Execute the foundation ETL: download → normalize → upsert."""
    # Step 1: Download
    print("\n[ETL 1/3] Downloading AtomicCards from MTGJson…")
    t0 = time.time()
    meta, data = download_and_parse()
    source_sha = fetch_remote_sha256()
    print(f"          ({time.time() - t0:.1f}s)")

    # Step 2: Normalize
    print("\n[ETL 2/3] Normalising to per-face rows…")
    t0 = time.time()
    rows = normalize_cards(data, source_version=meta["version"])
    print(f"          {len(rows):,} rows from {len(data):,} card names ({time.time() - t0:.1f}s)")

    # Step 3: Upsert
    print("\n[ETL 3/3] Upserting into Postgres…")
    t0 = time.time()
    stats = upsert_cards(rows, source_version=meta["version"], source_sha256=source_sha)
    elapsed = time.time() - t0

    print(f"\n  Source version : {meta['version']}")
    print(f"  Total rows     : {stats['total_cards']:,}")
    print(f"  Inserted       : {stats['inserted']:,}")
    print(f"  Updated        : {stats['updated']:,}")
    print(f"  Unchanged      : {stats['unchanged']:,}")
    print(f"  Deleted        : {stats['deleted']:,}")
    print(f"  Upsert time    : {elapsed:.1f}s")

    return stats


def run_embed(*, full_refresh: bool = False):
    """Execute the embedding pipeline for all sources."""
    from etl.embed import run as embed_run

    embed_run(full_refresh=full_refresh)


def run(
    *,
    init_only: bool = False,
    etl_only: bool = False,
    embed_only: bool = False,
    full_refresh: bool = False,
):
    """Execute the full pipeline."""
    print("━" * 60)
    print("  MTG RAG — Pipeline")
    print("━" * 60)

    # Step 0: Ensure tables exist
    print("\n[0] Ensuring database schema…")
    create_all_tables()

    if init_only:
        print("\n✓ Schema initialised. Exiting (--init flag).")
        return

    if not embed_only:
        run_etl()

    if not etl_only:
        run_embed(full_refresh=full_refresh)

    print("\n" + "━" * 60)
    print("  Pipeline Complete")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="MTG RAG pipeline")
    parser.add_argument(
        "--init", action="store_true",
        help="Create tables only, do not download/load data",
    )
    parser.add_argument(
        "--etl-only", action="store_true",
        help="Run foundation ETL only (skip embeddings)",
    )
    parser.add_argument(
        "--embed-only", action="store_true",
        help="Run embeddings only (skip ETL download/normalize/upsert)",
    )
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Re-embed all chunks, not just changed ones",
    )
    args = parser.parse_args()
    run(
        init_only=args.init,
        etl_only=args.etl_only,
        embed_only=args.embed_only,
        full_refresh=args.full_refresh,
    )


if __name__ == "__main__":
    main()
