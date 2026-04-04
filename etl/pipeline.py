"""
End-to-end ETL orchestrator for the MTG cards foundation table.

Usage:
    python -m etl.pipeline          # full run
    python -m etl.pipeline --init   # create tables only (no data load)
"""

import argparse
import sys
import time

from db.schema import create_all_tables
from etl.download import download_and_parse, fetch_remote_sha256
from etl.normalize import normalize_cards
from etl.upsert import upsert_cards


def run(*, init_only: bool = False):
    """Execute the full ETL pipeline."""
    print("━" * 60)
    print("  MTG RAG — Foundation ETL Pipeline")
    print("━" * 60)

    # Step 1: Ensure tables exist
    print("\n[1/4] Ensuring database schema…")
    create_all_tables()

    if init_only:
        print("\n✓ Schema initialised. Exiting (--init flag).")
        return

    # Step 2: Download
    print("\n[2/4] Downloading AtomicCards from MTGJson…")
    t0 = time.time()
    meta, data = download_and_parse()
    source_sha = fetch_remote_sha256()
    print(f"       ({time.time() - t0:.1f}s)")

    # Step 3: Normalize
    print("\n[3/4] Normalising to per-face rows…")
    t0 = time.time()
    rows = normalize_cards(data, source_version=meta["version"])
    print(f"       {len(rows):,} rows from {len(data):,} card names ({time.time() - t0:.1f}s)")

    # Step 4: Upsert
    print("\n[4/4] Upserting into Postgres…")
    t0 = time.time()
    stats = upsert_cards(rows, source_version=meta["version"], source_sha256=source_sha)
    elapsed = time.time() - t0

    # Summary
    print("\n" + "━" * 60)
    print("  ETL Complete")
    print("━" * 60)
    print(f"  Source version : {meta['version']}")
    print(f"  Total rows     : {stats['total_cards']:,}")
    print(f"  Inserted       : {stats['inserted']:,}")
    print(f"  Updated        : {stats['updated']:,}")
    print(f"  Unchanged      : {stats['unchanged']:,}")
    print(f"  Deleted        : {stats['deleted']:,}")
    print(f"  Upsert time    : {elapsed:.1f}s")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="MTG RAG foundation ETL")
    parser.add_argument(
        "--init", action="store_true",
        help="Create tables only, do not download/load data",
    )
    args = parser.parse_args()
    run(init_only=args.init)


if __name__ == "__main__":
    main()
