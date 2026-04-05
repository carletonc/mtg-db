"""
Upsert normalised card rows into Postgres.

Strategy:
---------
1. Load all existing (scryfall_oracle_id, face_index, content_hash) into memory.
   With ~34k rows this is trivially small.
2. For each incoming row:
   - Key doesn't exist → INSERT
   - Key exists but hash differs → UPDATE
   - Key exists and hash matches → skip (unchanged)
3. Optionally delete rows whose keys no longer appear (cards removed from dataset).
4. Record stats in sync_log.
"""

import psycopg2.extras
from db.connection import get_connection, get_cursor


# All columns in insert order (must match normalize.py output keys)
_COLUMNS = [
    "scryfall_oracle_id", "face_index",
    "name", "face_name", "side", "layout",
    "mana_cost", "mana_value", "face_mana_value",
    "colors", "color_identity", "color_indicator",
    "type_line", "supertypes", "types", "subtypes",
    "power", "toughness", "loyalty", "defense", "hand", "life",
    "oracle_text", "keywords",
    "legalities", "leadership_skills", "rulings",
    "purchase_urls", "related_cards",
    "printings", "produced_mana", "subsets",
    "is_funny", "is_game_changer", "is_reserved", "has_alt_deck_limit",
    "edhrec_rank", "edhrec_saltiness",
    "content_hash", "first_printing", "source_version",
]

# Columns to update on conflict (everything except the PK)
_UPDATE_COLS = [c for c in _COLUMNS if c not in ("scryfall_oracle_id", "face_index")]


def _build_upsert_sql() -> str:
    """Build INSERT ... ON CONFLICT DO UPDATE."""
    col_list = ", ".join(_COLUMNS)
    placeholders = ", ".join(f"%({c})s" for c in _COLUMNS)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in _UPDATE_COLS)

    return f"""
        INSERT INTO cards ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (scryfall_oracle_id, face_index)
        DO UPDATE SET {update_set}, updated_at = now()
    """


def upsert_cards(
    rows: list[dict],
    source_version: str,
    source_sha256: str,
    *,
    delete_stale: bool = True,
    batch_size: int = 500,
) -> dict:
    """
    Upsert normalised rows and return a stats dict.

    Parameters
    ----------
    rows : list[dict]
        Output of normalize.normalize_cards()
    source_version : str
        Version from MTGJson meta
    source_sha256 : str
        SHA-256 of the downloaded .gz file
    delete_stale : bool
        If True, delete DB rows no longer in incoming data.
    batch_size : int
        Rows per executemany batch.

    Returns
    -------
    dict with keys: inserted, updated, unchanged, deleted, total_cards
    """
    upsert_sql = _build_upsert_sql()
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}

    with get_connection() as conn:
        with get_cursor(conn) as cur:
            # 1. Start sync_log entry
            cur.execute(
                """INSERT INTO sync_log (source_version, source_sha256, total_cards)
                   VALUES (%s, %s, %s) RETURNING id""",
                (source_version, source_sha256, len(rows)),
            )
            sync_id = cur.fetchone()[0]

            # 2. Fetch existing hashes
            cur.execute(
                "SELECT scryfall_oracle_id, face_index, content_hash FROM cards"
            )
            existing = {(r[0], r[1]): r[2] for r in cur.fetchall()}

            # 3. Classify rows
            to_insert = []
            to_update = []
            incoming_keys = set()

            for row in rows:
                key = (row["scryfall_oracle_id"], row["face_index"])
                incoming_keys.add(key)
                old_hash = existing.get(key)

                if old_hash is None:
                    to_insert.append(row)
                elif old_hash != row["content_hash"]:
                    to_update.append(row)
                else:
                    stats["unchanged"] += 1

            # 4. Batch insert new rows
            if to_insert:
                for i in range(0, len(to_insert), batch_size):
                    batch = to_insert[i : i + batch_size]
                    psycopg2.extras.execute_batch(cur, upsert_sql, batch, page_size=batch_size)
                stats["inserted"] = len(to_insert)
                print(f"  ＋ Inserted {len(to_insert):,} new rows")

            # 5. Batch update changed rows
            if to_update:
                for i in range(0, len(to_update), batch_size):
                    batch = to_update[i : i + batch_size]
                    psycopg2.extras.execute_batch(cur, upsert_sql, batch, page_size=batch_size)
                stats["updated"] = len(to_update)
                print(f"  ✎ Updated {len(to_update):,} changed rows")

            # 6. Delete stale rows
            if delete_stale:
                stale_keys = set(existing.keys()) - incoming_keys
                if stale_keys:
                    for key in stale_keys:
                        cur.execute(
                            "DELETE FROM cards WHERE scryfall_oracle_id = %s AND face_index = %s",
                            key,
                        )
                    stats["deleted"] = len(stale_keys)
                    print(f"  ✕ Deleted {len(stale_keys):,} stale rows")

            if stats["inserted"] == 0 and stats["updated"] == 0 and stats["deleted"] == 0:
                print("  ＝ No changes detected — database is up to date.")

            # 7. Finalise sync_log
            cur.execute(
                """UPDATE sync_log
                   SET finished_at = now(),
                       inserted    = %s,
                       updated     = %s,
                       unchanged   = %s,
                       deleted     = %s,
                       status      = 'success'
                   WHERE id = %s""",
                (
                    stats["inserted"],
                    stats["updated"],
                    stats["unchanged"],
                    stats["deleted"],
                    sync_id,
                ),
            )

    stats["total_cards"] = len(rows)
    return stats
