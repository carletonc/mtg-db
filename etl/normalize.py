"""
Normalize the raw MTGJson AtomicCards dict into flat per-face rows.

ID ASSIGNMENT
─────────────
Each row gets a deterministic composite PK: (scryfall_oracle_id, face_index)

face_index is a per-oracle-id counter (0-based), NOT the global array
position.  This matters for cards like "Fast // Furious" where two
different oracle cards (UNK vs MH2 printings with different oracle text)
share one name key:

    Global array index (WRONG):          Per-oracle-id counter (CORRECT):
    (298a..., 0)  Fast                   (298a..., 0)  Fast
    (624...,  1)  Fast                   (624...,  0)  Fast
    (624...,  2)  Furious                (624...,  1)  Furious
    (298a..., 3)  Furious                (298a..., 1)  Furious

The per-oracle-id counter ensures face_index is stable even if MTGJson
reorders the array, and semantically correct (face 0 and 1 of a given
oracle card always mean front/back).

CHANGE DETECTION
────────────────
content_hash is SHA-256 of oracle-level fields only.  This means the
pipeline only marks a row as changed when game-relevant data changes:
  - Oracle text errata              → triggers update ✓
  - New ruling                      → triggers update ✓
  - Legality change (ban/unban)     → triggers update ✓
  - Stats/type change               → triggers update ✓

Fields EXCLUDED from the content_hash (but still stored and overwritten
on every sync):
  - edhrecRank / edhrecSaltiness    → community metrics, change weekly
  - purchaseUrls                    → redirect links, change every build
  - printings                       → grows with new sets, not oracle data
  - firstPrinting                   → static but not oracle data
  - foreignData                     → 94 MB of translations, excluded entirely

COLUMN EXCLUSIONS (confirmed duplicates via EDA)
────────────────────────────────────────────────
  - convertedManaCost    ≡ manaValue           (deprecated, 100% identical)
  - faceConvertedManaCost ≡ faceManaValue      (deprecated, 100% identical)
  - asciiName            ≡ NFKD(name)          (81 rows, 100% derivable)
  - foreignData          → excluded (93.8 MB)
"""

import hashlib
import json
from typing import Any


def _stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON blob with deterministic key ordering."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _pg_array(val: list | None) -> list | None:
    """Return a plain list or None — psycopg2 handles the rest."""
    if val is None:
        return None
    return list(val)


# Keys included in the oracle-level content hash.
# If any of these change in the source, the row is marked as updated.
_ORACLE_HASH_KEYS = frozenset({
    "colorIdentity",
    "colorIndicator",
    "colors",
    "defense",
    "faceManaValue",
    "faceName",
    "hand",
    "hasAlternativeDeckLimit",
    "identifiers",
    "isFunny",
    "isGameChanger",
    "isReserved",
    "keywords",
    "layout",
    "leadershipSkills",
    "legalities",
    "life",
    "loyalty",
    "manaCost",
    "manaValue",
    "name",
    "power",
    "producedMana",
    "relatedCards",
    "rulings",
    "side",
    "subsets",
    "subtypes",
    "supertypes",
    "text",
    "toughness",
    "type",
    "types",
})


def _oracle_hash(face: dict) -> str:
    """
    Hash only oracle-level fields.  Changes to edhrecRank, purchaseUrls,
    printings, foreignData, etc. will NOT trigger an update.
    """
    oracle_data = {k: v for k, v in face.items() if k in _ORACLE_HASH_KEYS}
    return _stable_hash(oracle_data)


def normalize_cards(
    data: dict[str, list[dict]],
    source_version: str,
) -> list[dict]:
    """
    Flatten the MTGJson AtomicCards dict into one dict per face.

    Parameters
    ----------
    data : dict
        The "data" key from AtomicCards.json → {"CardName": [face, ...]}
    source_version : str
        Version string from the file metadata (e.g. "5.3.0+20260323")

    Returns
    -------
    list[dict]  — one dict per face, ready for DB upsert.

    Notes
    -----
    MTGJson contains duplicate oracle IDs for "reversible_card" variants
    (e.g. "Sol Ring" and "Sol Ring // Sol Ring" share the same oracle ID).
    We deduplicate by preferring the entry whose card_name key is shorter
    (the canonical version).
    """
    # Collect rows keyed by (oracle_id, face_index) for dedup
    seen: dict[tuple[str, int], dict] = {}
    seen_name_len: dict[tuple[str, int], int] = {}

    for card_name, faces in data.items():
        # Per-oracle-id face counter (NOT global array index)
        oid_counter: dict[str, int] = {}

        for face in faces:
            oracle_id = face.get("identifiers", {}).get("scryfallOracleId")
            if not oracle_id:
                continue

            face_index = oid_counter.get(oracle_id, 0)
            oid_counter[oracle_id] = face_index + 1

            key = (oracle_id, face_index)

            # Dedup: keep the entry with the shorter card_name key
            if key in seen_name_len:
                if len(card_name) >= seen_name_len[key]:
                    continue

            # ── Build the row ──────────────────────────────────────

            legalities_raw = face.get("legalities", {})
            legalities = {k: v for k, v in legalities_raw.items() if v is not None}

            ls = face.get("leadershipSkills")
            leadership_skills = json.dumps(ls) if ls else None

            pu = face.get("purchaseUrls", {})
            rc = face.get("relatedCards")
            rulings = face.get("rulings", [])

            row = {
                "scryfall_oracle_id": oracle_id,
                "face_index": face_index,
                # Identity
                "name": face.get("name", card_name),
                "face_name": face.get("faceName"),
                "side": face.get("side"),
                "layout": face.get("layout", "normal"),
                # Casting
                "mana_cost": face.get("manaCost"),
                "mana_value": face.get("manaValue"),
                "face_mana_value": face.get("faceManaValue"),
                "colors": _pg_array(face.get("colors")),
                "color_identity": _pg_array(face.get("colorIdentity")),
                "color_indicator": _pg_array(face.get("colorIndicator")),
                # Types
                "type_line": face.get("type"),
                "supertypes": _pg_array(face.get("supertypes")),
                "types": _pg_array(face.get("types")),
                "subtypes": _pg_array(face.get("subtypes")),
                # Stats
                "power": face.get("power"),
                "toughness": face.get("toughness"),
                "loyalty": face.get("loyalty"),
                "defense": face.get("defense"),
                "hand": face.get("hand"),
                "life": face.get("life"),
                # Rules text
                "oracle_text": face.get("text"),
                "keywords": _pg_array(face.get("keywords")),
                # Game metadata (JSONB)
                "legalities": json.dumps(legalities),
                "leadership_skills": leadership_skills,
                "rulings": json.dumps(rulings),
                "purchase_urls": json.dumps(pu),
                "related_cards": json.dumps(rc) if rc else None,
                # Array columns
                "printings": _pg_array(face.get("printings")),
                "produced_mana": _pg_array(face.get("producedMana")),
                "subsets": _pg_array(face.get("subsets")),
                # Flags — coerce null → False
                "is_funny": bool(face.get("isFunny", False)),
                "is_game_changer": bool(face.get("isGameChanger", False)),
                "is_reserved": bool(face.get("isReserved", False)),
                "has_alt_deck_limit": bool(face.get("hasAlternativeDeckLimit", False)),
                # EDHREC (stored but NOT in content_hash)
                "edhrec_rank": face.get("edhrecRank"),
                "edhrec_saltiness": face.get("edhrecSaltiness"),
                # Housekeeping
                "content_hash": _oracle_hash(face),
                "first_printing": face.get("firstPrinting"),
                "source_version": source_version,
            }

            seen[key] = row
            seen_name_len[key] = len(card_name)

    return list(seen.values())
