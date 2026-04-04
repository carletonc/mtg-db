"""
Database schema definitions.

Design decisions (validated via json_normalize EDA on 34,475 rows):
──────────────────────────────────────────────────────────────────
1. **One row per card face/side.**  Multi-faced cards (transform, modal_dfc,
   split, adventure, flip, aftermath, reversible_card) → one row per face.

2. **Composite PK:**  (scryfall_oracle_id, face_index).

3. **Change detection via content_hash** (SHA-256 of the face JSON, excluding
   foreignData and deprecated fields).

4. **Dropped columns** (confirmed duplicates via EDA):
   - convertedManaCost   ≡ manaValue       (deprecated)
   - faceConvertedManaCost ≡ faceManaValue  (deprecated)
   - asciiName           ≡ NFKD(name)      (derivable)
   - foreignData         → excluded (93.8 MB translations, would eat free tier)

5. **JSONB blobs** for legalities (19 format keys), purchaseUrls (8 keys),
   leadershipSkills (3 booleans), rulings (array of {date, text}),
   relatedCards ({spellbook: [], tokens: []}).

6. **Boolean flags** coerced null→False at ETL time (MTGJson only includes
   isFunny/isReserved/isGameChanger/hasAlternativeDeckLimit when True).
"""

from db.connection import get_connection, get_cursor

# --------------------------------------------------------------------------- #
#  Foundational cards table                                                    #
# --------------------------------------------------------------------------- #
CARDS_DDL = """
CREATE TABLE IF NOT EXISTS cards (
    -- Identity (PK)
    scryfall_oracle_id  TEXT        NOT NULL,
    face_index          SMALLINT    NOT NULL DEFAULT 0,

    -- Card identity
    name                TEXT        NOT NULL,
    face_name           TEXT,
    side                TEXT,           -- 'a', 'b', etc. (1,711 rows non-null)
    layout              TEXT        NOT NULL,

    -- Casting
    mana_cost           TEXT,           -- e.g. '{2}{R}{R}{G}{G}' (93.3% non-null)
    mana_value          REAL        NOT NULL,  -- always present (was convertedManaCost)
    face_mana_value     REAL,           -- only multi-faced (4.6% non-null)
    colors              TEXT[],         -- always present, can be empty for colorless
    color_identity      TEXT[],         -- always present
    color_indicator     TEXT[],         -- rare: 374 rows (1.1%)

    -- Types
    type_line           TEXT        NOT NULL,  -- e.g. 'Legendary Creature — God'
    supertypes          TEXT[],         -- always present, often empty
    types               TEXT[],         -- always present, always non-empty
    subtypes            TEXT[],         -- always present, often empty

    -- Stats (creature/planeswalker/vehicle/battle/vanguard)
    power               TEXT,           -- 54.9% non-null (stored as text: *, X, etc.)
    toughness           TEXT,           -- 54.9% non-null
    loyalty             TEXT,           -- 1.0% non-null (planeswalkers)
    defense             TEXT,           -- 0.1% non-null (battles)
    hand                TEXT,           -- 0.3% non-null (vanguards)
    life                TEXT,           -- 0.3% non-null (vanguards)

    -- Rules text & keywords
    oracle_text         TEXT,           -- 99.0% non-null (lands have none)
    keywords            TEXT[],         -- 49.0% non-null

    -- Game metadata (JSONB)
    legalities          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    leadership_skills   JSONB,          -- 11.0% non-null (legendary creatures)
    rulings             JSONB       NOT NULL DEFAULT '[]'::jsonb,
    purchase_urls       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    related_cards       JSONB,          -- spellbook + tokens

    -- Array columns
    printings           TEXT[],         -- always present
    produced_mana       TEXT[],         -- 8.0% non-null (lands, mana dorks)
    subsets             TEXT[],         -- 0.3% non-null

    -- Flags (coerced null→False at ETL time)
    is_funny            BOOLEAN     NOT NULL DEFAULT FALSE,  -- 3.9%
    is_game_changer     BOOLEAN     NOT NULL DEFAULT FALSE,  -- 0.2%
    is_reserved         BOOLEAN     NOT NULL DEFAULT FALSE,  -- 1.7%
    has_alt_deck_limit  BOOLEAN     NOT NULL DEFAULT FALSE,  -- <0.1%

    -- EDHREC
    edhrec_rank         INTEGER,        -- 91.5% non-null
    edhrec_saltiness    REAL,           -- 91.6% non-null

    -- Housekeeping
    content_hash        TEXT        NOT NULL,
    first_printing      TEXT,
    source_version      TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (scryfall_oracle_id, face_index)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_cards_name        ON cards (name);
CREATE INDEX IF NOT EXISTS idx_cards_layout      ON cards (layout);
CREATE INDEX IF NOT EXISTS idx_cards_colors      ON cards USING GIN (colors);
CREATE INDEX IF NOT EXISTS idx_cards_types       ON cards USING GIN (types);
CREATE INDEX IF NOT EXISTS idx_cards_keywords    ON cards USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_cards_legalities  ON cards USING GIN (legalities);
"""

# --------------------------------------------------------------------------- #
#  Sync-tracking table                                                         #
# --------------------------------------------------------------------------- #
SYNC_LOG_DDL = """
CREATE TABLE IF NOT EXISTS sync_log (
    id              SERIAL      PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    source_version  TEXT,
    source_sha256   TEXT,
    total_cards     INTEGER,
    inserted        INTEGER     DEFAULT 0,
    updated         INTEGER     DEFAULT 0,
    unchanged       INTEGER     DEFAULT 0,
    deleted         INTEGER     DEFAULT 0,
    status          TEXT        DEFAULT 'running'
);
"""

# --------------------------------------------------------------------------- #
#  pgvector extension + downstream vector table                                #
# --------------------------------------------------------------------------- #
VECTOR_TABLE_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS card_embeddings (
    scryfall_oracle_id  TEXT        NOT NULL,
    face_index          SMALLINT    NOT NULL DEFAULT 0,
    chunk_index         SMALLINT    NOT NULL DEFAULT 0,
    chunk_text          TEXT        NOT NULL,
    embedding           vector(1536),
    model               TEXT        NOT NULL DEFAULT 'text-embedding-3-small',
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (scryfall_oracle_id, face_index, chunk_index),
    FOREIGN KEY (scryfall_oracle_id, face_index)
        REFERENCES cards (scryfall_oracle_id, face_index)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_card_embeddings_vec
    ON card_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def create_all_tables():
    """Run all DDL statements to bootstrap the database."""
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(CARDS_DDL)
            cur.execute(SYNC_LOG_DDL)
            cur.execute(VECTOR_TABLE_DDL)
    print("✓ All tables and indexes created.")