# MTG RAG Pipeline

An agentic RAG system for Magic: The Gathering, built on **Postgres + pgvector** (hosted free on [Neon](https://neon.tech)).

Downloads card data from MTGJson, normalizes it per-face, preprocesses MTG notation into natural language (preserving domain terminology for agent cross-referencing), creates vector embeddings for card names, card text, and game rules, and keeps everything up to date via weekly automated syncs.

## Architecture

```
MTGJson AtomicCards.json.gz          MTG Comprehensive Rules (.txt)
        │                                       │
        ▼                                       │
  ┌─────────────┐    ┌──────────────┐           │
  │  Download &  │───▶│  Normalize   │           │
  │  SHA-256     │    │  per-face    │           │
  │  verify      │    │  rows        │           │
  └─────────────┘    └──────┬───────┘           │
                            │                    │
                            ▼                    │
                   ┌────────────────┐            │
                   │  Upsert cards  │            │
                   │  (content_hash │            │
                   │   change det.) │            │
                   └────────┬───────┘            │
                            │                    │
                            ▼                    ▼
                   ┌──────────────────────────────────┐
                   │     Embedding Pipeline            │
                   │                                   │
                   │  card_name ─┐                     │
                   │  card_text ─┤─▶ HF / OpenAI ──▶ embeddings table
                   │  rules ─────┘                     │
                   │  (add more)                       │
                   └───────────────────────────────────┘
```

## Quick Start

```bash
git clone https://github.com/<you>/mtg-rag.git && cd mtg-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in DATABASE_URL
python -m etl.pipeline  # does everything: tables → ETL → embed all sources
```

### Selective Runs

```bash
python -m etl.pipeline --init          # create tables only
python -m etl.pipeline --etl-only      # foundation ETL (no embeddings)
python -m etl.pipeline --embed-only    # embeddings only (skip ETL)
python -m etl.pipeline --full-refresh  # re-embed everything

python -m etl.embed --source card_name # only card name embeddings
python -m etl.embed --source card_text # only card text embeddings
python -m etl.embed --source rules     # only rules embeddings
```

### Test

```bash
python scripts/query_test.py            # full test suite
python scripts/query_test.py --verbose  # detailed output
```

Tests: foundation table integrity, exact name lookup, text phrase search (SQL ILIKE), vector similarity on card names, vector similarity on card text, vector similarity on rules.

## Embedding Sources

### `card_name` — Card Names

Each card face name → one embedding. Enables fuzzy/semantic name matching (misspellings, partial names, nicknames).

### `card_text` — Card Oracle Text (Preprocessed)

Each card face → one embedding containing name, type line, mana cost, stats (P/T, loyalty, defense), oracle text, and keywords — all **preprocessed** from MTG notation to natural language before embedding.

Raw columns in the `cards` table stay untouched.  Preprocessing happens at embedding time only, via `utils/mtg_text.py`.

See [Text Preprocessing](#text-preprocessing) below for the full mapping.

### `rules` — MTG Comprehensive Rules

Downloads the [official Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260227.txt) from Wizards of the Coast. The document is ~940K characters with this structure:

```
Table of Contents
  1. Game Concepts                 ← section headers
  ...
  9. Casual Variants
  Glossary
  Credits

1. Game Concepts                   ← actual rules body starts here
100. General
100.1. These Magic rules apply ...
100.1a A two-player game ...
...
702.9. Flying
702.9a Flying is an evasion ability.
702.9b A creature with flying can't be blocked except by creatures
       with flying and/or reach. ...
...
702.17. Reach
702.17a Reach is a static ability.
702.17b A creature with flying can't be blocked except by creatures
        with flying and/or reach. ...
...

Glossary
  Flying
    A keyword ability that restricts how a creature may be blocked.
    See rule 702.9, "Flying."
  Reach
    A keyword ability that allows a creature to block an attacking
    creature with flying. See rule 702.17, "Reach." See also Flying.
  ...

Credits
```

#### Parsing Strategy

The parser produces two types of chunks:

| Type | Count | Example ID | What it contains |
|---|---|---|---|
| Numbered rules | ~2,900 | `rule:702.17` | Full rule text (e.g., Reach definition + subrules) |
| Glossary terms | ~725 | `glossary:Reach` | Term + definition + cross-references |

**Why this matters for agentic search:**  When a user says "cards that stop flyers", the agent needs to discover that Reach is the relevant keyword — this isn't something an LLM should answer from pretrained knowledge. The rules embeddings provide:

- **702.9 (Flying)** → "can't be blocked except by creatures with flying and/or reach"
- **702.17 (Reach)** → "allows a creature to block an attacking creature with flying"
- **Glossary: Reach** → "See also Flying"
- **Glossary: Evasion Ability** → "restricts what creatures can block"

The agent can vector-search rules to find these, then use the discovered keywords (reach, flying) to search card_text embeddings and SQL queries.

#### What the Rules Cover

| Section | Rules | Content |
|---|---|---|
| 1. Game Concepts | 100-199 | Players, objects, zones, game structure |
| 2. Parts of a Card | 200-299 | Name, mana cost, types, oracle text |
| 3. Card Types | 300-399 | Creatures, artifacts, enchantments, etc. |
| 4. Zones | 400-499 | Library, hand, battlefield, graveyard, exile, stack |
| 5. Turn Structure | 500-599 | Phases, steps, combat, declare blockers |
| 6. Spells & Abilities | 600-699 | Casting, resolving, targeting, replacement effects |
| 7. Additional Rules | 700-732 | **701: Keyword Actions** (68 — destroy, exile, fight, etc.) |
| | | **702: Keyword Abilities** (190 — flying, reach, trample, etc.) |
| | | 703-732: State-based actions, copying, DFCs, sagas, etc. |
| 8. Multiplayer | 800-899 | Two-Headed Giant, free-for-all, etc. |
| 9. Casual Variants | 900-999 | Commander, planechase, etc. |

**Note:** Wizards updates the rules URL with each set release (the date changes in the filename). When that happens, update `MTG_RULES_URL` in `.env`.

### Adding New Sources

To add a new embedding source:

```python
# In etl/embed.py

def my_source_chunks(cur, *, full_refresh: bool = False):
    """Yield (id, chunk_index, chunk_text) tuples."""
    yield ("my_source:key1", 0, "text to embed")

SOURCES = {
    "card_name": card_name_chunks,
    "card_text": card_text_chunks,
    "rules": rules_chunks,
    "my_source": my_source_chunks,  # ← add here
}
```

Then run: `python -m etl.embed --source my_source`

## Text Preprocessing

MTG oracle text uses domain-specific notation (`{T}`, `{W/P}`, `{E}`, etc.) that embedding models cannot interpret semantically.  The preprocessing module (`utils/mtg_text.py`) translates this notation into natural language before embedding, improving average retrieval similarity from **0.18 to 0.58** in benchmark tests.

### What Gets Preprocessed

| Notation | Raw | Preprocessed |
|---|---|---|
| Tap/Untap | `{T}: Add {G}.` | `Tap (tap this permanent): Add green mana.` |
| Phyrexian mana | `{W/P}` | `Phyrexian white mana (white mana or 2 life)` |
| Hybrid mana | `{2/U}` | `hybrid two generic or blue mana` |
| Energy | `{E}{E}{E}` | `energy (energy counter), energy (energy counter), ...` |
| Ticket counters | `{TK}` | `ticket (ticket counter)` |
| Snow mana | `{S}` | `snow mana` |
| Generic mana | `{3}` | `three generic mana` |
| Variable cost | `{X}` | `X (variable amount)` |
| Power/Toughness | `*/1+*` | `variable power / one plus variable toughness` |
| Mana cost | `{2}{R}{R}` | `two generic mana, red mana, red mana` |
| Em dash | `—` | ` -- ` |
| Bullet (modal) | `•` | ` * ` |
| Self-references | `Lightning Bolt deals 3...` | `this card deals 3...` |

Design choices:
- **Dual expansion — jargon + functional description**: `{W/P}` becomes "Phyrexian white mana (white mana or 2 life)".  This preserves the domain term so jargon queries ("Phyrexian mana") match, while the functional description ensures natural queries ("pay life instead of mana") also match.  The rules embeddings reference "Phyrexian mana" in their natural text, so keeping the term in card embeddings lets an agent cross-reference rules to self-correct retrieval.
- **Reminder text is kept**: 31% of cards include parenthesized rules reminders that add embedding context.
- **Raw data is never modified**: The `cards` table stores raw MTG notation.  Preprocessing is embedding-time only.

### Cross-Repo Consistency

`utils/mtg_text.py` and `utils/mtg_taxonomy.py` are both **self-contained** (Python stdlib only, zero dependencies).  Copy them into the retrieval repository.  Both repos must use the same preprocessing and taxonomy definitions.

```python
# Retrieval repo usage — preprocessing
from utils.mtg_text import preprocess_oracle_text

# User pastes raw card text → preprocess before embedding
query = preprocess_oracle_text(user_input)
vector = model.encode(query)
```

Normal English queries ("tap for green mana") don't need preprocessing.

## Gameplay Concept Taxonomy

MTG players use slang and high-level strategy terms that don't appear in oracle text.  The taxonomy module (`utils/mtg_taxonomy.py`) maps this vocabulary hierarchy so the retrieval layer can bridge the gap.

### The Problem

A user searching "I need card advantage" won't find cards because no card says "card advantage" in its oracle text.  Card advantage is a parent concept that branches into: direct draw ("draw a card"), tutoring ("search your library"), impulse draw ("exile the top... you may play"), cheating into play ("without paying its mana cost"), recursion ("return from your graveyard"), and card filtering (loot, rummage, scry, surveil).

This is true for most strategic concepts — "ramp" splits into land tutors, mana dorks, mana rocks, Treasure tokens, cost reduction, and mana doublers.  "Interaction" splits into targeted removal, board wipes, counterspells, protection, and stax.

### The Taxonomy Structure

12 top-level concepts, each with 2-6 children, covering ~90% of player vocabulary:

```
CARD ADVANTAGE
├── Draw (cantrip, wheel)
├── Tutor (to hand / top / battlefield / graveyard)
├── Impulse Draw (exile top, play temporarily)
├── Cheat into Play (cascade, free cast)
├── Card Filtering (loot, rummage, scry, surveil)
└── Recursion (reanimate, regrowth)

INTERACTION
├── Targeted Removal (destroy, exile, damage, shrink, edict, tuck)
├── Mass Removal (board wipe, sweeper, wrath)
├── Counterspells (hard counter, soft counter)
├── Protection (hexproof, ward, indestructible, phasing)
└── Stax / Denial (tax, lock, prison)

BUFF / ENHANCE
├── Anthems / Lords (static mass buff)
├── Counters (+1/+1, keyword counters)
├── Pump (temporary buff, combat tricks)
├── Keyword Granting (haste, flying, trample, etc.)
└── Equipment & Auras (voltron)

MANA ACCELERATION
├── Land Tutors (true ramp, land to hand)
├── Mana Dork (creature that taps for mana)
├── Mana Rock (artifact that taps for mana)
├── Treasure Tokens
├── Cost Reduction (affinity, convoke, delve)
├── Mana Doubler
└── Extra Land Drops

EVASION — flying, menace, trample, can't be blocked, etc.
GRAVEYARD — flashback, self-mill, mill opponent, graveyard hate
TOKENS — creature tokens, artifact tokens, copy/clone
SACRIFICE / ARISTOCRATS — sac outlets, dies triggers
FLICKER / BLINK — exile and return for ETB abuse
COMBAT — fog, goad, lure
LIFE — lifegain, drain
```

Each node carries: aliases (player slang), oracle text hints (for query expansion), and a description (optionally embeddable).

### How to Use It

The taxonomy is **decoupled** from the embedding pipeline.  It doesn't modify card text or embeddings.  The retrieval layer uses it for:

```python
from utils.mtg_taxonomy import get_all_aliases, get_oracle_hints, get_node

# 1. QUERY MATCHING — detect when a user query maps to a concept
aliases = get_all_aliases()
user_term = "board wipe"
if user_term.lower() in aliases:
    key = aliases[user_term.lower()]  # → "interaction.mass_removal"

# 2. QUERY EXPANSION — fan out to oracle text patterns
hints = get_oracle_hints(key)
# → ["destroy all creatures", "destroy all permanents", "exile all creatures",
#    "all creatures get -", "damage to each creature", "each player sacrifices"]
# Issue multiple vector searches, one per hint, merge results.

# 3. STRUCTURED FILTERING — color group names
from utils.mtg_taxonomy import COLOR_GROUPS
colors = COLOR_GROUPS.get("azorius")  # → ["W", "U"]
# Use for SQL WHERE clause on color_identity column.

# 4. ABBREVIATION EXPANSION
from utils.mtg_taxonomy import ABBREVIATIONS
expanded = ABBREVIATIONS.get("ETB")  # → "enters the battlefield"
```

### Future Integration Points

The codebase has comments marking where the taxonomy could optionally be integrated deeper:

| Location | Integration | Status |
|---|---|---|
| `etl/embed.py` → `_build_card_text_chunk()` | Tag cards with taxonomy keys at embedding time | Commented out |
| `etl/embed.py` → `SOURCES` registry | Embed taxonomy descriptions as a new source | Commented out |
| `db/schema.py` → `CARDS_DDL` | Add `taxonomy_tags TEXT[]` column with GIN index | Commented out |

None of these are active.  Enable them when you need tighter coupling between the taxonomy and the data pipeline.

### Change Detection

The embedding pipeline uses `md5(chunk_text + model_name)` to detect when re-embedding is needed.  This catches all three mutation vectors:

| Change | Detected? | Mechanism |
|---|---|---|
| Raw card data changed (errata) | Yes | Different preprocessed text → different md5 |
| Preprocessing logic changed | Yes | Different preprocessed text → different md5 |
| Embedding model changed | Yes | Model name is part of the hash |

No manual version bumping required.  Just run the pipeline and it figures out what needs re-embedding.

## Embedding Backend

Controlled by `EMBEDDING_BACKEND` in `.env`:

| Backend | Default model | Dimension | Cost |
|---|---|---|---|
| `huggingface` (default) | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Free |
| `openai` | `text-embedding-3-small` | 1536 | ~$0.02/1M tokens |

Other HuggingFace options:
- `sentence-transformers/all-mpnet-base-v2` — 768 dims, higher quality
- `BAAI/bge-small-en-v1.5` — 384 dims, strong benchmarks
- `thenlper/gte-small` — 384 dims, good general-purpose

To switch backends: change `EMBEDDING_BACKEND` in `.env`, then `python -m etl.embed --full-refresh` (dimensions differ, vectors must be regenerated).

## Project Structure

```
mtg-rag/
├── .env.example               # Template — copy to .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── utils/
│   ├── config.py              # Env loading, URL constants, model config
│   ├── embeddings.py          # Pluggable embedding backend (HF / OpenAI)
│   ├── mtg_text.py            # MTG notation → natural language preprocessor
│   └── mtg_taxonomy.py        # Hierarchical gameplay concept taxonomy
│
├── db/
│   ├── connection.py          # psycopg2 connection context managers
│   └── schema.py              # DDL: cards, sync_log, embeddings (unified)
│
├── etl/
│   ├── download.py            # Fetch + SHA-256 verify AtomicCards.json.gz
│   ├── normalize.py           # Explode multi-faced cards → flat rows
│   ├── upsert.py              # Hash-based change detection + batch upsert
│   ├── embed.py               # Multi-source embedding pipeline
│   └── pipeline.py            # End-to-end orchestrator (CLI entry point)
│
├── scripts/
│   ├── init_db.py             # Create tables
│   ├── run_etl.py             # Foundation ETL only
│   ├── run_embed.py           # Embedding pipeline only
│   └── query_test.py          # Test suite (7 tests)
│
└── .github/workflows/
    └── sync_cards.yml         # Weekly automated sync via GitHub Actions
```

## Database Schema

### `cards` — Foundation Table

41 columns per row. PK: `(scryfall_oracle_id, face_index)`.

Key columns: `name`, `oracle_text`, `type_line`, `legalities` (JSONB), `keywords` (TEXT[]), `content_hash` (SHA-256 for change detection).

### `embeddings` — Unified Vector Table

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT | Source-specific key |
| `source` | TEXT | `card_name`, `card_text`, `rules` |
| `chunk_index` | SMALLINT | For future multi-chunk support |
| `chunk_text` | TEXT | The preprocessed text that was embedded |
| `chunk_hash` | TEXT | md5(chunk_text + model) for change detection |
| `embedding` | vector(N) | Dimension set by active model |
| `model` | TEXT | Model name for tracking |

PK: `(id, source, chunk_index)`.

Source-specific ID formats:
- `card_name` / `card_text`: `"{oracle_id}:{face_index}"`
- `rules`: `"rule:{number}"` or `"glossary:{term}"`

### `sync_log` — ETL Audit Trail

Tracks every ETL run: version, timestamps, insert/update/unchanged/delete counts.

## GitHub Secrets

| Secret | Required | Value |
|---|---|---|
| `DATABASE_URL` | Yes | Neon connection string |
| `EMBEDDING_BACKEND` | No | `huggingface` (default) or `openai` |
| `HF_EMBED_MODEL` | No | HuggingFace model name |
| `OPENAI_API_KEY` | If using OpenAI | API key |
