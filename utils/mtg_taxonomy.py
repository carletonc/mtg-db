"""
Hierarchical taxonomy of MTG gameplay concepts.

Maps the vocabulary players actually use — high-level strategy terms,
slang, abbreviations — to the oracle text patterns that exist on cards
in the database.  This is a structured lookup, NOT an embedding.

This module is **intentionally self-contained** (stdlib only) so it can
be copied into the retrieval repository alongside mtg_text.py.

Architecture
────────────
The taxonomy is a tree of concepts.  Each node has:
  - key:          machine-readable identifier (e.g., "card_advantage.tutor")
  - name:         human-readable display name
  - aliases:      list of slang terms / abbreviations players use
  - description:  plain English explanation (embeddable as a reference chunk)
  - oracle_hints: list of oracle text fragments for query expansion
  - children:     list of child nodes (recursion)

The retrieval layer can use this in several ways:

  1. QUERY EXPANSION — When a user says "I need card advantage", the agent
     recognizes it as a taxonomy node, walks the children, and issues
     multiple targeted searches using each child's oracle_hints.

  2. REFERENCE CHUNKS — Each node's description can be embedded as a
     standalone document in the vector DB.  When a user searches a
     high-level concept, the taxonomy chunk ranks high, and the agent
     reads it to understand the sub-concepts before doing card searches.

  3. STRUCTURED FILTERS — For concepts like color pairs ("Azorius"),
     the taxonomy provides metadata (color codes) that the agent can
     use for SQL WHERE clauses instead of semantic search.

  4. CARD TAGGING (optional) — The oracle_hints patterns could be used
     to tag cards in the DB with taxonomy keys.  This is not currently
     implemented but the schema supports it (see schema.py comments).

The taxonomy is deliberately NOT exhaustive — it covers the ~90% of
player vocabulary that causes retrieval failures.  It will grow over
time as new mechanics and slang emerge.

Usage
─────
    from utils.mtg_taxonomy import TAXONOMY, get_node, get_all_aliases

    # Walk a specific branch
    node = get_node("card_advantage.tutor")
    print(node["oracle_hints"])
    # → ["search your library"]

    # Flat lookup of all aliases → node keys (for query matching)
    aliases = get_all_aliases()
    print(aliases["board wipe"])
    # → "interaction.mass_removal"
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
#  Taxonomy tree
# ═══════════════════════════════════════════════════════════════════════════════

TAXONOMY: list[dict] = [
    # ──────────────────────────────────────────────────────────────────────
    #  CARD ADVANTAGE
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "card_advantage",
        "name": "Card Advantage",
        "aliases": ["card advantage", "CA", "card draw", "card selection"],
        "description": (
            "Card advantage: gaining access to more cards than your opponent. "
            "Includes direct draw, tutoring, impulse draw (exile-based), "
            "cheating cards into play, card filtering (loot, rummage, surveil, "
            "scry), and recursion from the graveyard."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "card_advantage.draw",
                "name": "Draw",
                "aliases": ["draw", "card draw", "draw spells"],
                "description": "Direct card draw — adding cards from library to hand.",
                "oracle_hints": ["draw a card", "draw cards", "draw two", "draw three"],
                "children": [
                    {
                        "key": "card_advantage.draw.cantrip",
                        "name": "Cantrip",
                        "aliases": ["cantrip", "cantrips", "cycling"],
                        "description": (
                            "A cheap spell that replaces itself by drawing a card. "
                            "Cycling (discard to draw) is a related mechanic."
                        ),
                        "oracle_hints": ["draw a card", "cycling"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.draw.symmetric",
                        "name": "Symmetric Draw",
                        "aliases": ["wheel", "wheeling", "windfall"],
                        "description": (
                            "Each player discards their hand and draws a new one. "
                            "Named after Wheel of Fortune."
                        ),
                        "oracle_hints": [
                            "each player discards",
                            "each player draws",
                            "each player shuffles",
                        ],
                        "children": [],
                    },
                ],
            },
            {
                "key": "card_advantage.tutor",
                "name": "Tutor",
                "aliases": ["tutor", "tutoring", "search"],
                "description": (
                    "Search your library for a specific card. Named after "
                    "Demonic Tutor. Subtypes based on where the card goes."
                ),
                "oracle_hints": ["search your library"],
                "children": [
                    {
                        "key": "card_advantage.tutor.to_hand",
                        "name": "Tutor to Hand",
                        "aliases": ["tutor to hand"],
                        "description": "Search library, put card into hand.",
                        "oracle_hints": ["search your library", "put it into your hand"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.tutor.to_top",
                        "name": "Tutor to Top",
                        "aliases": ["topdeck tutor", "vampiric tutor"],
                        "description": "Search library, put card on top.",
                        "oracle_hints": ["search your library", "put it on top"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.tutor.to_battlefield",
                        "name": "Tutor to Battlefield",
                        "aliases": ["cheat into play"],
                        "description": "Search library, put directly onto the battlefield.",
                        "oracle_hints": [
                            "search your library",
                            "put it onto the battlefield",
                        ],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.tutor.entomb",
                        "name": "Entomb",
                        "aliases": ["entomb", "tutor to graveyard", "buried alive"],
                        "description": "Search library, put into graveyard (for recursion).",
                        "oracle_hints": [
                            "search your library",
                            "put it into your graveyard",
                        ],
                        "children": [],
                    },
                ],
            },
            {
                "key": "card_advantage.impulse",
                "name": "Impulse Draw",
                "aliases": ["impulse draw", "impulse", "exile draw", "red draw"],
                "description": (
                    "Exile cards from the top of your library and cast them "
                    "temporarily. Common in red."
                ),
                "oracle_hints": [
                    "exile the top",
                    "you may play",
                    "you may cast",
                    "until end of turn",
                ],
                "children": [],
            },
            {
                "key": "card_advantage.cheat",
                "name": "Cheat into Play",
                "aliases": ["cheat", "free cast", "cascade", "discover"],
                "description": (
                    "Put permanents onto the battlefield or cast spells "
                    "without paying their mana cost."
                ),
                "oracle_hints": [
                    "without paying its mana cost",
                    "without paying their mana cost",
                    "put onto the battlefield",
                ],
                "children": [],
            },
            {
                "key": "card_advantage.filtering",
                "name": "Card Filtering",
                "aliases": ["filtering", "card selection", "card quality"],
                "description": (
                    "Improving card quality without strict card advantage — "
                    "loot, rummage, scry, surveil."
                ),
                "oracle_hints": [],
                "children": [
                    {
                        "key": "card_advantage.filtering.loot",
                        "name": "Loot",
                        "aliases": ["loot", "looting"],
                        "description": "Draw a card, then discard a card.",
                        "oracle_hints": ["draw a card, then discard", "draw then discard"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.filtering.rummage",
                        "name": "Rummage",
                        "aliases": ["rummage", "rummaging"],
                        "description": "Discard a card, then draw a card.",
                        "oracle_hints": ["discard a card, then draw", "discard then draw"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.filtering.scry",
                        "name": "Scry",
                        "aliases": ["scry", "scrying"],
                        "description": (
                            "Look at the top N cards of your library, put any "
                            "on bottom in any order."
                        ),
                        "oracle_hints": ["scry"],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.filtering.surveil",
                        "name": "Surveil",
                        "aliases": ["surveil", "surveilling"],
                        "description": (
                            "Look at the top N cards, put any into your graveyard "
                            "and the rest back on top."
                        ),
                        "oracle_hints": ["surveil"],
                        "children": [],
                    },
                ],
            },
            {
                "key": "card_advantage.recursion",
                "name": "Recursion",
                "aliases": ["recursion", "graveyard recursion", "reanimation", "regrowth"],
                "description": (
                    "Return cards from the graveyard to hand, battlefield, or "
                    "top of library."
                ),
                "oracle_hints": ["return", "from your graveyard"],
                "children": [
                    {
                        "key": "card_advantage.recursion.reanimate",
                        "name": "Reanimation",
                        "aliases": [
                            "reanimate", "reanimation", "reanimate spell",
                            "rise from the dead",
                        ],
                        "description": "Return a creature from graveyard to battlefield.",
                        "oracle_hints": [
                            "return",
                            "creature",
                            "from your graveyard to the battlefield",
                        ],
                        "children": [],
                    },
                    {
                        "key": "card_advantage.recursion.regrowth",
                        "name": "Regrowth",
                        "aliases": ["regrowth", "raise dead", "return to hand"],
                        "description": "Return a card from graveyard to hand.",
                        "oracle_hints": [
                            "return",
                            "from your graveyard to your hand",
                        ],
                        "children": [],
                    },
                ],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  INTERACTION
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "interaction",
        "name": "Interaction",
        "aliases": ["interaction", "answers", "removal", "response"],
        "description": (
            "Answering your opponent's threats. Includes targeted removal, "
            "mass removal (board wipes), counterspells, protection, and "
            "stax/denial effects."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "interaction.targeted_removal",
                "name": "Targeted Removal",
                "aliases": ["removal", "spot removal", "single-target removal"],
                "description": (
                    "Destroy, exile, or otherwise remove a single permanent."
                ),
                "oracle_hints": [
                    "destroy target",
                    "exile target",
                    "deals damage to target",
                ],
                "children": [
                    {
                        "key": "interaction.targeted_removal.destroy",
                        "name": "Destroy",
                        "aliases": ["destroy", "kill spell", "murder"],
                        "description": "Destroy target creature or permanent.",
                        "oracle_hints": ["destroy target creature", "destroy target permanent"],
                        "children": [],
                    },
                    {
                        "key": "interaction.targeted_removal.exile",
                        "name": "Exile-Based Removal",
                        "aliases": ["exile removal", "path", "swords"],
                        "description": "Exile target creature or permanent.",
                        "oracle_hints": ["exile target creature", "exile target permanent"],
                        "children": [],
                    },
                    {
                        "key": "interaction.targeted_removal.damage",
                        "name": "Damage-Based Removal",
                        "aliases": ["burn", "bolt", "direct damage"],
                        "description": "Deal damage to a target creature or player.",
                        "oracle_hints": ["deals damage to", "damage to any target"],
                        "children": [],
                    },
                    {
                        "key": "interaction.targeted_removal.shrink",
                        "name": "Shrink",
                        "aliases": ["shrink", "debuff", "minus counters"],
                        "description": "Give a creature -N/-N to weaken or kill it.",
                        "oracle_hints": ["gets -", "-1/-1"],
                        "children": [],
                    },
                    {
                        "key": "interaction.targeted_removal.edict",
                        "name": "Edict",
                        "aliases": ["edict", "forced sacrifice", "sac effect"],
                        "description": (
                            "Force an opponent to sacrifice a creature "
                            "(bypasses hexproof/indestructible)."
                        ),
                        "oracle_hints": ["opponent sacrifices", "player sacrifices"],
                        "children": [],
                    },
                    {
                        "key": "interaction.targeted_removal.tuck",
                        "name": "Tuck",
                        "aliases": ["tuck", "put on bottom"],
                        "description": "Put a permanent on the bottom of its owner's library.",
                        "oracle_hints": ["on the bottom of", "into its owner's library"],
                        "children": [],
                    },
                ],
            },
            {
                "key": "interaction.mass_removal",
                "name": "Mass Removal",
                "aliases": [
                    "board wipe", "boardwipe", "wrath", "sweeper",
                    "mass removal", "wipe the board", "nuke",
                ],
                "description": (
                    "Remove all or most creatures/permanents from the battlefield. "
                    "Named after Wrath of God."
                ),
                "oracle_hints": [
                    "destroy all creatures",
                    "destroy all permanents",
                    "exile all creatures",
                    "all creatures get -",
                    "damage to each creature",
                    "each player sacrifices",
                ],
                "children": [],
            },
            {
                "key": "interaction.counterspell",
                "name": "Counterspell",
                "aliases": [
                    "counter", "counterspell", "negate", "cancel",
                    "permission", "counter magic",
                ],
                "description": "Counter a spell on the stack, preventing it from resolving.",
                "oracle_hints": ["counter target spell"],
                "children": [
                    {
                        "key": "interaction.counterspell.hard",
                        "name": "Hard Counter",
                        "aliases": ["hard counter"],
                        "description": "Unconditionally counter a spell.",
                        "oracle_hints": ["counter target spell"],
                        "children": [],
                    },
                    {
                        "key": "interaction.counterspell.soft",
                        "name": "Soft Counter",
                        "aliases": ["soft counter", "tax counter", "mana leak"],
                        "description": "Counter unless the opponent pays additional mana.",
                        "oracle_hints": ["counter target spell unless", "unless its controller pays"],
                        "children": [],
                    },
                ],
            },
            {
                "key": "interaction.protection",
                "name": "Protection",
                "aliases": [
                    "protection", "hexproof", "shroud", "ward",
                    "indestructible", "phase out",
                ],
                "description": (
                    "Protect your permanents from opponent interaction — "
                    "hexproof, ward, indestructible, protection from, phasing."
                ),
                "oracle_hints": [
                    "hexproof",
                    "ward",
                    "indestructible",
                    "protection from",
                    "phases out",
                    "can't be the target",
                ],
                "children": [],
            },
            {
                "key": "interaction.stax",
                "name": "Stax / Denial",
                "aliases": [
                    "stax", "tax", "hate bear", "hatebear",
                    "denial", "lock", "prison",
                ],
                "description": (
                    "Restrict what opponents can do — prevent untapping, "
                    "increase costs, prohibit casting/activating."
                ),
                "oracle_hints": [
                    "can't untap",
                    "can't cast",
                    "can't activate",
                    "costs more",
                    "cost more to cast",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  BUFF / ENHANCE
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "buff",
        "name": "Buff / Enhance",
        "aliases": ["buff", "enhance", "pump", "boost", "power up"],
        "description": (
            "Making your creatures stronger — through anthems, counters, "
            "temporary pump spells, keyword granting, or equipment/auras."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "buff.anthem",
                "name": "Anthem",
                "aliases": ["anthem", "lord", "mass buff", "team buff", "static buff"],
                "description": (
                    "Permanent effect that buffs all your creatures. 'Lord' "
                    "refers to type-specific buffs (all Elves get +1/+1)."
                ),
                "oracle_hints": [
                    "creatures you control get +",
                    "other creatures you control get +",
                ],
                "children": [],
            },
            {
                "key": "buff.counters",
                "name": "Counters",
                "aliases": [
                    "+1/+1 counters", "counters", "counter synergy",
                    "counter build-up", "grow",
                ],
                "description": (
                    "Permanent buffs via +1/+1 counters or keyword counters "
                    "(flying counter, trample counter, etc.)."
                ),
                "oracle_hints": [
                    "+1/+1 counter",
                    "flying counter",
                    "trample counter",
                    "put a counter",
                ],
                "children": [],
            },
            {
                "key": "buff.pump",
                "name": "Pump",
                "aliases": ["pump", "pump spell", "combat trick", "giant growth"],
                "description": (
                    "Temporary power/toughness boost, usually until end of turn. "
                    "Combat tricks are pump spells cast during combat."
                ),
                "oracle_hints": [
                    "gets +",
                    "until end of turn",
                    "target creature gets +",
                ],
                "children": [],
            },
            {
                "key": "buff.keywords",
                "name": "Keyword Granting",
                "aliases": [
                    "grant haste", "give flying", "give trample",
                    "keyword soup", "keyword buff",
                ],
                "description": (
                    "Give creatures keyword abilities — haste, flying, trample, "
                    "lifelink, deathtouch, indestructible, double strike, etc."
                ),
                "oracle_hints": [
                    "gains haste",
                    "gains flying",
                    "gains trample",
                    "gains lifelink",
                    "gains deathtouch",
                    "gains indestructible",
                    "gains double strike",
                    "has flying",
                    "has haste",
                ],
                "children": [],
            },
            {
                "key": "buff.equipment_aura",
                "name": "Equipment & Auras",
                "aliases": [
                    "equipment", "aura", "voltron", "suit up",
                    "enchant creature", "equip",
                ],
                "description": (
                    "Persistent buffs through Equipment (re-attachable artifacts) "
                    "or Auras (enchantments attached to creatures). Voltron is "
                    "the strategy of loading one creature with many."
                ),
                "oracle_hints": [
                    "equip",
                    "equipped creature",
                    "enchanted creature",
                    "enchant creature",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  MANA ACCELERATION (RAMP)
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "ramp",
        "name": "Mana Acceleration",
        "aliases": ["ramp", "mana ramp", "acceleration", "mana base", "fast mana"],
        "description": (
            "Getting ahead on mana — through land tutors, mana creatures, "
            "mana rocks, Treasure tokens, cost reduction, doublers, "
            "and extra land drops."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "ramp.land_tutor",
                "name": "Land Tutor",
                "aliases": ["land ramp", "rampant growth", "cultivate", "fetch land"],
                "description": (
                    "Search your library for a land and put it onto the "
                    "battlefield (true ramp) or into your hand."
                ),
                "oracle_hints": [
                    "search your library for a land",
                    "search your library for a basic land",
                    "put it onto the battlefield tapped",
                ],
                "children": [],
            },
            {
                "key": "ramp.mana_dork",
                "name": "Mana Dork",
                "aliases": ["mana dork", "mana creature", "mana elf", "birds"],
                "description": (
                    "A creature that taps to produce mana. Usually fragile "
                    "but provides early acceleration."
                ),
                "oracle_hints": ["add {", "add one mana"],
                "children": [],
            },
            {
                "key": "ramp.mana_rock",
                "name": "Mana Rock",
                "aliases": ["mana rock", "rock", "signet", "talisman", "sol ring"],
                "description": "An artifact that taps to produce mana.",
                "oracle_hints": ["tap: add", "add one mana of any color"],
                "children": [],
            },
            {
                "key": "ramp.treasure",
                "name": "Treasure Tokens",
                "aliases": ["treasure", "treasure token", "treasure maker"],
                "description": (
                    "Create Treasure tokens — artifact tokens you sacrifice "
                    "to add one mana of any color."
                ),
                "oracle_hints": ["create a Treasure token", "create Treasure tokens"],
                "children": [],
            },
            {
                "key": "ramp.cost_reduction",
                "name": "Cost Reduction",
                "aliases": [
                    "cost reduction", "cost reducer", "affinity",
                    "convoke", "delve", "improvise",
                ],
                "description": (
                    "Make spells cheaper to cast — flat cost reduction, "
                    "Affinity, Convoke (tap creatures), Delve (exile graveyard), "
                    "Improvise (tap artifacts)."
                ),
                "oracle_hints": [
                    "costs less",
                    "cost less to cast",
                    "affinity",
                    "convoke",
                    "delve",
                    "improvise",
                ],
                "children": [],
            },
            {
                "key": "ramp.doubler",
                "name": "Mana Doubler",
                "aliases": ["mana doubler", "mana multiplier", "double mana"],
                "description": (
                    "Effects that double or multiply mana production — "
                    "'whenever a land is tapped for mana, add one additional'."
                ),
                "oracle_hints": [
                    "add one additional",
                    "add that much mana",
                    "double the mana",
                    "produce additional mana",
                ],
                "children": [],
            },
            {
                "key": "ramp.extra_land",
                "name": "Extra Land Drops",
                "aliases": [
                    "extra land", "additional land", "explore",
                    "land per turn", "extra land drop",
                ],
                "description": "Play additional lands per turn beyond the normal one.",
                "oracle_hints": [
                    "play an additional land",
                    "play two additional lands",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  EVASION
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "evasion",
        "name": "Evasion",
        "aliases": ["evasion", "unblockable", "evasive", "get through"],
        "description": (
            "Abilities that make creatures hard or impossible to block — "
            "flying, menace, trample, shadow, fear, skulk, horsemanship, "
            "or full unblockability."
        ),
        "oracle_hints": [
            "flying",
            "menace",
            "trample",
            "can't be blocked",
            "shadow",
            "fear",
            "intimidate",
            "skulk",
            "horsemanship",
        ],
        "children": [],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  GRAVEYARD STRATEGIES
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "graveyard",
        "name": "Graveyard Strategies",
        "aliases": ["graveyard", "GY", "yard", "graveyard matters"],
        "description": (
            "Using the graveyard as a resource — casting from it, filling "
            "it (self-mill), milling opponents, or hating on it."
        ),
        "oracle_hints": ["graveyard"],
        "children": [
            {
                "key": "graveyard.cast_from",
                "name": "Cast from Graveyard",
                "aliases": [
                    "flashback", "unearth", "escape", "disturb",
                    "embalm", "eternalize", "retrace", "aftermath",
                ],
                "description": (
                    "Cast or activate cards directly from the graveyard — "
                    "Flashback, Unearth, Escape, Disturb, Embalm, Eternalize."
                ),
                "oracle_hints": [
                    "flashback",
                    "unearth",
                    "escape",
                    "disturb",
                    "embalm",
                    "eternalize",
                    "cast this card from your graveyard",
                ],
                "children": [],
            },
            {
                "key": "graveyard.self_mill",
                "name": "Self-Mill",
                "aliases": ["self-mill", "self mill", "fill graveyard", "dredge"],
                "description": (
                    "Intentionally fill your own graveyard as a resource. "
                    "Dredge replaces draws with graveyard fills."
                ),
                "oracle_hints": [
                    "mill",
                    "put the top",
                    "cards from the top of your library into your graveyard",
                    "dredge",
                ],
                "children": [],
            },
            {
                "key": "graveyard.mill_opponent",
                "name": "Mill (Win Condition)",
                "aliases": ["mill", "milling", "deck out", "mill deck"],
                "description": (
                    "Put cards from an opponent's library into their graveyard "
                    "as a win condition."
                ),
                "oracle_hints": [
                    "target player mills",
                    "target opponent mills",
                    "cards from the top of their library",
                ],
                "children": [],
            },
            {
                "key": "graveyard.hate",
                "name": "Graveyard Hate",
                "aliases": ["graveyard hate", "GY hate", "grave hate", "rest in peace"],
                "description": "Exile cards from graveyards to shut down graveyard strategies.",
                "oracle_hints": [
                    "exile all cards from",
                    "exile target card from a graveyard",
                    "exile all graveyards",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  TOKEN GENERATION
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "tokens",
        "name": "Token Generation",
        "aliases": ["tokens", "token maker", "token generator", "go wide"],
        "description": (
            "Creating creature tokens, artifact tokens (Treasure, Food, "
            "Clue, Blood), or copy tokens."
        ),
        "oracle_hints": ["create", "token"],
        "children": [
            {
                "key": "tokens.creature",
                "name": "Creature Tokens",
                "aliases": ["creature tokens", "token creatures", "army in a can"],
                "description": "Create creature tokens to build a board presence.",
                "oracle_hints": ["create a", "creature token", "creature tokens"],
                "children": [],
            },
            {
                "key": "tokens.artifact",
                "name": "Artifact Tokens",
                "aliases": [
                    "treasure tokens", "food tokens", "clue tokens",
                    "blood tokens", "powerstone tokens",
                ],
                "description": (
                    "Create artifact tokens with built-in abilities — "
                    "Treasure (mana), Food (life), Clue (draw), Blood (loot), "
                    "Powerstone (colorless mana), Map (explore)."
                ),
                "oracle_hints": [
                    "Treasure token",
                    "Food token",
                    "Clue token",
                    "Blood token",
                    "Powerstone token",
                    "Map token",
                ],
                "children": [],
            },
            {
                "key": "tokens.copy",
                "name": "Copy / Clone",
                "aliases": ["clone", "copy", "token copy", "populate"],
                "description": (
                    "Create token copies of existing creatures, spells, "
                    "or permanents."
                ),
                "oracle_hints": [
                    "create a token that's a copy",
                    "create a copy",
                    "becomes a copy",
                    "copy target",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  SACRIFICE / ARISTOCRATS
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "sacrifice",
        "name": "Sacrifice / Aristocrats",
        "aliases": [
            "aristocrats", "sacrifice", "sac", "sac outlet",
            "death triggers", "sacrifice synergy",
        ],
        "description": (
            "Sacrifice-for-value strategies — sacrifice outlets that let you "
            "sacrifice creatures/permanents, and dies-triggers that generate "
            "value when things die."
        ),
        "oracle_hints": ["sacrifice a", "whenever", "dies"],
        "children": [
            {
                "key": "sacrifice.outlet",
                "name": "Sacrifice Outlet",
                "aliases": ["sac outlet", "sacrifice outlet", "free sac"],
                "description": "Abilities that let you sacrifice permanents at will.",
                "oracle_hints": [
                    "sacrifice a creature",
                    "sacrifice a permanent",
                    "sacrifice an artifact",
                ],
                "children": [],
            },
            {
                "key": "sacrifice.payoff",
                "name": "Dies Payoff",
                "aliases": [
                    "dies trigger", "death trigger", "blood artist",
                    "aristocrat payoff",
                ],
                "description": (
                    "Effects that trigger when creatures die — drain life, "
                    "draw cards, create tokens."
                ),
                "oracle_hints": [
                    "whenever a creature you control dies",
                    "whenever another creature dies",
                    "whenever a creature dies",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  FLICKER / BLINK
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "blink",
        "name": "Flicker / Blink",
        "aliases": ["blink", "flicker", "exile and return", "ETB abuse"],
        "description": (
            "Exile a permanent then return it to the battlefield to retrigger "
            "enters-the-battlefield abilities."
        ),
        "oracle_hints": [
            "exile target",
            "then return it to the battlefield",
            "return that card to the battlefield",
            "enters the battlefield",
        ],
        "children": [],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  COMBAT MANIPULATION
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "combat",
        "name": "Combat Manipulation",
        "aliases": ["combat tricks", "combat", "fog", "goad"],
        "description": (
            "Controlling how combat works — first strike, deathtouch, "
            "lifelink, vigilance, fog effects, goad, lure."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "combat.fog",
                "name": "Fog",
                "aliases": ["fog", "damage prevention", "safe passage"],
                "description": "Prevent all combat damage for a turn.",
                "oracle_hints": ["prevent all combat damage"],
                "children": [],
            },
            {
                "key": "combat.goad",
                "name": "Goad",
                "aliases": ["goad", "forced attack", "must attack"],
                "description": "Force opponents' creatures to attack someone else.",
                "oracle_hints": ["goaded", "attacks each combat if able", "must attack"],
                "children": [],
            },
            {
                "key": "combat.lure",
                "name": "Lure",
                "aliases": ["lure", "must block", "provoke"],
                "description": "Force an opponent's creature to block yours.",
                "oracle_hints": [
                    "must be blocked",
                    "blocks if able",
                    "all creatures able to block",
                ],
                "children": [],
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    #  LIFE MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────
    {
        "key": "life",
        "name": "Life Management",
        "aliases": ["lifegain", "life gain", "drain", "life loss"],
        "description": (
            "Gaining life, draining opponents, or paying life as a resource."
        ),
        "oracle_hints": [],
        "children": [
            {
                "key": "life.gain",
                "name": "Life Gain",
                "aliases": ["lifegain", "life gain", "heal", "gain life"],
                "description": "Gain life to stabilize or trigger payoffs.",
                "oracle_hints": ["gain life", "gains life"],
                "children": [],
            },
            {
                "key": "life.drain",
                "name": "Drain",
                "aliases": ["drain", "life drain", "extort"],
                "description": "Opponent loses life and you gain that much life.",
                "oracle_hints": [
                    "loses life",
                    "you gain that much life",
                    "deals damage",
                    "you gain life equal",
                ],
                "children": [],
            },
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Color group mapping — for structured query filtering
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_GROUPS: dict[str, list[str]] = {
    # Ravnica guilds (2-color pairs)
    "azorius": ["W", "U"],
    "dimir": ["U", "B"],
    "rakdos": ["B", "R"],
    "gruul": ["R", "G"],
    "selesnya": ["G", "W"],
    "orzhov": ["W", "B"],
    "izzet": ["U", "R"],
    "golgari": ["B", "G"],
    "boros": ["R", "W"],
    "simic": ["G", "U"],
    # Alara shards (3-color)
    "bant": ["W", "U", "G"],
    "esper": ["W", "U", "B"],
    "grixis": ["U", "B", "R"],
    "jund": ["B", "R", "G"],
    "naya": ["R", "G", "W"],
    # Tarkir wedges (3-color)
    "abzan": ["W", "B", "G"],
    "jeskai": ["U", "R", "W"],
    "sultai": ["B", "G", "U"],
    "mardu": ["R", "W", "B"],
    "temur": ["G", "U", "R"],
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Abbreviation expansion — for query preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

ABBREVIATIONS: dict[str, str] = {
    "ETB": "enters the battlefield",
    "LTB": "leaves the battlefield",
    "CMC": "mana value",
    "MV": "mana value",
    "P/T": "power and toughness",
    "GY": "graveyard",
    "WUBRG": "white blue black red green",
    "cEDH": "competitive commander",
    "EDH": "commander",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


def _walk(nodes: list[dict]):
    """Depth-first walk of taxonomy nodes."""
    for node in nodes:
        yield node
        yield from _walk(node.get("children", []))


def get_node(key: str) -> dict | None:
    """Look up a taxonomy node by its dotted key."""
    for node in _walk(TAXONOMY):
        if node["key"] == key:
            return node
    return None


def get_all_aliases() -> dict[str, str]:
    """
    Build a flat dict mapping every alias (lowercased) to its node key.

    Returns
    -------
    dict[str, str] — e.g. {"board wipe": "interaction.mass_removal", ...}
    """
    result = {}
    for node in _walk(TAXONOMY):
        for alias in node.get("aliases", []):
            result[alias.lower()] = node["key"]
    return result


def get_oracle_hints(key: str) -> list[str]:
    """
    Collect oracle_hints for a node AND all its descendants.

    Useful for query expansion: if a user says "ramp", gather all
    oracle text fragments from every sub-type of ramp.
    """
    node = get_node(key)
    if not node:
        return []

    hints = list(node.get("oracle_hints", []))
    for child in _walk(node.get("children", [])):
        hints.extend(child.get("oracle_hints", []))
    return hints


def get_descriptions(key: str) -> list[str]:
    """
    Collect descriptions for a node AND all its descendants.

    Useful for generating embeddable reference chunks.
    """
    node = get_node(key)
    if not node:
        return []

    descs = [node["description"]]
    for child in _walk(node.get("children", [])):
        descs.append(child["description"])
    return descs
