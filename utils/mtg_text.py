"""
MTG text preprocessing — translates game notation to natural language
for embedding models.

This module is **intentionally self-contained** (stdlib only) so it can
be copied into the retrieval repository without adding dependencies.
Both repos must preprocess text identically to ensure query vectors land
in the same region of the embedding space as document vectors.

Usage (pipeline side — preprocessing card text before embedding):

    from utils.mtg_text import preprocess_oracle_text, preprocess_mana_cost

    clean = preprocess_oracle_text(
        raw_oracle_text,
        card_name="Tarmogoyf",      # optional: replaces self-references
    )
    clean_cost = preprocess_mana_cost("{2}{R}{R}")

Usage (retrieval side — preprocessing user queries that contain notation):

    from utils.mtg_text import preprocess_oracle_text

    clean_query = preprocess_oracle_text(user_input)

Design decisions
────────────────
- **Dual expansion — jargon + functional description** for every symbol
  that has a distinct game term.  {W/P} → "Phyrexian white mana (white
  mana or 2 life)".  This ensures:

    1. Jargon queries ("Phyrexian mana") match the domain term.
    2. Functional queries ("pay life instead of mana") match the
       description.
    3. The rules embeddings (which mention "Phyrexian mana" naturally)
       share vocabulary with the card embeddings, so an agent can
       cross-reference rules to self-correct.

  Benchmark results (cosine similarity, all-MiniLM-L6-v2):
    - "Phyrexian mana" query:  raw 0.14 → functional-only 0.45 → dual 0.63
    - "hybrid mana cost" query: raw 0.01 → functional-only 0.41 → dual 0.59
    - "tap to add green" query: raw 0.24 → functional-only 0.97 → dual 0.95
  The dual approach loses ~0.02 on pure functional queries but gains
  ~0.17 on jargon queries — a clear win for agentic retrieval.

- Reminder text is KEPT by default.  31% of cards include parenthesized
  rules reminders.  These add valuable context for embedding models.
  Pass strip_reminder=True to remove them.

- Self-references replaced with "this card" so embeddings generalize
  across cards with similar abilities.

- Non-ASCII normalized to ASCII equivalents where meaningful.
  Accented letters in card names are left as-is.
"""

import re


# ═══════════════════════════════════════════════════════════════════════════════
#  Symbol → natural-language expansion tables
# ═══════════════════════════════════════════════════════════════════════════════

# --- Color names for compound symbols ---
_COLOR_NAME = {
    "W": "white",
    "U": "blue",
    "B": "black",
    "R": "red",
    "G": "green",
    "C": "colorless",
}

# --- Basic mana (term = description, no dual needed) ---
_BASIC_MANA = {
    "W": "white mana",
    "U": "blue mana",
    "B": "black mana",
    "R": "red mana",
    "G": "green mana",
    "C": "colorless mana",
    "S": "snow mana",
}

# --- Generic mana (numerals) ---
_NUMBER_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
    "100": "one hundred", "1000000": "one million",
}

# --- Special mechanic symbols (dual: jargon + functional) ---
_SPECIAL = {
    "T":     "Tap (tap this permanent)",
    "Q":     "Untap (untap this permanent)",
    "E":     "energy (energy counter)",
    "TK":    "ticket (ticket counter)",
    "P":     "power marker",
    "A":     "acorn counter",
    "CHAOS": "chaos",
    "HR":    "heart",
    "PW":    "planeswalker symbol",
    "X":     "X (variable amount)",
    "½":     "half",
    "∞":     "infinity",
    # Rare / set-specific mechanic symbols (1 card each as of 2026)
    "D":     "land drop (give up a land drop to pay this cost)",
    "Z":     "multi-source mana (paid with mana from a source that could produce two or more colors)",
    "L":     "legendary mana (paid with mana from a legendary source)",
    "H":     "Phyrexian mana (paid with mana or 2 life)",
}

# --- Non-ASCII character normalization ---
_NON_ASCII = {
    "\u2014": " -- ",     # em dash (ability separator)
    "\u2022": " * ",      # bullet (modal choices)
    "\u2212": "-",         # math minus → hyphen
    "\u00BD": "1/2",       # ½
    "\u00B2": "squared",   # ²
    "\u00B9": "^1",        # ¹
    "\u00B3": "cubed",     # ³
    "\u2074": "^4",        # ⁴
    "\u2075": "^5",        # ⁵
    "\u221E": "infinity",  # ∞
    "\u03C0": "pi",        # π
    "\u221A": "sqrt",      # √
    "\u2192": "->",        # →
    "\u02E3": "^x",        # ˣ
    "\u2610": "[ ]",       # ☐ (checkbox)
    "\u2666": "<>",        # ♦
    "\uA789": ":",         # ꞉ (modifier colon)
    "\u00BA": "o",         # º
}

# --- Power/toughness non-numeric values ---
_PT_EXPAND = {
    "*": "variable",
    "1+*": "one plus variable",
    "2+*": "two plus variable",
    "*+1": "variable plus one",
    "*²": "variable squared",
    "7-*": "seven minus variable",
    "?": "unknown",
    "∞": "infinity",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Regex for {symbol} tokens — greedy non-} match inside braces.
_SYMBOL_RE = re.compile(r"\{([^}]+)\}")

# Adjacent symbols like {E}{E}{E} need separating after expansion.
_ADJACENT_SYMBOLS_RE = re.compile(r"\}\{")

# Regex for reminder text:  (text that may span lines, ending with .)
_REMINDER_RE = re.compile(r"\(([^()]*)\)")


def _expand_symbol(inner: str) -> str:
    """
    Expand the content between { and } into natural language.

    Priority order:
      1. Special mechanic symbols ({T}, {E}, {Q}, etc.) — dual expansion
      2. Basic colored mana ({W}, {U}, {B}, {R}, {G}, {C}, {S})
      3. Generic mana ({0}..{20}, {100}, {½}, {∞})
      4. Phyrexian hybrid ({W/P}, {G/W/P}, etc.) — jargon + functional
      5. Hybrid mana ({W/U}, {2/R}, etc.) — includes "hybrid" label
      6. Fallback: return the inner text as-is
    """
    # 1. Special / mechanic (already includes dual form)
    if inner in _SPECIAL:
        return _SPECIAL[inner]

    # 2. Basic colored mana (term = description)
    if inner in _BASIC_MANA:
        return _BASIC_MANA[inner]

    # 3. Generic mana (numeric)
    if inner in _NUMBER_WORD:
        return f"{_NUMBER_WORD[inner]} generic mana"

    # 4. Phyrexian mana (contains /P) — DUAL: jargon + functional
    #    {W/P}   → "Phyrexian white mana (white mana or 2 life)"
    #    {G/W/P} → "Phyrexian green-white mana (green or white mana or 2 life)"
    if "/P" in inner:
        colors = inner.replace("/P", "").split("/")
        color_names = [_COLOR_NAME.get(c, c) for c in colors]
        jargon = "Phyrexian " + "-".join(color_names) + " mana"
        functional = " or ".join(color_names) + " mana or 2 life"
        return f"{jargon} ({functional})"

    # 5. Hybrid mana (A/B) — includes "hybrid" label for searchability
    #    {W/U} → "hybrid white or blue mana"
    #    {2/U} → "hybrid two generic or blue mana"
    if "/" in inner:
        parts = inner.split("/")
        expanded = []
        for p in parts:
            if p in _COLOR_NAME:
                expanded.append(_COLOR_NAME[p])
            elif p in _NUMBER_WORD:
                expanded.append(f"{_NUMBER_WORD[p]} generic")
            else:
                expanded.append(p)
        return "hybrid " + " or ".join(expanded) + " mana"

    # 6. Fallback — unknown symbol, keep as-is
    return inner


def _expand_all_symbols(text: str) -> str:
    """Replace every {symbol} in the text with its natural-language form."""
    # Insert a comma+space between adjacent symbols: {E}{E} → {E}, {E}
    text = _ADJACENT_SYMBOLS_RE.sub("}, {", text)
    return _SYMBOL_RE.sub(lambda m: _expand_symbol(m.group(1)), text)


def _normalize_non_ascii(text: str) -> str:
    """Replace known non-ASCII characters with ASCII equivalents."""
    for char, replacement in _NON_ASCII.items():
        if char in text:
            text = text.replace(char, replacement)
    return text


def _replace_self_references(text: str, card_name: str) -> str:
    """
    Replace literal card-name self-references with a generic token.

    "Lightning Bolt deals 3 damage" → "this card deals 3 damage"

    For multi-faced cards with " // " in the name, we also try each face
    name individually (e.g., "Fire // Ice" → try "Fire" and "Ice").
    """
    if not card_name:
        return text

    # Full name first (handles "Nicol Bolas, the Ravager" etc.)
    text = text.replace(card_name, "this card")

    # Individual face names for split/transform/adventure cards
    if " // " in card_name:
        for face in card_name.split(" // "):
            face = face.strip()
            if len(face) > 2:  # avoid replacing single-letter names
                text = text.replace(face, "this card")

    return text


def _strip_reminder_text(text: str) -> str:
    """Remove parenthesized reminder text."""
    return _REMINDER_RE.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """Normalize runs of whitespace to single spaces; strip edges."""
    return re.sub(r"[ \t]+", " ", text).strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════


def preprocess_oracle_text(
    text: str,
    *,
    card_name: str | None = None,
    strip_reminder: bool = False,
) -> str:
    """
    Preprocess oracle text for embedding.

    Steps (in order):
      1. Optionally strip reminder text (parenthesized)
      2. Expand all {symbol} notation to natural language (dual form)
      3. Normalize non-ASCII characters
      4. Replace card self-references with "this card"
      5. Collapse whitespace

    Parameters
    ----------
    text : str
        Raw oracle text from the cards table.
    card_name : str, optional
        If provided, self-references are replaced with "this card".
    strip_reminder : bool
        If True, remove parenthesized reminder text.  Default False.

    Returns
    -------
    str — cleaned text ready for the embedding model.
    """
    if not text:
        return ""

    if strip_reminder:
        text = _strip_reminder_text(text)

    text = _expand_all_symbols(text)
    text = _normalize_non_ascii(text)

    if card_name:
        text = _replace_self_references(text, card_name)

    return _collapse_whitespace(text)


def preprocess_mana_cost(mana_cost: str) -> str:
    """
    Expand a mana cost string for embedding.

    "{2}{R}{R}" → "two generic mana, red mana, red mana"
    "{U/P}"     → "Phyrexian blue mana (blue mana or 2 life)"
    """
    if not mana_cost:
        return ""

    # Separate adjacent symbols before expanding
    separated = _ADJACENT_SYMBOLS_RE.sub("}, {", mana_cost)
    symbols = _SYMBOL_RE.findall(separated)
    parts = [_expand_symbol(s) for s in symbols]
    return ", ".join(parts)


def preprocess_power_toughness(
    power: str | None,
    toughness: str | None,
) -> str:
    """
    Expand non-numeric power/toughness for embedding.

    ("*", "1+*") → "variable power / one plus variable toughness"
    ("3", "4")   → "3/4"  (numeric passes through)
    """
    if power is None and toughness is None:
        return ""

    def _expand_pt(val: str | None, label: str) -> str:
        if val is None:
            return ""
        if val in _PT_EXPAND:
            return f"{_PT_EXPAND[val]} {label}"
        # Fractional values (Un-set)
        try:
            float(val)
            return val  # numeric, no label needed
        except (ValueError, TypeError):
            # Modifiers like +1, -1 (Vanguard)
            return f"{val} {label}"

    p = _expand_pt(power, "power")
    t = _expand_pt(toughness, "toughness")

    # If both are simple numeric, use compact P/T format
    try:
        float(power or "")
        float(toughness or "")
        return f"{power}/{toughness}"
    except (ValueError, TypeError):
        parts = [x for x in [p, t] if x]
        return " / ".join(parts)


def preprocess_loyalty(loyalty: str | None) -> str:
    """Expand non-standard loyalty values."""
    if loyalty is None:
        return ""
    if loyalty in ("*", "X"):
        return "variable loyalty"
    return f"Loyalty: {loyalty}"
