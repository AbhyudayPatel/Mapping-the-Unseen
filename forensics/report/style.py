"""Shared visual language for the report: the epistemic/confidence -> (colour, style)
mapping reused by the diagrams, the charts and the HTML.

Everything here is a *pure function of run data*. Colours come from the validated
colourblind-safe categorical palette (see the dataviz skill's ``references/palette.md``);
the same eight hues pass adjacent-pair CVD separation in both light and dark modes.

The core contract other modules rely on::

    color, line_style = style_for(level, confidence)

``level`` is one of the epistemic labels (``observed``/``inferred``/``estimated``/
``speculative``) or ``"unknown"``/``"non_identifiable"``; ``confidence`` is an optional
0..1 number (posterior mass, or an identifiability-derived confidence). ``color`` is a
light-mode hex string; ``line_style`` is one of ``solid``/``dashed``/``dotted``.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- palette
# Validated categorical palette (dataviz reference instance). Light + dark steps of the
# same eight hues; both modes clear the adjacent-pair CVD gate.
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)

# Single-hue sequential blue ramp (light -> dark) for magnitude encodings (heatmaps).
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

# Neutral ink that reads acceptably on BOTH a light and a dark surface (used for
# matplotlib axes/text so a single PNG works in either theme).
NEUTRAL_INK = "#6b6b6b"
GRID_INK = "#9a9a9a"


@dataclass(frozen=True)
class LevelStyle:
    """Colour + line style + a short human label for one epistemic level."""
    key: str
    label: str
    color: str        # light mode
    dark: str         # dark mode
    line: str         # solid | dashed | dotted


# The heart of the visual language: epistemic level -> colour + line style.
# observed  = green solid   (read straight off the wire)
# inferred  = blue solid    (strong testable argument)
# estimated = yellow dashed (statistical estimate + CI)
# speculative = red dotted  (plausible under assumptions only)
# unknown / non-identifiable = grey dashed (no observable trace)
_LEVELS: dict[str, LevelStyle] = {
    "observed":   LevelStyle("observed", "observed", "#008300", "#00a300", "solid"),
    "inferred":   LevelStyle("inferred", "inferred", "#2a78d6", "#3987e5", "solid"),
    "estimated":  LevelStyle("estimated", "estimated", "#eda100", "#c98500", "dashed"),
    "speculative": LevelStyle("speculative", "speculative", "#e34948", "#e66767", "dotted"),
    "unknown":    LevelStyle("unknown", "unresolved (U)", "#8a8a8a", "#9a9a9a", "dashed"),
}
# aliases -> canonical level key
_ALIASES = {
    "": "unknown",
    None: "unknown",
    "non_identifiable": "unknown",
    "non-identifiable": "unknown",
    "not_identifiable": "unknown",
    "assumed": "speculative",
}

# The five identifiability levels (1..5) -> a representative epistemic style + label.
IDENT_LABELS: dict[int, str] = {
    1: "directly observable",
    2: "statistically identifiable",
    3: "weakly identifiable",
    4: "identifiable only under assumptions",
    5: "unresolved by this estimator/interface (legacy level 5)",
}
_IDENT_TO_LEVEL: dict[int, str] = {
    1: "observed", 2: "inferred", 3: "estimated", 4: "speculative", 5: "unknown",
}


def canonical_level(level: str | None) -> str:
    """Map any level-ish string to one of the five canonical style keys."""
    if level in _ALIASES:
        return _ALIASES[level]
    lv = str(level).strip().lower()
    if lv in _LEVELS:
        return lv
    return _ALIASES.get(lv, "unknown")


def level_style(level: str | None) -> LevelStyle:
    return _LEVELS[canonical_level(level)]


def style_for(level: str | None, confidence: float | None = None) -> tuple[str, str]:
    """Pure map ``(level, confidence) -> (color, line_style)`` used everywhere.

    ``color`` is a light-mode hex string, ``line_style`` is ``solid|dashed|dotted``.
    Very low confidence (< 0.34) downgrades a solid line to dashed so uncertain nodes
    read as uncertain even when their level is nominally strong.
    """
    st = level_style(level)
    line = st.line
    if confidence is not None and confidence < 0.34 and line == "solid":
        line = "dashed"
    return st.color, line


def dark_color_for(level: str | None) -> str:
    return level_style(level).dark


def ident_to_level(identifiability: int | None) -> str:
    try:
        return _IDENT_TO_LEVEL.get(int(identifiability), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def ident_confidence(identifiability: int | None) -> float:
    """Turn an identifiability level (1 best .. 5 non-identifiable) into a 0..1 confidence."""
    try:
        i = int(identifiability)
    except (TypeError, ValueError):
        return 0.3
    return {1: 0.95, 2: 0.82, 3: 0.6, 4: 0.4, 5: 0.2}.get(i, 0.3)


def confidence_marker(confidence: float | None) -> str:
    """A three-dot glyph summarising confidence (accessible: also has a text label)."""
    if confidence is None:
        return "○○○"        # ○○○
    if confidence >= 0.8:
        return "●●●"        # ●●●
    if confidence >= 0.55:
        return "●●○"        # ●●○
    if confidence >= 0.3:
        return "●○○"        # ●○○
    return "○○○"            # ○○○


def confidence_word(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.55:
        return "medium"
    if confidence >= 0.3:
        return "low"
    return "very low"


def legend_rows() -> list[dict[str, str]]:
    """Rows describing the colour/line-style -> confidence + epistemic-level mapping,
    reused verbatim by the Markdown and HTML legends."""
    rows: list[dict[str, str]] = []
    for key in ("observed", "inferred", "estimated", "speculative", "unknown"):
        st = _LEVELS[key]
        rows.append({
            "level": st.label,
            "color": st.color,
            "dark": st.dark,
            "line": st.line,
            "meaning": _LEVEL_MEANING[key],
        })
    return rows


_LEVEL_MEANING = {
    "observed": "read directly off the interface (error text, usage counts, reported fields)",
    "inferred": "follows from observations by a strong, testable argument",
    "estimated": "a statistical estimate with a model and confidence interval",
    "speculative": "plausible only under stated modelling assumptions",
    "unknown": "unresolved; formal non-identifiability requires a separately scoped equivalence witness",
}
