"""Configuration management for litsearch.

Reads and validates litsearch.toml. Provides the canonical Config dataclass
used throughout the package.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── data classes ────────────────────────────────────────────────────────────

@dataclass
class Profile:
    """The researcher this digest is being run for."""

    field: str = ""


@dataclass
class KeywordGroup:
    """A research topic: terms to match, how much it counts, and an optional gate."""

    label: str
    terms: list[str] = field(default_factory=list)
    weight: int = 1
    must_have: list[str] = field(default_factory=list)


@dataclass
class Author:
    """An author to track; matching papers get a relevance boost."""

    name: str
    priority: str = "normal"  # high | medium | normal
    reason: str = ""


@dataclass
class Output:
    """Report format and filtering options."""

    format: str = "markdown"   # html | markdown
    theme: str = "litsearch"   # litsearch | none
    max_highlights: int = 20  # 0 = unlimited
    min_score: float = 0.0
    dir: str = ""             # empty = CWD


@dataclass
class Sources:
    """Which literature sources to query."""

    pubmed: bool = True
    lookback_days: int = 1


@dataclass
class LLMConfig:
    """Optional LLM provider settings for the report summary."""

    enabled: bool = False
    provider: str = "openai"  # openai | claude | local
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""


@dataclass
class Schedule:
    """Daily run time for `litsearch schedule` (system local time)."""

    time: str = "08:00"
    enabled: bool = False


@dataclass
class Config:
    """The fully-loaded contents of litsearch.toml."""

    profile: Profile = field(default_factory=Profile)
    keywords: list[KeywordGroup] = field(default_factory=list)
    authors: list[Author] = field(default_factory=list)
    output: Output = field(default_factory=Output)
    sources: Sources = field(default_factory=Sources)
    llm: LLMConfig = field(default_factory=LLMConfig)
    schedule: Schedule = field(default_factory=Schedule)


# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = """\
[profile]
# Your research field — used to brief the LLM summary, if enabled.
field = "your research field"

# ── Keyword Groups ──────────────────────────────────────────────────────
# Define your research topics. Each group gets a weight (higher = more
# important). Terms are regex-matched against title + abstract.
# `must_have` acts as a gate: a paper must match at least one must_have
# term to be included in this group.

[[keywords]]
label = "Main Topic"
terms = ["example", "keywords", "here"]
weight = 3
must_have = []   # e.g. ["membrane", "protein"]

# [[keywords]]
# label = "Secondary Topic"
# terms = ["other", "terms"]
# weight = 2

# ── Authors to Track ────────────────────────────────────────────────────
# Papers by these authors always included. Priority: high | medium | normal.

# [[authors]]
# name = "Last, FirstInitials"
# priority = "high"
# reason = "why you're tracking them"

# ── Sources ─────────────────────────────────────────────────────────────
[sources]
pubmed = true            # NCBI PubMed (free, no API key needed)
lookback_days = 1         # how many days to search

# ── Output ──────────────────────────────────────────────────────────────
[output]
format = "markdown"       # html or markdown
theme = "litsearch"       # litsearch | none  (html only)
max_highlights = 20       # top N papers in the digest; 0 = unlimited
# dir = ""                # output directory; empty = CWD (e.g. "~/notes/obsidian")

# ── LLM (optional) ──────────────────────────────────────────────────────
# If enabled, generates a brief AI summary at the top of each report.
# Requires `pip install litsearch[llm]`. Run `litsearch configure` to set up.

[llm]
enabled = false
provider = "openai"          # openai | claude | local
model = "gpt-4o-mini"        # gpt-4o-mini | claude-haiku-4-5-20251001 | your-local-model
api_key = ""                 # or set LITSEARCH_OPENAI_API_KEY / LITSEARCH_ANTHROPIC_API_KEY
base_url = ""                # local only, e.g. http://localhost:11434/v1

# ── Schedule (optional) ─────────────────────────────────────────────────
# litsearch schedule reads this section. Runs at `time` in system local time.

[schedule]
time = "08:00"
enabled = false
"""


# ── loading ────────────────────────────────────────────────────────────────

def _find_config() -> Path:
    """Return the path to litsearch.toml, searching upward from CWD."""
    candidate = Path.cwd()
    for _ in range(20):
        cfg = candidate / "litsearch.toml"
        if cfg.exists():
            return cfg
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    raise FileNotFoundError(
        "litsearch.toml not found. Run 'litsearch init' to create one."
    )


def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate litsearch.toml, returning a Config."""
    cfg_path = path or _find_config()
    raw = tomllib.loads(cfg_path.read_text())

    return Config(
        profile=Profile(**raw.get("profile", {})),
        keywords=[KeywordGroup(**kw) for kw in raw.get("keywords", [])],
        authors=[Author(**a) for a in raw.get("authors", [])],
        output=Output(**raw.get("output", {})),
        sources=Sources(**raw.get("sources", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        schedule=Schedule(**raw.get("schedule", {})),
    )


def write_default_config(path: Path) -> None:
    """Write a fresh litsearch.toml at the given path."""
    path.write_text(DEFAULT_CONFIG)


