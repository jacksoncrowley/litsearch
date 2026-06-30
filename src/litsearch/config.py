"""Configuration management for litsearch.

Reads and validates litsearch.toml. Provides the canonical Config dataclass
used throughout the package.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ── data classes ────────────────────────────────────────────────────────────

@dataclass
class Profile:
    name: str = ""
    field: str = ""


@dataclass
class KeywordGroup:
    label: str
    terms: list[str] = field(default_factory=list)
    weight: int = 1
    must_have: list[str] = field(default_factory=list)


@dataclass
class Author:
    name: str
    priority: str = "normal"  # high | medium | normal
    reason: str = ""


@dataclass
class Output:
    format: str = "html"      # html | markdown
    max_highlights: int = 20
    min_score: float = 0.0
    group_by: str = "category"  # category | relevance | date
    dir: str = ""             # empty = CWD


@dataclass
class Sources:
    pubmed: bool = True
    semantic_scholar: bool = False
    lookback_days: int = 1


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "openai"  # openai | custom
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""


@dataclass
class Schedule:
    time: str = "08:00"
    timezone: str = "UTC"
    enabled: bool = False


@dataclass
class Config:
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
# Your name and research field — used in report headers.
name = "Your Name"
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
semantic_scholar = false  # requires free API key
lookback_days = 1         # how many days to search

# ── Output ──────────────────────────────────────────────────────────────
[output]
format = "html"           # html or markdown
max_highlights = 20       # top N papers in the digest
group_by = "category"     # category | relevance | date

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
# litsearch schedule reads this section.

[schedule]
time = "08:00"
timezone = "UTC"
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


