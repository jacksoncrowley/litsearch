"""Relevance scoring engine.

Scores papers against keyword groups, author profiles, and optional
LLM-based relevance justifications.
"""

from __future__ import annotations

import re
import os
from typing import Optional

from litsearch.config import Config, KeywordGroup, Author
from litsearch.pubmed import Paper


def score_paper(paper: Paper, cfg: Config) -> Paper:
    """Score a paper against all keyword groups and author profiles.

    Returns the paper with matched_groups, relevance_score, and
    relevance_reason populated. Modifies paper in place but also
    returns it for chaining.
    """
    text = f"{paper.title} {paper.abstract}"
    total_score = 0.0
    matched: list[str] = []

    for kg in cfg.keywords:
        pattern = _build_pattern(kg)
        matches = len(pattern.findall(text))

        if matches == 0:
            continue

        # Must-have gate
        if kg.must_have:
            must_pattern = "|".join(re.escape(t) for t in kg.must_have)
            if not re.search(must_pattern, text, re.IGNORECASE):
                continue

        # Title bonus: matches in title count double
        title_matches = len(pattern.findall(paper.title))
        score = (title_matches * 2 + (matches - title_matches)) * kg.weight
        total_score += score
        matched.append(kg.label)

    # Author boost
    for author in cfg.authors:
        if _author_matches(author, paper.authors):
            boost = {"high": 0.5, "medium": 0.3, "normal": 0.1}.get(
                author.priority, 0.1
            )
            total_score += total_score * boost

    paper.matched_groups = matched
    paper.relevance_score = total_score
    return paper


def _build_pattern(kg: KeywordGroup) -> re.Pattern:
    """Build a compiled regex from a KeywordGroup's terms."""
    return re.compile("|".join(re.escape(t) for t in kg.terms), re.IGNORECASE)


def _author_matches(author: Author, author_string: str) -> bool:
    """Check if a tracked author appears in the paper's author list.

    Handles both 'Last, First' and 'First Last' formats.
    """
    # Parse the tracked author name
    name = author.name.strip()
    if "," in name:
        # "Last, First" format
        parts = [p.strip() for p in name.split(",", 1)]
        last = parts[0].lower()
        first = parts[1].lower() if len(parts) > 1 else ""
    else:
        # "First Last" or "First M Last" format
        parts = name.rsplit(None, 1)
        first = parts[0].lower() if len(parts) > 1 else ""
        last = parts[-1].lower()

    source = author_string.lower()

    # Must match the last name
    if last not in source:
        return False

    # If first name provided, must also match (first initial or full first name)
    if first:
        first_initial = first[0]
        # Check if first name or initial appears near the last name in source
        # PubMed format: "First Last" or "Last FI"
        if first not in source and first_initial not in source:
            return False

    return True


def score_all(papers: list[Paper], cfg: Config) -> list[Paper]:
    """Score and sort all papers by relevance."""
    for p in papers:
        score_paper(p, cfg)

    # Filter: only papers with at least one matched group
    scored = [p for p in papers if p.matched_groups]
    scored.sort(key=lambda p: p.relevance_score, reverse=True)
    return scored


def generate_relevance_reason(paper: Paper, cfg: Config) -> str:
    """Generate a 1-2 sentence relevance justification using an LLM.

    Requires `openai` package and an API key (via LITSEARCH_OPENAI_API_KEY
    or config).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return ""

    api_key = cfg.llm.api_key or os.environ.get("LITSEARCH_OPENAI_API_KEY", "")
    if not api_key:
        return ""

    client_kwargs = {"api_key": api_key}
    if cfg.llm.base_url:
        client_kwargs["base_url"] = cfg.llm.base_url

    client = OpenAI(**client_kwargs)

    groups = ", ".join(paper.matched_groups)
    prompt = (
        f"You are a research literature assistant. Given a paper and the "
        f"researcher's interests ({groups}), write 1-2 sentences explaining "
        f"why this paper is relevant. Be specific and concise.\n\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract[:800]}\n\n"
        f"Relevance:"
    )

    try:
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""
