"""Relevance scoring engine.

Scores papers against keyword groups, author profiles, and optional
LLM-based relevance justifications.
"""

from __future__ import annotations

import re
import os
import sys
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
    scored = [p for p in papers if p.matched_groups and p.relevance_score >= cfg.output.min_score]
    scored.sort(key=lambda p: p.relevance_score, reverse=True)
    return scored


def _llm_complete(prompt: str, cfg: Config) -> str:
    """Send a prompt to the configured LLM provider and return the text response."""
    provider = cfg.llm.provider
    api_key = cfg.llm.api_key or os.environ.get(
        "LITSEARCH_ANTHROPIC_API_KEY" if provider == "claude" else "LITSEARCH_OPENAI_API_KEY",
        "",
    )

    if provider == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("LLM warning: 'anthropic' package not installed. Run: uv pip install -e '.[llm]'", file=sys.stderr)
            return ""
        if not api_key:
            print("LLM warning: llm.enabled is true but no api_key set (LITSEARCH_ANTHROPIC_API_KEY or config)", file=sys.stderr)
            return ""
        try:
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=cfg.llm.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text or ""
        except Exception as e:
            print(f"LLM error ({type(e).__name__}): {e}", file=sys.stderr)
            return ""

    elif provider == "local":
        # ponytail: urllib avoids the openai package dep for local/custom endpoints
        import json
        import urllib.request

        base_url = (cfg.llm.base_url or "http://localhost:11434/v1").rstrip("/")
        payload = json.dumps({
            "model": cfg.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            req = urllib.request.Request(f"{base_url}/chat/completions", data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"LLM error ({type(e).__name__}): {e}", file=sys.stderr)
            return ""

    else:  # "openai"
        try:
            from openai import OpenAI
        except ImportError:
            print("LLM warning: 'openai' package not installed. Run: uv pip install -e '.[llm]'", file=sys.stderr)
            return ""
        if not api_key:
            print("LLM warning: llm.enabled is true but no api_key set (LITSEARCH_OPENAI_API_KEY or config)", file=sys.stderr)
            return ""
        client_kwargs: dict = {"api_key": api_key}
        if cfg.llm.base_url:
            client_kwargs["base_url"] = cfg.llm.base_url
        try:
            client = OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=cfg.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM error ({type(e).__name__}): {e}", file=sys.stderr)
            return ""


def generate_report_summary(papers: list[Paper], cfg: Config) -> str:
    """Generate a 2-4 sentence digest of all hits for the day."""
    lines = []
    for i, p in enumerate(papers, 1):
        snippet = p.abstract[:150].rstrip()
        lines.append(f"{i}. {p.title} ({', '.join(p.matched_groups)}): {snippet}")

    prompt = (
        f"You are a research assistant briefing a scientist in {cfg.profile.field}.\n"
        f"Here are today's {len(papers)} matched papers:\n\n"
        f"{chr(10).join(lines)}\n\n"
        f"Write 2-4 sentences summarising what matters today. Call out any genuinely "
        f"important papers by name and finding. If nothing stands out, say so directly. "
        f"Do not pad — if today is a slow day, one sentence is fine."
    )
    return _llm_complete(prompt, cfg)


def generate_relevance_reasons(papers: list[Paper], cfg: Config) -> None:
    """Batch-generate relevance justifications for a list of papers (one LLM call)."""
    if not papers:
        return

    entries = []
    for i, p in enumerate(papers, 1):
        groups = ", ".join(p.matched_groups)
        entries.append(
            f"[{i}] Title: {p.title}\nInterests: {groups}\nAbstract: {p.abstract[:400]}"
        )

    prompt = (
        "You are a research literature assistant. For each numbered paper below, "
        "write exactly one line starting with its number and a period, giving 1-2 sentences "
        "explaining why it is relevant to the listed researcher interests. "
        "Be specific and concise. Do not add any other text.\n\n"
        + "\n\n".join(entries)
    )

    raw = _llm_complete(prompt, cfg)
    if not raw:
        return

    # Parse "N. <reason>" lines — ponytail: simple split, handles most LLM outputs
    import re as _re
    for line in raw.splitlines():
        m = _re.match(r"^\[?(\d+)\]?[.)]\s+(.+)", line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(papers):
                papers[idx].relevance_reason = m.group(2).strip()
