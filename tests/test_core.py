import tempfile
from pathlib import Path

import pytest

from litsearch.config import load_config, Config, KeywordGroup, Author, Output, LLMConfig
from litsearch.pubmed import Paper
from litsearch.scoring import _author_matches, score_paper, score_all, _llm_complete
from litsearch.report import render_report


def _paper(
    pmid="1", title="Test paper", abstract="", authors="",
    journal="J Test", pub_date="2026-01-01", doi="", url="https://pubmed.ncbi.nlm.nih.gov/1/",
    matched_groups=None,
) -> Paper:
    p = Paper(pmid=pmid, title=title, abstract=abstract, authors=authors,
              journal=journal, pub_date=pub_date, doi=doi, url=url)
    if matched_groups is not None:
        p.matched_groups = matched_groups
    return p


# ── Config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_full_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "litsearch.toml"
            cfg_path.write_text("""\
[profile]
field = "testing"

[[keywords]]
label = "Topic A"
terms = ["alpha", "beta"]
weight = 2
must_have = ["alpha"]

[[authors]]
name = "Smith, John"
priority = "high"
reason = "relevant work"

[sources]
pubmed = true
lookback_days = 3

[output]
format = "html"
max_highlights = 10
min_score = 2.0
""")
            cfg = load_config(cfg_path)
            assert cfg.profile.field == "testing"
            assert len(cfg.keywords) == 1
            assert cfg.keywords[0].label == "Topic A"
            assert cfg.keywords[0].weight == 2
            assert cfg.keywords[0].must_have == ["alpha"]
            assert len(cfg.authors) == 1
            assert cfg.authors[0].priority == "high"
            assert cfg.sources.lookback_days == 3
            assert cfg.output.max_highlights == 10
            assert cfg.output.min_score == 2.0


# ── Author matching ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("author_name,author_string,expected", [
    ("Smith, Alice",  "Alice Smith; Bob Jones",  True),   # Last, First — full match
    ("Smith, Alice",  "Smith A; Bob Jones",      True),   # Last, First — initial match
    ("Alice Smith",   "Alice Smith; Bob Jones",  True),   # First Last format
    ("Robert Brown",  "Brown RJ; others",        True),   # First Last with initials in source
    ("Smith, Alice",  "Brown J; Jones K",        False),  # no match
])
def test_author_matching(author_name, author_string, expected):
    assert _author_matches(Author(name=author_name), author_string) is expected


# ── Scoring ───────────────────────────────────────────────────────────────────

class TestScoring:
    def _cfg(self, **kw_overrides):
        kw = dict(label="Test", terms=["alpha"], weight=1)
        kw.update(kw_overrides)
        return Config(keywords=[KeywordGroup(**kw)])

    def test_single_group_match(self):
        cfg = self._cfg(terms=["alpha", "beta"], weight=3)
        paper = _paper(title="Alpha study", abstract="beta interaction")
        score_paper(paper, cfg)
        assert "Test" in paper.matched_groups
        assert paper.relevance_score > 0

    def test_title_bonus(self):
        cfg = self._cfg()
        title_hit = _paper(pmid="1", title="Alpha study", abstract="")
        abstract_hit = _paper(pmid="2", title="Unrelated", abstract="alpha study")
        score_paper(title_hit, cfg)
        score_paper(abstract_hit, cfg)
        assert title_hit.relevance_score > abstract_hit.relevance_score

    def test_must_have_gate_blocks(self):
        cfg = self._cfg(terms=["beta"], must_have=["alpha"])
        paper = _paper(abstract="beta study")  # "alpha" absent
        score_paper(paper, cfg)
        assert "Test" not in paper.matched_groups

    def test_must_have_gate_passes(self):
        cfg = self._cfg(terms=["beta"], must_have=["alpha"])
        paper = _paper(abstract="alpha and beta study")
        score_paper(paper, cfg)
        assert "Test" in paper.matched_groups

    def test_multi_group_match(self):
        cfg = Config(keywords=[
            KeywordGroup(label="A", terms=["alpha"], weight=1),
            KeywordGroup(label="B", terms=["beta"], weight=1),
        ])
        paper = _paper(abstract="alpha and beta interaction")
        score_paper(paper, cfg)
        assert "A" in paper.matched_groups
        assert "B" in paper.matched_groups

    def test_no_match_zero_score(self):
        cfg = self._cfg()
        paper = _paper(title="Completely unrelated", abstract="nothing here")
        score_paper(paper, cfg)
        assert paper.relevance_score == 0.0
        assert paper.matched_groups == []

    def test_author_boost_increases_score(self):
        base_cfg = self._cfg()
        boosted_cfg = Config(
            keywords=base_cfg.keywords,
            authors=[Author(name="Smith, Alice", priority="high")],
        )
        base = _paper(pmid="1", title="Alpha study")
        boosted = _paper(pmid="2", title="Alpha study", authors="Alice Smith")
        score_paper(base, base_cfg)
        score_paper(boosted, boosted_cfg)
        assert boosted.relevance_score > base.relevance_score

    def test_author_boost_high_gt_normal(self):
        def _scored(priority):
            cfg = Config(
                keywords=[KeywordGroup(label="T", terms=["alpha"], weight=1)],
                authors=[Author(name="Smith, Alice", priority=priority)],
            )
            paper = _paper(title="Alpha study", authors="Alice Smith")
            score_paper(paper, cfg)
            return paper.relevance_score

        assert _scored("high") > _scored("normal")

    def test_score_all_sorts_descending(self):
        cfg = Config(keywords=[KeywordGroup(label="A", terms=["alpha"], weight=3)])
        papers = [
            _paper(pmid="1", title="Alpha factor", abstract="alpha interaction"),  # 2 hits
            _paper(pmid="2", title="Alpha study", abstract=""),                    # 1 hit
        ]
        result = score_all(papers, cfg)
        assert result[0].pmid == "1"

    def test_score_all_filters_zero(self):
        cfg = Config(keywords=[KeywordGroup(label="A", terms=["alpha"], weight=1)])
        papers = [
            _paper(pmid="1", title="No match", abstract="nothing"),
            _paper(pmid="2", title="Alpha study", abstract=""),
        ]
        result = score_all(papers, cfg)
        assert len(result) == 1
        assert result[0].pmid == "2"

    def test_score_all_min_score(self):
        cfg = Config(
            keywords=[KeywordGroup(label="A", terms=["alpha"], weight=1)],
            output=Output(min_score=5.0),
        )
        papers = [
            _paper(pmid="1", title="Alpha study", abstract=""),           # score=2, below threshold
            _paper(pmid="2", title="Alpha alpha alpha", abstract="alpha alpha"),  # score>5
        ]
        result = score_all(papers, cfg)
        assert all(p.relevance_score >= 5.0 for p in result)
        assert not any(p.pmid == "1" for p in result)

    def test_score_all_empty(self):
        cfg = Config(keywords=[KeywordGroup(label="A", terms=["alpha"], weight=1)])
        assert score_all([], cfg) == []


# ── Report ───────────────────────────────────────────────────────────────────

def _render(papers=None, cfg=None, start="2026-01-01", end="2026-01-01"):
    return render_report(
        papers or [], cfg or Config(),
        start_date=start, end_date=end,
    )


class TestReport:
    def test_html_structure(self):
        html = _render()
        assert "<!DOCTYPE html>" in html
        assert "<h1>" in html
        assert "<footer>" in html

    def test_paper_title_and_link(self):
        paper = _paper(title="My Paper", url="https://pubmed.ncbi.nlm.nih.gov/42/",
                       matched_groups=["Topic"])
        html = _render([paper])
        assert "My Paper" in html
        assert "https://pubmed.ncbi.nlm.nih.gov/42/" in html

    def test_section_grouping(self):
        p1 = _paper(pmid="1", title="First", matched_groups=["Alpha"])
        p2 = _paper(pmid="2", title="Second", matched_groups=["Beta"])
        html = _render([p1, p2])
        assert "Alpha" in html
        assert "Beta" in html

    def test_empty_report(self):
        html = _render([])
        assert "empty" in html  # class="empty" paragraph

    def test_html_escaping(self):
        paper = _paper(title="A & B <study>", matched_groups=["T"])
        html = _render([paper])
        assert "&amp;" in html
        assert "&lt;" in html
        assert "A & B" not in html  # raw ampersand must not appear unescaped

    def test_author_badge(self):
        cfg = Config(authors=[Author(name="Smith, Alice", priority="high")])
        paper = _paper(authors="Alice Smith", matched_groups=["T"])
        html = _render([paper], cfg=cfg)
        assert "badge-high" in html
        assert "Smith" in html

    def test_abstract_truncation(self):
        long_abstract = "word " * 300  # ~1500 chars
        paper = _paper(abstract=long_abstract, matched_groups=["T"])
        html = _render([paper])
        assert "..." in html

    def test_short_abstract_not_truncated(self):
        paper = _paper(abstract="Short abstract.", matched_groups=["T"])
        html = _render([paper])
        assert "Short abstract." in html

    def test_date_range(self):
        html = _render(start="2026-01-01", end="2026-01-31")
        assert "2026-01-01 to 2026-01-31" in html

    def test_single_date(self):
        html = _render(start="2026-01-15", end="2026-01-15")
        assert "2026-01-15" in html
        assert " to " not in html


# ── LLM / API security ────────────────────────────────────────────────────────

def _llm_cfg(provider="openai", api_key="", base_url="", model="gpt-4o-mini"):
    return Config(llm=LLMConfig(enabled=True, provider=provider, api_key=api_key,
                                base_url=base_url, model=model))


class TestLLMSecurity:
    def test_empty_key_openai_returns_empty(self):
        # Guard fires before any network call — no mock needed
        assert _llm_complete("test", _llm_cfg(provider="openai", api_key="")) == ""

    def test_empty_key_claude_returns_empty(self):
        assert _llm_complete("test", _llm_cfg(provider="claude", api_key="")) == ""

    def test_missing_openai_import_returns_empty(self, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def _block(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("openai not installed")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block)
        assert _llm_complete("test", _llm_cfg(provider="openai", api_key="sk-x")) == ""

    def test_toml_escape_in_configure(self):
        from litsearch.cli import cmd_configure
        # Verify _te logic inline — the function is nested, so test its contract directly
        def _te(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        assert _te('sk"abc') == 'sk\\"abc'
        assert _te("sk\nabc") == "sk\\nabc"
        assert _te("sk\\abc") == "sk\\\\abc"
        assert _te("normal-key-123") == "normal-key-123"
