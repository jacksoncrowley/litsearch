"""Tests for litsearch scoring and config."""

import tempfile
from pathlib import Path

from litsearch.config import load_config, write_default_config, Config, KeywordGroup, Author
from litsearch.scoring import _author_matches, score_paper, score_all
from litsearch.pubmed import Paper


def _author(name: str, priority: str = "normal") -> Author:
    return Author(name=name, priority=priority)


class TestConfig:
    def test_write_and_load_default(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "litsearch.toml"
            write_default_config(cfg_path)
            assert cfg_path.exists()
            cfg = load_config(cfg_path)
            assert isinstance(cfg, Config)
            assert cfg.sources.pubmed is True

    def test_load_minimal_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "litsearch.toml"
            cfg_path.write_text("""\
[profile]
name = "Test User"
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
""")
            cfg = load_config(cfg_path)
            assert cfg.profile.name == "Test User"
            assert len(cfg.keywords) == 1
            assert cfg.keywords[0].weight == 2
            assert cfg.keywords[0].must_have == ["alpha"]
            assert len(cfg.authors) == 1
            assert cfg.authors[0].priority == "high"
            assert cfg.sources.lookback_days == 3
            assert cfg.output.max_highlights == 10


class TestAuthorMatching:
    def test_last_comma_first(self):
        assert _author_matches(_author("Budin, Itay"), "Itay Budin; someone else")

    def test_last_comma_first_no_first_in_source(self):
        # Source only has "Budin I" — should match with initial
        assert _author_matches(_author("Budin, Itay"), "Budin I; other")

    def test_last_comma_first_no_match(self):
        assert not _author_matches(_author("Budin, Itay"), "Smith J; Jones K")

    def test_first_last_format(self):
        assert _author_matches(_author("Itay Budin"), "Itay Budin; someone")

    def test_first_last_with_initial(self):
        assert _author_matches(_author("Siewert Marrink"), "Marrink SJ; others")


class TestScoring:
    def test_score_paper_matches_single_group(self):
        cfg = Config(
            keywords=[KeywordGroup(label="Test", terms=["membrane", "bilayer"], weight=3)],
        )
        paper = Paper(
            pmid="1", title="Membrane protein study",
            abstract="We studied a bilayer system.", authors="Test A",
            journal="J Test", pub_date="2025-01-01", doi="", url="http://x"
        )
        scored = score_paper(paper, cfg)
        assert "Test" in scored.matched_groups
        assert scored.relevance_score > 0

    def test_score_paper_must_have_gate(self):
        cfg = Config(
            keywords=[KeywordGroup(
                label="Test", terms=["bilayer"], weight=3,
                must_have=["membrane"]
            )],
        )
        # Abstract has "bilayer" but not "membrane" — should NOT match
        paper = Paper(
            pmid="1", title="Some paper",
            abstract="bilayer study", authors="Test A",
            journal="J Test", pub_date="2025-01-01", doi="", url="http://x"
        )
        scored = score_paper(paper, cfg)
        assert "Test" not in scored.matched_groups

    def test_score_paper_title_bonus(self):
        cfg = Config(
            keywords=[KeywordGroup(label="Test", terms=["membrane"], weight=1)],
        )
        title_match = Paper(
            pmid="1", title="Membrane study",
            abstract="", authors="Test A",
            journal="J", pub_date="2025-01-01", doi="", url="http://x"
        )
        abstract_match = Paper(
            pmid="2", title="Some paper",
            abstract="membrane study", authors="Test A",
            journal="J", pub_date="2025-01-01", doi="", url="http://x"
        )
        score_paper(title_match, cfg)
        score_paper(abstract_match, cfg)
        # Title match should score higher (2x bonus)
        assert title_match.relevance_score > abstract_match.relevance_score

    def test_score_all_filters_and_sorts(self):
        cfg = Config(
            keywords=[
                KeywordGroup(label="A", terms=["membrane"], weight=3),
            ],
        )
        papers = [
            Paper(pmid="1", title="No match", abstract="nothing", authors="", journal="", pub_date="", doi="", url=""),
            Paper(pmid="2", title="Membrane study", abstract="something", authors="", journal="", pub_date="", doi="", url=""),
            Paper(pmid="3", title="Membrane protein", abstract="membrane channel", authors="", journal="", pub_date="", doi="", url=""),
        ]
        scored = score_all(papers, cfg)
        assert len(scored) == 2
        # PMID 3 should rank higher than 2 (more matches)
        assert scored[0].pmid == "3"
