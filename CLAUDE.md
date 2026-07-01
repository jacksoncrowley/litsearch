# CLAUDE.md
`litsearch` is an agent that scans the research literature for a user's areas of choice and returns personalised, scored summaries of what's important.


## Commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_core.py::TestScoring::test_score_paper_title_bonus  # single test
uv run litsearch run                   # run search, generate report
uv run litsearch run --start-date 2026-06-01 --end-date 2026-06-30
uv run litsearch init                  # create litsearch.toml in CWD
uv run litsearch schedule              # install systemd/launchd timer
uv run litsearch unschedule            # remove systemd/launchd timer
```

LLM features require the optional extra: `uv pip install -e ".[llm]"`.

## Architecture

The pipeline is linear: **config → pubmed → scoring → report**.

- `config.py` — loads `litsearch.toml` into `Config` (dataclass tree). `load_config()` walks up from CWD to find the file and unpacks each TOML section directly into its dataclass via `**kwargs` (defaults live in the dataclasses, not in the loader).
- `pubmed.py` — hits NCBI E-utilities (no API key needed). Defines the `Paper` dataclass. `search(terms, ...)` builds a broad PubMed OR query from all keyword group terms (flattened from config), fetches XML, parses into `Paper` objects. Client-side scoring does the fine-grained filtering.
- `scoring.py` — `score_all(papers, cfg)` runs `score_paper` on each `Paper`: counts regex hits per keyword group (title hits count double), applies the `must_have` gate, then adds author-priority boosts. Returns papers with `matched_groups` and `relevance_score` populated, sorted descending, zero-score papers excluded.
- `report.py` — `render_report()` groups papers, builds author badges into a local `paper_badges` dict (keyed by pmid), then renders self-contained HTML. No intermediate `_build_html` split. Called by `cli.py:cmd_run`.
- `cli.py` — argparse entry point wiring `init` / `run` / `schedule` subcommands.

`generate_report_summary()` in `scoring.py` optionally generates a whole-report LLM summary (not per-paper) via an OpenAI-compatible API (key from `LITSEARCH_OPENAI_API_KEY` / `LITSEARCH_ANTHROPIC_API_KEY` env var or config).

Tests live in `tests/test_core.py`.
