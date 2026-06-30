# litsearch

Daily PubMed digest, personalised to your research. Define your topics and authors in a config file; litsearch fetches recent papers, scores them for relevance, and writes an HTML report.

## Install

Requires Python 3.11+.

```bash
pip install git+https://github.com/jacksoncrowley/litsearch.git
```

Or clone and install locally:

```bash
git clone https://github.com/jacksoncrowley/litsearch.git
cd litsearch
pip install -e .
```

## Quick start

```bash
litsearch init        # creates litsearch.toml in the current directory
# edit litsearch.toml — see Configure below
litsearch run         # fetches papers and writes an HTML report
```

The report is saved as `litsearch_report_YYYY-MM-DD.html` in the current directory.

## Configure

`litsearch.toml` controls everything. Open it and fill in:

### Profile

```toml
[profile]
name = "Your Name"
field = "your research field"
```

Used in the report header.

### Keyword groups

Each group is a topic you care about. Papers are scored by how many of the `terms` they match — title matches count double. `must_have` is an optional gate: the paper must match at least one of those terms to count for this group.

```toml
[[keywords]]
label = "Topic A"
terms = ["keyword one", "keyword two", "related term"]
weight = 3          # higher weight = more influence on ranking
must_have = []      # leave empty to match any paper with a term hit

[[keywords]]
label = "Topic B"
terms = ["another topic", "subtopic"]
weight = 2
must_have = ["another topic"]   # only include if this term appears
```

Add as many groups as you like.

### Authors to track

Papers by tracked authors receive a relevance boost.

```toml
[[authors]]
name = "Last, First"    # or "First Last"
priority = "high"       # high | medium | normal
reason = "optional note to yourself"
```

### Sources and output

```toml
[sources]
pubmed = true
lookback_days = 1       # how many days back to search

[output]
format = "html"         # html | markdown
max_highlights = 20     # top N papers shown in the digest
```

## Run over a custom date range

```bash
litsearch run --start-date 2026-06-01 --end-date 2026-06-30
```

## Schedule daily runs

To get a digest every morning, enable scheduling in your config:

```toml
[schedule]
enabled = true
time = "08:00"
timezone = "Europe/London"
```

Then run:

```bash
litsearch schedule
```

This installs a systemd timer (Linux) or launchd agent (macOS) that runs `litsearch run` at the configured time.

## Optional: LLM summaries

Install the extra dependency:

```bash
pip install "litsearch[llm]"
```

Then in your config:

```toml
[llm]
enabled = true
model = "gpt-4o-mini"
api_key = ""    # or set LITSEARCH_OPENAI_API_KEY in your environment
```

When enabled, the top papers get a one-sentence explanation of why they matched your profile.
