from __future__ import annotations

import datetime
from pathlib import Path

from litsearch.config import Config
from litsearch.pubmed import Paper
from litsearch.scoring import _author_matches

_CSS = """
:root {
  --bg:#F8FAFC; --surface:#fff; --border:#E2E8F0;
  --text:#0F172A; --muted:#64748B; --accent:#2563EB; --accent-bg:#EFF6FF;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.6;padding:2rem 1rem}
.wrap{max-width:780px;margin:0 auto}

.hd{border-bottom:2px solid var(--text);padding-bottom:1rem;margin-bottom:1.5rem}
.hd-eyebrow{font:0.72rem/1 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.hd h1{font:bold 1.6rem/1.2 Georgia,serif;margin-top:.3rem}
.hd-meta{font:.78rem/1.4 ui-monospace,monospace;color:var(--muted);margin-top:.6rem;display:flex;gap:1.5rem;flex-wrap:wrap}

.summary-box{background:var(--accent-bg);border-left:3px solid var(--accent);padding:.75rem 1rem;margin-bottom:1.5rem;border-radius:0 4px 4px 0;font-style:italic;font-size:.9rem}

.toc{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:2rem}
.toc-chip{font:.72rem/1 ui-monospace,monospace;padding:.25rem .6rem;border:1px solid var(--border);border-radius:3px;color:var(--muted);background:var(--surface)}
.toc-chip b{color:var(--accent)}

.section{margin-bottom:2.5rem}
.sec-hd{display:flex;align-items:baseline;gap:.5rem;border-bottom:1px solid var(--border);padding-bottom:.4rem;margin-bottom:.8rem}
.sec-name{font:bold 1.1rem/1 Georgia,serif}
.sec-n{font:.75rem/1 ui-monospace,monospace;color:var(--muted);margin-left:auto}

.papers{list-style:none;display:flex;flex-direction:column;gap:2px}
.paper{background:var(--surface);border:1px solid var(--border);border-left-width:4px;border-left-color:var(--accent);border-radius:0 4px 4px 0;padding:.9rem 1rem .75rem}
.paper:hover{box-shadow:0 1px 6px rgba(37,99,235,.12)}
.paper-title{font:1rem/1.45 Georgia,serif;margin-bottom:.35rem}
.paper-title a{color:var(--text);text-decoration:none}
.paper-title a:hover{color:var(--accent);text-decoration:underline}
.paper-meta{font:.72rem/1.4 ui-monospace,monospace;color:var(--muted)}
.badges{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.35rem}
.badge{font:.62rem/1 ui-monospace,monospace;text-transform:uppercase;letter-spacing:.04em;padding:.15rem .45rem;border-radius:2px}
.badge-high{background:#DC2626;color:#fff}
.badge-normal,.badge-low{background:#94A3B8;color:#fff}
.reason{font-size:.85rem;font-style:italic;border-top:1px solid var(--border);margin-top:.6rem;padding-top:.5rem}
details{margin-top:.5rem}
summary{font:.72rem/1 ui-monospace,monospace;color:var(--accent);cursor:pointer}
summary:hover{text-decoration:underline}
.abstract{font-size:.82rem;color:#374151;line-height:1.65;margin-top:.4rem}

.empty{text-align:center;padding:3rem;color:var(--muted);font-family:ui-monospace,monospace}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);font:.68rem/1 ui-monospace,monospace;color:var(--muted)}
"""


def render_report(
    papers: list[Paper],
    cfg: Config,
    start_date: str,
    end_date: str,
    version: str = "0.1.0",
    summary: str = "",
) -> str:
    """Render scored papers as a single self-contained HTML report."""

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    sections: dict[str, list[Paper]] = {}
    paper_badges: dict[str, list[dict]] = {}
    for p in papers:
        sections.setdefault(p.matched_groups[0] if p.matched_groups else "Other", []).append(p)
        badges = [
            {"name": a.name.split(",")[0].strip() if "," in a.name else a.name.split()[-1], "priority": a.priority}
            for a in cfg.authors if _author_matches(a, p.authors)
        ]
        if badges:
            paper_badges[p.pmid] = badges

    max_score = max((p.relevance_score for p in papers), default=1.0) or 1.0

    def bar_style(s: float) -> str:
        pct = s / max_score
        w = max(2, round(2 + 8 * pct))
        color = next(c for t, c in [(0.75, "#1D4ED8"), (0.5, "#3B82F6"), (0.25, "#93C5FD"), (0.0, "#BFDBFE")] if pct >= t)
        return f"border-left-width:{w}px;border-left-color:{color}"

    date_str = f"{start_date} to {end_date}" if start_date != end_date else end_date
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    cwd = Path.cwd()
    try:
        cwd_display = "~/" + str(cwd.relative_to(Path.home()))
    except ValueError:
        cwd_display = cwd.name

    out = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>litsearch — {esc(date_str)}</title>",
        f"<style>{_CSS}</style>" if cfg.output.theme != "none" else "",
        "</head>",
        "<body><div class='wrap'>",
        "<header class='hd'>",
        "<div class='hd-eyebrow'>litsearch · PubMed</div>",
        f"<h1>{esc(date_str)}</h1>",
        f"<div class='hd-meta'><span>{len(papers)} papers</span><span>{esc(cwd_display)}</span></div>",
        "</header>",
    ]

    if summary:
        out.append(f"<div class='summary-box'>{esc(summary)}</div>")

    out.append("<div class='toc'>")
    for group, ps in sorted(sections.items()):
        out.append(f"<span class='toc-chip'>{esc(group)} <b>{len(ps)}</b></span>")
    out.append("</div>")

    for section_name, section_papers in sections.items():
        out.append("<section class='section'>")
        out.append(
            f"<div class='sec-hd'><span class='sec-name'>{esc(section_name)}</span>"
            f"<span class='sec-n'>{len(section_papers)} papers</span></div>"
        )
        out.append("<ul class='papers'>")
        for paper in section_papers:
            meta_parts = [esc(paper.journal)]
            if paper.pub_date:
                meta_parts.append(esc(paper.pub_date))
            if paper.doi:
                meta_parts.append(f"DOI: {esc(paper.doi)}")
            meta_parts.append(f"score: {paper.relevance_score:.1f}")

            out.append(f"<li class='paper' style='{bar_style(paper.relevance_score)}'>")
            out.append(f"<div class='paper-title'><a href='{esc(paper.url)}' target='_blank' rel='noopener'>{esc(paper.title)}</a></div>")
            out.append(f"<div class='paper-meta'>{' · '.join(meta_parts)}</div>")

            badges = paper_badges.get(paper.pmid, [])
            if badges:
                badges_html = " ".join(
                    f"<span class='badge badge-{b['priority']}'>{esc(b['name'])}</span>"
                    for b in badges
                )
                out.append(f"<div class='badges'>{badges_html}</div>")

            if paper.abstract:
                abs_text = paper.abstract[:1000] + ("..." if len(paper.abstract) > 1000 else "")
                out.append("<details><summary>Abstract</summary>")
                out.append(f"<div class='abstract'>{esc(abs_text)}</div>")
                out.append("</details>")

            out.append("</li>")
        out.append("</ul>")
        out.append("</section>")

    if not sections:
        out.append("<div class='empty'>No papers matched your keyword profile for this date range.</div>")

    out.append(f"<footer>litsearch v{esc(version)} · {now}</footer>")
    out.append("</div></body></html>")

    return "\n".join(out)
