"""CLI entry point for litsearch.

Commands:
    litsearch init       Create litsearch.toml in current directory
    litsearch run        Run search and generate report
    litsearch refine     Update keywords/authors from a plain-English description
    litsearch schedule   Set up systemd/launchd timer (Linux/macOS)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from dataclasses import asdict, fields as dc_fields
from pathlib import Path

from litsearch import __version__
from litsearch.config import Config, KeywordGroup, Author, load_config, write_default_config, _find_config
from litsearch.pubmed import search, Paper
from litsearch.scoring import score_all, generate_report_summary, _llm_complete
from litsearch.report import render_report


def cmd_init(args: argparse.Namespace) -> None:
    """Create a default litsearch.toml in the current directory."""
    target = Path(args.dir or ".") / "litsearch.toml"
    if target.exists() and not args.force:
        print(f"{target} already exists. Use --force to overwrite.")
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_default_config(target)
    print(f"Created {target}")
    print("Edit it with your keywords, authors, and preferences, then run:")
    print("  litsearch run")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a literature search and generate a report."""
    cfg = load_config()
    if args.api_key:
        cfg.llm.api_key = args.api_key
    if args.base_url:
        cfg.llm.base_url = args.base_url

    # Date handling
    end_date = args.end_date or datetime.date.today().isoformat()
    start_date = args.start_date or (
        datetime.date.today() - datetime.timedelta(days=cfg.sources.lookback_days)
    ).isoformat()

    # Search
    papers: list[Paper] = []
    if cfg.sources.pubmed:
        terms = [t for kg in cfg.keywords for t in kg.terms]
        papers = search(terms, start_date, end_date, lookback_days=cfg.sources.lookback_days)

    if not papers:
        print("No papers found.")
        return

    # Score
    print(f"Scoring {len(papers)} papers...")
    scored = score_all(papers, cfg)
    print(f"  {len(scored)} papers matched your keyword profile.")
    if cfg.output.max_highlights:
        scored = scored[: cfg.output.max_highlights]

    # Global AI summary
    summary = ""
    if cfg.llm.enabled and scored:
        print("Generating report summary...")
        summary = generate_report_summary(scored, cfg)

    # Render
    output_dir = next((Path(x) for x in (args.output_dir, cfg.output.dir) if x), Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output.format == "html":
        report_path = output_dir / f"litsearch_report_{end_date}.html"
        html = render_report(scored, cfg, start_date, end_date, version=__version__, summary=summary)
        report_path.write_text(html)
        print(f"\nReport saved: {report_path}")
    else:
        report_path = output_dir / f"litsearch_report_{end_date}.md"
        md = _render_markdown(scored, cfg, start_date, end_date, summary=summary)
        report_path.write_text(md)
        print(f"\nReport saved: {report_path}")


def _render_markdown(
    papers: list[Paper], cfg: Config, start_date: str, end_date: str, summary: str = ""
) -> str:
    """Simple markdown output (fallback)."""
    lines = [
        f"# litsearch Report: {start_date} to {end_date}",
        "",
        f"**Papers scanned:** {len(papers)}",
        f"**Source:** PubMed",
        "",
        "---",
    ]
    if summary:
        lines += ["", f"*{summary}*", "", "---"]

    # Group by first matched category
    sections: dict[str, list[Paper]] = {}
    for p in papers:
        group = p.matched_groups[0] if p.matched_groups else "Other"
        sections.setdefault(group, []).append(p)

    for section_name, section_papers in sections.items():
        lines.append(f"\n## {section_name} ({len(section_papers)})")
        lines.append("")
        for i, p in enumerate(section_papers, 1):
            lines.append(f"### {i}. {p.title}")
            lines.append(f"- **PMID:** [{p.pmid}]({p.url})")
            if p.doi:
                lines.append(f"- **DOI:** {p.doi}")
            lines.append(f"- **Score:** {p.relevance_score} | **Journal:** {p.journal} | **Date:** {p.pub_date}")
            authors = p.authors
            if len(authors) > 200:
                authors = authors[:200] + "..."
            lines.append(f"- **Authors:** {authors}")
            lines.append("")

    lines.append(f"\n*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


def _te(s: str) -> str:  # TOML-escape a basic string value
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def cmd_configure(_args: argparse.Namespace) -> None:
    """Interactive wizard to configure the LLM provider."""
    try:
        cfg_path = _find_config()
    except FileNotFoundError:
        print("No litsearch.toml found. Run 'litsearch init' first.")
        sys.exit(1)

    if _args.provider or _args.api_key or _args.base_url or _args.model:
        provider = _args.provider or "openai"
        api_key = _args.api_key or ""
        base_url = _args.base_url or ""
        default_model = {"claude": "claude-haiku-4-5-20251001", "local": "llama3"}.get(provider, "gpt-4o-mini")
        model = _args.model or default_model
    else:
        print("Which AI provider would you like to use?")
        print("  [1] OpenAI  (e.g. gpt-4o-mini)")
        print("  [2] Claude  (Anthropic)")
        print("  [3] Local   (Ollama / llama.cpp)")
        choice = input("Choice [1]: ").strip() or "1"

        if choice == "2":
            provider = "claude"
            default_model = "claude-haiku-4-5-20251001"
            api_key = input("Anthropic API key: ").strip()
            model = input(f"Model [{default_model}]: ").strip() or default_model
            base_url = ""
        elif choice == "3":
            provider = "local"
            default_url = "http://localhost:11434/v1"
            base_url = input(f"Base URL [{default_url}]: ").strip() or default_url
            model = input("Model name: ").strip()
            api_key = ""
        else:
            provider = "openai"
            default_model = "gpt-4o-mini"
            api_key = input("OpenAI API key: ").strip()
            model = input(f"Model [{default_model}]: ").strip() or default_model
            base_url = ""

    new_llm = f"[llm]\nenabled = true\nprovider = \"{_te(provider)}\"\nmodel = \"{_te(model)}\"\napi_key = \"{_te(api_key)}\"\nbase_url = \"{_te(base_url)}\"\n"
    cfg_path.write_text(re.sub(r"\[llm\].*?(?=\n\[|\Z)", new_llm, cfg_path.read_text(), flags=re.DOTALL))
    print(f"\nUpdated {cfg_path.name} — AI summaries now use {provider} / {model}.")
    if api_key:
        print("Note: API key stored in plaintext. Prefer LITSEARCH_OPENAI_API_KEY / LITSEARCH_ANTHROPIC_API_KEY env vars.")
    print("Run 'litsearch run' to try it.")


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM response, tolerating ```-fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|```\s*$", "", text, flags=re.M).strip()
    return json.loads(text)


def _coerce(cls, d: dict):
    """Build a dataclass instance from a dict, ignoring unknown keys."""
    known = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def _dump_keywords(groups: list[KeywordGroup]) -> str:
    parts = []
    for kg in groups:
        terms = ", ".join(f'"{_te(t)}"' for t in kg.terms)
        must_have = ", ".join(f'"{_te(t)}"' for t in kg.must_have)
        parts.append(
            f'[[keywords]]\nlabel = "{_te(kg.label)}"\nterms = [{terms}]\n'
            f"weight = {kg.weight}\nmust_have = [{must_have}]\n"
        )
    return "\n".join(parts)


def _dump_authors(authors: list[Author]) -> str:
    parts = []
    for a in authors:
        parts.append(
            f'[[authors]]\nname = "{_te(a.name)}"\npriority = "{_te(a.priority)}"\n'
            f'reason = "{_te(a.reason)}"\n'
        )
    return "\n".join(parts)


def _replace_array_table(content: str, name: str, new_block: str) -> str:
    """Replace all `[[name]]` tables in content with new_block.

    Inserts before `[sources]` (or appends) if the table isn't present yet.
    """
    pattern = re.compile(rf"(?:^\[\[{name}\]\]\n(?:(?!^\[).)*)+", re.M | re.S)
    if pattern.search(content):
        return pattern.sub(lambda _m: new_block, content, count=1)
    if not new_block:
        return content
    if "[sources]" in content:
        return content.replace("[sources]", new_block + "\n[sources]", 1)
    return content.rstrip("\n") + "\n\n" + new_block


def cmd_refine(args: argparse.Namespace) -> None:
    """Update keyword groups and tracked authors from a plain-English description."""
    cfg_path = _find_config()
    cfg = load_config(cfg_path)

    instructions = " ".join(args.instructions).strip()
    if not instructions:
        instructions = input("Describe the changes: ").strip()
    if not instructions:
        print("No instructions given, nothing to do.")
        return

    current = {
        "keywords": [asdict(kg) for kg in cfg.keywords],
        "authors": [asdict(a) for a in cfg.authors],
    }
    prompt = (
        "You maintain a researcher's literature-search config.\n"
        f"Current keyword groups and tracked authors (JSON):\n{json.dumps(current, indent=2)}\n\n"
        f'The researcher says: "{instructions}"\n\n'
        'Return ONLY a JSON object with the COMPLETE updated "keywords" and "authors" '
        "lists (include unchanged groups/authors as-is; add, edit, or remove per the "
        "instructions). Schema: keywords[].{label:str, terms:[str], weight:int, "
        'must_have:[str]}, authors[].{name:str, priority:"high"|"medium"|"normal", '
        "reason:str}. No prose, no markdown fences, JSON only."
    )

    raw = _llm_complete(prompt, cfg, max_tokens=2000)
    if not raw:
        sys.exit(1)

    try:
        data = _extract_json(raw)
        new_keywords = [_coerce(KeywordGroup, kg) for kg in data["keywords"]]
        new_authors = [_coerce(Author, a) for a in data.get("authors", [])]
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Could not parse LLM response ({type(e).__name__}: {e}):\n{raw}", file=sys.stderr)
        sys.exit(1)

    print("\nProposed keyword groups:")
    for kg in new_keywords:
        print(f"  - {kg.label} ({len(kg.terms)} terms, weight={kg.weight})")
    print("Proposed tracked authors:")
    for a in new_authors:
        print(f"  - {a.name} ({a.priority})")

    if input("\nApply these changes to litsearch.toml? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Aborted — no changes made.")
        return

    text = cfg_path.read_text()
    text = _replace_array_table(text, "keywords", _dump_keywords(new_keywords))
    text = _replace_array_table(text, "authors", _dump_authors(new_authors))
    cfg_path.write_text(text)
    print(f"Updated {cfg_path.name}.")


def cmd_schedule(args: argparse.Namespace) -> None:
    """Set up scheduled runs via systemd (Linux) or launchd (macOS)."""
    cfg = load_config()

    if not cfg.schedule.enabled and not args.force:
        print("Schedule not enabled in config. Set [schedule] enabled = true or use --force.")
        sys.exit(1)

    platform = sys.platform
    if platform.startswith("linux"):
        _install_systemd(cfg)
    elif platform == "darwin":
        _install_launchd(cfg)
    else:
        print(f"Automatic scheduling not supported on {platform}.")
        print("Use cron instead — add this line to your crontab:")
        h, m = cfg.schedule.time.split(":")
        print(f"  {m} {h} * * * cd {Path.cwd()} && litsearch run")
        sys.exit(1)


def _install_systemd(cfg: Config) -> None:
    """Install a systemd user timer."""
    service_dir = Path.home() / ".config/systemd/user"
    service_dir.mkdir(parents=True, exist_ok=True)

    time_parts = cfg.schedule.time.split(":")
    hour, minute = time_parts[0], time_parts[1] if len(time_parts) > 1 else "00"

    cwd = Path.cwd()
    python = sys.executable

    service = f"""[Unit]
Description=litsearch daily literature search

[Service]
Type=oneshot
ExecStart={python} -m litsearch run
WorkingDirectory={cwd}
"""

    timer = f"""[Unit]
Description=litsearch daily timer

[Timer]
OnCalendar=*-*-* {hour}:{minute}:00
Persistent=true

[Install]
WantedBy=timers.target
"""

    service_path = service_dir / "litsearch.service"
    timer_path = service_dir / "litsearch.timer"

    service_path.write_text(service)
    timer_path.write_text(timer)

    os.system("systemctl --user daemon-reload")
    os.system("systemctl --user enable litsearch.timer")
    os.system("systemctl --user start litsearch.timer")

    print(f"Installed systemd timer: daily at {cfg.schedule.time}")
    print(f"  Service: {service_path}")
    print(f"  Timer:   {timer_path}")
    print("Check status: systemctl --user status litsearch.timer")


def _install_launchd(cfg: Config) -> None:
    """Install a launchd agent (macOS)."""
    time_parts = cfg.schedule.time.split(":")
    hour, minute = int(time_parts[0]), int(time_parts[1]) if len(time_parts) > 1 else 0

    agents_dir = Path.home() / "Library/LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    python = sys.executable

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.litsearch.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>litsearch</string>
        <string>run</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{cwd}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{agents_dir}/litsearch.log</string>
    <key>StandardErrorPath</key>
    <string>{agents_dir}/litsearch.err</string>
</dict>
</plist>"""

    plist_path = agents_dir / "com.litsearch.daily.plist"
    plist_path.write_text(plist)

    os.system(f"launchctl load {plist_path}")

    print(f"Installed launchd agent: daily at {cfg.schedule.time}")
    print(f"  Plist: {plist_path}")
    print("Check status: launchctl list com.litsearch.daily")


def cmd_unschedule(_args: argparse.Namespace) -> None:
    """Remove the scheduled litsearch timer."""
    if sys.platform.startswith("linux"):
        _remove_systemd()
    elif sys.platform == "darwin":
        _remove_launchd()
    else:
        print("Nothing to remove — scheduling is not managed on this platform.")
        sys.exit(1)


def _remove_systemd() -> None:
    service_dir = Path.home() / ".config/systemd/user"
    os.system("systemctl --user stop litsearch.timer 2>/dev/null")
    os.system("systemctl --user disable litsearch.timer 2>/dev/null")
    for name in ("litsearch.timer", "litsearch.service"):
        p = service_dir / name
        if p.exists():
            p.unlink()
            print(f"Removed {p}")
    os.system("systemctl --user daemon-reload")
    print("litsearch timer removed.")


def _remove_launchd() -> None:
    plist_path = Path.home() / "Library/LaunchAgents/com.litsearch.daily.plist"
    os.system(f"launchctl unload {plist_path} 2>/dev/null")
    if plist_path.exists():
        plist_path.unlink()
        print(f"Removed {plist_path}")
    print("litsearch agent removed.")


def main() -> None:
    """Parse CLI args and dispatch to the matching `cmd_*` handler."""
    parser = argparse.ArgumentParser(
        description="litsearch — personalised literature search",
        prog="litsearch",
    )
    parser.add_argument(
        "--version", action="version", version=f"litsearch {__version__}"
    )

    sub = parser.add_subparsers(dest="command", title="commands")

    # init
    p_init = sub.add_parser("init", help="Create litsearch.toml")
    p_init.add_argument("--dir", help="Target directory (default: current)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing")

    # run
    p_run = sub.add_parser("run", help="Run search and generate report")
    p_run.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    p_run.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    p_run.add_argument("--output-dir", help="Directory to write the report (overrides config)")
    p_run.add_argument("--api-key", help="LLM API key (overrides config and env var)")
    p_run.add_argument("--base-url", help="LLM base URL (overrides config, e.g. http://localhost:11434/v1)")

    # configure
    p_configure = sub.add_parser("configure", help="Set up AI provider and API key")
    p_configure.add_argument("--provider", choices=["openai", "claude", "local"], help="LLM provider")
    p_configure.add_argument("--api-key", help="API key")
    p_configure.add_argument("--base-url", help="Base URL (for local/custom endpoints)")
    p_configure.add_argument("--model", help="Model name")

    # refine
    p_refine = sub.add_parser("refine", help="Update keywords/authors from a plain-English description")
    p_refine.add_argument(
        "instructions", nargs="*",
        help="e.g. 'add more on cryo-EM, drop the MD Simulation group' (prompts interactively if omitted)",
    )

    # schedule
    p_sched = sub.add_parser("schedule", help="Install scheduled runs")
    p_sched.add_argument("--force", action="store_true", help="Schedule even if disabled in config")

    # unschedule
    sub.add_parser("unschedule", help="Remove scheduled runs")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "configure":
        cmd_configure(args)
    elif args.command == "refine":
        cmd_refine(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    elif args.command == "unschedule":
        cmd_unschedule(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
