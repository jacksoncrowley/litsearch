"""CLI entry point for litsearch.

Commands:
    litsearch init       Create litsearch.toml in current directory
    litsearch run        Run search and generate report
    litsearch schedule   Set up systemd/launchd timer (Linux/macOS)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
from pathlib import Path

from litsearch import __version__
from litsearch.config import load_config, write_default_config, _find_config
from litsearch.pubmed import search, Paper
from litsearch.scoring import score_all, generate_relevance_reason, generate_report_summary
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

    # LLM relevance (optional, only for top N)
    top_n = min(cfg.output.max_highlights, len(scored))
    if cfg.llm.enabled and top_n > 0:
        print(f"Generating relevance justifications for top {top_n} papers...")
        for paper in scored[:top_n]:
            reason = generate_relevance_reason(paper, cfg)
            if reason:
                paper.relevance_reason = reason
        print("  Done.")

    # Global AI summary
    summary = ""
    if cfg.llm.enabled and scored:
        print("Generating report summary...")
        summary = generate_report_summary(scored, cfg)

    # Render
    output_dir = Path(args.output_dir) if args.output_dir else \
                 (Path(cfg.output.dir) if cfg.output.dir else Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output.format == "html":
        report_path = output_dir / f"litsearch_report_{end_date}.html"
        html = render_report(scored, cfg, start_date, end_date, version=__version__, summary=summary)
        report_path.write_text(html)
        print(f"\nReport saved: {report_path}")
    else:
        report_path = output_dir / f"litsearch_report_{end_date}.md"
        md = _render_markdown(scored, cfg, start_date, end_date)
        report_path.write_text(md)
        print(f"\nReport saved: {report_path}")


def _render_markdown(papers: list[Paper], cfg, start_date, end_date) -> str:
    """Simple markdown output (fallback)."""
    lines = [
        f"# litsearch Report: {start_date} to {end_date}",
        "",
        f"**Papers scanned:** {len(papers)}",
        f"**Source:** PubMed",
        "",
        "---",
    ]

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
            if p.relevance_reason:
                lines.append(f"- **Relevance:** {p.relevance_reason}")
            lines.append("")

    lines.append(f"\n*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)


def cmd_configure(_args: argparse.Namespace) -> None:
    """Interactive wizard to configure the LLM provider."""
    try:
        cfg_path = _find_config()
    except FileNotFoundError:
        print("No litsearch.toml found. Run 'litsearch init' first.")
        sys.exit(1)

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

    _patch_llm_section(cfg_path, provider=provider, model=model,
                       api_key=api_key, base_url=base_url)
    print(f"\nUpdated {cfg_path.name} — AI summaries now use {provider} / {model}.")
    print("Run 'litsearch run' to try it.")


def _patch_llm_section(path: Path, *, provider: str, model: str,
                        api_key: str, base_url: str) -> None:
    """Replace the [llm] section in litsearch.toml, preserving all other content."""
    new_section = (
        f"[llm]\n"
        f"enabled = true\n"
        f"provider = \"{provider}\"\n"
        f"model = \"{model}\"\n"
        f"api_key = \"{api_key}\"\n"
        f"base_url = \"{base_url}\"\n"
    )
    text = path.read_text()
    text = re.sub(r"\[llm\].*?(?=\n\[|\Z)", new_section, text, flags=re.DOTALL)
    path.write_text(text)


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


def _install_systemd(cfg) -> None:
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


def _install_launchd(cfg) -> None:
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

    # configure
    sub.add_parser("configure", help="Set up AI provider and API key")

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
    elif args.command == "schedule":
        cmd_schedule(args)
    elif args.command == "unschedule":
        cmd_unschedule(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
