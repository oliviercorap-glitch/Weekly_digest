#!/usr/bin/env python3
"""
weekly_digest_agent.py
========================
Weekly intelligence digest: consolidates the HTML reports produced by
TLD Group's other watch agents (GSE competitive intelligence, China economic
watch, APAC economic watch, APAC GSE watch, China tax & corporate law watch,
APAC FX risk watch), re-analyzes them together with DeepSeek, and emails a
single newsletter every Monday morning -- so there is no more need to open
GitHub Actions manually and download HTML files one by one.

Architecture
------------
Unlike the source agents, this one does NOT scrape the web or query Tavily:
it only consumes reports that already exist. The pipeline is:

1. For each configured source repository (public GitHub repos), list the
   `reports/` folder via the GitHub Contents API and fetch every HTML report
   dated within the last N days (falls back to `reports/latest.html` if no
   dated files are found), with retry/backoff on every network call.
2. Extract clean text + every source link from each report (BeautifulSoup),
   regardless of that report's exact internal HTML structure -- this makes
   the digest robust even for source agents whose markup we don't control.
3. Feed all of that week's extracted content to DeepSeek in a single prompt,
   asking for ONE consolidated, de-duplicated newsletter using the same
   delimiter-based structured output as the other agents
   (===SIGNAL_START===...===SIGNAL_END===), plus an executive "week in
   review" and a "top risks" section, with truncation detection.
4. Build a single self-contained HTML newsletter (impact levels
   CRITICAL/IMPORTANT/WATCH/INFO, executive summary, top risks, and an
   appendix of every original source link grouped by agent -- so nothing is
   ever lost even if the AI narrative omits an item).
5. Email the newsletter via SMTP (Office365 / Outlook) and also archive it
   in the repo + as a workflow artifact.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import os
import random
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ORG_NAME = "TLD Group"
ORG_CONTEXT = "APAC Finance Department (CFO)"
REPORT_LANG = "en"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_REPORTS_TOKEN", "")  # optional, public repos work without it

SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_TO = os.environ.get("EMAIL_TO", "")  # comma-separated list of recipients

# NOTE: GitHub Actions injects an EMPTY STRING (not an unset variable) for any
# secret referenced in a workflow's `env:` block that hasn't been configured
# in the repo settings. `os.environ.get(key, default)` only falls back to
# `default` when the key is entirely absent, so it does NOT protect against
# an empty string here -- hence the explicit `or` fallback below, which
# treats "" the same as "not set".
SMTP_HOST = os.environ.get("SMTP_HOST") or "smtp.office365.com"
SMTP_PORT = int(os.environ.get("SMTP_PORT") or "587")
EMAIL_FROM = os.environ.get("EMAIL_FROM") or SMTP_USERNAME

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 7  # gather reports generated in the last 7 days
TEST_MODE = "--test" in os.sys.argv or os.environ.get("TEST_MODE") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("weekly_digest_agent")

# ---------------------------------------------------------------------------
# SOURCE REPOSITORIES
# ---------------------------------------------------------------------------
# EDIT ME: replace the "<owner>/<repo>" placeholders below with the actual
# GitHub paths of your other agents. The two repos built alongside this one
# are already filled in; the first three placeholders correspond to your
# pre-existing GSE aviation / China economic / APAC GSE agents -- update
# them to match their real repository names.
#
# This list can also be overridden entirely at runtime without touching the
# code, by setting the SOURCE_REPOS_JSON environment variable / secret to a
# JSON array in the same shape.

DEFAULT_SOURCE_REPOS = [
    {
        "name": "GSE Aviation Competitive Intelligence (China)",
        "repo": "oliviercorap-glitch/agent_aviation_industry_china",
        "reports_path": "reports",
    },
    {
        "name": "GSE Aviation Competitive Intelligence (APAC)",
        "repo": "oliviercorap-glitch/agent_aviation_industry_APAC",
        "reports_path": "reports",
    },
    {
        "name": "China Economic Watch",
        "repo": "oliviercorap-glitch/China_eco_agent",
        "reports_path": "reports",
    },
    {
        "name": "APAC Economic Watch (ex-Mainland China)",
        "repo": "oliviercorap-glitch/apac_eco_agent",
        "reports_path": "reports",
    },
    {
        "name": "China Tax & Corporate Law Watch",
        "repo": "oliviercorap-glitch/China_tax_law",
        "reports_path": "reports",
    },
    {
        "name": "APAC FX Risk Watch",
        "repo": "oliviercorap-glitch/APAC_FOREX",
        "reports_path": "reports",
    },
]


def load_source_repos() -> list[dict]:
    raw_override = os.environ.get("SOURCE_REPOS_JSON", "").strip()
    if raw_override:
        try:
            parsed = json.loads(raw_override)
            if isinstance(parsed, list) and parsed:
                return parsed
        except json.JSONDecodeError as exc:
            logger.warning("SOURCE_REPOS_JSON could not be parsed, using default list: %s", exc)
    return DEFAULT_SOURCE_REPOS


SOURCE_REPOS = load_source_repos()

DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ---------------------------------------------------------------------------
# RETRY / BACKOFF
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries=3, base_delay=1.5, max_delay=20.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, ValueError) as exc:
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s",
                        func.__name__, attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        sleep_time = min(delay, max_delay) + random.uniform(0, 0.75)
                        time.sleep(sleep_time)
                        delay *= 2
            logger.error("%s gave up after %d attempts: %s", func.__name__, max_retries, last_exc)
            return None
        return wrapper
    return decorator


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


@retry_with_backoff(max_retries=3, base_delay=1.5)
def github_get(url: str, timeout: int = 15):
    resp = requests.get(url, headers=_github_headers(), timeout=timeout)
    if resp.status_code == 404:
        return None  # repo/path genuinely absent - not a transient failure, don't retry loudly
    resp.raise_for_status()
    return resp.json()


@retry_with_backoff(max_retries=3, base_delay=2.0)
def post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> Optional[dict]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# GITHUB REPORT COLLECTION
# ---------------------------------------------------------------------------

def list_report_files(repo: str, reports_path: str) -> list[dict]:
    """Tries the configured reports_path first, then a set of common
    fallback locations, since not all agents in the suite necessarily use
    the exact same folder convention. Returns as soon as a location with at
    least one .html file is found."""
    candidates = [reports_path] if reports_path else []
    for fallback in ["reports", "Reports", "report", "output", "outputs", ""]:
        if fallback not in candidates:
            candidates.append(fallback)

    tried_summary = []
    for path in candidates:
        url = f"https://api.github.com/repos/{repo}/contents/{path}" if path else f"https://api.github.com/repos/{repo}/contents"
        listing = github_get(url)
        if listing is None:
            tried_summary.append(f"{path or '(root)'}: not found (404)")
            continue
        if not isinstance(listing, list):
            tried_summary.append(f"{path or '(root)'}: not a directory")
            continue
        html_files = [item for item in listing if item.get("type") == "file" and item.get("name", "").endswith(".html")]
        if html_files:
            logger.info("Found %d html report(s) in %s/%s", len(html_files), repo, path or "(root)")
            return html_files
        tried_summary.append(f"{path or '(root)'}: directory exists but no .html files")

    logger.warning(
        "No HTML reports found in repo %s after trying: %s. "
        "Check that the agent has run at least once and that its reports are committed to the repo.",
        repo, "; ".join(tried_summary),
    )
    return []


def fetch_file_content(repo: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    data = github_get(url)
    if not data or "content" not in data:
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except (ValueError, TypeError) as exc:
        logger.warning("Could not decode content for %s/%s: %s", repo, path, exc)
        return None


def select_recent_files(files: list[dict], lookback_days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    dated_files = []
    for f in files:
        if f["name"] == "latest.html":
            continue
        match = DATE_IN_FILENAME_RE.search(f["name"])
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date >= cutoff:
            dated_files.append(f)

    if dated_files:
        return dated_files

    # Fallback: no dated files found in the freshness window -> use latest.html if present
    latest = [f for f in files if f["name"] == "latest.html"]
    if latest:
        logger.info("No dated reports within %d days, falling back to latest.html", lookback_days)
    return latest


def extract_report_content(html: str, source_name: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    links = []
    seen_hrefs = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        label = a_tag.get_text(strip=True)
        if not href.startswith("http") or href in seen_hrefs or len(label) < 8:
            continue
        seen_hrefs.add(href)
        links.append({"label": label, "url": href, "source": source_name})

    return {"text": text, "links": links}


def collect_weekly_reports() -> tuple[list[dict], list[dict]]:
    """Returns (report_digests, all_links).
    report_digests: [{"source_name", "file_name", "text"}]
    all_links: [{"label", "url", "source"}]
    """
    digests, all_links = [], []

    for source in SOURCE_REPOS:
        repo, reports_path, name = source["repo"], source.get("reports_path", "reports"), source["name"]
        if "<owner>" in repo:
            logger.warning("Skipping '%s': repository placeholder not replaced (%s)", name, repo)
            continue

        files = list_report_files(repo, reports_path)
        if not files:
            continue

        recent_files = select_recent_files(files, LOOKBACK_DAYS)
        if TEST_MODE:
            recent_files = recent_files[:1]

        for f in recent_files:
            html = fetch_file_content(repo, f["path"])
            if not html:
                logger.warning("Could not fetch %s from %s", f["path"], repo)
                continue
            extracted = extract_report_content(html, name)
            digests.append({
                "source_name": name,
                "file_name": f["name"],
                "text": extracted["text"][:6000],  # cap per-file size to control token budget
            })
            all_links.extend(extracted["links"])
            logger.info("Collected %s from %s (%d chars, %d links)", f["name"], repo, len(extracted["text"]), len(extracted["links"]))

    return digests, all_links


# ---------------------------------------------------------------------------
# DEEPSEEK - API CALL
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(messages: list[dict], max_tokens: int = 4000, temperature: float = 0.3):
    """Returns (text, truncated: bool). text is None on total failure."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing: cannot call DeepSeek")
        return None, False

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = post_json(DEEPSEEK_URL, headers, payload)
    if not data:
        return None, False

    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        truncated = choice.get("finish_reason") == "length"
        return text, truncated
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected DeepSeek response shape: %s", exc)
        return None, False


# ---------------------------------------------------------------------------
# CONSOLIDATED ANALYSIS (delimiter-based structured output)
# ---------------------------------------------------------------------------

SIGNAL_BLOCK_RE = re.compile(r"===SIGNAL_START===(.*?)===SIGNAL_END===", re.DOTALL)
EXEC_SUMMARY_RE = re.compile(r"===EXEC_SUMMARY_START===(.*?)===EXEC_SUMMARY_END===", re.DOTALL)
TOP_RISK_RE = re.compile(r"===TOP_RISK_START===(.*?)===TOP_RISK_END===", re.DOTALL)

SIGNAL_FIELD_RE = re.compile(r"^(TITLE|IMPACT|CATEGORY|SUMMARY|IMPLICATIONS|SOURCE_AGENT):\s*(.*)$")

VALID_IMPACTS = {"CRITICAL", "IMPORTANT", "WATCH", "INFO"}


def build_consolidation_prompt(digests: list[dict]) -> str:
    if not digests:
        return ""
    blocks = []
    for i, d in enumerate(digests, start=1):
        blocks.append(f"[Report {i} - Agent: {d['source_name']} - File: {d['file_name']}]\n{d['text']}\n")
    reports_text = "\n".join(blocks)

    return f"""You are the Chief Intelligence Editor for {ORG_NAME} ({ORG_CONTEXT}).
Each week you receive the raw text extracted from several independent watch-agent
reports (competitive intelligence, China macroeconomic watch, APAC macroeconomic
watch, APAC GSE market watch, China tax & corporate law watch, APAC FX risk
watch). Your job is to consolidate
them into ONE clean weekly newsletter for the CFO -- not to just concatenate them.

RAW EXTRACTED REPORT CONTENT FOR THIS WEEK:
{reports_text}

INSTRUCTIONS:
1. Identify genuinely significant items. Merge or de-duplicate overlapping items
   that appear in more than one source report (e.g. a China macro item that is
   also referenced in the FX report) into a single signal, and mention in
   SOURCE_AGENT that it was corroborated by multiple agents if relevant.
2. Where you can see a cross-cutting connection between reports (for example a
   PBOC move noted in the FX report that also explains a trend in the China
   economic report), call that out explicitly.
3. Ignore items that are purely routine/no-action or clearly low-value noise.

For each significant item, produce a block in the EXACT following format
(nothing before or after the delimiters):

===SIGNAL_START===
TITLE: <short, clear title in English>
IMPACT: <CRITICAL|IMPORTANT|WATCH|INFO>
CATEGORY: <Competitive Intelligence|China Macro|APAC Macro|APAC Market|Tax & Legal|FX & Treasury|Cross-cutting|Other>
SUMMARY: <2-4 factual sentences in English>
IMPLICATIONS: <concrete implication for TLD Group's APAC finance leadership, in English>
SOURCE_AGENT: <which agent report(s) this came from>
===SIGNAL_END===

Impact scale:
- CRITICAL: requires the CFO's attention this week, binding deadline, or material financial/legal exposure.
- IMPORTANT: significant development worth a close read and possible follow-up.
- WATCH: weak signal or early-stage development to keep an eye on.
- INFO: useful context, no action needed.

Then add EXACTLY one "week in review" executive summary block:
===EXEC_SUMMARY_START===
<5-8 sentences in English giving a genuine week-in-review across all source reports,
written for a time-pressed CFO who has not read the underlying reports>
===EXEC_SUMMARY_END===

Then EXACTLY one top risks block:
===TOP_RISK_START===
<3-5 sentences identifying the 1-3 most important things to watch going into next week,
across all domains (competitive, macro, tax/legal, FX)>
===TOP_RISK_END===

If the raw content contains nothing significant, say so plainly in EXEC_SUMMARY and
TOP_RISK and produce no SIGNAL blocks."""


def parse_signals(raw_text: str) -> tuple[list[dict], str, str, bool]:
    signals = []
    for block in SIGNAL_BLOCK_RE.findall(raw_text):
        fields = {}
        for line in block.strip().splitlines():
            match = SIGNAL_FIELD_RE.match(line.strip())
            if match:
                fields[match.group(1)] = match.group(2).strip()
        if fields.get("TITLE") and fields.get("IMPACT") in VALID_IMPACTS:
            signals.append(fields)
        elif fields.get("TITLE"):
            fields["IMPACT"] = "INFO"
            signals.append(fields)

    exec_summary_match = EXEC_SUMMARY_RE.search(raw_text)
    top_risk_match = TOP_RISK_RE.search(raw_text)
    exec_summary = exec_summary_match.group(1).strip() if exec_summary_match else ""
    top_risk = top_risk_match.group(1).strip() if top_risk_match else ""

    truncation_suspected = raw_text.count("===SIGNAL_START===") > raw_text.count("===SIGNAL_END===")
    if not exec_summary_match and "===EXEC_SUMMARY_START===" in raw_text:
        truncation_suspected = True
    if not top_risk_match and "===TOP_RISK_START===" in raw_text:
        truncation_suspected = True

    return signals, exec_summary, top_risk, truncation_suspected


def analyze_weekly_reports(digests: list[dict]):
    if not digests:
        return [], ("No source reports could be retrieved this week. Check that the "
                     "SOURCE_REPOS repository names are correct and that each agent ran "
                     "successfully."), "N/A", False

    prompt = build_consolidation_prompt(digests)
    raw_text, api_truncated = call_deepseek(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )
    if not raw_text:
        logger.error("DeepSeek consolidation unavailable: emailing a degraded digest")
        return [], "AI consolidation unavailable this week (DeepSeek call failed). See the appendix below for all source links collected this week.", "N/A", False

    signals, exec_summary, top_risk, parse_truncated = parse_signals(raw_text)
    truncated = api_truncated or parse_truncated
    if truncated:
        logger.warning("Truncation detected in DeepSeek response: some signals may be missing")
    return signals, exec_summary, top_risk, truncated


# ---------------------------------------------------------------------------
# HTML NEWSLETTER (email-safe: inline styles)
# ---------------------------------------------------------------------------

IMPACT_ORDER = ["CRITICAL", "IMPORTANT", "WATCH", "INFO"]
IMPACT_STYLE = {
    "CRITICAL":  {"color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "text": "#991b1b", "label": "Critical",  "icon": "🔴"},
    "IMPORTANT": {"color": "#d97706", "bg": "#fffbeb", "border": "#fde68a", "text": "#92400e", "label": "Important", "icon": "🟠"},
    "WATCH":     {"color": "#0369a1", "bg": "#f0f9ff", "border": "#bae6fd", "text": "#0c4a6e", "label": "Watch",     "icon": "🔵"},
    "INFO":      {"color": "#6b7280", "bg": "#f9fafb", "border": "#e5e7eb", "text": "#374151", "label": "Info",     "icon": "⚪"},
}

CATEGORY_ICON = {
    "Competitive Intelligence": "📡",
    "China Macro":              "🇨🇳",
    "APAC Macro":               "🌏",
    "APAC Market":              "🛬",
    "Tax & Legal":              "⚖️",
    "FX & Treasury":            "💱",
    "Cross-cutting":            "🔗",
    "Other":                    "📌",
}


def html_escape(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


SIGNAL_ROW = """
<tr>
  <td style="padding:0 0 14px 0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="background:#ffffff; border:1px solid #e5e7eb; border-left:4px solid {color}; border-radius:10px; box-shadow:0 1px 2px rgba(16,24,40,.04);">
      <tr>
        <td style="padding:16px 20px 14px 18px;">
          <span style="display:inline-block; padding:3px 11px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:.02em; background:{bg}; color:{text}; border:1px solid {border}; margin-right:6px;">{impact_icon} {impact_label}</span>
          <span style="display:inline-block; padding:3px 11px; border-radius:999px; font-size:11px; font-weight:600; background:#f1f5f9; color:#475569; margin-right:6px;">{category_icon} {category}</span>
          <h3 style="margin:12px 0 6px 0; font-size:16px; line-height:1.4; color:#0f172a; font-weight:600;">{title}</h3>
          <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; font-weight:600; margin-bottom:10px;">Source &middot; {source_agent}</div>
          <p style="margin:0 0 10px 0; font-size:14px; line-height:1.65; color:#334155;">{summary}</p>
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">
            <tr>
              <td style="padding:10px 14px;">
                <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8; margin-bottom:3px;">Implication for TLD</div>
                <div style="font-size:14px; line-height:1.6; color:#0f172a; font-weight:500;">{implications}</div>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </td>
</tr>
"""

LINK_APPENDIX_GROUP = """
<div style="margin-bottom:14px;">
  <h3 style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:#64748b; margin:0 0 8px 0; padding-bottom:6px; border-bottom:1px solid #e2e8f0;">{source_name}</h3>
  <ul style="margin:0; padding-left:18px; font-size:13px; color:#2563eb; line-height:1.9;">
    {items}
  </ul>
</div>
"""


def render_signals(signals: list[dict]) -> str:
    grouped = {impact: [] for impact in IMPACT_ORDER}
    for sig in signals:
        grouped.setdefault(sig.get("IMPACT", "INFO"), []).append(sig)

    parts = []
    for impact in IMPACT_ORDER:
        items = grouped.get(impact, [])
        if not items:
            continue
        style = IMPACT_STYLE[impact]
        parts.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:22px 0 10px 0;">'
            f'<tr><td style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#64748b;">'
            f'{style["icon"]} {style["label"]} &middot; {len(items)}'
            f'</td></tr></table>'
        )
        parts.append('<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tbody>')
        for sig in items:
            category = sig.get("CATEGORY", "Other")
            parts.append(SIGNAL_ROW.format(
                color=style["color"],
                bg=style["bg"],
                border=style["border"],
                text=style["text"],
                impact_icon=style["icon"],
                impact_label=style["label"],
                category_icon=CATEGORY_ICON.get(category, "📌"),
                category=html_escape(category),
                title=html_escape(sig.get("TITLE", "")),
                source_agent=html_escape(sig.get("SOURCE_AGENT", "unspecified")),
                summary=html_escape(sig.get("SUMMARY", "")),
                implications=html_escape(sig.get("IMPLICATIONS", "")),
            ))
        parts.append("</tbody></table>")

    if not parts:
        parts.append(
            '<p style="font-size:14px; color:#64748b; font-style:italic; padding:12px 0;">'
            "No significant signal detected this week.</p>"
        )
    return "\n".join(parts)


def render_link_appendix(all_links: list[dict]) -> str:
    if not all_links:
        return '<p style="font-size:13px; color:#6b7280;">No source links were collected this week.</p>'

    by_source: dict[str, list[dict]] = {}
    seen = set()
    for link in all_links:
        key = (link["source"], link["url"])
        if key in seen:
            continue
        seen.add(key)
        by_source.setdefault(link["source"], []).append(link)

    groups = []
    for source_name, links in by_source.items():
        items_html = "\n".join(
            f'<li style="margin-bottom:4px;"><a href="{l["url"]}" style="color:#2563eb; text-decoration:none;" target="_blank" rel="noopener">{html_escape(l["label"])[:140]}</a></li>'
            for l in links[:40]
        )
        groups.append(LINK_APPENDIX_GROUP.format(source_name=html_escape(source_name), items=items_html))
    return "\n".join(groups)


NEWSLETTER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weekly Intelligence Digest - {run_date}</title>
<!--[if mso]>
<style>table {{border-collapse:collapse;}}</style>
<![endif]-->
</head>
<body style="margin:0; padding:0; background:#eef1f6; font-family:'Segoe UI',-apple-system,Helvetica,Arial,sans-serif; color:#0f172a;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef1f6;">
<tr><td align="center" style="padding:28px 12px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:900px;">

<!-- MASTHEAD -->
<tr><td style="background:#0f172a; padding:32px 32px 26px 32px; border-radius:14px 14px 0 0;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td>
      <div style="font-family:Consolas,'Courier New',monospace; font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#64748b; margin-bottom:10px;">{org_name} &middot; {org_context}</div>
      <div style="font-size:25px; font-weight:700; letter-spacing:-.01em; color:#ffffff; margin-bottom:6px;">Weekly Intelligence Digest</div>
      <div style="font-size:14px; color:#94a3b8; margin-bottom:18px;">Week of {run_date} &middot; {nb_reports} source report(s) consolidated across {nb_agents} watch agents</div>
    </td></tr>
    <tr><td style="border-top:1px solid #1e293b; padding-top:16px;">
      {counter_pills}
    </td></tr>
  </table>
</td></tr>

<tr><td style="padding:0;">{truncation_html}</td></tr>

<!-- EXEC SUMMARY -->
<tr><td style="background:#ffffff; padding:0 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:22px;">
    <tr><td style="background:#0f172a; border-radius:12px; padding:22px 26px;">
      <div style="font-family:Consolas,'Courier New',monospace; font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:#64748b; margin-bottom:10px;">Week in review</div>
      <p style="margin:0; font-size:14.5px; line-height:1.75; color:#e2e8f0;">{exec_summary}</p>
    </td></tr>
  </table>
</td></tr>

<!-- TOP RISKS -->
<tr><td style="background:#ffffff; padding:16px 32px 0 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:18px 22px;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#92400e; margin-bottom:8px;">&#9888;&nbsp; Top risks to watch this week</div>
      <p style="margin:0; font-size:14px; line-height:1.7; color:#78350f;">{top_risk}</p>
    </td></tr>
  </table>
</td></tr>

<!-- SIGNALS -->
<tr><td style="background:#ffffff; padding:26px 32px 6px 32px;">
  <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8; padding-bottom:10px; margin-bottom:6px; border-bottom:1px solid #e2e8f0;">Signals this week</div>
  {signals_html}
</td></tr>

<!-- APPENDIX -->
<tr><td style="background:#ffffff; padding:8px 32px 28px 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px 22px;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8; margin-bottom:12px;">Appendix &middot; all source links this week</div>
      {link_appendix_html}
    </td></tr>
  </table>
</td></tr>

<!-- FOOTER -->
<tr><td style="background:#ffffff; border-radius:0 0 14px 14px; padding:0 32px 26px 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="border-top:1px solid #e2e8f0; padding-top:16px; font-size:11px; color:#94a3b8; font-family:Consolas,'Courier New',monospace; letter-spacing:.02em;">
      Automatically generated by weekly_digest_agent.py &middot; consolidating: {agent_names}
    </td></tr>
  </table>
</td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def generate_newsletter_html(signals: list[dict], exec_summary: str, top_risk: str,
                              all_links: list[dict], nb_reports: int, truncated: bool) -> str:
    truncation_html = ""
    if truncated:
        truncation_html = (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#ffffff; padding:16px 32px 0 32px;">'
            '<tr><td style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 18px; font-size:13px; color:#991b1b;">'
            "&#9888;&nbsp; Warning: the AI consolidation response appears to have been truncated. "
            "Some signals may be incomplete -- check the appendix below for the full list of "
            "source links."
            "</td></tr></table>"
        )

    active_repos = [s for s in SOURCE_REPOS if "<owner>" not in s["repo"]]
    agent_names = ", ".join(s["name"] for s in active_repos)

    counts = {impact: 0 for impact in IMPACT_ORDER}
    for sig in signals:
        counts[sig.get("IMPACT", "INFO")] = counts.get(sig.get("IMPACT", "INFO"), 0) + 1

    counter_pills = "".join(
        f'<span style="display:inline-block; padding:4px 12px; border-radius:999px; '
        f'font-size:11px; font-weight:600; background:{IMPACT_STYLE[lvl]["bg"]}; '
        f'color:{IMPACT_STYLE[lvl]["text"]}; border:1px solid {IMPACT_STYLE[lvl]["border"]}; '
        f'margin-right:8px;">{IMPACT_STYLE[lvl]["icon"]} {counts[lvl]} {IMPACT_STYLE[lvl]["label"]}</span>'
        for lvl in IMPACT_ORDER
        if counts[lvl] > 0
    ) or '<span style="font-size:12px; color:#64748b;">No signals to report this week</span>'

    return NEWSLETTER_TEMPLATE.format(
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        org_name=ORG_NAME,
        org_context=ORG_CONTEXT,
        nb_reports=nb_reports,
        nb_agents=len(active_repos) or len(SOURCE_REPOS),
        counter_pills=counter_pills,
        truncation_html=truncation_html,
        exec_summary=html_escape(exec_summary) or "No summary available.",
        top_risk=html_escape(top_risk) or "No particular risk identified.",
        signals_html=render_signals(signals),
        link_appendix_html=render_link_appendix(all_links),
        agent_names=agent_names or "no agents configured yet",
    )


# ---------------------------------------------------------------------------
# EMAIL DELIVERY (Office365 / Outlook SMTP)
# ---------------------------------------------------------------------------

def send_email(html_body: str, subject: str) -> bool:
    if not (SMTP_USERNAME and SMTP_PASSWORD and EMAIL_TO):
        logger.error("SMTP_USERNAME, SMTP_PASSWORD or EMAIL_TO missing: cannot send email")
        return False

    recipients = [addr.strip() for addr in EMAIL_TO.split(",") if addr.strip()]
    if not recipients:
        logger.error("EMAIL_TO did not contain any valid address")
        return False

    plain_text = BeautifulSoup(html_body, "lxml").get_text(separator="\n", strip=True)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    delay = 2.0
    for attempt in range(1, 4):
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(EMAIL_FROM, recipients, msg.as_string())
            logger.info("Email sent to %s", ", ".join(recipients))
            return True
        except smtplib.SMTPException as exc:
            logger.warning("Email send failed (attempt %d/3): %s", attempt, exc)
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
    logger.error("Email delivery failed after 3 attempts")
    return False


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Starting weekly_digest_agent.py (TEST_MODE=%s) ===", TEST_MODE)

    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing: consolidation will be degraded.")
    if not GITHUB_TOKEN:
        logger.info("GITHUB_REPORTS_TOKEN not set: relying on unauthenticated GitHub API calls (fine for public repos, lower rate limit).")

    digests, all_links = collect_weekly_reports()
    logger.info("Collected %d report file(s) with %d total source link(s)", len(digests), len(all_links))

    signals, exec_summary, top_risk, truncated = analyze_weekly_reports(digests)

    html_newsletter = generate_newsletter_html(
        signals=signals,
        exec_summary=exec_summary,
        top_risk=top_risk,
        all_links=all_links,
        nb_reports=len(digests),
        truncated=truncated,
    )

    run_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"weekly_digest_{run_date_str}.html"
    report_path.write_text(html_newsletter, encoding="utf-8")
    (REPORTS_DIR / "latest.html").write_text(html_newsletter, encoding="utf-8")
    logger.info("Newsletter written: %s", report_path)

    subject = f"Weekly Intelligence Digest - {ORG_NAME} APAC - {run_date_str}"
    send_email(html_newsletter, subject)

    logger.info("=== Done: %d signal(s) across %d report(s) consolidated ===", len(signals), len(digests))


if __name__ == "__main__":
    main()
