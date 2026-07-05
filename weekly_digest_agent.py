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
import io
import json
import logging
import math
import os
import random
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend -- no display available on GitHub Actions runners
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
                "links": extracted["links"][:30],  # cap link list sent to DeepSeek per report
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
FX_BLOCK_RE = re.compile(r"===FX_START===(.*?)===FX_END===", re.DOTALL)

SIGNAL_FIELD_RE = re.compile(r"^(TITLE|IMPACT|CATEGORY|SUMMARY|IMPLICATIONS|SOURCE_AGENT|SOURCE_INDEX):\s*(.*)$")
FX_FIELD_RE = re.compile(r"^(CURRENCY|CHANGE_PCT):\s*(.*)$")

VALID_IMPACTS = {"CRITICAL", "IMPORTANT", "WATCH", "INFO"}
CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def build_numbered_links(all_links: list[dict]) -> list[dict]:
    """Deduplicate all_links by URL and assign each a stable 1-based index.

    This numbered list is shown to DeepSeek exactly once; DeepSeek must cite
    a signal's source by this index (SOURCE_INDEX) rather than copying a URL
    itself. Citation by number is unambiguous by construction -- there is
    exactly one article behind each number, so DeepSeek cannot accidentally
    reference an unrelated-but-real article the way it could when asked to
    recall/retype a URL (the residual failure mode of the URL-citation
    approach: a genuine URL from this week's set, just the wrong one)."""
    seen = set()
    numbered = []
    for link in all_links:
        key = _normalize_url(link["url"])
        if key in seen:
            continue
        seen.add(key)
        numbered.append({
            "index": len(numbered) + 1,
            "label": link["label"],
            "url": link["url"],
            "source": link["source"],
        })
    return numbered


def build_consolidation_prompt(digests: list[dict], numbered_links: list[dict]) -> str:
    if not digests:
        return ""
    blocks = []
    for i, d in enumerate(digests, start=1):
        blocks.append(f"[Report {i} - Agent: {d['source_name']} - File: {d['file_name']}]\n{d['text']}\n")
    reports_text = "\n".join(blocks)

    articles_list = "\n".join(
        f"[{l['index']}] {l['label'][:140]} — {l['source']}"
        for l in numbered_links
    ) or "(no source articles collected this week)"

    return f"""You are the Chief Intelligence Editor for {ORG_NAME} ({ORG_CONTEXT}).
Each week you receive the raw text extracted from several independent watch-agent
reports (competitive intelligence, China macroeconomic watch, APAC macroeconomic
watch, APAC GSE market watch, China tax & corporate law watch, APAC FX risk
watch). Your job is to consolidate
them into ONE clean weekly newsletter for the CFO -- not to just concatenate them.

RAW EXTRACTED REPORT CONTENT FOR THIS WEEK:
{reports_text}

AVAILABLE SOURCE ARTICLES (cite ONLY by number in SOURCE_INDEX -- never write a URL):
{articles_list}

INSTRUCTIONS:
1. Identify genuinely significant items. Merge or de-duplicate overlapping items
   that appear in more than one source report (e.g. a China macro item that is
   also referenced in the FX report) into a single signal, and mention in
   SOURCE_AGENT that it was corroborated by multiple agents if relevant.
2. Where you can see a cross-cutting connection between reports (for example a
   PBOC move noted in the FX report that also explains a trend in the China
   economic report), call that out explicitly.
3. Ignore items that are purely routine/no-action or clearly low-value noise.
4. For SOURCE_INDEX, only ever cite a single number from the "AVAILABLE SOURCE
   ARTICLES" list above -- never a URL, never a range, never a guess. Read the
   article at that number before citing it, to confirm it is genuinely the one
   this specific signal is based on (not just a different article that happens
   to mention a similar company or topic). An empty SOURCE_INDEX is always
   preferable to a wrong one.

For each significant item, produce a block in the EXACT following format
(nothing before or after the delimiters):

===SIGNAL_START===
TITLE: <short, clear title in English>
IMPACT: <CRITICAL|IMPORTANT|WATCH|INFO>
CATEGORY: <Competitive Intelligence|China Macro|APAC Macro|APAC Market|Tax & Legal|FX & Treasury|Cross-cutting|Other>
SUMMARY: <2-4 factual sentences in English>
IMPLICATIONS: <what TLD Group should do about this, in English -- phrase it as company action ("TLD should...", "TLD needs to...", "TLD's regional teams should..."), not as a personal instruction to the CFO ("the CFO should..."). Be concrete and specific (which team, which market, what decision), not generic.>
SOURCE_AGENT: <which agent report(s) this came from>
SOURCE_INDEX: <the single integer number from the "AVAILABLE SOURCE ARTICLES" list above that this exact signal is based on. If this signal synthesizes multiple reports, pick the one article number most directly relevant to the HEADLINE of this specific signal. If truly no article above corresponds to this signal, leave this field EMPTY rather than guessing.>
===SIGNAL_END===

MANDATORY FX EXTRACTION STEP -- do this for every currency mentioned with a
number, even in passing: scan the APAC FX Risk Watch report AND every other
report for ANY currency percentage move this week (e.g. "JPY weakened 1.8%
vs EUR", "CNY depreciated 0.6%", "both down ~0.3% weekly"). Approximate
figures explicitly given in the source reports (e.g. "~0.3%", "about 2%")
DO count and MUST be captured -- "never estimate or invent" means never make
up a number that wasn't in the source text, it does NOT mean you should skip
an approximate figure the source itself reported. For EACH such currency,
output one FX snapshot block, in addition to (not instead of) any narrative
mention of it in TOP_RISK or the executive summary:

===FX_START===
CURRENCY: <3-letter ISO code, e.g. JPY, KRW, CNY, THB, VND, INR, AUD>
CHANGE_PCT: <signed number only, no % sign -- e.g. -1.8 or 0.6 or -0.3. Negative = currency weakened/depreciated vs EUR or USD this week (use whichever base the source report used); positive = strengthened/appreciated.>
===FX_END===

Self-check before finalizing your response: if TOP_RISK or the executive
summary mentions a currency by name alongside a percentage, there MUST be a
matching FX_START block for that currency -- a number should never exist
only in prose. Only skip FX blocks entirely if truly no report gives any
currency percentage move this week.

Impact scale:
- CRITICAL: requires the CFO's attention this week, binding deadline, or material financial/legal exposure.
- IMPORTANT: significant development worth a close read and possible follow-up.
- WATCH: weak signal or early-stage development to keep an eye on.
- INFO: useful context, no action needed.

Then add EXACTLY one "week in review" executive summary block. Write it as 3
SHORT paragraphs, separated by a blank line, so a time-pressed CFO can scan it
in seconds rather than parse one dense block:
  - Paragraph 1 (1-2 sentences): the single most urgent/critical development
    this week and why it matters right now.
  - Paragraph 2 (2-3 sentences): the next most important developments —
    competitive, macro, or FX — grouped logically, not just concatenated.
  - Paragraph 3 (1-2 sentences): the net picture and what TLD should do about
    it going into next week -- phrase this as company action ("TLD should...",
    "TLD needs to..."), not as a personal instruction to the CFO.
Each paragraph must be genuinely short (max ~3 sentences). Do not write a
single 5-8 sentence wall of text -- that defeats the purpose of splitting
into paragraphs.
===EXEC_SUMMARY_START===
<paragraph 1>

<paragraph 2>

<paragraph 3>
===EXEC_SUMMARY_END===

Then EXACTLY one top risks block:
===TOP_RISK_START===
<3-5 sentences identifying the 1-3 most important things to watch going into next week,
across all domains (competitive, macro, tax/legal, FX). Where a specific response is
warranted, phrase it as what TLD should do ("TLD should review...", "TLD's regional
team should assess..."), not as a personal instruction to the CFO.>
===TOP_RISK_END===

If the raw content contains nothing significant, say so plainly in EXEC_SUMMARY and
TOP_RISK and produce no SIGNAL blocks.

WORDING RULE THROUGHOUT (IMPLICATIONS, executive summary, top risks): always frame
recommended responses as company-level action ("TLD should...", "TLD needs to...",
"TLD's [team/entity] should..."). Never phrase them as a personal directive to an
individual ("the CFO should...", "the CFO needs to..."). The CFO is the reader of
this newsletter, not the subject of its recommendations."""


def parse_fx_data(raw_text: str) -> list[dict]:
    """Extract structured FX moves DeepSeek cited this week (see FX_START/
    FX_END in the prompt). Strict validation: a currency is only kept if it
    has a valid 3-letter code AND a numeric percentage -- anything malformed
    is dropped rather than guessed at, since a chart with a wrong number is
    worse than no chart at all."""
    fx_data = []
    seen_currencies = set()
    for block in FX_BLOCK_RE.findall(raw_text):
        fields = {}
        for line in block.strip().splitlines():
            match = FX_FIELD_RE.match(line.strip())
            if match:
                fields[match.group(1)] = match.group(2).strip()

        currency = fields.get("CURRENCY", "").strip().upper()
        change_raw = fields.get("CHANGE_PCT", "").strip().replace("%", "")

        if not CURRENCY_CODE_RE.match(currency):
            if currency:
                logger.warning("FX snapshot: dropping invalid currency code '%s'", currency)
            continue
        if currency in seen_currencies:
            continue
        try:
            change_pct = float(change_raw)
        except ValueError:
            logger.warning("FX snapshot: dropping %s, non-numeric CHANGE_PCT '%s'", currency, change_raw)
            continue

        seen_currencies.add(currency)
        fx_data.append({"currency": currency, "change_pct": change_pct})

    return fx_data


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


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def resolve_signal_source_index(signals: list[dict], numbered_links: list[dict]) -> list[dict]:
    """Resolve each signal's cited SOURCE_INDEX to its real URL.

    Citation is by number (see build_numbered_links / build_consolidation_prompt),
    so this is a simple, unambiguous lookup rather than a validation step: a
    given index maps to exactly one article, so there is no way for a valid
    index to resolve to "the wrong but real" article the way a copied URL
    could. The only failure modes are a missing/out-of-range/non-numeric
    index (model declined to cite, or malformed output) -- both are treated
    the same way: no link shown, rather than guessing.
    """
    by_index = {l["index"]: l for l in numbered_links}
    dropped = 0
    for sig in signals:
        raw_index = (sig.get("SOURCE_INDEX") or "").strip()
        link = None
        if raw_index:
            try:
                link = by_index.get(int(raw_index))
            except ValueError:
                link = None
        if link:
            sig["SOURCE_URL"] = link["url"]
        else:
            if raw_index:
                dropped += 1
                logger.warning(
                    "Dropping unresolvable SOURCE_INDEX for signal '%s': %r",
                    sig.get("TITLE", "?"), raw_index,
                )
            sig["SOURCE_URL"] = ""
    if dropped:
        logger.warning("%d signal(s) cited a SOURCE_INDEX that didn't resolve to a known article and were cleared", dropped)
    return signals


def analyze_weekly_reports(digests: list[dict], numbered_links: list[dict]):
    if not digests:
        return [], ("No source reports could be retrieved this week. Check that the "
                     "SOURCE_REPOS repository names are correct and that each agent ran "
                     "successfully."), "N/A", False, []

    prompt = build_consolidation_prompt(digests, numbered_links)
    raw_text, api_truncated = call_deepseek(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )
    if not raw_text:
        logger.error("DeepSeek consolidation unavailable: emailing a degraded digest")
        return [], "AI consolidation unavailable this week (DeepSeek call failed). See the appendix below for all source links collected this week.", "N/A", False, []

    signals, exec_summary, top_risk, parse_truncated = parse_signals(raw_text)
    fx_data = parse_fx_data(raw_text)
    truncated = api_truncated or parse_truncated
    if truncated:
        logger.warning("Truncation detected in DeepSeek response: some signals may be missing")
    return signals, exec_summary, top_risk, truncated, fx_data


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


# ---------------------------------------------------------------------------
# FX CHART (static PNG -- email clients don't run JavaScript, so an
# interactive chart is not an option here; this is the standard approach
# used by financial newsletters: render server-side, embed as an image)
# ---------------------------------------------------------------------------

def generate_fx_chart_png(fx_data: list[dict]) -> Optional[bytes]:
    """Render a horizontal bar chart of this week's FX moves as a PNG.

    Returns None if matplotlib isn't installed or there's no FX data --
    callers must handle that by simply omitting the chart section, never
    by inventing placeholder data.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed: skipping FX chart (pip install matplotlib)")
        return None
    if not fx_data:
        return None

    ordered = sorted(fx_data, key=lambda d: d["change_pct"])
    currencies = [d["currency"] for d in ordered]
    changes = [d["change_pct"] for d in ordered]
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in changes]

    height = max(2.2, 0.55 * len(ordered) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, height), dpi=180)

    bars = ax.barh(currencies, changes, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color="#334155", linewidth=1, zorder=2)

    for bar, val in zip(bars, changes):
        label = f"{val:+.1f}%"
        offset = 0.08 if val >= 0 else -0.08
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, label,
                 va="center", ha=ha, fontsize=10, color="#0f172a", fontweight="medium")

    ax.set_title("FX moves this week (vs EUR/USD)", fontsize=12, fontweight="bold",
                  color="#0f172a", pad=12, loc="left")
    ax.set_xlabel("% change", fontsize=9, color="#64748b")
    ax.tick_params(axis="both", labelsize=10, colors="#334155")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    xmax = max(abs(min(changes, default=0)), abs(max(changes, default=0)), 1) * 1.35
    ax.set_xlim(-xmax, xmax)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_fx_chart_html(chart_src: Optional[str]) -> str:
    """Build the HTML block for the FX chart section. Omitted entirely if
    there's no chart to show -- never renders a broken/empty image."""
    if not chart_src:
        return ""
    return f"""
<tr><td style="background:#ffffff; padding:16px 32px 0 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px 22px; text-align:center;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8; margin-bottom:14px; text-align:left;">💱 FX snapshot this week</div>
      <img src="{chart_src}" alt="FX moves this week" width="560" style="max-width:100%; height:auto; display:block; margin:0 auto;">
    </td></tr>
  </table>
</td></tr>"""


def html_escape(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_into_paragraphs(text: str, target_groups: int = 3) -> list[str]:
    """Split prose into readable paragraphs.

    Primary path: DeepSeek was asked to separate paragraphs with a blank
    line -- if present, just use those.

    Fallback: if the model ignored that instruction and returned one dense
    block (as observed in production), split it into ~target_groups
    roughly-equal sentence groups instead of showing one unreadable wall of
    text. A short summary (<=2 sentences) is left as a single paragraph.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) <= 2:
        return [text.strip()]

    groups = min(target_groups, len(sentences))
    size = math.ceil(len(sentences) / groups)
    chunks = [sentences[i:i + size] for i in range(0, len(sentences), size)]
    return [" ".join(chunk) for chunk in chunks]


def render_prose_paragraphs(text: str, font_size: str, line_height: str, color: str, fallback: str) -> str:
    """Render prose as properly spaced <p> tags instead of one dense block."""
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        paragraphs = [fallback]

    parts = []
    for i, p in enumerate(paragraphs):
        margin = "0 0 12px 0" if i < len(paragraphs) - 1 else "0"
        parts.append(
            f'<p style="margin:{margin}; font-size:{font_size}; '
            f'line-height:{line_height}; color:{color};">{html_escape(p)}</p>'
        )
    return "".join(parts)


def _build_url_lookup(all_links: list[dict]) -> dict:
    """Exact (not fuzzy) lookup from normalized URL -> link dict, used only to
    fetch the display label for a SOURCE_URL that DeepSeek already cited
    verbatim -- never to guess which link a signal 'probably' refers to."""
    lookup = {}
    for l in all_links:
        lookup.setdefault(_normalize_url(l["url"]), l)
    return lookup


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
          {source_link_block}
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


def render_signals(signals: list[dict], all_links: list[dict]) -> str:
    grouped = {impact: [] for impact in IMPACT_ORDER}
    for sig in signals:
        grouped.setdefault(sig.get("IMPACT", "INFO"), []).append(sig)

    url_lookup = _build_url_lookup(all_links)

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
            # SOURCE_URL was already resolved (resolve_signal_source_index) from
            # DeepSeek's cited SOURCE_INDEX -- it is either the URL of the exact
            # article DeepSeek pointed to by number, or an empty string if the
            # index was missing/invalid. No guessing.
            source_url = (sig.get("SOURCE_URL") or "").strip()
            if source_url:
                matched_link = url_lookup.get(_normalize_url(source_url))
                label_esc = html_escape(matched_link["label"])[:140] if matched_link else html_escape(source_url)[:80]
                source_link_block = (
                    '<div style="padding-top:10px; margin-top:10px; border-top:1px dashed #e2e8f0; '
                    'font-size:12px; display:flex; flex-wrap:wrap; gap:4px; align-items:center;">'
                    '<span style="font-weight:700; text-transform:uppercase; letter-spacing:.06em; '
                    'font-size:10px; color:#94a3b8; margin-right:4px;">Read the source</span>'
                    f'<a href="{source_url}" target="_blank" rel="noopener" '
                    f'style="color:#2563eb; text-decoration:none; font-weight:500;">&#128279;&nbsp;{label_esc}</a>'
                    '</div>'
                )
            else:
                source_link_block = ""
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
                source_link_block=source_link_block,
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
      {exec_summary}
    </td></tr>
  </table>
</td></tr>

<!-- TOP RISKS -->
<tr><td style="background:#ffffff; padding:16px 32px 0 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td style="background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:18px 22px;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#92400e; margin-bottom:8px;">&#9888;&nbsp; Top risks to watch this week</div>
      {top_risk}
    </td></tr>
  </table>
</td></tr>

{fx_chart_html}

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
                              all_links: list[dict], nb_reports: int, truncated: bool,
                              fx_chart_src: Optional[str] = None) -> str:
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
        exec_summary=render_prose_paragraphs(
            exec_summary, font_size="14.5px", line_height="1.75",
            color="#e2e8f0", fallback="No summary available.",
        ),
        top_risk=render_prose_paragraphs(
            top_risk, font_size="14px", line_height="1.7",
            color="#78350f", fallback="No particular risk identified.",
        ),
        fx_chart_html=render_fx_chart_html(fx_chart_src),
        signals_html=render_signals(signals, all_links),
        link_appendix_html=render_link_appendix(all_links),
        agent_names=agent_names or "no agents configured yet",
    )


# ---------------------------------------------------------------------------
# EMAIL DELIVERY (Office365 / Outlook SMTP)
# ---------------------------------------------------------------------------

def send_email(html_body: str, subject: str, inline_images: Optional[list[tuple[str, bytes]]] = None) -> bool:
    """inline_images: list of (content_id, png_bytes) to embed inline via
    cid: references in html_body -- e.g. [("fx_chart", png_bytes)] matches
    an <img src="cid:fx_chart"> tag. This is the standard, reliable way to
    embed images in email (works across Outlook/Gmail/Apple Mail), unlike
    base64 data URIs which some Outlook versions strip."""
    if not (SMTP_USERNAME and SMTP_PASSWORD and EMAIL_TO):
        logger.error("SMTP_USERNAME, SMTP_PASSWORD or EMAIL_TO missing: cannot send email")
        return False

    recipients = [addr.strip() for addr in EMAIL_TO.split(",") if addr.strip()]
    if not recipients:
        logger.error("EMAIL_TO did not contain any valid address")
        return False

    plain_text = BeautifulSoup(html_body, "lxml").get_text(separator="\n", strip=True)

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_text, "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    for cid, png_bytes in (inline_images or []):
        img = MIMEImage(png_bytes, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

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

    numbered_links = build_numbered_links(all_links)
    logger.info("%d unique source article(s) available for citation this week", len(numbered_links))

    signals, exec_summary, top_risk, truncated, fx_data = analyze_weekly_reports(digests, numbered_links)
    signals = resolve_signal_source_index(signals, numbered_links)

    if not fx_data:
        prose_to_check = f"{exec_summary}\n{top_risk}"
        mentions_currency_and_pct = bool(
            re.search(r"\b(JPY|KRW|CNY|CNH|THB|VND|INR|AUD|NZD|SGD|MYR|IDR|PHP|TWD|HKD)\b", prose_to_check)
            and "%" in prose_to_check
        )
        if mentions_currency_and_pct:
            logger.warning(
                "No structured FX data extracted, but the exec summary / top "
                "risk text mentions a currency alongside a percentage -- "
                "DeepSeek likely narrated an FX move without emitting the "
                "matching FX_START block this week. The FX chart will be "
                "omitted this run; no action needed unless this recurs often."
            )

    fx_chart_png = generate_fx_chart_png(fx_data)
    if fx_data and not fx_chart_png:
        logger.warning("FX data was extracted (%d currencies) but chart generation failed or matplotlib is unavailable", len(fx_data))

    # Archived copy (committed to the repo / workflow artifact): embed the
    # chart as a base64 data URI so the file is fully self-contained when
    # opened directly in a browser.
    fx_chart_data_uri = None
    if fx_chart_png:
        fx_chart_data_uri = "data:image/png;base64," + base64.b64encode(fx_chart_png).decode("ascii")

    html_for_archive = generate_newsletter_html(
        signals=signals,
        exec_summary=exec_summary,
        top_risk=top_risk,
        all_links=all_links,
        nb_reports=len(digests),
        truncated=truncated,
        fx_chart_src=fx_chart_data_uri,
    )

    run_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"weekly_digest_{run_date_str}.html"
    report_path.write_text(html_for_archive, encoding="utf-8")
    (REPORTS_DIR / "latest.html").write_text(html_for_archive, encoding="utf-8")
    logger.info("Newsletter written: %s", report_path)

    # Email copy: reference the chart via cid: and attach it inline --
    # more reliable across email clients than a base64 data URI.
    inline_images = []
    if fx_chart_png:
        html_for_email = generate_newsletter_html(
            signals=signals,
            exec_summary=exec_summary,
            top_risk=top_risk,
            all_links=all_links,
            nb_reports=len(digests),
            truncated=truncated,
            fx_chart_src="cid:fx_chart",
        )
        inline_images.append(("fx_chart", fx_chart_png))
    else:
        html_for_email = html_for_archive

    subject = f"Weekly Intelligence Digest - {ORG_NAME} APAC - {run_date_str}"
    send_email(html_for_email, subject, inline_images=inline_images)

    logger.info("=== Done: %d signal(s), %d FX data point(s) across %d report(s) consolidated ===", len(signals), len(fx_data), len(digests))


if __name__ == "__main__":
    main()
