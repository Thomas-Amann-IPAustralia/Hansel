# -*- coding: utf-8 -*-
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import time
import requests
from http import HTTPStatus
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http.cookiejar as cookiejar
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as html2md

from pipeline.parse_md import parse_markdown
from pipeline.regex_checks import run_regex_checks
from pipeline.nlp_checks import run_heuristic_checks
from pipeline.semantic_search import KBIndex
from pipeline.utils import slugify, is_title_case, to_title_case
from pipeline.polite import can_fetch, polite_sleep

# Optional Playwright import (lazy)
try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except Exception:
    HAVE_PLAYWRIGHT = False


def _make_session(
    retries: int = 5,
    backoff: float = 1.0,
    connect_timeout: float = 15.0,
    read_timeout: float = 120.0,
):
    sess = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
        "Referer": "https://www.google.com/",
    })

    # store timeouts and a cookie jar
    sess.request_timeout = (connect_timeout, read_timeout)
    sess.cookies = cookiejar.CookieJar()
    return sess


def fetch_url_to_markdown(
    url: str,
    retries: int = 5,
    backoff: float = 1.0,
    connect_timeout: float = 15.0,
    read_timeout: float = 120.0,
) -> Dict[str, str]:
    """
    Fetch URL -> clean HTML (Readability + BS4) -> Markdown.
    Raises on final failure.
    """
    if not can_fetch(url):
        raise RuntimeError("robots.txt disallows fetching this URL for the configured user-agent.")

    sess = _make_session(retries, backoff, connect_timeout, read_timeout)
    attempts = retries + 1
    resp = None

    for i in range(attempts):
        try:
            resp = sess.get(url, timeout=sess.request_timeout, allow_redirects=True)
            if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                retry_after = int(resp.headers.get("Retry-After", "0") or "0")
                # honor Retry-After plus a small jitter
                polite_sleep(1.0 + retry_after, 0.25)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.ReadTimeout:
            if i == attempts - 1:
                raise
            polite_sleep(1.5 * (i + 1), 0.5)
        except requests.exceptions.RequestException:
            if i == attempts - 1:
                raise
            polite_sleep(1.0 * (i + 1), 0.5)

    if resp is None:
        raise RuntimeError("Unable to fetch the URL after retries.")

    html = resp.text

    # Readability extraction
    try:
        doc = Document(html)
        title = doc.short_title()
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
    except Exception:
        # fall back to basic soup if readability fails
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else url

    # remove noisy sections
    for sel in ["nav", "footer", "header", "aside", "script", "style"]:
        for el in soup.select(sel):
            el.decompose()

    clean_html = str(soup)
    md = html2md(clean_html)
    return {"title": title, "markdown": md}


def fetch_html_via_playwright(url: str, wait_until: str = "networkidle", timeout_ms: int = 45000) -> str:
    """
    Render the page in headless Chromium and return the HTML.
    """
    if not HAVE_PLAYWRIGHT:
        raise RuntimeError("Playwright not installed.")
    if not can_fetch(url):
        raise RuntimeError("robots.txt disallows fetching this URL for the configured user-agent.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ])
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-AU",
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(url, wait_until=wait_until)

        # gentle scroll to trigger lazy-loading
        page.evaluate("""() => new Promise(resolve => {
            let y=0, step=400; (function scroll(){ y+=step; window.scrollTo(0,y);
            if(y<document.body.scrollHeight){ setTimeout(scroll, 150); } else { resolve(); } })();
        })""")
        page.wait_for_timeout(800)

        html = page.content()
        ctx.close()
        browser.close()
        return html


def apply_layer1_rules(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Layer 1 (AST) rules: H2 Title Case; no citations in headings.
    """
    issues = []
    if block["type"] == "heading":
        lvl = block.get("level")
        text = block.get("text", "")

        if lvl == 2:
            if not is_title_case(text):
                issues.append({
                    "rule_id": "APS-H2-TITLECASE",
                    "tags": ["heading", "level-2-heading", "title-case"],
                    "message": "Level 2 headings must be in Title Case.",
                    "excerpt": text,
                    "suggested_fix": f'"{text}" → "{to_title_case(text)}".',
                })

        # discourage citations in headings
        if "(" in text and ")" in text and any(c.isdigit() for c in text):
            issues.append({
                "rule_id": "APS-NO-CITATION-IN-HEADING",
                "tags": ["heading", "citation"],
                "message": "Avoid including citations inside headings.",
                "excerpt": text,
                "suggested_fix": "Move the citation out of the heading and place it in the body text.",
            })
    return issues


def attach_kb_matches(kb: Optional[KBIndex], text: str, tag_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    if kb is None or not text.strip():
        return []
    hits = kb.search(text, top_k=3, tag_hint=tag_hint)
    out = []
    for entry, score in hits:
        out.append({
            "kb_entry_id": entry.entry_id,
            "kb_text": entry.text[:300] + ("…" if len(entry.text) > 300 else ""),
            "kb_tags": entry.tags,
            "kb_source_url": entry.source_url,
            "similarity": round(score, 3),
        })
    return out


def analyze_markdown(md_text: str, kb: Optional[KBIndex]) -> Dict[str, Any]:
    blocks = parse_markdown(md_text)
    issues: List[Dict[str, Any]] = []

    for b in blocks:
        # Layer 1
        l1 = apply_layer1_rules(b)
        for it in l1:
            it["layer"] = "L1"
            it["location"] = {"type": b["type"], "level": b.get("level"), "line_start": b.get("line_start")}
            it["kb_matches"] = attach_kb_matches(
                kb,
                (it.get("message", "") + " " + (it.get("excerpt") or "")),
                tag_hint="heading" if "heading" in it["tags"] else None,
            )
        issues.extend(l1)

        # Layer 2
        l2 = run_regex_checks(b)
        for it in l2:
            it["layer"] = "L2"
            it["location"] = {"type": b["type"], "level": b.get("level"), "line_start": b.get("line_start")}
            it["excerpt"] = it.get("excerpt") or b.get("text", "")[:240]
            it["kb_matches"] = attach_kb_matches(
                kb,
                (it.get("message", "") + " " + (it.get("excerpt") or "")),
                tag_hint="citation" if "citation" in it["tags"] else None,
            )
        issues.extend(l2)

        # Layer 3
        l3 = run_heuristic_checks(b)
        for it in l3:
            it["layer"] = "L3"
            it["location"] = {"type": b["type"], "level": b.get("level"), "line_start": b.get("line_start")}
            it["kb_matches"] = attach_kb_matches(kb, it.get("excerpt", ""))
        issues.extend(l3)

    return {"blocks": blocks, "issues": issues}


def write_reports(out_dir: Path, page_meta: Dict[str, str], analysis: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    slug = slugify(page_meta.get("title") or page_meta.get("url") or "page")
    base = f"{ts}_{slug}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    payload = {
        "page": page_meta,
        "summary": {
            "issues_found": len(analysis["issues"]),
            "by_layer": {
                "L1": sum(1 for i in analysis["issues"] if i["layer"] == "L1"),
                "L2": sum(1 for i in analysis["issues"] if i["layer"] == "L2"),
                "L3": sum(1 for i in analysis["issues"] if i["layer"] == "L3"),
            },
        },
        "issues": analysis["issues"],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Audit Report – {page_meta.get('title') or page_meta.get('url')}")
    lines.append(f"_Generated: {ts} UTC_")
    lines.append("")
    lines.append(f"- **URL**: {page_meta.get('url', '(local file)')}")
    lines.append(f"- **Issues found**: {len(analysis['issues'])}")
    lines.append("")
    for i, issue in enumerate(analysis["issues"], 1):
        lines.append(f"## {i}. [{issue['layer']}] {issue['message']}")
        if issue.get("excerpt"):
            lines.append("")
            lines.append("**Excerpt**")
            lines.append("")
            lines.append("> " + issue["excerpt"].replace("\n", " ")[:500])
        loc = issue.get("location", {})
        lines.append("")
        lines.append(f"- **Location**: `{loc.get('type')}` {('level ' + str(loc.get('level'))) if loc.get('level') else ''} line ~{loc.get('line_start')}")
        lines.append(f"- **Rule ID**: `{issue['rule_id']}`")
        if issue.get("suggested_fix"):
            lines.append(f"- **Suggested fix**: {issue['suggested_fix']}")
        if issue.get("kb_matches"):
            lines.append("- **Relevant APS guidance (similarity)**:")
            for m in issue["kb_matches"]:
                src = f" — source: {m['kb_source_url']}" if m.get("kb_source_url") else ""
                lines.append(f"  - {m['similarity']}: {m['kb_text']}{src}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-url")
    group.add_argument("--markdown-file")
    group.add_argument("--html-file", help="Provide a pre-fetched HTML file to convert & audit")
    ap.add_argument("--kb-index", default="pipeline/kb/kb_index.json")
    ap.add_argument("--output-dir", default="reports/")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=1.0)
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    ap.add_argument("--read-timeout", type=float, default=120.0)
    ap.add_argument("--use-playwright", action="store_true", help="Allow Playwright fallback")
    args = ap.parse_args()

    kb = None
    kb_path = Path(args.kb_index)
    if kb_path.exists():
        kb = KBIndex.load(kb_path)

    # If a pre-fetched HTML file is provided, use it
    if args.html_file:
        p = Path(args.html_file)
        html = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for sel in ["nav", "footer", "header", "aside", "script", "style"]:
            for el in soup.select(sel):
                el.decompose()
        title = soup.title.get_text(strip=True) if soup.title else p.name
        md_text = html2md(str(soup))
        page_meta = {"url": None, "title": title}

    # Otherwise, audit a local Markdown file
    elif args.markdown_file:
        p = Path(args.markdown_file)
        page_meta = {"url": None, "title": p.name}
        md_text = p.read_text(encoding="utf-8")

    # Otherwise, fetch from the target URL
    else:
        # If the workflow pre-downloaded HTML via curl, prefer it
        curl_path = Path(".cache/page.html")
        if curl_path.exists():
            html = curl_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            for sel in ["nav", "footer", "header", "aside", "script", "style"]:
                for el in soup.select(sel):
                    el.decompose()
            title = soup.title.get_text(strip=True) if soup.title else args.target_url
            md_text = html2md(str(soup))
            page_meta = {"url": args.target_url, "title": title}
        else:
            # Try robust requests path
            try:
                page = fetch_url_to_markdown(
                    args.target_url,
                    retries=args.retries,
                    backoff=args.backoff,
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                )
                page_meta = {"url": args.target_url, "title": page["title"]}
                md_text = page["markdown"]
            except Exception:
                # Optionally try Playwright
                if args.use_playwright and HAVE_PLAYWRIGHT:
                    html = fetch_html_via_playwright(args.target_url)
                    soup = BeautifulSoup(html, "lxml")
                    for sel in ["nav", "footer", "header", "aside", "script", "style"]:
                        for el in soup.select(sel):
                            el.decompose()
                    title = soup.title.get_text(strip=True) if soup.title else args.target_url
                    md_text = html2md(str(soup))
                    page_meta = {"url": args.target_url, "title": title}
                else:
                    raise

    analysis = analyze_markdown(md_text, kb)
    out = write_reports(Path(args.output_dir), page_meta, analysis)
    print(json.dumps({"report_json": out["json"], "report_md": out["md"]}, indent=2))


if __name__ == "__main__":
    main()
