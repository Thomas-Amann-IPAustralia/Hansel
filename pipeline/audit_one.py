# -*- coding: utf-8 -*-
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as html2md

from pipeline.parse_md import parse_markdown
from pipeline.regex_checks import run_regex_checks
from pipeline.nlp_checks import run_heuristic_checks
from pipeline.semantic_search import KBIndex
from pipeline.utils import slugify, is_title_case, to_title_case

def fetch_url_to_markdown(url: str) -> Dict[str, str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text
    doc = Document(html)
    title = doc.short_title()
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, 'lxml')
    # remove nav/aside/footers commonly kept
    for sel in ['nav', 'footer', 'header', 'aside']:
        for el in soup.select(sel):
            el.decompose()
    clean_html = str(soup)
    md = html2md(clean_html)
    return {'title': title, 'markdown': md}

def apply_layer1_rules(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    if block['type'] == 'heading':
        lvl = block['level']
        text = block['text']
        if lvl == 2:
            if not is_title_case(text):
                issues.append({
                    'rule_id': 'APS-H2-TITLECASE',
                    'tags': ['heading', 'level-2-heading', 'title-case'],
                    'message': 'Level 2 headings must be in Title Case.',
                    'excerpt': text,
                    'suggested_fix': f"“{text}” → “{to_title_case(text)}”."
                })
        # No citations in headings
        if '(' in text and ')' in text and any(c.isdigit() for c in text):
            issues.append({
                'rule_id': 'APS-NO-CITATION-IN-HEADING',
                'tags': ['heading', 'citation'],
                'message': 'Avoid including citations inside headings.',
                'excerpt': text,
                'suggested_fix': 'Move the citation out of the heading and place it in the body text.'
            })
    return issues

def attach_kb_matches(kb: Optional[KBIndex], text: str, tag_hint: Optional[str] = None):
    if kb is None:
        return []
    hits = kb.search(text, top_k=3, tag_hint=tag_hint)
    out = []
    for entry, score in hits:
        out.append({
            'kb_entry_id': entry.entry_id,
            'kb_text': entry.text[:300] + ('…' if len(entry.text) > 300 else ''),
            'kb_tags': entry.tags,
            'kb_source_url': entry.source_url,
            'similarity': round(score, 3)
        })
    return out

def analyze_markdown(md_text: str, kb: Optional[KBIndex]) -> Dict[str, Any]:
    blocks = parse_markdown(md_text)
    issues = []
    for b in blocks:
        # Layer 1
        l1 = apply_layer1_rules(b)
        for it in l1:
            it['layer'] = 'L1'
            it['location'] = {'type': b['type'], 'level': b.get('level'), 'line_start': b.get('line_start')}
            it['kb_matches'] = attach_kb_matches(kb, it.get('message','') + ' ' + (it.get('excerpt') or ''), tag_hint='heading' if 'heading' in it['tags'] else None)
        issues.extend(l1)

        # Layer 2
        l2 = run_regex_checks(b)
        for it in l2:
            it['layer'] = 'L2'
            it['location'] = {'type': b['type'], 'level': b.get('level'), 'line_start': b.get('line_start')}
            it['excerpt'] = it.get('excerpt') or b.get('text','')[:240]
            it['kb_matches'] = attach_kb_matches(kb, it.get('message','') + ' ' + (it.get('excerpt') or ''), tag_hint='citation' if 'citation' in it['tags'] else None)
        issues.extend(l2)

        # Layer 3
        l3 = run_heuristic_checks(b)
        for it in l3:
            it['layer'] = 'L3'
            it['location'] = {'type': b['type'], 'level': b.get('level'), 'line_start': b.get('line_start')}
            it['kb_matches'] = attach_kb_matches(kb, it.get('excerpt',''))
        issues.extend(l3)

    return {'blocks': blocks, 'issues': issues}

def write_reports(out_dir: Path, page_meta: Dict[str, str], analysis: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    slug = slugify(page_meta.get('title') or page_meta.get('url') or 'page')
    base = f"{ts}_{slug}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    # JSON
    payload = {
        'page': page_meta,
        'summary': {
            'issues_found': len(analysis['issues']),
            'by_layer': {
                'L1': sum(1 for i in analysis['issues'] if i['layer']=='L1'),
                'L2': sum(1 for i in analysis['issues'] if i['layer']=='L2'),
                'L3': sum(1 for i in analysis['issues'] if i['layer']=='L3'),
            }
        },
        'issues': analysis['issues']
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Markdown
    lines = []
    lines.append(f"# Audit Report – {page_meta.get('title') or page_meta.get('url')}")
    lines.append(f"_Generated: {ts} UTC_")
    lines.append("")
    lines.append(f"- **URL**: {page_meta.get('url','(local file)')}")
    lines.append(f"- **Issues found**: {len(analysis['issues'])}  ")
    lines.append("")
    for i, issue in enumerate(analysis['issues'], 1):
        lines.append(f"## {i}. [{issue['layer']}] {issue['message']}")
        if issue.get('excerpt'):
            lines.append("")
            lines.append("**Excerpt**")
            lines.append("")
            lines.append("> " + issue['excerpt'].replace('\n', ' ')[:500])
        loc = issue.get('location', {})
        lines.append("")
        lines.append(f"- **Location**: `{loc.get('type')}` {('level ' + str(loc.get('level'))) if loc.get('level') else ''} line ~{loc.get('line_start')}")
        lines.append(f"- **Rule ID**: `{issue['rule_id']}`")
        if issue.get('suggested_fix'):
            lines.append(f"- **Suggested fix**: {issue['suggested_fix']}")
        if issue.get('kb_matches'):
            lines.append("- **Relevant APS guidance (similarity)**:")
            for m in issue['kb_matches']:
                src = f" — source: {m['kb_source_url']}" if m.get('kb_source_url') else ''
                lines.append(f"  - {m['similarity']}: {m['kb_text']}{src}")
        lines.append("")
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    return {'json': str(json_path), 'md': str(md_path)}

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--target-url')
    group.add_argument('--markdown-file')
    ap.add_argument('--kb-index', default='pipeline/kb/kb_index.json')
    ap.add_argument('--output-dir', default='reports/')
    args = ap.parse_args()

    kb = None
    kb_path = Path(args.kb_index)
    if kb_path.exists():
        kb = KBIndex.load(kb_path)

    if args.target_url:
        page = fetch_url_to_markdown(args.target_url)
        page_meta = {'url': args.target_url, 'title': page['title']}
        md_text = page['markdown']
    else:
        p = Path(args.markdown_file)
        page_meta = {'url': None, 'title': p.name}
        md_text = p.read_text(encoding='utf-8')

    analysis = analyze_markdown(md_text, kb)
    out = write_reports(Path(args.output_dir), page_meta, analysis)
    print(json.dumps({'report_json': out['json'], 'report_md': out['md']}, indent=2))

if __name__ == '__main__':
    main()
