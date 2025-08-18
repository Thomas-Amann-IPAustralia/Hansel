# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from pipeline.semantic_search import KBIndex, KBEntry, extract_rule_candidates
from pipeline.utils import load_urls_map, read_jsonl_lines, stable_id

def load_chunks_from_jsonl_glob(glob_pattern: str) -> List[Dict[str, Any]]:
    chunks = []
    for p in Path('.').glob(glob_pattern):
        for obj in read_jsonl_lines(p):
            chunks.append(obj)
    return chunks

def load_chunks_from_xlsx(path: Path) -> List[Dict[str, Any]]:
    # The exemplar .xlsx appears to contain one JSON object per cell/row in the first column
    df = pd.read_excel(path, engine='openpyxl')
    col = df.columns[0]
    out = []
    for _, row in df.iterrows():
        cell = str(row[col])
        try:
            obj = json.loads(cell)
            out.append(obj)
        except Exception:
            continue
    return out

def to_text(obj: Dict[str, Any]) -> str:
    # Try common keys used in your chunker
    for k in ('text', 'content', 'chunk_text', 'body'):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k]
    # Sometimes text is inside a 'nodes' array
    if 'nodes' in obj and isinstance(obj['nodes'], list):
        txt = ' '.join([n.get('text','') for n in obj['nodes'] if isinstance(n, dict)])
        if txt.strip():
            return txt
    # Fallback to joining all string values
    vals = [v for v in obj.values() if isinstance(v, str)]
    return ' '.join(vals)

def infer_tags(obj: Dict[str, Any]) -> List[str]:
    tags = set()
    frag = (obj.get('fragment') or obj.get('jump_url') or '')
    heading = (obj.get('heading') or obj.get('title') or '')
    if 'heading' in obj or 'title' in obj:
        tags.add('heading')
    if isinstance(obj.get('level'), int):
        tags.add(f'level-{obj["level"]}-heading')
    if isinstance(frag, str) and frag:
        parts = [p for p in frag.split('/') if p]
        tags.update(parts)
    # General APS tags
    txt = to_text(obj).lower()
    for hint in ('heading', 'title case', 'inclusive', 'bias', 'accessibility', 'punctuation', 'citation'):
        if hint in txt:
            tags.add(hint)
    return sorted(tags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks-glob', default='pipeline/kb/chunks/**/*.jsonl', help='Glob to JSONL chunk files')
    ap.add_argument('--xlsx', default='', help='Optional path to a single .xlsx containing JSON rows (like ChunkingExample.xlsx)')
    ap.add_argument('--urls-map', default='kb/urls.map.csv', help='CSV mapping filename->url to hydrate source_url')
    ap.add_argument('--out', default='pipeline/kb/kb_index.json', help='Where to write the TF-IDF index')
    args = ap.parse_args()

    entries = []
    if args.xlsx:
        raw = load_chunks_from_xlsx(Path(args.xlsx))
    else:
        raw = load_chunks_from_jsonl_glob(args.chunks_glob)

    urlmap = load_urls_map(Path(args.urls_map))

    for obj in raw:
        text = to_text(obj).strip()
        if not text:
            continue
        if not extract_rule_candidates(text):
            continue  # keep the index compact & rule-focused
        source_file = (obj.get('file_name') or obj.get('filename') or obj.get('file') or '')
        source_url = obj.get('source_url') or urlmap.get(source_file or '', None)
        tags = infer_tags(obj)
        entries.append(KBEntry(
            entry_id=obj.get('chunk_id') or stable_id(text),
            text=text,
            tags=tags,
            source_url=source_url,
            source_file=source_file
        ))

    kb = KBIndex()
    kb.build(entries)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kb.save(Path(args.out))
    print(f"KB index built with {len(entries)} entries → {args.out}")

if __name__ == '__main__':
    main()
