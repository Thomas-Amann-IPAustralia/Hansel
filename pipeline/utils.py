# -*- coding: utf-8 -*-
import re
import json
import hashlib
from pathlib import Path

SMALL_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor',
    'as', 'at', 'by', 'for', 'from', 'in', 'into', 'near', 'of',
    'on', 'onto', 'to', 'up', 'with', 'over', 'via'
}

def slugify(text: str) -> str:
    base = re.sub(r'[^a-zA-Z0-9]+', '-', text.strip().lower())
    return re.sub(r'-{2,}', '-', base).strip('-')[:80] or 'report'

def is_title_case(s: str) -> bool:
    words = re.split(r'\s+', s.strip())
    if not words:
        return True
    for i, w in enumerate(words):
        clean = re.sub(r"[^A-Za-z0-9’'’-]", "", w)  # keep apostrophes/hyphens
        if not clean:
            continue
        lw = clean.lower()
        if i in (0, len(words)-1):
            if clean[0].upper() != clean[0]:
                return False
        else:
            if lw in SMALL_WORDS:
                if clean[0].lower() != clean[0]:
                    return False  # small word should be lowercase
            else:
                if clean[0].upper() != clean[0]:
                    return False
    return True

def to_title_case(s: str) -> str:
    words = re.split(r'(\s+)', s.strip())
    out = []
    idx = 0
    for token in words:
        if token.isspace():
            out.append(token)
            continue
        clean = re.sub(r"[^A-Za-z0-9’'’-]", "", token)
        lw = clean.lower()
        if idx == 0 or idx == len([w for w in words if not w.isspace()]) - 1:
            out.append(token[:1].upper() + token[1:])
        else:
            if lw in SMALL_WORDS:
                out.append(token[:1].lower() + token[1:])
            else:
                out.append(token[:1].upper() + token[1:])
        idx += 1
    return ''.join(out)

def load_urls_map(path: Path) -> dict:
    """Accepts a well-formed CSV or a single-column CSV containing 'filename,url' strings."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding='utf-8', errors='ignore')
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    header = lines[0].lower()
    mapping = {}
    for i, line in enumerate(lines[1:] if 'filename' in header and 'url' in header else lines):
        if not line:
            continue
        # tolerate malformed quoting by splitting on first comma
        if ',' in line:
            fname, url = line.split(',', 1)
        else:
            parts = re.split(r'\s+', line, 1)
            if len(parts) != 2: 
                continue
            fname, url = parts
        mapping[fname.strip()] = url.strip()
    return mapping

def read_jsonl_lines(path: Path):
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            # tolerate trailing commas or subtle format issues
            line = re.sub(r',\s*}', '}', line)
            line = re.sub(r',\s*]', ']', line)
            yield json.loads(line)

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
