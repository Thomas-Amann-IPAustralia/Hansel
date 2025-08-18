# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Any

CITATION_PAREN_YEAR = re.compile(r'\([^)]*\b\d{4}\b[^)]*\)')
ET_AL_MISSING_PERIOD = re.compile(r'\bet al(?!\.)\b', flags=re.IGNORECASE)
DOUBLE_SPACE_AFTER_PERIOD = re.compile(r'\.\s{2,}')
MALFORMED_URL_SPACE = re.compile(r'https?://\S*\s+\S*')
DOI_PATTERN = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b')

def run_regex_checks(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = block.get('text', '') or ''
    issues = []
    # Citations (any (...) containing a 4-digit year)
    for m in CITATION_PAREN_YEAR.finditer(text):
        issues.append({
            'rule_id': 'REGEX-CITATION-YEAR',
            'tags': ['citation'],
            'message': 'Potential citation detected.',
            'span': (m.start(), m.end()),
            'suggested_fix': 'Verify citation format per APS; avoid placing citations in headings.'
        })
    # et al missing period
    for m in ET_AL_MISSING_PERIOD.finditer(text):
        issues.append({
            'rule_id': 'APS-ET-AL-PERIOD',
            'tags': ['citation', 'et-al'],
            'message': "In 'et al.', a period must follow 'al'.",
            'span': (m.start(), m.end()),
            'suggested_fix': "Change 'et al' → 'et al.'."
        })
    # Double space after period
    for m in DOUBLE_SPACE_AFTER_PERIOD.finditer(text):
        issues.append({
            'rule_id': 'PUNCT-DOUBLE-SPACE',
            'tags': ['punctuation'],
            'message': 'Double space after a period.',
            'span': (m.start(), m.end()),
            'suggested_fix': 'Replace with a single space.'
        })
    # Malformed URLs (space breaks)
    for m in MALFORMED_URL_SPACE.finditer(text):
        issues.append({
            'rule_id': 'URL-SPACE',
            'tags': ['url'],
            'message': 'URL appears to contain a space.',
            'span': (m.start(), m.end()),
            'suggested_fix': 'Remove spaces; ensure full URL is contiguous.'
        })
    # DOI presence (does not validate beyond pattern)
    for m in DOI_PATTERN.finditer(text):
        issues.append({
            'rule_id': 'DOI-DETECTED',
            'tags': ['doi'],
            'message': 'DOI detected; ensure APS formatting.',
            'span': (m.start(), m.end()),
            'suggested_fix': 'Check that DOI is correct and formatted per APS.'
        })
    return issues
