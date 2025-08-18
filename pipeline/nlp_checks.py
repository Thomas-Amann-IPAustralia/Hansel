# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Any

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')

# Rough passive voice heuristic: BE + past participle (…ed|…en), often followed by 'by'
PASSIVE = re.compile(
    r'\b(am|is|are|was|were|be|been|being)\b\s+(\w+(ed|en))(\s+by\b)?',
    flags=re.IGNORECASE
)

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]

def run_heuristic_checks(block: Dict[str, Any], max_words: int = 30) -> List[Dict[str, Any]]:
    issues = []
    text = block.get('text', '') or ''
    if not text.strip():
        return issues

    for sent in split_sentences(text):
        # Long sentences
        words = re.findall(r'\b\w[\w’\'-]*\b', sent)
        if len(words) > max_words:
            issues.append({
                'rule_id': 'HEUR-LONG-SENTENCE',
                'tags': ['readability'],
                'message': f'Sentence longer than {max_words} words.',
                'excerpt': sent,
                'suggested_fix': 'Split into shorter sentences; aim for one idea per sentence.'
            })
        # Passive voice
        if PASSIVE.search(sent):
            issues.append({
                'rule_id': 'HEUR-PASSIVE-VOICE',
                'tags': ['voice'],
                'message': 'Possible passive voice.',
                'excerpt': sent,
                'suggested_fix': 'Prefer active voice where possible.'
            })
    return issues
