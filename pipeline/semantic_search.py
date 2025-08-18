# -*- coding: utf-8 -*-
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

IMPERATIVE_HINTS = re.compile(
    r'\b(must|should|do not|avoid|use|write|spell|capitali[sz]e|heading|title case|inclusive|bias|accessible)\b',
    flags=re.IGNORECASE
)

@dataclass
class KBEntry:
    entry_id: str
    text: str
    tags: List[str]
    source_url: Optional[str]
    source_file: Optional[str]

class KBIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.entries: List[KBEntry] = []

    def build(self, entries: List[KBEntry]):
        self.entries = entries
        corpus = [e.text for e in entries]
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def save(self, path: Path):
        # minimal pickle-free save to avoid compatibility issues
        data = {
            'entries': [e.__dict__ for e in self.entries],
            'vocab': self.vectorizer.vocabulary_,
            'idf': self.vectorizer.idf_.tolist()
        }
        path.write_text(json.dumps(data), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> "KBIndex":
        raw = json.loads(path.read_text(encoding='utf-8'))
        inst = KBIndex()
        inst.entries = [KBEntry(**d) for d in raw['entries']]
        inst.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        inst.vectorizer.vocabulary_ = raw['vocab']
        inst.vectorizer.idf_ = np.array(raw['idf'])
        inst.vectorizer._tfidf._idf_diag = None  # rebuilt lazily
        # rebuild matrix
        corpus = [e.text for e in inst.entries]
        inst.matrix = inst.vectorizer.transform(corpus)
        return inst

    def search(self, text: str, top_k: int = 5, tag_hint: Optional[str] = None) -> List[Tuple[KBEntry, float]]:
        if not text.strip():
            return []
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1]
        results = []
        for idx in order[:top_k*3]:
            entry = self.entries[idx]
            if tag_hint and tag_hint not in entry.tags:
                continue
            results.append((entry, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results

def extract_rule_candidates(chunk_text: str) -> bool:
    # Basic filter: keep chunks likely to be normative rules
    return bool(IMPERATIVE_HINTS.search(chunk_text))
