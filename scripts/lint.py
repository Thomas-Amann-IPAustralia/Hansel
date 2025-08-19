# scripts/lint.py

import os
import re
import json
import spacy
import logging
import bisect
from typing import List, Dict, Callable, Any, Optional, Pattern
from spacy.tokens import Doc, Span, Token

# --- Configuration ---
MARKDOWN_DIR: str = 'markdown'
REPORT_FILE: str = 'report.json'
LOG_DIR: str = 'logs'
RULEBOOK_FILE: str = 'Trinity.json' 

# --- Setup Structured Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'linter.log')),
        logging.StreamHandler()
    ]
)

# --- spaCy Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please ensure it's in your requirements.txt or run 'python -m spacy download en_core_web_sm'")
    exit()

# --- Helper Functions ---
def get_line_number_from_offset(offset: int, line_offsets: List[int]) -> int:
    """Finds the line number for a given character offset using binary search."""
    return bisect.bisect_right(line_offsets, offset)

def _add_finding(findings: List[Dict], line_number: int, offending_text: str):
    """Helper to prevent duplicate findings for the same line."""
    if not any(f['line_number'] == line_number for f in findings):
        findings.append({
            "line_number": line_number,
            "offending_text": offending_text.strip()
        })

# --- Heuristic Rule Implementations ---
# Each function now returns a list of dictionaries, each containing the line number and offending text.

def check_passive_voice(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic check for passive voice (Rule: APS-GPC-Partsofsentences-H-009)."""
    findings = []
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            _add_finding(findings, get_line_number_from_offset(token.sent.start_char, line_offsets), token.sent.text)
    return findings

def check_complete_sentence(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic to check for sentence fragments (Rule: APS-GPC-Partsofsentences-H-001)."""
    findings = []
    for sent in doc.sents:
        has_root = any(token.dep_ == "ROOT" for token in sent)
        has_subject = any("subj" in token.dep_ for token in sent)
        # Check for fragments, but ignore short lines which might be headings or list items.
        if not (has_root and has_subject) and len(sent.text.strip().split()) > 3:
            _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_collective_noun_agreement(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for plural verbs with singular collective nouns (Rule: APS-GPC-Nouns-R-004)."""
    findings = []
    collective_nouns = {"government", "committee", "crowd", "team", "family", "group", "staff"}
    plural_verbs = {"are", "were", "have", "do"}
    for token in doc:
        if token.lemma_.lower() in collective_nouns and token.head.lemma_.lower() in plural_verbs:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_hyphenated_modifier(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for unhyphenated compound modifiers before a noun (Rule: APS-GPC-Adjectives-H-002)."""
    findings = []
    for i in range(len(doc) - 2):
        token1, token2, token3 = doc[i], doc[i+1], doc[i+2]
        # Pattern: Adjective/Adverb -> Noun/Adjective -> Noun
        is_potential_compound = (token1.pos_ in ['ADJ', 'ADV']) and (token2.pos_ in ['NOUN', 'ADJ'])
        is_before_noun = token3.pos_ == 'NOUN'
        if is_potential_compound and is_before_noun and token2.head == token3 and token1.head == token2:
             _add_finding(findings, get_line_number_from_offset(token1.idx, line_offsets), f"{token1.text} {token2.text} {token3.text}")
    return findings

def check_that_vs_which(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for 'which' without a preceding comma, suggesting it might need to be 'that' (Rule: APS-GPC-Pronouns-H-005)."""
    findings = []
    for i, token in enumerate(doc):
        if token.text.lower() == 'which' and i > 0 and doc[i-1].text != ',':
            # Check if it's introducing a relative clause
            if token.dep_ == 'relcl':
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_missing_determiner(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for singular countable nouns that might be missing a determiner (Rule: APS-GPC-Nouns-H-001)."""
    findings = []
    for token in doc:
        # If a singular noun is a subject but has no determiner ('a', 'the', etc.)
        if token.pos_ == 'NOUN' and token.tag_ == 'NN' and 'subj' in token.dep_:
            children_deps = {child.dep_ for child in token.children}
            if 'det' not in children_deps and 'poss' not in children_deps:
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_exclamation_marks(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags the use of exclamation marks (Rule: APS-GPC-Exclamationmarks-H-001)."""
    findings = []
    for token in doc:
        if token.text == '!':
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_matched_correlatives(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for mismatched correlative conjunctions like 'either/nor' (Rule: APS-GPC-Conjunctions-H-001)."""
    findings = []
    text = doc.text.lower()
    if ('either' in text and 'nor' in text) or ('neither' in text and 'or' in text):
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if ('either' in sent_text and 'nor' in sent_text) or ('neither' in sent_text and 'or' in sent_text):
                _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_prefer_english_forms(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags common Latin abbreviations that should be written in English (Rule: APS-GPC-Latinshortenedforms-H-001)."""
    findings = []
    latin_forms = {'e.g.', 'i.e.', 'etc.'}
    for token in doc:
        if token.text.lower() in latin_forms:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings
    
def check_unique_grading(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags phrases like 'very unique' or 'more unique' (Rule: APS-GPC-Adjectives-H-002)."""
    findings = []
    graders = {"very", "more", "most", "less", "least", "extremely", "highly", "quite"}
    for i, token in enumerate(doc):
        if token.lemma_.lower() == "unique" and i > 0:
            if doc[i-1].lemma_.lower() in graders:
                _add_finding(findings, get_line_number_from_offset(doc[i-1].idx, line_offsets), f"{doc[i-1].text} {token.text}")
    return findings

HEURISTIC_CHECKS: Dict[str, Callable[[Doc, List[int]], List[Dict[str, Any]]]] = {
    # Existing Rules
    "APS-GPC-Partsofsentences-H-009": check_passive_voice,
    "APS-GPC-Partsofsentences-H-001": check_complete_sentence,
    "APS-GPC-Nouns-R-004": check_collective_noun_agreement,
    # New Rules
    "APS-GPC-Adjectives-H-002": check_hyphenated_modifier,
    "APS-GPC-Pronouns-H-005": check_that_vs_which,
    "APS-GPC-Nouns-H-001": check_missing_determiner,
    "APS-GPC-Exclamationmarks-H-001": check_exclamation_marks,
    "APS-GPC-Conjunctions-H-001": check_matched_correlatives,
    "APS-GPC-Latinshortenedforms-H-001": check_prefer_english_forms,
    "APS-GPC-Adjectives-R-002": check_unique_grading,
}

def load_rules_from_rulebook(file_path: str) -> List[Dict[str, Any]]:
    """Loads, validates, and compiles linting rules from the specified JSON rulebook."""
    if not os.path.exists(file_path):
        logging.error(f"Rulebook file '{file_path}' not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rule_sets = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from rulebook: {e}")
        return []

    all_rules = [rule for rule_set in rule_sets for rule in rule_set.get('rules', [])]
    
    transformed_rules = []
    unimplemented_heuristics = 0
    for rule in all_rules:
        rule_type = rule.get("category")
        rule_id = rule.get("id")

        new_rule = {
            "id": rule_id,
            "description": rule.get("message"),
            "severity": rule.get("severity"),
            "type": rule_type
        }

        if rule_type == "regex" and "pattern" in rule:
            pattern = rule.get("pattern", "")
            try:
                flags = re.IGNORECASE if not pattern.startswith("(?i)") else 0
                new_rule["compiled_pattern"] = re.compile(pattern, flags)
                transformed_rules.append(new_rule)
            except re.error as e:
                logging.warning(f"Skipping invalid regex for rule '{rule_id}': {e}")

        elif rule_type == "heuristic":
            if rule_id in HEURISTIC_CHECKS:
                new_rule["check"] = HEURISTIC_CHECKS[rule_id]
                transformed_rules.append(new_rule)
            else:
                unimplemented_heuristics += 1
    
    if unimplemented_heuristics > 0:
        logging.warning(f"Skipped {unimplemented_heuristics} heuristic rules that do not have an implementation.")

    return transformed_rules

def build_github_url(file_name: str, line_number: int) -> str:
    """Constructs a permalink to a specific line in a file on GitHub."""
    server_url = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    sha = os.getenv("GITHUB_SHA")

    if not all([server_url, repository, sha]):
        return f"local://{file_name}#L{line_number}"

    return f"{server_url}/{repository}/blob/{sha}/{MARKDOWN_DIR}/{file_name}#L{line_number}"

def lint_file(file_path: str, file_name: str, linting_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies all defined linting rules to a single file."""
    findings: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Could not find file {file_path}")
        return []

    lines = content.splitlines()
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    doc = nlp(content)

    for rule in linting_rules:
        try:
            if rule.get('type') == 'regex':
                compiled_pattern = rule.get("compiled_pattern")
                if not compiled_pattern:
                    continue
                for line_num, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        findings.append({
                            "fileName": file_name, "lineNumber": line_num,
                            "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                            "severity": rule.get('severity'), "offendingText": line.strip(),
                            "githubUrl": build_github_url(file_name, line_num)
                        })
            elif rule.get('type') == 'heuristic':
                heuristic_findings = rule['check'](doc, line_offsets)
                for h_finding in heuristic_findings:
                    # Prevent adding duplicate findings from heuristics
                    if not any(f['lineNumber'] == h_finding['line_number'] and f['ruleId'] == rule.get('id') for f in findings):
                        findings.append({
                            "fileName": file_name, "lineNumber": h_finding['line_number'],
                            "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                            "severity": rule.get('severity'), "offendingText": h_finding['offending_text'],
                            "githubUrl": build_github_url(file_name, h_finding['line_number'])
                        })
        except Exception as e:
            logging.error(f"Error applying rule '{rule.get('id', 'N/A')}' to {file_name}: {e}")

    return findings

def main() -> None:
    """Main function to orchestrate the linting process and generate the report."""
    all_findings = []
    linting_rules = load_rules_from_rulebook(RULEBOOK_FILE)
    
    if not linting_rules:
        logging.warning("No linting rules were loaded. An empty report will be created.")
    else:
        logging.info(f"Successfully loaded {len(linting_rules)} rules from {RULEBOOK_FILE}.")

    if os.path.exists(MARKDOWN_DIR):
        for file_name in os.listdir(MARKDOWN_DIR):
            if file_name.endswith('.md'):
                file_path = os.path.join(MARKDOWN_DIR, file_name)
                logging.info(f"Linting {file_path}...")
                findings = lint_file(file_path, file_name, linting_rules)
                all_findings.extend(findings)
    else:
        logging.warning(f"Markdown directory '{MARKDOWN_DIR}' not found. No files to lint.")

    # Sort findings for consistency
    all_findings.sort(key=lambda x: (x['fileName'], x['lineNumber']))

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_findings, f, indent=2)

    logging.info(f"Linting complete. Report generated at {REPORT_FILE}")
    logging.info(f"Found {len(all_findings)} issues.")

if __name__ == "__main__":
    main()
