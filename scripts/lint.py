# scripts/lint.py

import os
import re
import json
import spacy
import logging
import bisect
from typing import List, Dict, Callable, Any
from spacy.tokens import Doc

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
    # The model is now installed via requirements.txt, making this more reliable.
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please ensure it's in your requirements.txt or run 'python -m spacy download en_core_web_sm'")
    exit()

# --- Helper Functions ---
def get_line_number_from_offset(offset: int, line_offsets: List[int]) -> int:
    """
    Finds the line number for a given character offset using binary search.
    This is crucial for mapping spaCy's findings back to the original file lines.
    """
    # bisect_right finds the insertion point, which corresponds to the line number.
    return bisect.bisect_right(line_offsets, offset)

# --- Heuristic Rule Implementations ---
# Each heuristic function now accepts a spaCy Doc object and line offsets.
# They return a list of findings, each with a line number and the offending text.

def check_passive_voice(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic check for passive voice using spaCy's dependency parser."""
    findings = []
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            sent = token.sent
            line_num = get_line_number_from_offset(sent.start_char, line_offsets)
            # Avoid adding the same sentence multiple times
            if not any(f['line_number'] == line_num for f in findings):
                findings.append({
                    "line_number": line_num,
                    "offending_text": sent.text.strip()
                })
    return findings

def check_complete_sentence(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic to check if a sentence is complete (has a subject and a root verb)."""
    findings = []
    for sent in doc.sents:
        # A simple check for sentence fragments. Imperative sentences might be flagged.
        has_root = any(token.dep_ == "ROOT" for token in sent)
        has_subject = any("subj" in token.dep_ for token in sent)
        
        # Ignore very short lines (likely headings/list items) and check for fragments.
        if not (has_root and has_subject) and len(sent.text.strip().split()) > 3:
            line_num = get_line_number_from_offset(sent.start_char, line_offsets)
            findings.append({
                "line_number": line_num,
                "offending_text": sent.text.strip()
            })
    return findings

def check_collective_noun_agreement(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic to check for plural verbs with singular collective nouns."""
    findings = []
    collective_nouns = {"government", "committee", "crowd", "team", "family", "group", "staff"}
    plural_verbs = {"are", "were", "have", "do"}
    for token in doc:
        # Check if a known collective noun is followed by a plural verb
        if token.lemma_.lower() in collective_nouns and token.head.lemma_.lower() in plural_verbs:
            line_num = get_line_number_from_offset(token.idx, line_offsets)
            findings.append({
                "line_number": line_num,
                "offending_text": token.sent.text.strip()
            })
    return findings

# This dictionary maps rule IDs from Trinity.json to our implemented functions.
# This addresses the supervisor's note about implementing more heuristics.
HEURISTIC_CHECKS: Dict[str, Callable[[Doc, List[int]], List[Dict[str, Any]]]] = {
    "APS-GPC-Partsofsentences-H-009": check_passive_voice,
    "APS-GPC-Partsofsentences-H-001": check_complete_sentence,
    "APS-GPC-Nouns-R-004": check_collective_noun_agreement,
}

def load_rules_from_rulebook(file_path: str) -> List[Dict[str, Any]]:
    """Loads and parses linting rules from the specified JSON rulebook."""
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
            new_rule["pattern"] = rule.get("pattern")
            transformed_rules.append(new_rule)
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
    """
    Applies all defined linting rules to a single file.
    This function is refactored to read the whole file at once, enabling
    context-aware heuristic checks that span multiple lines.
    """
    findings: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Could not find file {file_path}")
        return []

    lines = content.splitlines()
    # Pre-calculate the starting character offset of each line for fast lookups.
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    # Process the entire document with spaCy once for efficiency.
    doc = nlp(content)

    for rule in linting_rules:
        try:
            if rule.get('type') == 'regex':
                # Regex checks still operate line-by-line.
                for line_num, line in enumerate(lines, 1):
                    if re.search(rule['pattern'], line, re.IGNORECASE):
                        findings.append({
                            "fileName": file_name, "lineNumber": line_num,
                            "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                            "severity": rule.get('severity'), "offendingText": line.strip(),
                            "githubUrl": build_github_url(file_name, line_num)
                        })
            elif rule.get('type') == 'heuristic':
                # Heuristic checks receive the full spaCy doc for context.
                heuristic_findings = rule['check'](doc, line_offsets)
                for h_finding in heuristic_findings:
                    findings.append({
                        "fileName": file_name, "lineNumber": h_finding['line_number'],
                        "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                        "severity": rule.get('severity'), "offendingText": h_finding['offending_text'],
                        "githubUrl": build_github_url(file_name, h_finding['line_number'])
                    })
        except re.error as e:
            logging.warning(f"Skipping invalid regex for rule '{rule.get('id', 'N/A')}': {e}")
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

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_findings, f, indent=2)

    logging.info(f"Linting complete. Report generated at {REPORT_FILE}")
    logging.info(f"Found {len(all_findings)} issues.")

if __name__ == "__main__":
    main()
