# scripts/lint.py

import os
import re
import json
import spacy
import logging

# --- Configuration ---
MARKDOWN_DIR = 'markdown'
REPORT_FILE = 'report.json'
LOG_DIR = 'logs'
# The new, improved rulebook file.
RULEBOOK_FILE = 'Trinity.json' 

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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()


# --- Heuristic Rule Implementation ---

def check_passive_voice(text):
    """
    Heuristic check for passive voice using spaCy's dependency parser.
    """
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            return True
    return False

HEURISTIC_CHECKS = {
    "APS-GPC-Partsofsentences-H-009": check_passive_voice,
}


def load_rules_from_rulebook(file_path):
    """
    Loads and parses linting rules from the Trinity.json file.
    This version is simpler because Trinity.json is a valid JSON array.
    """
    if not os.path.exists(file_path):
        logging.error(f"Rulebook file '{file_path}' not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rule_sets = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from rulebook: {e}")
        return []

    all_rules = []
    for rule_set in rule_sets:
        if 'rules' in rule_set:
            all_rules.extend(rule_set['rules'])

    transformed_rules = []
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
        elif rule_type == "heuristic" and rule_id in HEURISTIC_CHECKS:
            new_rule["check"] = HEURISTIC_CHECKS[rule_id]
            transformed_rules.append(new_rule)
            
    return transformed_rules


def build_github_url(file_name, line_number):
    """
    Constructs a permalink to a specific line in a file on GitHub.
    """
    server_url = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    sha = os.getenv("GITHUB_SHA")

    if not all([server_url, repository, sha]):
        return f"local://{file_name}#L{line_number}"

    return f"{server_url}/{repository}/blob/{sha}/{MARKDOWN_DIR}/{file_name}#L{line_number}"


def lint_file(file_path, file_name, linting_rules):
    """
    Applies all defined linting rules to a single file.
    """
    findings = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for rule in linting_rules:
                    issue_found = False
                    try:
                        if rule.get('type') == 'regex':
                            if re.search(rule['pattern'], line, re.IGNORECASE):
                                issue_found = True
                        elif rule.get('type') == 'heuristic':
                            if rule['check'](line):
                                issue_found = True
                    except re.error as e:
                        logging.warning(f"Skipping invalid regex for rule '{rule.get('id', 'N/A')}': {e}")
                        rule['type'] = 'invalid' # Mark as invalid to avoid re-checking
                        continue

                    if issue_found:
                        finding = {
                            "fileName": file_name,
                            "lineNumber": line_num,
                            "ruleId": rule.get('id'),
                            "ruleDescription": rule.get('description'),
                            "severity": rule.get('severity'),
                            "offendingText": line.strip(),
                            "githubUrl": build_github_url(file_name, line_num)
                        }
                        findings.append(finding)
    except FileNotFoundError:
        logging.error(f"Could not find file {file_path}")

    return findings


def main():
    """
    Main function to orchestrate the linting process and generate the report.
    """
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
