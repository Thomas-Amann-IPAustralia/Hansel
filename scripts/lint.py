# scripts/lint.py

import os
import re
import json
import spacy

# --- Configuration ---
MARKDOWN_DIR = 'markdown'
REPORT_FILE = 'report.json'
RULEBOOK_FILE = 'Codebook.json' 

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
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
    Loads and parses linting rules from the rulebook file, which contains a
    stream of multiple JSON objects. This function is designed to be robust
    against extra whitespace or newlines between the JSON objects.
    """
    if not os.path.exists(file_path):
        print(f"Error: Rulebook file '{file_path}' not found.")
        return []

    all_rules = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Use a decoder to handle multiple JSON objects in a single file
        decoder = json.JSONDecoder()
        content = f.read().strip()
        pos = 0
        while pos < len(content):
            try:
                # Decode one JSON object at a time
                obj, end_pos = decoder.raw_decode(content, pos)
                if 'rules' in obj:
                    all_rules.extend(obj['rules'])
                # Move position to the start of the next object
                pos = end_pos
                # Skip any whitespace/newlines between objects
                while pos < len(content) and content[pos].isspace():
                    pos += 1
            except json.JSONDecodeError:
                print(f"Error decoding JSON object near position {pos}. Skipping rest of file.")
                break # Stop if we hit an unrecoverable error

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
                    if rule['type'] == 'regex':
                        if re.search(rule['pattern'], line, re.IGNORECASE):
                            issue_found = True
                    elif rule['type'] == 'heuristic':
                        if rule['check'](line):
                            issue_found = True

                    if issue_found:
                        finding = {
                            "fileName": file_name,
                            "lineNumber": line_num,
                            "ruleId": rule['id'],
                            "ruleDescription": rule['description'],
                            "severity": rule['severity'],
                            "offendingText": line.strip(),
                            "githubUrl": build_github_url(file_name, line_num)
                        }
                        findings.append(finding)
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")

    return findings


def main():
    """
    Main function to orchestrate the linting process and generate the report.
    """
    all_findings = []
    linting_rules = load_rules_from_rulebook(RULEBOOK_FILE)
    
    if not linting_rules:
        print("No linting rules were loaded. An empty report will be created.")
    else:
        print(f"Successfully loaded {len(linting_rules)} rules from {RULEBOOK_FILE}.")

    if os.path.exists(MARKDOWN_DIR):
        for file_name in os.listdir(MARKDOWN_DIR):
            if file_name.endswith('.md'):
                file_path = os.path.join(MARKDOWN_DIR, file_name)
                print(f"Linting {file_path}...")
                findings = lint_file(file_path, file_name, linting_rules)
                all_findings.extend(findings)
    else:
        print(f"Markdown directory '{MARKDOWN_DIR}' not found. No files to lint.")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_findings, f, indent=2)

    print(f"Linting complete. Report generated at {REPORT_FILE}")
    print(f"Found {len(all_findings)} issues.")


if __name__ == "__main__":
    main()
