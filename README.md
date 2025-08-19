Web Content Scraper & APS Style Linter
This project is a GitHub Application designed to automate the process of scraping web content, converting it to Markdown, and auditing it against a set of rules based on the Australian Public Service (APS) style guide.

How It Works
The process is orchestrated by a GitHub Action (.github/workflows/audit.yml) and is composed of two main Python scripts:

scripts/scrape.py: This script reads a list of target URLs from config.json. For each URL, it uses Playwright to fetch the page, aggressively clean the HTML of unwanted elements (like ads, social sharing buttons, and scripts), and then convert the cleaned content into a Markdown file.

scripts/lint.py: This is a unified linting engine that analyzes the generated Markdown files. It applies two types of rules:

Regex-based rules: Simple, pattern-matching rules for common style issues.

Heuristic (NLP) rules: Complex, context-aware rules using the spaCy library to detect grammatical issues, such as the use of passive voice.

The final output is a report.json file, which is uploaded as a workflow artifact. This report provides a detailed list of all issues found, including the file, line number, and a direct permalink to the offending line in the GitHub repository for easy remediation.

Configuration
To add new pages to the audit, simply edit the config.json file in the root of the repository. Add a new object to the array with the following keys:

url: The full URL of the page to scrape.

selector: The CSS selector for the main content container on the page.

output: The desired filename for the resulting Markdown file (e.g., page-one.md).

The GitHub Action is configured to run automatically whenever config.json is modified on the main branch. It can also be triggered manually from the Actions tab in GitHub.
