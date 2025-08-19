# scripts/scrape.py

import json
import os
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'
# List of selectors for elements to be aggressively removed before markdown conversion.
# This is crucial for cleaning the content of ads, navs, footers, etc.
UNWANTED_ELEMENTS_SELECTORS = [
    'script',
    'style',
    'header',
    'footer',
    'nav',
    '.social-share',
    '.related-articles',
    '#comments',
    '[aria-hidden="true"]',
    'iframe'
]

def clean_html_content(page, selector):
    """
    Uses Playwright's DOM evaluation to remove unwanted elements from the main content.
    This pre-processing step is vital for generating clean Markdown.
    """
    main_content_handle = page.query_selector(selector)
    if not main_content_handle:
        print(f"Warning: Content selector '{selector}' not found on page.")
        return ""

    # This JavaScript function will be executed in the browser's context.
    # It finds the main content element, clones it, removes unwanted children from the clone,
    # and then returns the cleaned HTML.
    cleaned_html = main_content_handle.evaluate(f"""(element) => {{
        const clonedElement = element.cloneNode(true);
        const selectorsToRemove = {json.dumps(UNWANTED_ELEMENTS_SELECTORS)};

        // Remove unwanted elements from the cloned node
        selectorsToRemove.forEach(sel => {{
            clonedElement.querySelectorAll(sel).forEach(el => el.remove());
        }});

        return clonedElement.innerHTML;
    }}""")

    return cleaned_html


def scrape_url(page, target):
    """
    Scrapes a single URL based on the configuration provided.
    """
    url = target['url']
    selector = target['selector']
    output_filename = target['output']

    print(f"Scraping {url}...")
    try:
        page.goto(url, wait_until='domcontentloaded')
        
        # Get the cleaned HTML content from the specified selector
        html_content = clean_html_content(page, selector)

        if not html_content:
            print(f"Could not extract content for {url}. Skipping.")
            return

        # Convert the cleaned HTML to Markdown
        markdown_content = md(html_content, heading_style="ATX")

        # Ensure the output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Save the Markdown content to a file
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Successfully saved content to {output_path}")

    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")


def main():
    """
    Main function to orchestrate the scraping process.
    """
    # Load configuration from JSON file
    try:
        with open(CONFIG_FILE, 'r') as f:
            targets = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for target in targets:
            scrape_url(page, target)

        browser.close()

if __name__ == "__main__":
    main()
