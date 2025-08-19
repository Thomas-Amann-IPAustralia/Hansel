# scripts/scrape.py

import json
import os
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'

def scrape_url(page, target):
    """
    Scrapes a single URL based on the configuration provided.
    This version scrapes the entire body for a "noisy" but complete capture.
    """
    url = target['url']
    output_filename = target['output']

    print(f"Scraping {url}...")
    try:
        page.goto(url, wait_until='domcontentloaded')
        
        # Get the full HTML content of the page
        html_content = page.content()

        if not html_content:
            print(f"Could not retrieve content for {url}. Skipping.")
            return

        # Convert the HTML to Markdown, stripping out script and style tags which are pure noise.
        markdown_content = md(html_content, heading_style="ATX", strip=['script', 'style', 'header', 'footer', 'nav'])

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
