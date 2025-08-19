# scripts/scrape.py

import json
import os
import random
import time
from playwright.sync_api import sync_playwright, Error
import trafilatura

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'
MAX_RETRIES = 3
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/108.0',
]

def scrape_url(context, target):
    """
    Scrapes a single URL by using Playwright to load the page and then
    trafilatura to intelligently extract the main content.
    """
    url = target['url']
    output_filename = target['output']

    if url.lower().endswith('.pdf'):
        print(f"Skipping PDF link: {url}")
        return

    print(f"Processing {url}...")
    page = None
    for attempt in range(MAX_RETRIES):
        try:
            page = context.new_page()
            # Use Playwright to robustly navigate and load the page content.
            page.goto(url, wait_until='networkidle', timeout=60000)
            html_content = page.content()
            page.close() # Close the page to conserve resources

            if not html_content:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Page loaded but content is empty. Retrying...")
                time.sleep(2 ** attempt)
                continue

            # Now, use trafilatura on the fetched HTML to extract the main article.
            markdown_content = trafilatura.extract(html_content, output_format='markdown', include_comments=False, include_tables=True)

            if not markdown_content:
                print(f"Could not extract main content from {url}. Skipping.")
                return

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Successfully extracted and saved content to {output_path}")
            return

        except Error as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: A Playwright error occurred for {url}: {e}")
            if page:
                page.close()
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to scrape {url} after {MAX_RETRIES} attempts.")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: A general error occurred for {url}: {e}")
            if page:
                page.close()
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to process {url} after {MAX_RETRIES} attempts.")


def main():
    """
    Main function to orchestrate the scraping process.
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            targets = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Create a single browser context with a random user agent.
        context = browser.new_context(user_agent=random.choice(USER_AGENTS))
        
        for target in targets:
            scrape_url(context, target)

        browser.close()

if __name__ == "__main__":
    main()
