# scripts/scrape.py

import json
import os
import random
import time
from playwright.sync_api import sync_playwright, Error
# trafilatura is a powerful library for extracting the main content from a webpage.
import trafilatura

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'
MAX_RETRIES = 3 # Number of times to retry a failed scrape

# A list of common browser user agents to rotate through.
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/108.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/108.0',
]

def scrape_url(browser, target):
    """
    Scrapes a single URL using intelligent content extraction and retry logic.
    """
    url = target['url']
    output_filename = target['output']

    # Gracefully handle links that point directly to PDF files.
    if url.lower().endswith('.pdf'):
        print(f"Skipping PDF link: {url}")
        return

    print(f"Processing {url}...")
    
    for attempt in range(MAX_RETRIES):
        try:
            # Use trafilatura to download and intelligently extract the main content.
            # It handles many anti-scraping measures and focuses on the core text.
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded is None:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Failed to download content from {url}. Retrying...")
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s
                continue

            # Extract the main content and convert it to Markdown.
            # This is much cleaner than scraping the whole body.
            markdown_content = trafilatura.extract(downloaded, output_format='markdown', include_comments=False, include_tables=True)

            if not markdown_content:
                print(f"Could not extract main content from {url}. The page might be empty or a non-standard format. Skipping.")
                return

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Successfully extracted and saved content to {output_path}")
            return # Exit the retry loop on success

        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: An error occurred for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt) # Wait before retrying
            else:
                print(f"Failed to scrape {url} after {MAX_RETRIES} attempts.")

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

    # Playwright is still useful for its robust browser environment,
    # even though trafilatura is handling the direct download.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # We don't need to create a page or context here anymore,
        # but we pass the browser instance for potential future use.
        for target in targets:
            scrape_url(browser, target)

        browser.close()

if __name__ == "__main__":
    main()
