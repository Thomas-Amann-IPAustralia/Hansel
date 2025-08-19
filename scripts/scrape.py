# scripts/scrape.py

import json
import os
import random
import time
import logging
from playwright.sync_api import sync_playwright, Error, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'
LOG_DIR = 'logs'
MAX_RETRIES = 2 # Set to 2 for 1 initial attempt + 2 retries = 3 total

# A list of common browser user agents to rotate through.
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
]

# --- Setup Structured Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'scraper.log')),
        logging.StreamHandler()
    ]
)

# --- Content Cleaning & Detection Configuration ---
TAGS_TO_EXCLUDE = ['nav', 'footer', 'header', 'script', 'style', 'aside', '.noprint', '#sidebar', 'iframe']
BLOCK_PAGE_SIGNATURES = [
    "access denied", "enable javascript", "checking if the site connection is secure",
    "just a moment...", "verifying you are human", "ddos protection by", "site can’t be reached"
]

def clean_html_content(html):
    """
    Uses BeautifulSoup to parse HTML and remove unwanted tags before text extraction.
    """
    soup = BeautifulSoup(html, 'html.parser')
    page_body = soup.body
    if page_body:
        for tag_selector in TAGS_TO_EXCLUDE:
            for tag in page_body.select(tag_selector):
                tag.decompose()
        return str(page_body)
    return ""

def scrape_url(context, target):
    """
    Scrapes a single URL using Playwright with advanced stealth options and robust error handling.
    """
    url = target['url']
    output_filename = target['output']

    if url.lower().endswith('.pdf'):
        logging.warning(f"Skipping PDF link: {url}")
        return

    logging.info(f"Processing URL: {url}")
    page = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            page = context.new_page()
            # Navigate with a generous timeout and wait for the page to be mostly idle.
            page.goto(url, wait_until='networkidle', timeout=60000)

            # --- Human-like Interaction ---
            logging.info(f"Simulating user interaction for {url}")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 4);")
            time.sleep(random.uniform(1.5, 3.5))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(1.0, 2.5))

            html_content = page.content()
            
            # Check for block page signatures
            if any(sig in html_content.lower() for sig in BLOCK_PAGE_SIGNATURES):
                logging.warning(f"Block page or browser error detected at {url}. Aborting scrape for this URL.")
                if page: page.close()
                return

            cleaned_html = clean_html_content(html_content)
            
            if not cleaned_html:
                logging.warning(f"Could not extract body content from {url}. Skipping.")
                if page: page.close()
                return

            markdown_content = md(cleaned_html, heading_style="ATX")

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logging.info(f"Successfully saved content to {output_path}")
            if page: page.close()
            return

        except PlaywrightTimeoutError:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: Timed out loading {url}")
        except Error as e:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: A Playwright error occurred for {url}: {type(e).__name__} - {e}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: A general error occurred for {url}: {e}")
        
        if page:
            page.close()
        
        if attempt < MAX_RETRIES:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
        else:
            logging.error(f"Failed to scrape {url} after {MAX_RETRIES + 1} attempts.")

def main():
    try:
        with open(CONFIG_FILE, 'r') as f:
            targets = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{CONFIG_FILE}' not found. Exiting.")
        return

    # --- Advanced Browser Configuration ---
    # These arguments help mask the fact that it's an automated browser.
    browser_args = [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-blink-features=AutomationControlled',
        '--disable-gpu',
        '--window-size=1920,1080',
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=browser_args)
        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={'width': 1920, 'height': 1080},
            java_script_enabled=True, # Ensure JS is enabled
            ignore_https_errors=True, # Helps with some government sites with certificate issues
        )
        
        for target in targets:
            scrape_url(context, target)

        browser.close()
    logging.info("Scraping process complete.")

if __name__ == "__main__":
    main()
