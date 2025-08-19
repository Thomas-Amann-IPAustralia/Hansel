# scripts/scrape.py

import json
import os
import random
import time
import logging
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# --- Configuration ---
CONFIG_FILE = 'config.json'
OUTPUT_DIR = 'markdown'
LOG_DIR = 'logs'
MAX_RETRIES = 2

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

def initialize_driver():
    """Initializes a stealth-configured Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Apply selenium-stealth modifications
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
        )
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        return None

def clean_html_content(html):
    """Uses BeautifulSoup to parse HTML and remove unwanted tags."""
    soup = BeautifulSoup(html, 'html.parser')
    page_body = soup.body
    if page_body:
        for tag_selector in TAGS_TO_EXCLUDE:
            for tag in page_body.select(tag_selector):
                tag.decompose()
        return str(page_body)
    return ""

def scrape_url(driver, target):
    """Scrapes a single URL using the provided Selenium driver."""
    url = target['url']
    output_filename = target['output']

    if url.lower().endswith('.pdf'):
        logging.warning(f"Skipping PDF link: {url}")
        return

    logging.info(f"Processing URL: {url}")
    for attempt in range(MAX_RETRIES + 1):
        try:
            driver.get(url)
            # Wait for the body tag to ensure the page has started loading.
            WebDriverWait(driver, 45).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Human-like interaction
            logging.info(f"Simulating user interaction for {url}")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 4);")
            time.sleep(random.uniform(1.5, 3.5))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(1.0, 2.5))

            html_content = driver.page_source
            
            if any(sig in html_content.lower() for sig in BLOCK_PAGE_SIGNATURES):
                logging.warning(f"Block page detected at {url}. Aborting.")
                return

            cleaned_html = clean_html_content(html_content)
            
            if not cleaned_html:
                logging.warning(f"Could not extract body content from {url}. Skipping.")
                return

            markdown_content = md(cleaned_html, heading_style="ATX")

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logging.info(f"Successfully saved content to {output_path}")
            return # Success

        except TimeoutException:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: Timed out loading {url}")
        except WebDriverException as e:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: A WebDriver error occurred for {url}: {type(e).__name__}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: A general error occurred for {url}: {e}")
        
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

    driver = initialize_driver()
    if not driver:
        logging.error("Could not start the browser. Aborting scrape.")
        return
        
    try:
        for target in targets:
            scrape_url(driver, target)
    finally:
        driver.quit()
        
    logging.info("Scraping process complete.")

if __name__ == "__main__":
    main()
