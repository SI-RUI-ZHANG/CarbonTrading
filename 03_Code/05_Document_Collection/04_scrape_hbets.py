"""
Scrape Hubei Emission Trading Scheme (HBETS) - Center Dynamics (Parallel Version)
Historical carbon trading news from 2013-present with parallel processing
Run: python 04_scrape_hbets.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    clean_text, calculate_hash, extract_date,
    ensure_output_dir, save_document_json, 
    load_progress, update_progress
)

# Configuration
BASE_URL = "https://www.hbets.cn"
OUTPUT_DIR = "../../01_Data_Raw/03_Policy_Documents/HBETS"
SECTION_NAME = "Center_Dynamics"
MAX_PAGES = 69  # Total pages as of analysis
MAX_WORKERS = 10  # Number of parallel document scrapers
BATCH_DELAY = 1  # Seconds between pages (not individual documents)
REQUEST_TIMEOUT = 30  # Timeout per request
RETRY_ATTEMPTS = 2  # Number of retry attempts for failed documents

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

# Thread-safe lock for file operations
save_lock = threading.Lock()

def scrape_hbets():
    """Main function to scrape HBETS Center Dynamics with parallel processing"""
    
    print("=" * 60)
    print("HBETS Center Dynamics Scraper - PARALLEL VERSION")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("Scraping 12 years of Hubei carbon trading history")
    print("=" * 60)
    
    # Setup
    ensure_output_dir(OUTPUT_DIR)
    progress = load_progress(OUTPUT_DIR)
    scraped_urls = {item['url'] for item in progress.get('scraped_urls', [])}
    print(f"Found {len(scraped_urls)} previously scraped documents\n")
    
    stats = {
        'total_scraped': 0,
        'skipped': 0,
        'failed': 0,
        'start_time': datetime.now()
    }
    
    # Scrape all pages with parallel processing
    scrape_center_dynamics_parallel(scraped_urls, stats)
    
    # Final report
    elapsed = (datetime.now() - stats['start_time']).total_seconds()
    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETED")
    print(f"Total new documents: {stats['total_scraped']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Speed: {stats['total_scraped']/(elapsed+0.1):.1f} docs/second")
    print(f"Data saved to: {OUTPUT_DIR}/{SECTION_NAME}/")
    print("=" * 60)

def scrape_center_dynamics_parallel(scraped_urls, stats):
    """Scrape Center Dynamics section with parallel document processing"""
    
    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}/list_10.html?page={page}"
        print(f"\nPage {page}/{MAX_PAGES}: {url}")
        
        try:
            # Fetch page (sequential)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract document links
            links = extract_document_links(soup)
            print(f"  Found {len(links)} articles")
            
            if not links and page < MAX_PAGES:
                print(f"  Warning: No links found on page {page}")
                continue
            
            # Filter out already scraped documents
            new_links = [link for link in links if link not in scraped_urls]
            
            if not new_links:
                print(f"  All documents already scraped")
                stats['skipped'] += len(links)
                continue
            
            print(f"  Processing {len(new_links)} new documents...")
            
            # Process documents in parallel
            page_stats = process_documents_parallel(new_links, scraped_urls)
            
            # Update global stats
            stats['total_scraped'] += page_stats['success']
            stats['failed'] += page_stats['failed']
            stats['skipped'] += len(links) - len(new_links)
            
            print(f"  ✓ Scraped {page_stats['success']} new documents")
            
            # Small delay between pages (not documents)
            if page < MAX_PAGES:
                time.sleep(BATCH_DELAY)
            
        except Exception as e:
            print(f"  ✗ Error on page {page}: {e}")

def process_documents_parallel(links, scraped_urls):
    """Process multiple documents in parallel"""
    page_stats = {'success': 0, 'failed': 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all scraping tasks
        future_to_url = {
            executor.submit(scrape_document_safe, url): url 
            for url in links
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(links), desc="    Scraping", leave=False) as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        page_stats['success'] += 1
                        scraped_urls.add(url)
                    else:
                        page_stats['failed'] += 1
                except Exception as e:
                    print(f"\n      Error processing {url}: {e}")
                    page_stats['failed'] += 1
                finally:
                    pbar.update(1)
    
    return page_stats

def scrape_document_safe(url, retry_count=0):
    """Thread-safe document scraping with retry logic"""
    try:
        # Create a new session for each thread
        session = requests.Session()
        session.headers.update(HEADERS)
        
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('span', class_='title') or \
                    soup.find('div', class_='title') or \
                    soup.find('h1')
        
        if not title_elem:
            if retry_count < RETRY_ATTEMPTS:
                time.sleep(2)  # Brief pause before retry
                return scrape_document_safe(url, retry_count + 1)
            return False
        
        title = clean_text(title_elem.get_text())
        
        # Extract content
        content_elem = soup.find('div', class_='wrapIntel') or \
                      soup.find('div', class_='cont_left')
        
        if content_elem:
            # Remove script and style elements
            for elem in content_elem(['script', 'style', 'img']):
                elem.decompose()
            content = clean_text(content_elem.get_text())
        else:
            content = ""
        
        # Handle image-only content
        if len(content) < 50:
            content = "[Image-based content - text extraction limited]"
        
        # Extract date from the time element
        # The HTML structure is: <div class="time...">发布时间：YYYY-MM-DD</div>
        date_text = soup.find('div', class_='time') or soup.find('span', class_='time')
        
        if date_text:
            date_str = date_text.get_text()
            # Remove the "发布时间：" prefix if present
            date_str = date_str.replace('发布时间：', '').replace('发布时间:', '').strip()
            publish_date = extract_date(date_str)
        else:
            publish_date = None
        
        # Extract view count
        view_count = None
        if date_text:
            view_elem = date_text.find_all('i')
            if len(view_elem) >= 2:
                view_text = view_elem[1].get_text()
                view_match = re.findall(r'\d+', view_text)
                view_count = int(view_match[0]) if view_match else None
        
        # Create document
        doc_data = {
            'doc_id': calculate_hash(url)[:16],
            'url': url,
            'title': title,
            'content': content,
            'section': SECTION_NAME,
            'publish_date': publish_date,
            'view_count': view_count,
            'content_length': len(content),
            'content_hash': calculate_hash(content),
            'scraped_at': datetime.now(),
            'source': 'HBETS',
            'source_name_cn': 'Hubei Carbon Emission Trading Center',
            'source_name_en': 'Hubei Emission Trading Scheme'
        }
        
        # Thread-safe save with lock
        with save_lock:
            save_document_json(doc_data, SECTION_NAME, OUTPUT_DIR)
            update_progress(url, SECTION_NAME, OUTPUT_DIR)
        
        session.close()
        return True
        
    except requests.exceptions.Timeout:
        if retry_count < RETRY_ATTEMPTS:
            time.sleep(2)
            return scrape_document_safe(url, retry_count + 1)
        return False
    except Exception as e:
        if retry_count < RETRY_ATTEMPTS:
            time.sleep(2)
            return scrape_document_safe(url, retry_count + 1)
        return False

def extract_document_links(soup):
    """Extract document links from list page"""
    links = []
    
    # Find all dl elements with news items
    news_items = soup.find_all('dl', class_='disflex')
    
    for item in news_items:
        # Find link in dt or dd tag
        link_elem = item.find('a', href=True)
        if link_elem:
            href = link_elem['href']
            # Handle relative and absolute URLs
            if href.startswith('/'):
                full_url = BASE_URL + href
            elif not href.startswith('http'):
                full_url = BASE_URL + '/' + href
            else:
                full_url = href
            
            # Only include view_XXX.html links
            if 'view_' in full_url and '.html' in full_url:
                links.append(full_url)
    
    return list(set(links))  # Remove duplicates

if __name__ == "__main__":
    scrape_hbets()