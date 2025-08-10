"""
Scrape Ministry of Ecology and Environment (MEE) website - Parallel Version
With incremental JSON saving, progress tracking, and parallel document processing
Run: python 01_scrape_mee.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
from tqdm import tqdm
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    clean_text, calculate_hash, extract_date, extract_doc_number,
    ensure_output_dir, is_valid_document_url, normalize_url,
    save_document_json, load_progress, update_progress
)

# Configuration
BASE_URL = "https://www.mee.gov.cn"
OUTPUT_DIR = "../../01_Data_Raw/03_Policy_Documents/MEE"
MAX_WORKERS = 8  # Number of parallel document scrapers (conservative for government site)
BATCH_DELAY = 1  # Seconds between pages (not individual documents)
REQUEST_TIMEOUT = 30  # Timeout per request
RETRY_ATTEMPTS = 2  # Number of retry attempts for failed documents
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Thread-safe lock for file operations
save_lock = threading.Lock()

def scrape_mee():
    """Main function to scrape MEE website with parallel processing"""
    
    print("=" * 60)
    print("MEE Scraper - PARALLEL VERSION")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("Scraping Ministry of Ecology and Environment documents")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)
    
    # Load progress to check what's already scraped
    progress = load_progress(OUTPUT_DIR)
    scraped_urls = {item['url'] for item in progress.get('scraped_urls', [])}
    print(f"Found {len(scraped_urls)} previously scraped documents")
    
    # Track statistics
    stats = {
        'total_scraped': 0,
        'skipped': 0,
        'failed': 0,
        'start_time': datetime.now()
    }
    
    # Scrape sections
    sections = [
        {"name": "Decrees", "url": "/zcwj/bwj/ling/", "max_pages": 7},
        {"name": "Notices", "url": "/zcwj/bwj/gg/", "max_pages": 34}
    ]
    
    for section in sections:
        print(f"\n{'='*40}")
        print(f"Scraping {section['name']}...")
        print(f"{'='*40}")
        
        scraped_count = scrape_section_parallel(
            section['url'], 
            section['name'],
            max_pages=section['max_pages'],
            scraped_urls=scraped_urls,
            stats=stats
        )
        
        stats['total_scraped'] += scraped_count
        print(f"✓ Scraped {scraped_count} new documents from {section['name']}")
    
    # Final report
    elapsed = (datetime.now() - stats['start_time']).total_seconds()
    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Total new documents: {stats['total_scraped']}")
    print(f"Skipped (already scraped): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Speed: {stats['total_scraped']/(elapsed+0.1):.1f} docs/second")
    print(f"Data saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

def scrape_section_parallel(section_url, section_name, max_pages=10, scraped_urls=None, stats=None):
    """Scrape a section with parallel document processing"""
    scraped_urls = scraped_urls or set()
    stats = stats or {}
    index_number = None
    scraped_count = 0
    
    for page in range(1, max_pages + 1):
        # Build page URL
        if page == 1:
            url = BASE_URL + section_url
        else:
            if not index_number:
                index_number = "8597"  # Default based on our research
            url = BASE_URL + section_url + f"index_{index_number}_{page}.shtml"
        
        print(f"\nPage {page}/{max_pages}: {url}")
        
        try:
            # Fetch page (sequential to be respectful to government site)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detect pagination index from first page
            if page == 1 and not index_number:
                index_number = detect_pagination_index(soup)
                if index_number != "8597":
                    print(f"  Detected pagination index: {index_number}")
            
            # Find document links
            links = find_document_links(soup, section_url)
            print(f"  Found {len(links)} articles")
            
            if not links and page > 2:
                print(f"  No more documents found, stopping at page {page}")
                break
            
            # Filter out already scraped documents
            new_links = [link for link in links if link not in scraped_urls]
            
            if not new_links:
                print(f"  All documents already scraped")
                stats['skipped'] = stats.get('skipped', 0) + len(links)
                continue
            
            print(f"  Processing {len(new_links)} new documents...")
            
            # Process documents in parallel
            page_stats = process_documents_parallel(new_links, section_name, scraped_urls)
            
            # Update global stats
            scraped_count += page_stats['success']
            stats['failed'] = stats.get('failed', 0) + page_stats['failed']
            stats['skipped'] = stats.get('skipped', 0) + (len(links) - len(new_links))
            
            print(f"  ✓ Scraped {page_stats['success']} new documents")
            
            # Small delay between pages (not documents)
            if page < max_pages:
                time.sleep(BATCH_DELAY)
            
        except Exception as e:
            print(f"  ✗ Error scraping page {page}: {e}")
            continue
    
    return scraped_count

def process_documents_parallel(links, section_name, scraped_urls):
    """Process multiple documents in parallel"""
    page_stats = {'success': 0, 'failed': 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all scraping tasks
        future_to_url = {
            executor.submit(scrape_document_safe, url, section_name): url 
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

def scrape_document_safe(url, section_name, retry_count=0):
    """Thread-safe document scraping with retry logic"""
    try:
        # Create a new session for each thread
        session = requests.Session()
        session.headers.update(HEADERS)
        
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title - prefer <title> tag first to avoid redirect warning
        title = None
        title_selectors = [
            ('title', None),  # Check title tag first
            ('h1', None),
            ('div', {'class': 'title'}),
            ('div', {'class': 'article-title'}),
            ('div', {'id': 'title'})
        ]
        
        for tag, attrs in title_selectors:
            element = soup.find(tag, attrs)
            if element:
                title = clean_text(element.get_text())
                if title and len(title) > 5 and '您访问的链接' not in title:
                    break
        
        # If we got redirect warning, extract title from content
        if title and '您访问的链接' in title:
            # Extract content first to get real title
            content = None
            content_selectors = [
                ('div', {'class': 'Custom_UnionStyle'}),
                ('div', {'class': 'content_body'}),
                ('div', {'class': 'TRS_Editor'}),
                ('div', {'class': 'content'}),
                ('div', {'class': 'article-content'}),
                ('div', {'id': 'content'}),
                ('div', {'class': 'view TRS_UEDITOR'}),
                ('div', {'class': 'main-content'})
            ]
            
            for tag, attrs in content_selectors:
                element = soup.find(tag, attrs)
                if element:
                    content = clean_text(element.get_text())
                    if content and len(content) > 100:
                        break
            
            if not content:
                # Try to get main text if specific content div not found
                for script in soup(["script", "style"]):
                    script.decompose()
                content = clean_text(soup.get_text())
            
            # Extract title from beginning of content
            if content and len(content) > 20:
                import re
                # Look for common title patterns
                # Pattern 1: Text ending with "公告" or "通知"
                match = re.match(r'^([^。]+(?:公告|通知|办法|规定|标准|名录|目录|决定|意见|批复))', content)
                if match:
                    title = match.group(1).strip()
                else:
                    # Pattern 2: First 100 chars before first period
                    first_sentence = content.split('。')[0][:100]
                    if len(first_sentence) > 10:
                        title = first_sentence.strip()
        
        if not title:
            if retry_count < RETRY_ATTEMPTS:
                time.sleep(2)  # Brief pause before retry
                return scrape_document_safe(url, section_name, retry_count + 1)
            return False
        
        # Extract content
        content = None
        content_selectors = [
            ('div', {'class': 'Custom_UnionStyle'}),
            ('div', {'class': 'content_body'}),
            ('div', {'class': 'TRS_Editor'}),
            ('div', {'class': 'content'}),
            ('div', {'class': 'article-content'}),
            ('div', {'id': 'content'}),
            ('div', {'class': 'view TRS_UEDITOR'}),
            ('div', {'class': 'main-content'})
        ]
        
        for tag, attrs in content_selectors:
            element = soup.find(tag, attrs)
            if element:
                content = clean_text(element.get_text())
                if content and len(content) > 100:
                    break
        
        if not content:
            # Try to get main text if specific content div not found
            for script in soup(["script", "style"]):
                script.decompose()
            content = clean_text(soup.get_text())
            
            if len(content) < 100:
                return False
        
        # Create document data
        doc_data = {
            'doc_id': calculate_hash(url)[:16],
            'url': url,
            'title': title,
            'content': content,
            'section': section_name,
            'publish_date': extract_date(str(soup)),
            'doc_number': extract_doc_number(title + ' ' + content[:500]),
            'content_length': len(content),
            'content_hash': calculate_hash(content),
            'scraped_at': datetime.now(),
            'source': 'MEE',
            'source_name_cn': '生态环境部',
            'source_name_en': 'Ministry of Ecology and Environment'
        }
        
        # Thread-safe save with lock
        with save_lock:
            save_document_json(doc_data, section_name, OUTPUT_DIR)
            update_progress(url, section_name, OUTPUT_DIR)
        
        session.close()
        return True
        
    except requests.exceptions.Timeout:
        if retry_count < RETRY_ATTEMPTS:
            time.sleep(2)
            return scrape_document_safe(url, section_name, retry_count + 1)
        return False
    except Exception:
        if retry_count < RETRY_ATTEMPTS:
            time.sleep(2)
            return scrape_document_safe(url, section_name, retry_count + 1)
        return False

def detect_pagination_index(soup):
    """Detect the pagination index pattern from the page's JavaScript"""
    import re
    
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string:
            match = re.search(r'index_(\d+)_.*\.shtml', script.string)
            if match:
                return match.group(1)
    
    return "8597"  # Default

def find_document_links(soup, section_path):
    """Extract document links from list page"""
    links = []
    
    # Look for links in containers or entire page
    containers = soup.find_all(['ul', 'ol', 'div'], 
                              class_=lambda x: x and ('list' in str(x).lower() or 'content' in str(x).lower()))
    
    if not containers:
        containers = [soup]
    
    for container in containers:
        for a in container.find_all('a', href=True):
            href = a['href']
            
            # Skip navigation and non-document links
            if '#' in href or 'javascript:' in href.lower():
                continue
            
            # Build full URL
            if href.startswith('..'):
                href = href.replace('../', '')
                full_url = BASE_URL + '/' + href
            elif href.startswith('/'):
                full_url = BASE_URL + href
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = BASE_URL + section_path + href
            
            # Normalize URL
            full_url = normalize_url(full_url, BASE_URL)
            
            # Check if it's a valid document URL - UPDATED PATTERNS
            valid_patterns = ['xxgk', 'gkml', 'content', 'article', 't20', 'hbb']
            if is_valid_document_url(full_url) and any(pattern in full_url for pattern in valid_patterns):
                links.append(full_url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    return unique_links

if __name__ == "__main__":
    scrape_mee()