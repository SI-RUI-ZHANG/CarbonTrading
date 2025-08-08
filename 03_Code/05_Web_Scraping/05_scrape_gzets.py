"""
Scrape Guangzhou Emission Trading Scheme (GZETS) - Multiple Sections (Parallel Version)
Trading announcements, center dynamics, and provincial/municipal news with parallel processing
Run: python 05_scrape_gzets.py
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from tqdm import tqdm
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    clean_text, calculate_hash, extract_date,
    ensure_output_dir, save_document_json, 
    load_progress, update_progress
)

# Configuration
BASE_URL = "https://www.cnemission.com"
OUTPUT_DIR = "../../01_Data_Raw/03_Policy_Documents/GZETS"
MAX_WORKERS = 12  # Number of parallel document scrapers
BATCH_DELAY = 1  # Seconds between pages (not individual documents)
REQUEST_TIMEOUT = 20  # Timeout per request
RETRY_ATTEMPTS = 2  # Number of retry attempts for failed documents

# Section configurations
SECTIONS = [
    {
        'name': 'Trading_Announcements',
        'url': '/article/news/jysgg/',
        'pages': 23,
        'name_cn': '交易公告'
    },
    {
        'name': 'Center_Dynamics',
        'url': '/article/news/jysdt/',
        'pages': 30,
        'name_cn': '交易中心动态'
    },
    {
        'name': 'Provincial_Municipal',
        'url': '/article/news/ssdt/',
        'pages': 22,
        'name_cn': '省市动态'
    }
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

# Thread-safe lock for file operations
save_lock = threading.Lock()

def scrape_gzets():
    """Main function to scrape GZETS sections with parallel processing"""
    
    print("=" * 60)
    print("GZETS Multi-Section Scraper - PARALLEL VERSION")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("Scraping trading announcements and carbon market news")
    print("=" * 60)
    
    # Setup
    ensure_output_dir(OUTPUT_DIR)
    progress = load_progress(OUTPUT_DIR)
    scraped_urls = {item['url'] for item in progress.get('scraped_urls', [])}
    print(f"Found {len(scraped_urls)} previously scraped documents\n")
    
    overall_stats = {
        'total_scraped': 0,
        'total_skipped': 0,
        'total_failed': 0,
        'start_time': datetime.now()
    }
    
    # Scrape each section
    for section in SECTIONS:
        print(f"\n{'=' * 60}")
        print(f"Scraping: {section['name']} ({section['name_cn']})")
        print(f"Pages: {section['pages']}")
        print("=" * 60)
        
        section_stats = scrape_section_parallel(section, scraped_urls)
        
        # Update overall stats
        overall_stats['total_scraped'] += section_stats['scraped']
        overall_stats['total_skipped'] += section_stats['skipped']
        overall_stats['total_failed'] += section_stats['failed']
    
    # Final report
    elapsed = (datetime.now() - overall_stats['start_time']).total_seconds()
    print(f"\n{'=' * 60}")
    print("ALL SECTIONS COMPLETED")
    print(f"Total new documents: {overall_stats['total_scraped']}")
    print(f"Total skipped: {overall_stats['total_skipped']}")
    print(f"Total failed: {overall_stats['total_failed']}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Speed: {overall_stats['total_scraped']/(elapsed+0.1):.1f} docs/second")
    print(f"Data saved to: {OUTPUT_DIR}/")
    print("=" * 60)

def scrape_section_parallel(section_config, scraped_urls):
    """Scrape one section with parallel document processing"""
    
    section_name = section_config['name']
    section_url = section_config['url']
    max_pages = section_config['pages']
    
    stats = {
        'scraped': 0,
        'skipped': 0,
        'failed': 0
    }
    
    for page in range(1, max_pages + 1):
        if page == 1:
            url = f"{BASE_URL}{section_url}"
        else:
            url = f"{BASE_URL}{section_url}?{page}"
        
        print(f"\nPage {page}/{max_pages}: {url}")
        
        try:
            # Fetch page (sequential)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract document links
            links = extract_document_links(soup)
            print(f"  Found {len(links)} articles")
            
            if not links and page < max_pages:
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
            page_stats = process_documents_parallel(new_links, section_name, scraped_urls)
            
            # Update section stats
            stats['scraped'] += page_stats['success']
            stats['failed'] += page_stats['failed']
            stats['skipped'] += len(links) - len(new_links)
            
            print(f"  ✓ Scraped {page_stats['success']} new documents")
            
            # Small delay between pages
            if page < max_pages:
                time.sleep(BATCH_DELAY)
            
        except Exception as e:
            print(f"  ✗ Error on page {page}: {e}")
    
    return stats

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
        
        # Extract title from <h3> tag
        title_elem = soup.find('h3')
        
        if not title_elem:
            if retry_count < RETRY_ATTEMPTS:
                time.sleep(2)  # Brief pause before retry
                return scrape_document_safe(url, section_name, retry_count + 1)
            return False
        
        title = clean_text(title_elem.get_text())
        
        # Extract date from JavaScript variables
        # Pattern: var writetime = "2025-01-06 14:03:31"
        date_pattern = r'var\s+writetime\s*=\s*"([^"]+)"'
        date_match = re.search(date_pattern, response.text)
        
        if date_match:
            date_str = date_match.group(1)
            publish_date = extract_date(date_str)
        else:
            # Try alternative date extraction from HTML
            publish_date = None
        
        # Extract content - main body after header
        # Remove navigation, scripts, styles
        for elem in soup(['script', 'style', 'nav', 'header', 'footer']):
            elem.decompose()
        
        # Try to find main content area
        content_elem = None
        # Try common content containers
        for selector in ['div.content', 'div.article', 'div.main', 'body']:
            content_elem = soup.select_one(selector)
            if content_elem:
                break
        
        if content_elem:
            # Get text after h3 title
            content = ""
            found_title = False
            for elem in content_elem.descendants:
                if isinstance(elem, str):
                    # Text node
                    if found_title:
                        content += elem + " "
                elif hasattr(elem, 'name'):
                    if elem.name == 'h3' and title in elem.get_text():
                        found_title = True
                        continue
                    if found_title and elem.name not in ['script', 'style', 'nav', 'header', 'footer']:
                        text = elem.get_text()
                        if text and text not in content:
                            content += text + " "
            
            content = clean_text(content)
        else:
            content = clean_text(soup.get_text())
        
        # Remove title from content if it appears at the beginning
        if content.startswith(title):
            content = content[len(title):].strip()
        
        # Handle short content
        if len(content) < 50:
            content = "[Limited text content available]"
        
        # Extract source if available
        source_pattern = r'var\s+source\s*=\s*"([^"]*)"'
        source_match = re.search(source_pattern, response.text)
        if source_match and source_match.group(1):
            source = source_match.group(1)
        else:
            source = "广州碳排放权交易中心"
        
        # Create document
        doc_data = {
            'doc_id': calculate_hash(url)[:16],
            'url': url,
            'title': title,
            'content': content,
            'section': section_name,
            'publish_date': publish_date,
            'view_count': None,  # GZETS doesn't show view counts
            'content_length': len(content),
            'content_hash': calculate_hash(content),
            'scraped_at': datetime.now(),
            'source': 'GZETS',
            'source_detail': source,
            'source_name_cn': '广州碳排放权交易中心',
            'source_name_en': 'Guangzhou Emission Trading Scheme'
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

def extract_document_links(soup):
    """Extract document links from GZETS list page"""
    links = []
    
    # Pattern for article links: /article/news/{section}/YYYYMM/YYYYMMDDXXXXX.shtml
    article_pattern = r'/article/news/\w+/\d{6}/\d{14,}\.shtml'
    
    # Find all links on the page
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Check if it matches article pattern
        if re.match(article_pattern, href):
            # Build full URL
            if href.startswith('/'):
                full_url = BASE_URL + href
            elif not href.startswith('http'):
                full_url = BASE_URL + '/' + href
            else:
                full_url = href
            
            links.append(full_url)
    
    # Remove duplicates and return max 15 (standard per page)
    unique_links = list(set(links))
    return unique_links[:15]

if __name__ == "__main__":
    scrape_gzets()