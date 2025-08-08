"""
Scrape Ministry of Ecology and Environment (MEE) website - Version 2
With incremental JSON saving and progress tracking
Run: python 01_scrape_mee_v2.py
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
from utils import (
    clean_text, calculate_hash, extract_date, extract_doc_number,
    ensure_output_dir, is_valid_document_url, normalize_url,
    save_document_json, load_progress, update_progress
)

# Configuration
BASE_URL = "https://www.mee.gov.cn"
OUTPUT_DIR = "../../01_Data_Raw/03_Policy_Documents/MEE"
DELAY = 2  # Seconds between requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_mee():
    """Main function to scrape MEE website with incremental saving"""
    
    print("=" * 60)
    print("MEE Scraper V2 - Incremental JSON Storage")
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
        
        scraped_count = scrape_section_incremental(
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
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Data saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

def scrape_section_incremental(section_url, section_name, max_pages=10, scraped_urls=None, stats=None):
    """Scrape a section with incremental saving"""
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
            # Get page
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detect pagination index from first page
            if page == 1 and not index_number:
                index_number = detect_pagination_index(soup)
                if index_number != "8597":
                    print(f"  Detected pagination index: {index_number}")
            
            # Find document links
            links = find_document_links(soup, section_url)
            print(f"  Found {len(links)} links on page")
            
            if not links and page > 2:
                print(f"  No more documents found, stopping at page {page}")
                break
            
            # Process each document
            new_on_page = 0
            for link in tqdm(links, desc="  Processing", leave=False):
                # Skip if already scraped
                if link in scraped_urls:
                    stats['skipped'] = stats.get('skipped', 0) + 1
                    continue
                
                # Scrape and save document
                success = scrape_and_save_document(link, section_name)
                
                if success:
                    scraped_urls.add(link)
                    scraped_count += 1
                    new_on_page += 1
                    update_progress(link, section_name, OUTPUT_DIR)
                else:
                    stats['failed'] = stats.get('failed', 0) + 1
                
                time.sleep(DELAY)  # Be polite to the server
            
            print(f"  ✓ Scraped {new_on_page} new documents from this page")
            
        except Exception as e:
            print(f"  ✗ Error scraping page {page}: {e}")
            continue
    
    return scraped_count

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

def scrape_and_save_document(url, section_name):
    """Scrape a document and save it immediately"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = None
        title_selectors = [
            ('h1', None),
            ('div', {'class': 'title'}),
            ('div', {'class': 'article-title'}),
            ('div', {'id': 'title'}),
            ('title', None)
        ]
        
        for tag, attrs in title_selectors:
            element = soup.find(tag, attrs)
            if element:
                title = clean_text(element.get_text())
                if title and len(title) > 5:
                    break
        
        if not title:
            print(f"\n    Warning: No title found for {url}")
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
                print(f"\n    Warning: Content too short for {url}")
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
        
        # Save document immediately
        save_document_json(doc_data, section_name, OUTPUT_DIR)
        
        return True
        
    except Exception as e:
        print(f"\n    Error scraping document {url}: {e}")
        return False

if __name__ == "__main__":
    scrape_mee()