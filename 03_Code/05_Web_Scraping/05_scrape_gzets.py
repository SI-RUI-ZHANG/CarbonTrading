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
        
        # Try multiple methods to extract date
        publish_date = None
        
        # Method 1: Extract from JavaScript variables
        # Pattern 1: var writetime = "2025-01-06 14:03:31"
        # Pattern 2: var tm = "2013-01-18 09:25:25"
        date_patterns_js = [
            r'var\s+writetime\s*=\s*"([^"]+)"',
            r'var\s+tm\s*=\s*"([^"]+)"'
        ]
        
        for pattern in date_patterns_js:
            date_match = re.search(pattern, response.text)
            if date_match:
                date_str = date_match.group(1)
                publish_date = extract_date(date_str)
                if publish_date:
                    break
        
        # Method 2: Extract from content (most reliable for GZETS)
        # Look for dates at the end of content like "广州碳排放权交易所 2019年6月12日"
        if not publish_date:
            content_elem = soup.find('div', class_='wzfbxx')
            if content_elem:
                content = content_elem.get_text()
            else:
                # Fallback to whole page text
                content = soup.get_text()
            
            # Look for date patterns in content
            # Pattern: "YYYY年MM月DD日" typically at the end of announcements
            date_patterns = [
                r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # Full date with year
                r'(\d{4})-(\d{1,2})-(\d{1,2})',       # Alternative format
                r'(\d{1,2})月(\d{1,2})日',            # Chinese date without year (e.g., "11月15日")
            ]
            
            for pattern in date_patterns:
                matches = list(re.finditer(pattern, content))
                if matches:
                    # For dates without year, try to get year from URL or use current/recent year
                    if '(\d{4})' not in pattern:  # Pattern without year
                        # First try to get year from URL
                        url_year_match = re.search(r'/(\d{4})\d{2}/', url)
                        if url_year_match:
                            year = int(url_year_match.group(1))
                        else:
                            # Use first found year in content as context, or fallback to 2022
                            year_in_content = re.search(r'(\d{4})年', content)
                            year = int(year_in_content.group(1)) if year_in_content else 2022
                        
                        # Use first occurrence for month-day only patterns (usually at beginning)
                        match = matches[0]  
                        month, day = map(int, match.groups())
                    else:
                        # Use the last date found for full dates (usually the publication date)
                        last_match = matches[-1]
                        if '年' in pattern:
                            year, month, day = map(int, last_match.groups())
                        else:
                            year, month, day = map(int, last_match.groups())
                    
                    try:
                        publish_date = datetime(year, month, day)
                        break
                    except ValueError:
                        continue
        
        # Method 3: Extract from URL pattern (common for news articles)
        # Pattern: /201403/20140300000663.shtml -> 2014-03
        if not publish_date:
            url_date_match = re.search(r'/(\d{4})(\d{2})/', url)
            if url_date_match:
                year, month = map(int, url_date_match.groups())
                try:
                    # Use first day of month as approximation
                    publish_date = datetime(year, month, 1)
                except ValueError:
                    pass
        
        # Method 4: Extract from title as fallback
        if not publish_date:
            # e.g., "关于2019年清明节休市安排的公告"
            year_in_title = re.search(r'(\d{4})年', title)
            if year_in_title:
                year = int(year_in_title.group(1))
                # Try to find month and day
                date_in_title = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
                if date_in_title:
                    year, month, day = map(int, date_in_title.groups())
                    try:
                        publish_date = datetime(year, month, day)
                    except ValueError:
                        publish_date = datetime(year, 1, 1)
                else:
                    # Just use year with January 1st as approximation
                    publish_date = datetime(year, 1, 1)
        
        # Extract content - main body after header
        # Remove navigation, scripts, styles
        for elem in soup(['script', 'style', 'nav', 'header', 'footer']):
            elem.decompose()
        
        # Step 1: Extract raw content
        raw_content = ""
        
        # Try multiple content div classes that GZETS uses
        # wzfbxx is one pattern, cont is another common one
        content_elem = soup.find('div', class_='wzfbxx')
        if not content_elem:
            content_elem = soup.find('div', class_='cont')
        
        if content_elem:
            raw_content = content_elem.get_text()
        else:
            # Fallback to full page text
            raw_content = soup.get_text()
        
        # Step 2: ALWAYS clean navigation from raw content
        content = raw_content
        
        # Primary cleaning method: Remove everything before "来源： 时间："
        # This is the most reliable separator between navigation and content
        if '来源： 时间：' in content[:500]:  # Check in first 500 chars
            pos = content.find('来源： 时间：')
            content = content[pos + len('来源： 时间：'):].strip()
        
        # Remove common navigation prefixes
        if content.startswith('-广州碳排放权交易所'):
            content = content[len('-广州碳排放权交易所'):].strip()
        
        # Remove navigation header patterns
        nav_header_patterns = [
            '首页 交易中心概况 关于交易中心 组织架构 人力资源 大事记 About us',
            '新闻中心 交易中心公告 交易中心动态 省市动态 国内资讯 全球碳新闻 绿金委动态',
            '交易大厅 交易规则及细则 交易提示 交易信息披露 市场研究 指数分析',
            '碳交易系统 碳配额登记系统 会员服务 开户指引',
            '政策法规 本所规章 国家政策文件 省市政策',
            '教育培训 产品介绍 最新班次 培训讲师 培训证书 证书查询',
            '国际合作 欧洲能源交易所 联系我们',
            '当前位置：首页'
        ]
        
        # Remove navigation patterns from the beginning
        for pattern in nav_header_patterns:
            if pattern in content[:300]:
                # Find the pattern and remove everything up to and including it
                pos = content.find(pattern)
                if pos >= 0:
                    content = content[pos + len(pattern):].strip()
        
        # If content still has navigation at the beginning, use smart patterns to find real content start
        # Patterns that indicate start of real content
        content_start_patterns = [
            r'一、',  # Chinese numbering
            r'二、',
            r'三、',
            r'(?:南方日报|新华社|中新社|人民日报|广州日报)(?:讯|电)',  # News sources
            r'[^：]{2,}(?:省|市|区|局|厅|委|部|院|所|中心)：',  # Official letters
            r'根据《[^》]+》',  # Legal references
            r'为(?:了|进一步|贯彻|落实|推进|加强)',  # Purpose statements
            r'各(?:会员单位|有关单位|相关企业)：',  # Announcements
            r'尊敬的',  # Formal greetings
            r'关于[^，]+的(?:通知|公告|决定|意见|办法)',  # Official notices
        ]
        
        # Check if content starts with navigation keywords
        nav_keywords = [
            '交易中心动态', '省市动态', '国内资讯', '国际资讯', '绿金委动态',
            '新闻中心', '交易中心公告', '首页', '交易中心概况', '关于交易中心'
        ]
        
        # If content starts with navigation keywords, find where real content starts
        has_nav_at_start = any(content[:100].startswith(nav) for nav in nav_keywords)
        
        if has_nav_at_start or '新闻中心' in content[:200]:
            # Find where real content starts using smart patterns
            content_start_pos = -1
            for pattern in content_start_patterns:
                match = re.search(pattern, content)
                if match:
                    if content_start_pos == -1 or match.start() < content_start_pos:
                        content_start_pos = match.start()
            
            # If we found a content start pattern, extract from there
            if content_start_pos >= 0:
                content = content[content_start_pos:]
        
        # Remove title from content if it appears after navigation
        # Look for title in content and remove everything before it if it's preceded by navigation
        if title in content[:500]:
            title_pos = content.find(title)
            # Check if what's before the title looks like navigation
            before_title = content[:title_pos]
            if any(nav in before_title for nav in nav_keywords):
                # Skip past the title
                content = content[title_pos + len(title):].strip()
        
        # Remove footer patterns
        footer_patterns = ['地址：广州市', '电话：020', '版权所有', 'ICP备', '粤公网安备', '网站安全检测平台']
        for footer in footer_patterns:
            footer_pos = content.find(footer)
            if footer_pos > 0:
                content = content[:footer_pos]
                break
        
        # Final cleanup
        content = clean_text(content)
        
        # Check if content is mostly navigation (empty announcement case)
        if content:
            words = content.split()
            nav_word_count = sum(1 for word in words if any(nav in word for nav in nav_keywords))
            
            # If more than 50% are navigation words, it's likely empty content
            if len(words) > 0 and nav_word_count > len(words) * 0.5:
                content = "[No article content available - only navigation]"
        
        # Handle very short content (only mark as limited if really minimal)
        if len(content) < 20:
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