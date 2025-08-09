#!/usr/bin/env python3
"""Test the updated GZETS scraper fix"""

import requests
from bs4 import BeautifulSoup
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text

def test_extraction_updated(url):
    """Test with the updated extraction logic"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
    
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract title
    title_elem = soup.find('h3')
    title = clean_text(title_elem.get_text()) if title_elem else "No title"
    
    # Remove scripts and styles
    for elem in soup(['script', 'style', 'nav', 'header', 'footer']):
        elem.decompose()
    
    # Updated extraction logic
    raw_content = ""
    
    # Try multiple content div classes
    content_elem = soup.find('div', class_='wzfbxx')
    if not content_elem:
        content_elem = soup.find('div', class_='cont')
    
    if content_elem:
        raw_content = content_elem.get_text()
        print(f"   Found content div: class='{' '.join(content_elem.get('class', []))}'")
    else:
        raw_content = soup.get_text()
        print(f"   Using full page text (no content div found)")
    
    # Clean navigation
    content = raw_content
    
    # Primary cleaning: Remove everything before "来源： 时间："
    if '来源： 时间：' in content[:500]:
        pos = content.find('来源： 时间：')
        content = content[pos + len('来源： 时间：'):].strip()
    
    # Remove prefixes
    if content.startswith('-广州碳排放权交易所'):
        content = content[len('-广州碳排放权交易所'):].strip()
    
    content = clean_text(content)
    
    return title, content

# Test URLs
test_urls = [
    "https://www.cnemission.com/article/news/jysgg/201912/20191200001849.shtml",
    "https://www.cnemission.com/article/news/jysdt/202203/20220300002481.shtml"
]

print("Testing Updated GZETS Fix")
print("=" * 60)

for url in test_urls:
    print(f"\n{url}")
    print("-" * 40)
    
    title, content = test_extraction_updated(url)
    
    print(f"   Title: {title}")
    
    # Check for navigation patterns
    nav_patterns = ['首页 交易中心概况', '新闻中心', '-广州碳排放权交易所']
    has_nav = False
    for pattern in nav_patterns:
        if pattern in content[:200]:
            print(f"   ❌ Navigation found: '{pattern}'")
            has_nav = True
            break
    
    if not has_nav:
        print(f"   ✅ No navigation in content!")
        print(f"   Content starts with: {content[:150]}...")
    else:
        print(f"   Content (first 300 chars): {content[:300]}")

print("\n" + "=" * 60)