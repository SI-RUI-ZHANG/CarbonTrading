#!/usr/bin/env python3
"""Test the fixed GZETS scraper by directly scraping and checking content"""

import requests
from bs4 import BeautifulSoup
import re
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text

def test_content_extraction(url):
    """Test content extraction with the fixed logic"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h3')
        if not title_elem:
            return None, None, "No title found"
        title = clean_text(title_elem.get_text())
        
        # Remove scripts and styles
        for elem in soup(['script', 'style', 'nav', 'header', 'footer']):
            elem.decompose()
        
        # Step 1: Extract raw content
        raw_content = ""
        content_elem = soup.find('div', class_='wzfbxx')
        if content_elem:
            raw_content = content_elem.get_text()
        else:
            raw_content = soup.get_text()
        
        # Step 2: ALWAYS clean navigation from raw content
        content = raw_content
        
        # Primary cleaning method: Remove everything before "来源： 时间："
        if '来源： 时间：' in content[:500]:
            pos = content.find('来源： 时间：')
            content = content[pos + len('来源： 时间：'):].strip()
        
        # Remove common navigation prefixes
        if content.startswith('-广州碳排放权交易所'):
            content = content[len('-广州碳排放权交易所'):].strip()
        
        # Remove navigation header patterns
        nav_header_patterns = [
            '首页 交易中心概况 关于交易中心',
            '新闻中心 交易中心公告',
            '交易大厅 交易规则',
            '当前位置：首页'
        ]
        
        for pattern in nav_header_patterns:
            if pattern in content[:300]:
                pos = content.find(pattern)
                if pos >= 0:
                    content = content[pos + len(pattern):].strip()
        
        # Smart content start patterns
        content_start_patterns = [
            r'尊敬的',
            r'根据《[^》]+》',
            r'各(?:会员单位|有关单位)',
            r'为(?:了|进一步)',
            r'关于[^，]+的(?:通知|公告)'
        ]
        
        # Check if content starts with navigation keywords
        nav_keywords = ['交易中心动态', '省市动态', '新闻中心', '交易中心公告', '首页']
        has_nav_at_start = any(content[:100].startswith(nav) for nav in nav_keywords)
        
        if has_nav_at_start or '新闻中心' in content[:200]:
            content_start_pos = -1
            for pattern in content_start_patterns:
                match = re.search(pattern, content)
                if match:
                    if content_start_pos == -1 or match.start() < content_start_pos:
                        content_start_pos = match.start()
            
            if content_start_pos >= 0:
                content = content[content_start_pos:]
        
        # Remove footer patterns
        footer_patterns = ['地址：广州市', '电话：020', '版权所有']
        for footer in footer_patterns:
            footer_pos = content.find(footer)
            if footer_pos > 0:
                content = content[:footer_pos]
                break
        
        content = clean_text(content)
        
        return title, content, None
        
    except Exception as e:
        return None, None, str(e)

# Test URLs
test_cases = [
    {
        'url': "https://www.cnemission.com/article/news/jysgg/201912/20191200001849.shtml",
        'expected_start': '尊敬的各交易参与人',
        'should_not_contain': ['首页 交易中心概况', '-广州碳排放权交易所']
    },
    {
        'url': "https://www.cnemission.com/article/news/jysdt/202203/20220300002481.shtml",
        'expected_contains': '香港交易所',
        'should_not_contain': ['首页 交易中心概况', '新闻中心']
    }
]

print("Testing Fixed GZETS Content Extraction")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['url']}")
    print("-" * 40)
    
    title, content, error = test_content_extraction(test['url'])
    
    if error:
        print(f"  ❌ Error: {error}")
        continue
    
    print(f"  Title: {title}")
    
    # Check that navigation is removed
    passed = True
    for pattern in test.get('should_not_contain', []):
        if pattern in content[:300]:
            print(f"  ❌ FAILED: Navigation pattern found: '{pattern}'")
            passed = False
            break
    
    # Check expected content
    if 'expected_start' in test:
        if content.startswith(test['expected_start']):
            print(f"  ✅ Content starts correctly with: '{test['expected_start'][:30]}...'")
        else:
            print(f"  ⚠️  Content doesn't start as expected")
            print(f"     Actual start: '{content[:50]}...'")
            passed = False
    
    if 'expected_contains' in test:
        if test['expected_contains'] in content:
            print(f"  ✅ Content contains: '{test['expected_contains']}'")
        else:
            print(f"  ❌ Content missing expected text: '{test['expected_contains']}'")
            passed = False
    
    if passed:
        print(f"  ✅ SUCCESS: Content properly cleaned")
        print(f"     Content preview: {content[:150]}...")
    else:
        print(f"  ❌ Test failed")
        print(f"     Full content start (first 300 chars):")
        print(f"     {content[:300]}")

print("\n" + "=" * 60)
print("Testing complete!")