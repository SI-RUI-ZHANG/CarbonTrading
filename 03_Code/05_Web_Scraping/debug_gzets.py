#!/usr/bin/env python3
"""Debug GZETS page structure to understand the issue"""

import requests
from bs4 import BeautifulSoup

url = "https://www.cnemission.com/article/news/jysgg/201912/20191200001849.shtml"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

response = requests.get(url, headers=HEADERS, timeout=20)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.content, 'html.parser')

print("=" * 60)
print("DEBUGGING GZETS PAGE STRUCTURE")
print("=" * 60)

# Check if wzfbxx div exists
wzfbxx_div = soup.find('div', class_='wzfbxx')
print(f"\n1. wzfbxx div exists: {wzfbxx_div is not None}")

if wzfbxx_div:
    wzfbxx_text = wzfbxx_div.get_text()
    print(f"   wzfbxx content length: {len(wzfbxx_text)}")
    print(f"   wzfbxx first 200 chars: {wzfbxx_text[:200]}")
    print(f"   wzfbxx contains '来源： 时间：': {'来源： 时间：' in wzfbxx_text}")
    if '来源： 时间：' in wzfbxx_text:
        pos = wzfbxx_text.find('来源： 时间：')
        print(f"   Position of '来源： 时间：': {pos}")
        print(f"   Text after '来源： 时间：': {wzfbxx_text[pos+10:pos+100]}")

# Check full page text structure
print("\n2. Full page text analysis:")
full_text = soup.get_text()
print(f"   Total length: {len(full_text)}")

# Find "来源： 时间：" in full text
if '来源： 时间：' in full_text:
    pos = full_text.find('来源： 时间：')
    print(f"   Position of '来源： 时间：': {pos}")
    print(f"   Text before (last 100 chars): ...{full_text[max(0,pos-100):pos]}")
    print(f"   Text after (first 100 chars): {full_text[pos+10:pos+110]}")

# Check page structure
print("\n3. Page structure analysis:")
print("   Main divs found:")
for div in soup.find_all('div', class_=True)[:10]:
    classes = ' '.join(div.get('class', []))
    text_preview = div.get_text()[:50].replace('\n', ' ')
    print(f"   - class='{classes}': {text_preview}...")

# Check for navigation elements
print("\n4. Navigation elements:")
nav_elems = soup.find_all(['nav', 'ul', 'div'], class_=lambda x: x and ('nav' in str(x).lower() or 'menu' in str(x).lower()))
print(f"   Found {len(nav_elems)} navigation-related elements")

# Check title location
print("\n5. Title analysis:")
h3 = soup.find('h3')
if h3:
    print(f"   H3 title: {h3.get_text().strip()}")
    print(f"   H3 parent tag: {h3.parent.name}")
    if h3.parent.get('class'):
        print(f"   H3 parent class: {' '.join(h3.parent.get('class'))}")

# Look for the actual content area
print("\n6. Content area search:")
# Common content area classes/ids
content_areas = soup.find_all(['div', 'article'], class_=lambda x: x and any(
    keyword in str(x).lower() for keyword in ['content', 'article', 'main', 'body', 'text', 'wzfbxx']
))
for area in content_areas[:5]:
    classes = ' '.join(area.get('class', []))
    text_len = len(area.get_text())
    print(f"   - class='{classes}': {text_len} chars")