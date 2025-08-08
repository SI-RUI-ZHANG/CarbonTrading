"""
Shared utility functions for all scrapers
"""

import hashlib
import re
from datetime import datetime
import os
from pathlib import Path

def clean_text(text):
    """Clean Chinese text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags if any remain
    text = re.sub(r'<[^>]+>', '', text)
    # Keep Chinese characters, numbers, English letters, and common punctuation
    text = re.sub(r'[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\s，。！？；：""''（）《》【】、\-—_]', '', text)
    return text.strip()

def calculate_hash(content):
    """Calculate MD5 hash for deduplication"""
    if not content:
        return None
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_date(text):
    """Extract date from Chinese text"""
    if not text:
        return None
    
    # Pattern for YYYY年MM月DD日
    pattern1 = r'(\d{4})年(\d{1,2})月(\d{1,2})日'
    match = re.search(pattern1, text)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    
    # Pattern for YYYY-MM-DD
    pattern2 = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern2, text)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    
    # Pattern for YYYY/MM/DD
    pattern3 = r'(\d{4})/(\d{1,2})/(\d{1,2})'
    match = re.search(pattern3, text)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    
    return None

def extract_doc_number(text):
    """Extract document number from Chinese government documents"""
    if not text:
        return None
    
    # Common patterns for document numbers
    patterns = [
        r'([京津沪渝冀豫云辽黑湘皖鲁苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]\w*[〔\[]\d{4}[〕\]]\d+号)',
        r'(国\w*[〔\[]\d{4}[〕\]]\d+号)',
        r'(环\w*[〔\[]\d{4}[〕\]]\d+号)',
        r'(\w+发[〔\[]\d{4}[〕\]]\d+号)',
        r'(第\d+号)',
        r'(公告\s*\d{4}年\s*第\d+号)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def save_checkpoint(scraper_name, data):
    """Save progress checkpoint for resuming if interrupted"""
    checkpoint_file = f'{scraper_name}_checkpoint.json'
    import json
    from datetime import datetime
    
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_serial)

def load_checkpoint(scraper_name):
    """Load last checkpoint"""
    checkpoint_file = f'{scraper_name}_checkpoint.json'
    import json
    from datetime import datetime
    
    def datetime_parser(dct):
        """Parse datetime strings back to datetime objects"""
        for k, v in dct.items():
            if isinstance(v, str):
                try:
                    # Try to parse ISO format datetime strings
                    if 'T' in v and ('+' in v or 'Z' in v or v.count(':') >= 2):
                        dct[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                except:
                    pass
        return dct
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f, object_hook=datetime_parser)
    except:
        return None

def delete_checkpoint(scraper_name):
    """Delete checkpoint file after successful completion"""
    checkpoint_file = f'{scraper_name}_checkpoint.json'
    try:
        os.remove(checkpoint_file)
    except:
        pass

def ensure_output_dir(output_dir):
    """Ensure output directory exists"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def get_existing_urls(output_dir):
    """Get list of already scraped URLs to avoid duplicates"""
    existing_urls = set()
    
    # Check all parquet files in the directory
    for file in Path(output_dir).glob('*.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(file)
            if 'url' in df.columns:
                existing_urls.update(df['url'].tolist())
        except:
            pass
    
    return existing_urls

def is_valid_document_url(url):
    """Check if URL looks like a valid document URL"""
    # Skip images, css, js files
    invalid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico']
    for ext in invalid_extensions:
        if url.lower().endswith(ext):
            return False
    
    # Should contain some indication it's a document
    valid_patterns = ['.html', '.htm', '.pdf', '.doc', '.docx', 'article', 'content', 'detail']
    return any(pattern in url.lower() for pattern in valid_patterns)

def normalize_url(url, base_url):
    """Normalize URL to absolute form"""
    from urllib.parse import urljoin, urlparse
    
    # Handle relative URLs
    if not url.startswith('http'):
        url = urljoin(base_url, url)
    
    # Remove fragments
    parsed = urlparse(url)
    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        url += f"?{parsed.query}"
    
    return url

def save_document_json(doc_data, section_name, output_dir):
    """Save individual document to JSON file"""
    import json
    from datetime import datetime
    
    # Create section subdirectory
    section_dir = os.path.join(output_dir, section_name.replace(' ', '_'))
    ensure_output_dir(section_dir)
    
    # Generate unique document ID if not present
    if 'doc_id' not in doc_data:
        doc_data['doc_id'] = calculate_hash(doc_data.get('url', ''))[:16]
    
    # Save individual JSON file
    filename = os.path.join(section_dir, f"{doc_data['doc_id']}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2, default=json_serial)
    
    # Also append to JSONL file for easy processing
    jsonl_file = os.path.join(section_dir, "all_documents.jsonl")
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, default=json_serial)
        f.write('\n')
    
    return filename

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def load_progress(output_dir):
    """Load scraping progress"""
    import json
    progress_file = os.path.join(output_dir, "progress.json")
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'scraped_urls': [], 'last_update': None}

def update_progress(url, section, output_dir):
    """Update scraping progress"""
    import json
    from datetime import datetime
    
    progress_file = os.path.join(output_dir, "progress.json")
    progress = load_progress(output_dir)
    
    # Add new URL
    progress['scraped_urls'].append({
        'url': url,
        'section': section,
        'timestamp': datetime.now().isoformat()
    })
    progress['last_update'] = datetime.now().isoformat()
    
    # Save updated progress
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)
    
    return progress