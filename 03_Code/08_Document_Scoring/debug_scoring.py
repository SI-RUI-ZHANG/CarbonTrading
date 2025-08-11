#!/usr/bin/env python3
"""Debug script to test document scoring and understand the output patterns."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from api_client import DocumentPositioner
import config

def debug_single_document():
    """Test scoring on a single document to debug the process."""
    
    # Initialize the positioner
    positioner = DocumentPositioner()
    
    # Test document 1: Should be neutral (informational)
    test_doc1 = {
        'doc_id': 'test_neutral',
        'title': '关于发布2023年电力碳足迹因子数据的公告',
        'content': '为落实《关于建立碳足迹管理体系的实施方案》相关要求，生态环境部、国家统计局、国家能源局组织中国电力企业联合会等单位计算了2023年燃煤发电、燃气发电、水力发电、核能发电、风力发电、光伏发电、光热发电、生物质发电碳足迹因子和输配电碳足迹因子以及全国电力平均碳足迹因子，供各行业产品核算电力生产和消费产生的碳足迹使用。'
    }
    
    # Test document 2: Should have significant impact
    test_doc2 = {
        'doc_id': 'test_policy',
        'title': '关于2024年度碳排放配额分配方案的通知',
        'content': '为严格控制碳排放总量，现决定：一、将2024年度碳排放配额总量较2023年减少15%。二、重点排放企业配额削减20%。三、新增企业暂停配额发放。四、违规企业将面临配额清零处罚。五、本方案自发布之日起立即执行，违反者将追究法律责任。'
    }
    
    # Build and print the prompt to understand what's being sent
    print("=" * 80)
    print("TESTING DOCUMENT 1 (Should be neutral/informational):")
    print("=" * 80)
    prompt1 = positioner._build_position_prompt(test_doc1)
    print("\nPROMPT PREVIEW (first 2000 chars):")
    print(prompt1[:2000])
    print("\n... [truncated] ...")
    
    # Test the scoring
    print("\nCalling API for document 1...")
    result1 = positioner.get_document_positions(test_doc1)
    if result1:
        print(f"RESULT 1: {json.dumps(result1, indent=2)}")
    else:
        print("ERROR: Failed to get positions for document 1")
    
    print("\n" + "=" * 80)
    print("TESTING DOCUMENT 2 (Should have significant negative supply impact):")
    print("=" * 80)
    
    # Test the second document
    print("\nCalling API for document 2...")
    result2 = positioner.get_document_positions(test_doc2)
    if result2:
        print(f"RESULT 2: {json.dumps(result2, indent=2)}")
    else:
        print("ERROR: Failed to get positions for document 2")
    
    # Also check what anchors are being loaded
    print("\n" + "=" * 80)
    print("CHECKING LOADED ANCHORS:")
    print("=" * 80)
    for dimension in ['supply', 'demand', 'policy_strength']:
        print(f"\n{dimension.upper()}:")
        if dimension in positioner.anchors:
            for category, anchor in positioner.anchors[dimension].items():
                if anchor:
                    print(f"  {category}: {anchor.get('title', 'NO TITLE')[:50]}...")
                else:
                    print(f"  {category}: NO ANCHOR")

if __name__ == "__main__":
    debug_single_document()