"""Configuration for anchor document selection."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key. "
        "See .env.example for the required format."
    )

MODEL = "gpt-4o"
TEMPERATURE = 0.3
MAX_TOKENS = 1000

# Dimensions for document classification
DIMENSIONS = {
    'supply': {
        'name': '配额供给',
        'description': '影响碳配额供给量的政策措施'
    },
    'demand': {
        'name': '配额需求', 
        'description': '影响碳配额需求量的政策措施'
    },
    'policy_strength': {
        'name': '政策强度',
        'description': '政策的约束力和执行强度'
    }
}

# Categories now specific to each dimension
DIMENSION_CATEGORIES = {
    'supply': {
        'restrict': '限制供给',
        'reduce': '减少供给',
        'increase': '增加供给',
        'expand': '扩大供给'
    },
    'demand': {
        'restrict': '限制需求',
        'reduce': '减少需求',
        'increase': '增加需求',
        'expand': '扩大需求'
    },
    'policy_strength': {
        'mandatory': '强制执行',
        'binding': '约束指导',
        'encouraged': '鼓励支持',
        'informational': '信息公告'
    }
}

# Flattened categories for backward compatibility
CATEGORIES = {
    # Supply/Demand categories
    'restrict': '限制',
    'reduce': '减少',
    'increase': '增加',
    'expand': '扩大',
    # Policy strength categories
    'mandatory': '强制执行',
    'binding': '约束指导',
    'encouraged': '鼓励支持',
    'informational': '信息公告'
}

# Processing configuration
DEFAULT_BATCH_SIZE = 20  # Default documents per group (> 16 anchor slots)
MAX_PARALLEL_GROUPS = 30  # Optimized for Tier 2 rate limits
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Rate limiting configuration - optimized for Tier 2
API_CALLS_PER_SECOND = 50  # 3,000 per minute
RATE_LIMIT_BUFFER = 0.90  # Use 90% of limits
MAX_RETRY_ATTEMPTS = 5
EXPONENTIAL_BACKOFF_BASE = 2

# File paths
DATA_BASE = "/Users/siruizhang/Desktop/碳交易/Project/02_Data_Processed"
CODE_BASE = "/Users/siruizhang/Desktop/碳交易/Project/03_Code"

# Input paths
FILTERED_DOCS_PATH = f"{DATA_BASE}/05_Policy_Doc_Filtered"

# Output paths
OUTPUT_BASE = f"{DATA_BASE}/06_Anchor_Analysis"
BATCH_RESULTS_PATH = f"{OUTPUT_BASE}/batch_results"
ANCHORS_PATH = f"{OUTPUT_BASE}/anchors"
SELECTION_LOG_PATH = f"{OUTPUT_BASE}/selection_log.json"
FINAL_ANCHORS_PATH = f"{OUTPUT_BASE}/anchors/final_anchors.json"