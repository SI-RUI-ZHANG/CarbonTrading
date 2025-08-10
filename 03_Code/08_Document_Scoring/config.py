"""Configuration for document scoring system with spectrum positioning."""

import os
from pathlib import Path
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

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3  # Low temperature for consistent scoring
MAX_TOKENS = 1500  # Enough for all 12 similarities

# Spectrum position configuration
SPECTRUM_POSITIONS = {
    'supply': {
        'major_decrease': -100,
        'minor_decrease': -50,
        'minor_increase': 50,
        'major_increase': 100
    },
    'demand': {
        'major_decrease': -100,
        'minor_decrease': -50,
        'minor_increase': 50,
        'major_increase': 100
    },
    'policy_strength': {
        'informational': 0,
        'encouraged': 33,
        'binding': 67,
        'mandatory': 100
    }
}

# Processing configuration
DEFAULT_BATCH_SIZE = 50  # Documents per batch
MAX_PARALLEL_WORKERS = 30  # Maximum parallel API calls
SAVE_CHECKPOINT_EVERY = 50  # Save progress every N documents

# Rate limiting configuration - optimized for Tier 2
API_CALLS_PER_SECOND = 50  # 3,000 per minute
RATE_LIMIT_BUFFER = 0.90  # Use 90% of limits
MAX_RETRY_ATTEMPTS = 5
EXPONENTIAL_BACKOFF_BASE = 2
RETRY_DELAY = 2  # seconds

# File paths
PROJECT_BASE = Path("/Users/siruizhang/Desktop/碳交易/Project")
DATA_BASE = PROJECT_BASE / "02_Data_Processed"
CODE_BASE = PROJECT_BASE / "03_Code"

# Input paths
ANCHOR_PATH = DATA_BASE / "06_Anchor_Analysis/anchor_summary.json"
FILTERED_DOCS_PATH = DATA_BASE / "05_Policy_Doc_Filtered"

# Output paths
OUTPUT_BASE = DATA_BASE / "07_Document_Scores"
BATCH_SCORES_PATH = OUTPUT_BASE / "batch_scores"
FINAL_SCORES_PATH = OUTPUT_BASE / "document_scores.parquet"
SUMMARY_PATH = OUTPUT_BASE / "scoring_summary.json"
DISTRIBUTIONS_PATH = OUTPUT_BASE / "score_distributions.json"
CHECKPOINT_PATH = OUTPUT_BASE / "checkpoint.json"

# Create output directories if they don't exist
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
BATCH_SCORES_PATH.mkdir(parents=True, exist_ok=True)

# Similarity prompt template
SIMILARITY_PROMPT_TEMPLATE = """
You are analyzing carbon trading policy documents. Compare the given document to 12 anchor documents and assess their similarity.

IMPORTANT: Focus ONLY on policy content and meaning:
- Ignore document length differences
- Ignore writing style or format
- Ignore publication date or source
- Focus on: What carbon market mechanism is being addressed? What is the policy impact?

Document to analyze:
Title: {doc_title}
Content: {doc_content}

Anchor documents to compare against:

Supply dimension anchors:
1. major_decrease: {supply_major_decrease}
2. minor_decrease: {supply_minor_decrease}
3. minor_increase: {supply_minor_increase}
4. major_increase: {supply_major_increase}

Demand dimension anchors:
5. major_decrease: {demand_major_decrease}
6. minor_decrease: {demand_minor_decrease}
7. minor_increase: {demand_minor_increase}
8. major_increase: {demand_major_increase}

Policy strength anchors:
9. informational: {policy_informational}
10. encouraged: {policy_encouraged}
11. binding: {policy_binding}
12. mandatory: {policy_mandatory}

Return a JSON object with similarity scores (0-100) for each anchor:
{{
    "supply": {{
        "major_decrease": <score>,
        "minor_decrease": <score>,
        "minor_increase": <score>,
        "major_increase": <score>
    }},
    "demand": {{
        "major_decrease": <score>,
        "minor_decrease": <score>,
        "minor_increase": <score>,
        "major_increase": <score>
    }},
    "policy_strength": {{
        "informational": <score>,
        "encouraged": <score>,
        "binding": <score>,
        "mandatory": <score>
    }},
    "confidence": <overall confidence 0-100>
}}
"""