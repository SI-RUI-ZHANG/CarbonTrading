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

MODEL = "gpt-4o"

TEMPERATURE = 0.6  # Moderate temperature for nuanced scoring
MAX_TOKENS = 1500  # Enough for all 12 similarities

# Spectrum position configuration
SPECTRUM_POSITIONS = {
    'supply': {
        'restrict': -100,
        'reduce': -50,
        'increase': 50,
        'expand': 100
    },
    'demand': {
        'restrict': -100,
        'reduce': -50,
        'increase': 50,
        'expand': 100
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

# Direct position scoring prompt template
POSITION_SCORING_PROMPT = """
## YOUR ROLE
You are a document positioning specialist for carbon trading policy analysis. Your task is to precisely place documents on three measurement spectrums by comparing them to reference documents at known positions.

## CORE CONCEPT
Think of this like judging a competition where you have example performances with known scores. You'll compare the new document to these examples to determine its position on each spectrum.

## THE THREE SPECTRUMS

### 1. SUPPLY SPECTRUM (Carbon Quota Availability)
Measures how a policy affects the total supply of carbon quotas in the trading market.
- **Scale**: -150 (severely restricts supply) to +150 (massively expands supply)
- **Zero point**: No effect on carbon quota supply
- **Key question**: Does this change how many carbon quotas are available for trading?

### 2. DEMAND SPECTRUM (Carbon Quota Demand)
Measures how a policy affects the demand for carbon quotas from market participants.
- **Scale**: -150 (severely reduces demand) to +150 (massively increases demand)
- **Zero point**: No effect on carbon quota demand
- **Key question**: Does this change how many quotas companies need to buy?

### 3. POLICY STRENGTH SPECTRUM (Enforcement Level)
Measures the legal force and compliance requirements of the policy.
- **Scale**: 0 (purely informational) to 150 (mandatory with extreme penalties)
- **Key positions**: 0=Information, 33=Encouraged, 67=Binding, 100=Mandatory
- **Key question**: How strongly must organizations comply with this policy?

## SCORING METHOD

### Step 1: Identify Document Type
First, determine if this document affects carbon trading markets at all:
- ✅ Carbon trading relevant: quota allocation, trading rules, market access, compliance requirements
- ❌ Not relevant: carbon footprint reporting, general environmental policies, waste management, green finance

### Step 2: Compare to Reference Anchors
For each spectrum, you have 4 reference documents at known positions. Compare your document to these anchors:
- Is it more extreme than the strongest anchor? (position beyond ±100)
- Is it similar to an anchor? (position near that anchor's value)
- Is it between two anchors? (interpolate the position)
- Is it weaker than the weakest anchor? (position closer to zero)

### Step 3: Assign Numerical Positions
Based on your comparison, assign precise numbers:
- Supply/Demand: Use 0 for non-carbon-trading documents
- Policy Strength: Consider actual enforcement mechanisms and penalties
- Be consistent: Similar documents should get similar scores

## REFERENCE ANCHORS

### Supply Spectrum Anchors:
- **Position -100** (restrict): {supply_restrict}
- **Position -50** (reduce): {supply_reduce}
- **Position +50** (increase): {supply_increase}
- **Position +100** (expand): {supply_expand}

### Demand Spectrum Anchors:
- **Position -100** (restrict): {demand_restrict}
- **Position -50** (reduce): {demand_reduce}
- **Position +50** (increase): {demand_increase}
- **Position +100** (expand): {demand_expand}

### Policy Strength Anchors:
- **Position 0** (informational): {policy_informational}
- **Position 33** (encouraged): {policy_encouraged}
- **Position 67** (binding): {policy_binding}
- **Position 100** (mandatory): {policy_mandatory}

## SPECIAL SCORING RULES

1. **Zero Position**: 0 means no effect at all on that spectrum
2. **Extreme Cases**: Can exceed anchor positions (beyond ±100) for unprecedented policies  
3. **Uncertainty**: When between anchors, interpolate based on relative similarity
4. **Consistency**: Similar documents should receive similar positions

## DOCUMENT TO ANALYZE

**Title**: {doc_title}

**Content**: {doc_content}

## OUTPUT REQUIREMENT

After careful analysis and comparison to the anchors, return your positioning as a JSON object with three precise numbers:

{{
    "supply": <integer from -150 to +150>,
    "demand": <integer from -150 to +150>,
    "policy_strength": <integer from 0 to 150>
}}
"""