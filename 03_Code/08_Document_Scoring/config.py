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
MAX_PARALLEL_WORKERS = 6  # Maximum parallel API calls (reduced by 5x for GPT-4o)
SAVE_CHECKPOINT_EVERY = 50  # Save progress every N documents

# Rate limiting configuration - adjusted for GPT-4o (÷5 scaling)
API_CALLS_PER_SECOND = 10  # Reduced by 5x from original 50
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
You are a document positioning specialist for carbon market policy analysis. Your task is to precisely place documents on three measurement spectrums by comparing them to reference documents at known positions.

## CORE CONCEPT
China's carbon markets include both mandatory emissions trading (碳排放权交易) and voluntary reduction markets (CCER/碳普惠). You should consider impacts on BOTH types of markets when scoring documents.

## THE THREE SPECTRUMS

### 1. SUPPLY SPECTRUM (Carbon Asset Availability)
Measures how a policy affects the total supply of tradeable carbon assets.
- **Scale**: -150 (severely restricts) to +150 (massively expands)
- **Zero point**: No effect on carbon asset supply
- **Includes**: Emission quotas, CCER credits, carbon offsets, green certificates
- **Key question**: Does this change the amount of carbon assets available in ANY market?

### 2. DEMAND SPECTRUM (Carbon Asset Demand)
Measures how a policy affects demand for carbon assets from market participants.
- **Scale**: -150 (severely reduces) to +150 (massively increases)
- **Zero point**: No effect on carbon asset demand
- **Includes**: Compliance needs, voluntary offsetting, carbon neutrality goals
- **Key question**: Does this change how many carbon assets entities need or want?

### 3. POLICY STRENGTH SPECTRUM (Enforcement Level)
Measures the legal force and compliance requirements of the policy.
- **Scale**: 0 (informational) to 150 (mandatory with severe penalties)
- **Key positions**: 0=Information, 33=Encouraged, 67=Binding, 100=Mandatory
- **Key question**: How binding is this policy on affected parties?

## SCORING METHOD

### Step 1: Identify Carbon Market Relevance
Look for ANY connection to carbon markets, including:

**DIRECT IMPACTS** (typically ±50 to ±150):
- Quota allocation rules (配额分配)
- Trading regulations (交易规则)
- CCER/voluntary reduction programs (自愿减排)
- Carbon offset mechanisms (碳抵消)
- Market access rules (市场准入)
- Compliance and penalties (履约处罚) 
- Feel free to expand.

**INDIRECT IMPACTS** (typically ±10 to ±50):
- Carbon accounting methods (碳核算)
- MRV systems (监测报告核查)
- Carbon neutrality targets (碳中和目标)
- Green finance for carbon projects (绿色金融)
- Technology standards affecting emissions (技术标准)
- Industry guidance affecting carbon intensity (行业指导)
- Feel free to expance.

**SUPPORTING INFRASTRUCTURE** (typically ±5 to ±15):
- Registry systems for trading (登记系统)
- Market analysis affecting trading decisions (市场分析)
- Training for market operators/traders (not general education)
- Feel free to expand.

### Step 2: Compare to Reference Anchors
For each spectrum, compare your document to the 4 reference anchors:
- Beyond strongest anchor? → Position beyond ±100
- Similar to an anchor? → Position near that anchor's value  
- Between two anchors? → Interpolate based on similarity
- Weaker than weakest? → Position between 0 and weakest anchor

### Step 3: Assign Numerical Positions

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

1. **Consider cumulative effects**: Multiple impacts can justify moderate scores
2. **Focus on market operations**: Document must affect how carbon markets function
3. **Be flexible**: You don't need to be too rigid, give a number that "feels right".

## EXCLUSIONS (Score 0 only if ALL these conditions met)
- No mention of carbon, emissions, or climate
- No connection to regulated entities or carbon market participants
- No impact on carbon-intensive industries
- Pure nature conservation without carbon considerations
- General waste/water/air pollution without carbon aspects

## DOCUMENT TO ANALYZE

**Title**: {doc_title}

**Content**: {doc_content}

## OUTPUT REQUIREMENT

Return your positioning as a JSON object:

{{
    "supply": <integer from -150 to +150>,
    "demand": <integer from -150 to +150>,
    "policy_strength": <integer from 0 to 150>
}}
"""