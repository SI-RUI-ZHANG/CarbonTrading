"""Advanced scoring configuration for document analysis using anchor references."""

import os
from pathlib import Path

# Score range configuration - using 0-100 for maximum granularity
SCORE_RANGES = {
    'relevance': {
        'min': 0,
        'max': 100,
        'thresholds': {
            'unrelated': (0, 10),      # Document is unrelated to dimension
            'tangential': (10, 30),    # Loosely related, mentions in passing
            'moderate': (30, 60),      # Moderately related, secondary focus
            'strong': (60, 85),        # Strongly related, primary focus
            'core': (85, 100)          # Core policy document for this dimension
        }
    },
    'intensity': {
        'min': 0,
        'max': 100,
        'thresholds': {
            'negligible': (0, 20),     # Minimal impact/effect
            'minor': (20, 40),         # Small impact
            'moderate': (40, 60),      # Medium impact
            'significant': (60, 80),   # Large impact
            'major': (80, 100)         # Transformative impact
        }
    },
    'confidence': {
        'min': 0,
        'max': 100,
        'thresholds': {
            'very_low': (0, 20),       # Very uncertain
            'low': (20, 40),           # Somewhat uncertain
            'medium': (40, 60),        # Moderately confident
            'high': (60, 80),          # Confident
            'very_high': (80, 100)     # Very confident
        }
    }
}

# Category mappings to numeric values
CATEGORY_MAPPINGS = {
    'supply': {
        # Direction score: -100 (major decrease) to +100 (major increase)
        'major_decrease': {
            'direction': -100,
            'direction_range': (-100, -75),  # Range for similarity matching
            'intensity_multiplier': 1.0,      # Full intensity for major changes
            'name': '大幅减少供给'
        },
        'minor_decrease': {
            'direction': -50,
            'direction_range': (-75, -25),
            'intensity_multiplier': 0.6,      # Reduced intensity for minor changes
            'name': '适度减少供给'
        },
        'minor_increase': {
            'direction': 50,
            'direction_range': (25, 75),
            'intensity_multiplier': 0.6,
            'name': '适度增加供给'
        },
        'major_increase': {
            'direction': 100,
            'direction_range': (75, 100),
            'intensity_multiplier': 1.0,
            'name': '大幅增加供给'
        }
    },
    'demand': {
        # Direction score: -100 (major decrease) to +100 (major increase)
        'major_decrease': {
            'direction': -100,
            'direction_range': (-100, -75),
            'intensity_multiplier': 1.0,
            'name': '大幅减少需求'
        },
        'minor_decrease': {
            'direction': -50,
            'direction_range': (-75, -25),
            'intensity_multiplier': 0.6,
            'name': '适度减少需求'
        },
        'minor_increase': {
            'direction': 50,
            'direction_range': (25, 75),
            'intensity_multiplier': 0.6,
            'name': '适度增加需求'
        },
        'major_increase': {
            'direction': 100,
            'direction_range': (75, 100),
            'intensity_multiplier': 1.0,
            'name': '大幅增加需求'
        }
    },
    'policy_strength': {
        # Ordinal score: 25 (informational) to 100 (mandatory)
        'informational': {
            'strength': 25,
            'strength_range': (0, 37.5),
            'enforcement_level': 'voluntary',
            'name': '信息公告'
        },
        'encouraged': {
            'strength': 50,
            'strength_range': (37.5, 62.5),
            'enforcement_level': 'incentivized',
            'name': '鼓励支持'
        },
        'binding': {
            'strength': 75,
            'strength_range': (62.5, 87.5),
            'enforcement_level': 'guided',
            'name': '约束指导'
        },
        'mandatory': {
            'strength': 100,
            'strength_range': (87.5, 100),
            'enforcement_level': 'required',
            'name': '强制执行'
        }
    }
}

# Score aggregation weights for final composite score
AGGREGATION_WEIGHTS = {
    'supply': {
        'relevance': 0.4,      # How related to supply dimension
        'intensity': 0.3,      # How strong the supply effect
        'direction': 0.2,      # Which direction (increase/decrease)
        'confidence': 0.1      # Model confidence
    },
    'demand': {
        'relevance': 0.4,
        'intensity': 0.3,
        'direction': 0.2,
        'confidence': 0.1
    },
    'policy_strength': {
        'relevance': 0.45,     # Slightly higher weight on relevance
        'strength': 0.35,      # Policy enforcement level
        'confidence': 0.2      # Model confidence
    }
}

# Similarity calculation parameters
SIMILARITY_PARAMS = {
    'method': 'weighted_comparison',  # 'embeddings' or 'weighted_comparison'
    'comparison_weights': {
        'title_similarity': 0.2,
        'content_similarity': 0.5,
        'temporal_proximity': 0.1,     # How close in time
        'source_alignment': 0.1,       # Same source gets bonus
        'keyword_overlap': 0.1         # Shared key terms
    },
    'min_similarity_threshold': 0.1,   # Below this, consider unrelated
    'anchor_weight_decay': 0.8         # How much to weight non-best anchors
}

# Batch processing configuration
BATCH_CONFIG = {
    'batch_size': 50,                  # Documents per batch
    'max_parallel_batches': 10,        # Parallel processing
    'save_intermediate': True,          # Save after each batch
    'retry_failed': True,              # Retry failed documents
    'max_retries': 3
}

# Output configuration
OUTPUT_CONFIG = {
    'scores_dir': '/Users/siruizhang/Desktop/碳交易/Project/02_Data_Processed/07_Document_Scores',
    'format': 'parquet',               # 'json' or 'parquet'
    'include_metadata': True,          # Include document metadata in output
    'include_explanations': True,      # Include scoring explanations
    'compression': 'snappy'            # For parquet files
}

# API configuration for scoring
SCORING_API_CONFIG = {
    'model': 'gpt-4o-mini',
    'temperature': 0.3,                # Low temperature for consistent scoring
    'max_tokens': 1500,                # Enough for detailed analysis
    'system_prompt': """You are an expert at analyzing carbon trading policy documents.
    Your task is to score documents against reference anchors with high precision.
    
    Consider:
    1. Direct policy implications for carbon markets
    2. Regulatory strength and enforcement mechanisms
    3. Quantitative vs qualitative impacts
    4. Temporal aspects (immediate vs long-term effects)
    5. Scope of application (local vs systemic)
    
    Provide numeric scores with detailed reasoning."""
}

# Scoring prompts for different dimensions
SCORING_PROMPTS = {
    'supply': """Compare this document to the supply dimension anchor documents.
    
    Evaluate:
    1. RELEVANCE (0-100): How much does this document relate to carbon quota supply?
       - 0-10: Unrelated to supply
       - 10-30: Mentions supply tangentially
       - 30-60: Discusses supply as secondary topic
       - 60-85: Supply is primary focus
       - 85-100: Core supply policy document
    
    2. DIRECTION (-100 to +100): What is the supply effect?
       - -100 to -75: Major supply decrease
       - -75 to -25: Minor supply decrease
       - -25 to +25: Neutral/mixed effects
       - +25 to +75: Minor supply increase
       - +75 to +100: Major supply increase
    
    3. INTENSITY (0-100): How strong is the supply impact?
       - 0-20: Negligible impact
       - 20-40: Minor impact
       - 40-60: Moderate impact
       - 60-80: Significant impact
       - 80-100: Major/transformative impact
    
    4. CONFIDENCE (0-100): How confident are you in this assessment?
    
    Return JSON with these four scores and brief explanation.""",
    
    'demand': """Compare this document to the demand dimension anchor documents.
    
    Evaluate:
    1. RELEVANCE (0-100): How much does this document relate to carbon quota demand?
       - 0-10: Unrelated to demand
       - 10-30: Mentions demand tangentially
       - 30-60: Discusses demand as secondary topic
       - 60-85: Demand is primary focus
       - 85-100: Core demand policy document
    
    2. DIRECTION (-100 to +100): What is the demand effect?
       - -100 to -75: Major demand decrease
       - -75 to -25: Minor demand decrease
       - -25 to +25: Neutral/mixed effects
       - +25 to +75: Minor demand increase
       - +75 to +100: Major demand increase
    
    3. INTENSITY (0-100): How strong is the demand impact?
       - 0-20: Negligible impact
       - 20-40: Minor impact
       - 40-60: Moderate impact
       - 60-80: Significant impact
       - 80-100: Major/transformative impact
    
    4. CONFIDENCE (0-100): How confident are you in this assessment?
    
    Return JSON with these four scores and brief explanation.""",
    
    'policy_strength': """Compare this document to the policy strength anchor documents.
    
    Evaluate:
    1. RELEVANCE (0-100): How much does this document relate to policy enforcement?
       - 0-10: Unrelated to policy enforcement
       - 10-30: Mentions enforcement tangentially
       - 30-60: Discusses enforcement as secondary topic
       - 60-85: Enforcement is primary focus
       - 85-100: Core enforcement policy document
    
    2. STRENGTH (0-100): What is the enforcement level?
       - 0-25: Informational only
       - 25-50: Encouraged/voluntary
       - 50-75: Binding guidance
       - 75-100: Mandatory with penalties
    
    3. CONFIDENCE (0-100): How confident are you in this assessment?
    
    Return JSON with these three scores and brief explanation."""
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'min_valid_relevance': 5,          # Below this, mark as unrelated
    'outlier_detection': {
        'z_score_threshold': 3,         # Flag scores > 3 std devs from mean
        'check_distribution': True       # Verify normal distribution of scores
    },
    'consistency_check': {
        'max_dimension_conflict': 30,   # Max difference between related dimensions
        'require_explanation': True      # Require explanation for conflicts
    }
}

# File paths
BASE_DIR = Path('/Users/siruizhang/Desktop/碳交易/Project')
ANCHOR_PATH = BASE_DIR / '02_Data_Processed/06_Anchor_Analysis/anchor_summary.json'
FILTERED_DOCS_PATH = BASE_DIR / '02_Data_Processed/05_Policy_Doc_Filtered'
OUTPUT_PATH = BASE_DIR / '02_Data_Processed/07_Document_Scores'

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)