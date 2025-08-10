# Anchor Document Selection System

## Overview

The Anchor Selection System identifies representative policy documents across different dimensions to provide contextual signals for LSTM carbon price prediction models. By selecting "anchor" documents that exemplify specific policy categories, the system creates a diverse reference set that helps the LSTM model understand how different types of policies might influence carbon market dynamics.

## System Architecture

### MapReduce Pattern

The system employs a MapReduce architecture for efficient parallel processing:

1. **Map Phase**: Documents are divided into groups and processed in parallel
2. **Reduce Phase**: Group results are merged using binary tournament selection
3. **Final Selection**: Best documents are selected as anchors for each category

### Key Components

```
document_loader.py    → Load filtered carbon-relevant documents
     ↓
group_processor.py    → Process groups in parallel (Map)
     ↓
api_client.py        → GPT-4o-mini classification & comparison
     ↓
merger.py            → Binary tournament merge (Reduce)
     ↓
anchor_summary.json  → Final anchors with full content
```

## Dimension System

The system uses **3 dimensions** with **4 categories each**, creating **12 total anchor slots**:

### 1. Supply Dimension (配额供给)
Policies affecting carbon quota supply:
- `major_decrease`: 大幅减少供给 - Policies significantly reducing quota supply
- `minor_decrease`: 适度减少供给 - Policies moderately reducing quota supply
- `minor_increase`: 适度增加供给 - Policies moderately increasing quota supply
- `major_increase`: 大幅增加供给 - Policies significantly increasing quota supply

### 2. Demand Dimension (配额需求)
Policies affecting carbon quota demand:
- `major_increase`: 大幅增加需求 - Policies significantly increasing demand
- `minor_increase`: 适度增加需求 - Policies moderately increasing demand
- `minor_decrease`: 适度减少需求 - Policies moderately decreasing demand
- `major_decrease`: 大幅减少需求 - Policies significantly decreasing demand

### 3. Policy Strength Dimension (政策强度)
Enforcement and binding levels:
- `mandatory`: 强制执行 - Mandatory enforcement with penalties
- `binding`: 约束指导 - Binding guidance with targets
- `encouraged`: 鼓励支持 - Encouraged with incentives
- `informational`: 信息公告 - Informational announcements

## Processing Pipeline

### 1. Document Loading
```python
# Load filtered carbon-relevant documents from all sources
documents = load_filtered_documents()
# Sources: MEE, HBETS, GZETS (2,617 total after filtering)
```

### 2. Group Processing (Map Phase)
```python
# Default configuration
BATCH_SIZE = 20  # Documents per group
MAX_PARALLEL_GROUPS = 30  # Optimized for rate limits

# Each group independently selects best documents
# Uses GPT-4o-mini for classification and comparison
```

### 3. Binary Tournament Merge (Reduce Phase)
```python
# Groups are merged pairwise in tournament style
# At each level, best documents are selected via API comparison
# Process continues until single set of anchors remains
```

### 4. Output Generation
The system generates a JSON file with full document content and metadata.

## API Configuration

### Model Settings
- **Model**: GPT-4o-mini
- **Temperature**: 0.3 (for consistent classification)
- **Max Tokens**: 1,000 per API call

### Rate Limiting
Smart rate limiting with adaptive throttling:
- **Target**: 3,000 calls/minute (Tier 2 limits)
- **Buffer**: 90% utilization to avoid hitting limits
- **Retry**: Exponential backoff with 5 max attempts
- **Monitoring**: Real-time tracking of call rates and token usage

## Usage

### Basic Run
```bash
cd 03_Code/07_Anchor_Selection/
python run_pipeline.py
```

### Custom Configuration
```bash
# Adjust batch size and parallelism
python run_pipeline.py --batch-size 30 --max-parallel 20

# Test mode with subset of documents
python run_pipeline.py --test --test-size 30
```

### Command Line Arguments
- `--batch-size`: Documents per group (default: 20)
- `--max-parallel`: Maximum parallel groups (default: 30)
- `--test`: Enable test mode
- `--test-size`: Number of documents in test mode (default: 10)

## Output Format

### JSON Structure
```json
{
  "summary": {
    "total_slots": 12,
    "filled": 12,
    "fill_rate": "100.0%",
    "empty": 0,
    "total_comparisons": 245,
    "documents_processed": 2617,
    "processing_time": "5m 32s"
  },
  "anchors": {
    "supply": {
      "major_decrease": {
        "doc_id": "mee_decree_2024_001",
        "title": "关于严格控制碳配额发放的通知",
        "content": "...",
        "publish_date": "2024-01-15",
        "source": "MEE",
        "url": "...",
        "classification_scores": {...}
      },
      // ... other supply categories
    },
    "demand": {
      // ... demand categories with full document content
    },
    "policy_strength": {
      // ... policy strength categories with full document content
    }
  }
}
```

### File Locations
- **Input**: `02_Data_Processed/05_Policy_Doc_Filtered/*/` - Filtered documents
- **Output**: `02_Data_Processed/06_Anchor_Analysis/anchors/anchor_summary.json`
- **Logs**: `02_Data_Processed/06_Anchor_Analysis/selection_log.json`

## Performance Metrics

With 2,617 filtered carbon-relevant documents:
- **Fill Rate**: 100% (12/12 slots filled)
- **Processing Time**: ~5-10 minutes depending on API latency
- **API Calls**: ~250-300 total (classification + comparisons)
- **Token Usage**: ~500K tokens total

## Integration with LSTM Model

The anchor documents serve as contextual references for LSTM price prediction:

1. **Policy Context**: Provides examples of how different policy types are expressed
2. **Market Signals**: Each anchor represents a potential market influence pattern
3. **Feature Enhancement**: Can be used to:
   - Generate policy similarity features
   - Create categorical policy indicators
   - Provide context for NLP-based feature extraction

### Example Integration
```python
# Load anchor documents
with open('anchor_summary.json', 'r') as f:
    anchors = json.load(f)

# Extract policy patterns for each dimension
supply_patterns = anchors['anchors']['supply']
demand_patterns = anchors['anchors']['demand']
strength_patterns = anchors['anchors']['policy_strength']

# Use for feature engineering or model context
# e.g., Calculate similarity of new policies to anchors
# e.g., Create one-hot encoding based on closest anchor
```

## Technical Notes

### Why 3 Dimensions?
Through empirical testing with 2,617 documents, these three dimensions were found to:
- Capture the primary policy levers affecting carbon markets
- Achieve 100% fill rate (all categories have representative documents)
- Provide orthogonal perspectives (supply vs demand vs enforcement)

### MapReduce Advantages
- **Scalability**: Can process thousands of documents efficiently
- **Parallelism**: Leverages API rate limits effectively
- **Robustness**: Failures in one group don't affect others
- **Quality**: Tournament selection ensures best documents rise to top

### Design Decisions
1. **Binary Classification**: Each document classified into most relevant category
2. **Tournament Merge**: Ensures global optimum through pairwise comparisons
3. **Full Content Storage**: Preserves complete document for downstream analysis
4. **Single Anchor per Category**: Provides clear exemplar for each policy type

## Dependencies

```python
# Core requirements
openai >= 1.0.0
pandas
numpy
asyncio
aiohttp

# Configuration
API_KEY in config.py
MODEL = "gpt-4o-mini"
```

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Reduce `MAX_PARALLEL_GROUPS` in config.py
   - Increase `RETRY_DELAY` for more conservative pacing

2. **Empty Slots**
   - Increase `BATCH_SIZE` to ensure enough documents per group
   - Check that input documents are properly filtered

3. **API Timeouts**
   - Check network connection
   - Verify API key is valid and has credits
   - Consider reducing `MAX_PARALLEL_GROUPS`

## Future Enhancements

Potential improvements for the anchor selection system:

1. **Dynamic Dimensions**: Automatically discover dimensions from document corpus
2. **Multi-Anchor Support**: Select top-N documents per category
3. **Temporal Anchors**: Select anchors from different time periods
4. **Cross-Market Anchors**: Compare anchors across GDEA and HBEA markets
5. **Embedding-Based Selection**: Use semantic embeddings instead of API classification