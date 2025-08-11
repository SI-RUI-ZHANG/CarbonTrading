# Document Scoring: Direct Spectrum Positioning

## The Core Insight

Every policy document exists at a specific position on three spectrums. By comparing to reference anchor documents, the LLM directly places each document at its appropriate position.

## The Spectrum Model

We use three continuous scales to capture policy dimensions:

```
Supply Impact (carbon quota availability):
←[RESTRICT]───────────────[NEUTRAL]───────────────[EXPAND]→
   -150    -100    -50        0        +50    +100    +150
           ▼       ▼                    ▼       ▼
        restrict  reduce            increase  expand
        (anchor)  (anchor)          (anchor)  (anchor)

Demand Impact (carbon quota demand):
←[SUPPRESS]───────────────[NEUTRAL]───────────────[BOOST]→
   -150    -100    -50        0        +50    +100    +150
           ▼       ▼                    ▼       ▼
        restrict  reduce            increase  expand
        (anchor)  (anchor)          (anchor)  (anchor)

Policy Strength (enforcement level):
[WEAK]────────────────────────────────────────[STRONG]→
  0        33         67        100       150
  ▼         ▼          ▼         ▼
inform  encourage  binding  mandatory
(anchor) (anchor)  (anchor)  (anchor)
```

**Why bidirectional for supply/demand?**
- Policies don't just have magnitude, they have direction
- Negative values = restrictive/reducing policies
- Positive values = expansive/increasing policies
- Zero = neutral/no impact

**Why unidirectional for policy strength?**
- Enforcement only varies in intensity, not direction
- From informational (0) to mandatory with penalties (100+)

## The Direct Positioning Algorithm

Unlike traditional similarity-based scoring, we use direct position placement:

```python
def score_document(positions):
    # No calculation needed - direct passthrough
    return {
        'supply': positions['supply'],        # -150 to +150
        'demand': positions['demand'],        # -150 to +150  
        'policy_strength': positions['policy_strength']  # 0 to 150
    }
```

**The simplicity is the innovation**: The LLM examines the document, compares it to the reference anchors at known positions, and directly determines where on each spectrum the document belongs.

## Why Extended Range (-150 to +150)?

Documents can be MORE extreme than our reference anchors:

- **Position -120**: More restrictive than the "restrict" anchor at -100
- **Position +130**: More expansive than the "expand" anchor at +100
- **Position 75**: Between "binding" (67) and "mandatory" (100)

This prevents ceiling effects where extreme documents would cluster at ±100.

## Single API Call Design

Traditional similarity approach would need complex calculations:
```python
# OLD: Get similarities, then calculate position
similarities = get_12_similarities(document)  # 12 values
position = weighted_interpolation(similarities)  # Complex math
```

Our direct approach:
```python
# NEW: Get position directly
positions = get_positions(document)  # 3 values, done!
```

**Advantages:**
- **90% less code**: ~200 lines reduced to ~20 lines
- **Transparent**: What you see is what you get
- **Accurate**: LLM considers document holistically
- **Efficient**: 3 outputs instead of 12

## Key Findings from 2,617 Documents

### Distribution Characteristics
Most documents cluster near neutral, which reflects market maturity:
- **~60% near neutral** (-25 to +25): Routine operational documents
- **~30% moderate impact** (±25 to ±75): Regular policy updates
- **~10% extreme scores** (beyond ±75): Major reforms and structural changes

### Supply vs Demand Dynamics
- **Mean Supply**: +4.82 (slight expansion bias)
- **Mean Demand**: +7.68 (stronger increase bias)
- **Net Effect**: +2.86 demand over supply = mild upward price pressure

### Policy Strength Distribution
- **Mean**: 43.6 (between encouraged and binding)
- **Majority**: 35-55 range (voluntary to guided)
- **Strong enforcement**: Only 10% score >60

### Source Patterns
- **MEE**: Balanced positioning, higher policy strength (regulatory authority)
- **GZETS**: Supply-focused, moderate policy strength (market operations)
- **HBETS**: Demand-focused, lower policy strength (market facilitation)

## Design Decisions Explained

### Why Not Calculate from Similarities?
Initial approach used weighted interpolation from 12 similarity scores.

**Problems:**
- Complex algorithm obscured the actual positioning logic
- Smoothing and extrapolation added artificial adjustments
- Difficult to debug and validate

**Solution:** Let the LLM do what it does best - understand context and make direct judgments.

### Why Not Keywords or Embeddings?
**Keywords** miss context:
- "减少" (reduce) could mean reduce burden OR reduce quotas
- Negations like "不减少" (not reduce) break keyword logic

**Embeddings** measure semantic similarity, not policy impact:
- Two documents about carbon markets have high embedding similarity
- But one could be routine, another could announce major changes

### Why Trust Direct Positioning?
The LLM can:
1. Understand policy context and implications
2. Compare relative intensity to reference anchors
3. Place documents on a continuous spectrum
4. Handle documents more extreme than anchors

## Practical Applications

The positioning system enables:

1. **Trend Detection**: Track policy direction over time
2. **Price Pressure Analysis**: Net supply/demand indicates market direction
3. **Regulatory Monitoring**: Policy strength shows enforcement trends
4. **Extreme Event Detection**: Identify major policy shifts beyond ±100

## Implementation Details

### Processing Efficiency
- **Total documents**: 2,617 filtered documents processed
- **Unique documents**: 989 after deduplication (37 MEE, 563 HBETS, 389 GZETS)
- **Processing time**: ~7-10 minutes (depending on API tier)
- **API calls**: 2,617 (one per document)
- **Success rate**: Typically >99%

### Batch Processing Architecture
The scoring pipeline uses resilient batch processing with checkpoint/resume functionality:

- **Batch Size**: 50 documents per batch for optimal API throughput
- **Persistent Batch Numbering**: `total_batches_processed` counter ensures unique batch numbers across interrupted runs
- **Checkpoint System**: Saves progress after each batch with processed doc IDs and batch counter
- **Automatic Resume**: Detects interrupted runs and continues from last checkpoint
- **Batch File Storage**: Each batch saved as `batch_XXX.json` for incremental persistence

### Automatic Deduplication
The final merge process ensures data integrity:

```python
def _save_final_results():
    # Read ALL batch files from directory
    batch_files = sorted(config.BATCH_SCORES_PATH.glob("batch_*.json"))
    
    # Merge all batches
    all_scores = []
    for batch_file in batch_files:
        batch_scores = json.load(batch_file)
        all_scores.extend(batch_scores)
    
    # Deduplicate by doc_id (keeps first occurrence)
    df = pd.DataFrame(all_scores)
    df = df.drop_duplicates(subset=['doc_id'], keep='first')
```

This ensures:
- No document scores are lost during interruptions
- Batch files from multiple runs are automatically merged
- Duplicate documents are removed (keeps earliest score)

### Score Interpretation
```python
# Supply/Demand interpretation
if score <= -110:
    interpretation = "Extreme restriction"
elif score <= -75:
    interpretation = "Strong restriction"
elif score <= -25:
    interpretation = "Moderate reduction"
elif score <= 25:
    interpretation = "Neutral"
elif score <= 75:
    interpretation = "Moderate increase"
elif score <= 110:
    interpretation = "Strong expansion"
else:
    interpretation = "Extreme expansion"

# Price pressure (demand - supply)
price_pressure = demand_score - supply_score
# Positive = upward pressure, Negative = downward pressure
```

## Key Innovation: Simplicity

The elegance of this system is its simplicity:
- **No complex algorithms** - just direct positioning
- **No artificial adjustments** - pure LLM judgment
- **No hidden calculations** - transparent scoring
- **No confidence metrics** - positions are document properties, not uncertainties

This makes the system more reliable, maintainable, and interpretable for downstream applications like LSTM price prediction.