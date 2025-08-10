# Document Scoring: Spectrum Positioning via Weighted Interpolation

## The Core Insight

Every policy document exists somewhere on three spectrums. By measuring similarity to anchor documents at spectrum extremes, we can precisely position any document in policy space.

## The Spectrum Model

Unlike traditional classification, we use continuous bidirectional scales:

```
Supply:  -100 ←── Major Decrease ── Neutral ── Major Increase ──→ +100
Demand:  -100 ←── Major Decrease ── Neutral ── Major Increase ──→ +100  
Policy:     0 ←── Informational ─── Voluntary ─── Mandatory ────→ 100
```

**Why bidirectional for supply/demand?**
- Policies don't just have magnitude, they have direction
- A supply decrease is fundamentally different from supply increase
- The scales capture market pressure: negative supply + positive demand = upward price pressure

**Why unidirectional for policy strength?**
- Enforcement only varies in intensity, not direction
- From "FYI" announcements (0) to "must comply or face penalties" (100)

## The Scoring Algorithm

The elegance lies in weighted interpolation based on similarity:

```python
def calculate_spectrum_score(similarities, dimension):
    # Normalize similarities to weights (sum to 1)
    total = sum(similarities.values())
    weights = {cat: sim/total for cat, sim in similarities.items()}
    
    # Weighted average of anchor positions
    score = sum(weights[cat] * SPECTRUM_POSITIONS[cat] 
                for cat in weights)
    return score
```

**Example calculation:**
If a document has similarities to supply anchors:
- major_decrease: 70 (high similarity)
- minor_decrease: 30  
- minor_increase: 10
- major_increase: 5

Weights: [0.61, 0.26, 0.09, 0.04]  
Score: 0.61×(-100) + 0.26×(-50) + 0.09×(50) + 0.04×(100) = -65.2

This document strongly decreases supply.

## Single API Call Design

Traditional approach would need 12 API calls:
```python
# Inefficient: 12 separate calls
for anchor in all_anchors:
    similarity = get_similarity(document, anchor)
```

Our approach uses one comprehensive call:
```python
# Efficient: All similarities in one request
similarities = get_all_similarities(document, all_12_anchors)
```

**Advantages:**
- 12× reduction in API calls (2,617 vs 31,404)
- Consistent comparison context across all anchors
- Better token efficiency with batched prompt
- Total processing: 7.4 minutes for 2,617 documents

## Why Neutral Clustering is Good

Initial concern: "Most documents score near neutral, is this a bug?"

**Reality: It's a feature.**

<img width="600" alt="Distribution shows most documents cluster around neutral with few extremes">

The distribution reflects market maturity:
- **~60% near neutral**: Routine operational documents (trading hours, minor adjustments)
- **~30% moderate impact**: Regular policy updates and clarifications
- **~10% extreme scores**: Major reforms and structural changes

This matches carbon market reality where stability is the norm and shocks are rare.

## Key Findings from 2,617 Documents

### Supply vs Demand Dynamics
- **Mean Supply**: +4.82 (slight increase bias)
- **Mean Demand**: +7.68 (stronger increase bias)
- **Net Effect**: +2.86 demand over supply = mild upward price pressure

### Policy Strength Distribution
- **Mean**: 43.6 (between encouraged and binding)
- **Most documents**: 35-55 range (voluntary to guided)
- **Few extremes**: Only 3.5% score >67 (mandatory)

### Source Patterns
- **MEE**: Balanced supply/demand, higher policy strength (regulatory authority)
- **GZETS**: Supply-focused, moderate policy strength (market operations)
- **HBETS**: Demand-focused, lower policy strength (market facilitation)

## Design Choices Explained

### Why Not Keywords?
Initial thought: "Just search for '配额减少' (quota reduction)"

**Problem**: Context matters enormously
- "考虑减少" (considering reduction) ≠ "立即减少" (immediate reduction)  
- "减少企业负担" (reduce enterprise burden) ≠ "减少配额" (reduce quota)
- Negations: "不减少" (not reduce) would match keyword search

### Why Not Embeddings?
Embeddings measure semantic similarity, not policy impact:
- "碳市场交易公告" and "碳市场发展报告" have high embedding similarity
- But one is operational (neutral) and one could announce major changes (extreme)

### Why One-Shot Similarity?
Getting all 12 similarities in one API call forces the model to:
1. Consider document holistically
2. Make relative comparisons
3. Maintain consistent evaluation criteria

## The Confidence Dimension

Confidence isn't about policy content, it's about classification certainty:

```python
confidence = 0.7 * max_similarity + 0.3 * (1 - variance_factor)
```

- **High confidence**: Document clearly matches certain anchors
- **Low confidence**: Document is unique or ambiguous

**Mean confidence: 42.3** - Most documents are somewhat unique, which is expected given the diverse nature of policy documents.

## Practical Impact

The scoring system enables:

1. **Trend Detection**: Track policy direction over time
2. **Market Prediction**: Net supply/demand indicates price pressure
3. **Regulatory Analysis**: Policy strength shows enforcement trends
4. **Anomaly Detection**: Low confidence flags unusual documents

Most importantly, any document can now be understood in context: "This announcement is similar to the 2024 quota tightening but with weaker enforcement" - actionable intelligence from mathematical scoring.