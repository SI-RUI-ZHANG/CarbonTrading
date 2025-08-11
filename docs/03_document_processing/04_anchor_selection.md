# Anchor Selection: MapReduce for Document Exemplars

## The Core Idea

Finding 12 exemplar documents from 2,617 carbon trading policies that best represent the spectrum of policy impacts. These anchors become reference points for scoring all other documents.

## Why Anchors?

Traditional clustering fails in policy analysis because:
- Policies don't form neat clusters - they exist on continuous spectrums
- We need interpretable reference points, not abstract centroids
- Human experts can validate actual documents, not statistical constructs

Anchors provide concrete examples: "This policy is similar to the 2024 Guangdong quota reduction announcement" is more meaningful than "This policy has cluster score 0.73".

## The Design: 3×4 Matrix

```
Supply Dimension:       Restrict  Reduce  Increase  Expand
Demand Dimension:       Restrict  Reduce  Increase  Expand  
Policy Strength:        Info      Encour  Binding   Mandatory
```

**Why these dimensions?**
- **Supply & Demand**: Carbon markets are fundamentally about quota supply and demand
- **Policy Strength**: Enforcement level determines market impact magnitude
- **4 categories each**: Captures spectrum extremes and moderate positions

## MapReduce Architecture

The challenge: 2,617 documents require 31,404 pairwise comparisons to find the best exemplars.

**Our solution:**
```python
# Phase 1: Map - Parallel group processing
groups = split_documents(2617, batch_size=20)  # 131 groups
anchors = parallel_map(select_anchors, groups)  # 131 local anchors

# Phase 2: Reduce - Binary tournament merging  
while len(anchors) > 1:
    anchors = binary_merge(anchors)  # Log(n) merges
```

**Key advantages:**
1. **Parallelization**: Process 100+ groups simultaneously
2. **Fault tolerance**: Failed groups don't crash the pipeline
3. **Incremental progress**: Each group produces valid local anchors
4. **Logarithmic merging**: Only log(131) ≈ 7 merge rounds needed

## Smart Comparison Strategy

Instead of comparing all documents globally, we use tournament selection:

```python
def compare_documents(doc_a, doc_b, dimension, category):
    # One focused comparison instead of global ranking
    prompt = f"Which document better represents {category} in {dimension}?"
    return winner
```

**Why binary comparisons?**
- LLMs excel at A/B comparisons but struggle with ranking 20+ items
- Each comparison has clear context and criteria
- Transitive property: if A > B and B > C, then A > C
- Reduces cognitive load on the model

## Intelligent Rate Limiting

The system adapts to API limits in real-time:

```python
class SmartRateLimiter:
    def hit_rate_limit(self):
        self.current_rate *= 0.5  # Reduce by 50%
    
    def success(self):
        if consecutive_successes > threshold:
            self.current_rate *= 1.5  # Gradually recover
```

**Results:**
- 807 API calls completed without a single 429 error
- Processing time: 13 minutes for 2,617 documents
- Automatic recovery maintains optimal throughput

## Key Insights from Implementation

### 1. Fill Rate: 100%
All 12 anchor slots filled successfully, proving the distribution of policy types across our corpus.

### 2. Dimension Correlation
Supply and demand anchors often come from the same documents, confirming that policies typically affect both sides of the market.

### 3. Source Distribution
- GZETS dominates anchors (8/12) - more detailed policy documents
- MEE provides regulatory anchors (3/12) - national-level mandates
- HBETS has fewer distinct policies (1/12) - follows national guidelines

### 4. Temporal Clustering
Recent documents (2024-2025) dominate anchors, reflecting policy evolution and increasing sophistication.

## Why Not Embeddings?

We explicitly chose LLM comparison over embedding similarity:

1. **Semantic nuance**: "减少配额" (reduce quota) vs "收紧配额" (tighten quota) have similar embeddings but different market implications
2. **Context awareness**: LLMs understand that "暂时冻结" (temporarily freeze) in a title signals major supply impact
3. **Multi-dimensional evaluation**: One document can be strong in supply but weak in policy strength

## The Outcome

The selected anchors successfully span the policy spectrum:
- **Supply anchors**: From 2024 quota tightening (-100) to 2019 auction releases (+100)
- **Demand anchors**: From market exit policies (-100) to coverage expansion (+100)  
- **Policy anchors**: From information notices (0) to compliance mandates (100)

These 12 documents now serve as the coordinate system for scoring all 2,617 documents, transforming an abstract policy space into a navigable spectrum.