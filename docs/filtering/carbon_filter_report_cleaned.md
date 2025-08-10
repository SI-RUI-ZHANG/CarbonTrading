# Carbon Document Filtering Report (Cleaned Documents)

Generated: 2025-08-10 22:16:03

## Configuration

- **Filter Keyword**: `碳`
- **Search Fields**: title, content
- **Case Sensitive**: False
- **Input**: Cleaned documents from `04_Documents_Cleaned/`
- **Output**: `05_Policy_Doc_Filtered/`

## Results Summary

- **Total Documents Processed**: 3,312
- **Documents Retained**: 2,617 (79.0%)
- **Documents Removed**: 695

## Results by Source

| Source | Category | Original | Filtered | Retention Rate |
|--------|----------|----------|----------|---------------|
| MEE | Decrees | 89 | 4 | 4.5% |
| MEE | Notices | 484 | 33 | 6.8% |
| HBETS | Center_Dynamics | 684 | 675 | 98.7% |
| GZETS | Trading_Announcements | 684 | 576 | 84.2% |
| GZETS | Center_Dynamics | 876 | 858 | 97.9% |
| GZETS | Provincial_Municipal | 495 | 471 | 95.2% |

## Pipeline Summary

1. **Raw Documents**: 3,312 total
2. **After Cleaning**: 3,312 documents (navigation garbage removed)
3. **After Carbon Filter**: 2,617 documents retained

### Impact of Two-Stage Processing
- Navigation cleaning reduced document sizes (especially GZETS)
- Carbon filtering removed non-carbon environmental documents
- Final dataset is clean and focused on carbon trading
