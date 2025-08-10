# Document Cleaning Report

Generated: 2025-08-10 22:15:55

## Summary

- **Total Documents Processed**: 3,312
- **Documents Cleaned**: 914
- **Total Characters Removed**: 401,234
- **Average Characters Removed**: 439 per cleaned document

## Patterns Applied

| Pattern | Times Applied |
|---------|---------------|
| navigation_header | 460 |
| footer_navigation | 450 |
| trailing_whitespace | 450 |
| empty_metadata | 55 |
| footer_simple | 55 |
| multiple_spaces | 1 |

## Impact Analysis

### GZETS Documents
- Primary source of navigation garbage
- Typical removal: 300-400 characters per document
- Patterns: navigation headers, footers, copyright info

### HBETS & MEE Documents
- Generally clean, minimal processing needed
- Only global whitespace cleaning applied
