# Anchor Selection System

MapReduce-based document selection system for identifying exemplar carbon trading policy documents.

## Setup

1. **API Key Configuration**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your-actual-api-key-here
   ```

2. **Install Dependencies**
   ```bash
   pip install openai python-dotenv pandas numpy
   ```

## Usage

```bash
# Run full pipeline (2,617 documents)
python run_pipeline.py

# Test mode with subset
python run_pipeline.py --test --test-size 30

# Custom configuration
python run_pipeline.py --batch-size 30 --max-parallel 20
```

## Security Note

**IMPORTANT**: Never commit your `.env` file containing the API key. The `.env` file is already in `.gitignore` to prevent accidental commits.

## Output

Results are saved to:
- `02_Data_Processed/06_Anchor_Analysis/anchor_summary.json` - Final anchors with full content
- `02_Data_Processed/06_Anchor_Analysis/anchors/final_anchors.json` - Anchor documents
- `02_Data_Processed/06_Anchor_Analysis/run_statistics.json` - Processing statistics

## Documentation

See [docs/anchor_selection/AnchorSelection.md](../../docs/anchor_selection/AnchorSelection.md) for detailed documentation.