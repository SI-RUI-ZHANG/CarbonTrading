#!/bin/bash
# Run the document processing pipeline

echo "Starting Document Processing Pipeline..."
echo "======================================="

# Change to the script directory
cd "$(dirname "$0")"

# Run the pipeline
python3 03_pipeline_runner.py

echo ""
echo "Pipeline execution complete!"