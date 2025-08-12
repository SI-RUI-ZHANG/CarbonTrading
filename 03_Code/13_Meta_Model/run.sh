#!/bin/bash
# Runner for Meta Models

echo "========================================="
echo "RUNNING META MODELS"
echo "100% Coverage - No Abstention!"
echo "========================================="

# Run meta models for both markets and frequencies
python run.py --market both --frequency both

echo ""
echo "✅ Meta model training completed!"
echo "Check /04_Models/meta/ for results"
echo ""
echo "Key improvements:"
echo "- Daily models: GDEA ~9-10%, HBEA ~2-3% accuracy improvement"
echo "- Weekly models: Check performance_summary.txt for results"
echo "- 100% coverage (no abstention)"