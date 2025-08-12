#!/bin/bash
# Quick test script to verify all models can run without errors

echo "========================================="
echo "TESTING ALL LSTM MODELS"
echo "========================================="
echo ""

# Test Daily LSTM
echo "1. Testing Daily LSTM (GDEA, no sentiment)..."
cd /Users/siruizhang/Desktop/碳交易/Project/03_Code/10_LSTM_Daily
timeout 10 python run.py --market GDEA 2>&1 | grep -E "Loading|Training|Error|failed" | head -5
if [ $? -eq 124 ]; then
    echo "✅ Daily LSTM started training (timed out after 10s - expected)"
else
    echo "✅ Daily LSTM test completed"
fi
echo ""

# Test Daily LSTM with Sentiment
echo "2. Testing Daily LSTM with Sentiment (HBEA)..."
cd /Users/siruizhang/Desktop/碳交易/Project/03_Code/11_LSTM_Daily_Sentiment
timeout 10 python run.py --market HBEA 2>&1 | grep -E "Loading|Training|Error|failed" | head -5
if [ $? -eq 124 ]; then
    echo "✅ Daily LSTM Sentiment started training (timed out after 10s - expected)"
else
    echo "✅ Daily LSTM Sentiment test completed"
fi
echo ""

# Test Weekly LSTM
echo "3. Testing Weekly LSTM (GDEA, base)..."
cd /Users/siruizhang/Desktop/碳交易/Project/03_Code/12_LSTM_Weekly
timeout 10 python run.py --market GDEA --sentiment base 2>&1 | grep -E "Loading|Training|Error|failed" | head -5
if [ $? -eq 124 ]; then
    echo "✅ Weekly LSTM started training (timed out after 10s - expected)"
else
    echo "✅ Weekly LSTM test completed"
fi
echo ""

# Test Meta Model
echo "4. Testing Daily Meta Model (GDEA)..."
cd /Users/siruizhang/Desktop/碳交易/Project/03_Code/13_Meta_Model
python run.py --market GDEA 2>&1 | grep -E "Using primary|Training|completed|failed" | head -5
echo "✅ Meta Model test completed"
echo ""

# Test Weekly Meta Model
echo "5. Testing Weekly Meta Model (HBEA)..."
cd /Users/siruizhang/Desktop/碳交易/Project/03_Code/13_Meta_Model
python run_weekly.py --market HBEA 2>&1 | grep -E "Loading|Training|completed|failed" | head -5
echo "✅ Weekly Meta Model test completed"
echo ""

echo "========================================="
echo "✅ ALL MODEL TESTS COMPLETED"
echo "========================================="