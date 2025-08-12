#!/bin/bash
# Script to train all LSTM models with different configurations
# This will force actual training runs (not just quick checks)

echo "========================================="
echo "COMPREHENSIVE LSTM MODEL TRAINING"
echo "Started at: $(date)"
echo "========================================="

# Base directory
BASE_DIR="/Users/siruizhang/Desktop/碳交易/Project/03_Code"

# Create log directory
LOG_DIR="$BASE_DIR/training_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo ""

# Function to run a model and log output
run_model() {
    local model_name=$1
    local working_dir=$2
    local command=$3
    local log_file="$LOG_DIR/${model_name}.log"
    
    echo "========================================="
    echo "Training: $model_name"
    echo "Command: $command"
    echo "Log: $log_file"
    echo "Starting at: $(date)"
    echo "========================================="
    
    cd "$working_dir"
    
    # Run the command and capture output
    if $command > "$log_file" 2>&1; then
        echo "✅ $model_name completed successfully"
        # Extract key metrics from log
        echo "Key metrics:"
        grep -i "accuracy\|loss\|completed" "$log_file" | tail -5
    else
        echo "❌ $model_name failed"
        echo "Error details:"
        tail -10 "$log_file"
    fi
    
    echo ""
}

# 1. Daily LSTM without sentiment
echo "========================================"
echo "1. DAILY LSTM MODELS (NO SENTIMENT)"
echo "========================================"
run_model "daily_lstm_both" \
    "$BASE_DIR/10_LSTM_Daily" \
    "python run.py"

# 2. Daily LSTM with sentiment  
echo "========================================"
echo "2. DAILY LSTM MODELS WITH SENTIMENT"
echo "========================================"
run_model "daily_lstm_sentiment_both" \
    "$BASE_DIR/11_LSTM_Daily_Sentiment" \
    "python run.py"

# 3. Weekly LSTM - all combinations
echo "========================================"
echo "3. WEEKLY LSTM MODELS"
echo "========================================"
run_model "weekly_lstm_all" \
    "$BASE_DIR/12_LSTM_Weekly" \
    "python run.py --market both --sentiment both"

# 4. Meta models - Daily
echo "========================================"
echo "4. DAILY META MODELS"
echo "========================================"
run_model "daily_meta_both" \
    "$BASE_DIR/13_Meta_Model" \
    "python run.py"

# 5. Meta models - Weekly
echo "========================================"
echo "5. WEEKLY META MODELS"
echo "========================================"
run_model "weekly_meta_both" \
    "$BASE_DIR/13_Meta_Model" \
    "python run_weekly.py"

# Generate summary
echo "========================================="
echo "TRAINING SUMMARY"
echo "========================================="
echo "Completed at: $(date)"
echo "Log directory: $LOG_DIR"
echo ""
echo "Models trained:"
ls -la "$LOG_DIR"/*.log 2>/dev/null | wc -l
echo ""
echo "Individual logs:"
ls -la "$LOG_DIR"/*.log 2>/dev/null

# Check for failures
echo ""
echo "Checking for failures..."
if grep -l "Error\|Failed\|Traceback" "$LOG_DIR"/*.log 2>/dev/null; then
    echo "⚠️ Some models may have failed. Check logs for details."
else
    echo "✅ All models appear to have completed successfully!"
fi

echo ""
echo "========================================="
echo "✅ TRAINING SCRIPT COMPLETED"
echo "========================================="