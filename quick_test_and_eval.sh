#!/bin/bash
# Quick test and evaluation script
# Run this to verify all fixes and start evaluation

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         PIDS Framework - Quick Test & Evaluation          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Test fixes
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Testing Code Fixes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/test_fixes.py

if [ $? -ne 0 ]; then
    echo "❌ Code tests failed. Please check the output above."
    exit 1
fi

echo ""
read -p "✓ Code tests passed. Press Enter to continue..."
echo ""

# Step 2: Test dataset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Testing Dataset Loading"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/test_dataset_fix.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset tests failed. Please check the output above."
    exit 1
fi

echo ""
read -p "✓ Dataset tests passed. Press Enter to start evaluation..."
echo ""

# Step 3: Run evaluation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Running Model Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting evaluation on custom_soc dataset..."
echo "This may take 10-30 minutes depending on your system."
echo ""

./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    EVALUATION COMPLETE!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results are saved in: results/evaluation_*/"
echo ""
echo "To view results:"
echo "  ls -lh results/evaluation_*/"
echo "  cat results/evaluation_*/comparison_report.json"
echo ""
