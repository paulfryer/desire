#!/bin/bash
# DESIRE Research - Full Reproduction Script
#
# This script reproduces all results from the paper:
# "Desire as Thermodynamic Imperative: Testing Active Inference Agents
#  for Desire-Like Behavior"
#
# Usage:
#   ./scripts/reproduce_all.sh [--quick]
#
# Options:
#   --quick    Run with reduced trials for faster verification
#
# Requirements:
#   - Python 3.10+ with dependencies installed
#   - Or run via Docker: docker build -t desire-research . && docker run desire-research

set -e  # Exit on error

# Configuration
SEED=${DESIRE_SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/reproduction"}
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
echo "=========================================="
echo "DESIRE Research - Full Reproduction"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo "  Quick mode: $QUICK_MODE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/agents"
mkdir -p "$OUTPUT_DIR/test_results"
mkdir -p "$OUTPUT_DIR/figures"
mkdir -p "$OUTPUT_DIR/extended_analysis"
mkdir -p "$OUTPUT_DIR/logs"

# Select config based on mode
if [ "$QUICK_MODE" = true ]; then
    CONFIG="experiments/configs/quick_test.yaml"
    echo "Using quick test configuration (reduced trials)"
else
    CONFIG="experiments/configs/publication_rigorous.yaml"
    echo "Using publication configuration (full trials)"
fi
echo ""

# Step 1: Train agents
echo "=========================================="
echo "Step 1: Training Agents"
echo "=========================================="
echo ""

python experiments/train_agents.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR/agents" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"

echo ""
echo "Agent training complete."
echo ""

# Step 2: Run predicate tests
echo "=========================================="
echo "Step 2: Running Predicate Tests"
echo "=========================================="
echo ""

python experiments/run_tests.py \
    --agents-dir "$OUTPUT_DIR/agents" \
    --output "$OUTPUT_DIR/test_results" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/logs/tests.log"

echo ""
echo "Predicate tests complete."
echo ""

# Step 3: Analyze results
echo "=========================================="
echo "Step 3: Analyzing Results"
echo "=========================================="
echo ""

python experiments/analyze_results.py \
    --results "$OUTPUT_DIR/test_results/test_results.json" \
    --output "$OUTPUT_DIR/test_results" \
    2>&1 | tee "$OUTPUT_DIR/logs/analyze.log"

echo ""
echo "Analysis complete."
echo ""

# Step 4: Generate publication analysis
echo "=========================================="
echo "Step 4: Generating Publication Analysis"
echo "=========================================="
echo ""

python experiments/generate_publication_analysis.py \
    --results "$OUTPUT_DIR/test_results/test_results.json" \
    --output-dir "$OUTPUT_DIR/test_results" \
    2>&1 | tee "$OUTPUT_DIR/logs/publication.log"

echo ""
echo "Publication analysis complete."
echo ""

# Step 5: Generate figures
echo "=========================================="
echo "Step 5: Generating Figures"
echo "=========================================="
echo ""

python scripts/generate_figures.py \
    --results "$OUTPUT_DIR/test_results/test_results.json" \
    --output "$OUTPUT_DIR/figures" \
    2>&1 | tee "$OUTPUT_DIR/logs/figures.log"

echo ""
echo "Figures generated."
echo ""

# Step 6: Extended analysis (optional, skip in quick mode)
if [ "$QUICK_MODE" = false ]; then
    echo "=========================================="
    echo "Step 6: Running Extended Analysis"
    echo "=========================================="
    echo ""

    python experiments/run_extended_analysis.py \
        --test-results "$OUTPUT_DIR/test_results/test_results.json" \
        --training-logs "$OUTPUT_DIR/agents" \
        --output-dir "$OUTPUT_DIR/extended_analysis" \
        2>&1 | tee "$OUTPUT_DIR/logs/extended.log"

    echo ""
    echo "Extended analysis complete."
    echo ""
fi

# Summary
echo "=========================================="
echo "Reproduction Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Trained agents:     $OUTPUT_DIR/agents/"
echo "  Test results:       $OUTPUT_DIR/test_results/test_results.json"
echo "  Publication HTML:   $OUTPUT_DIR/test_results/publication_analysis.html"
echo "  Figures:            $OUTPUT_DIR/figures/"
if [ "$QUICK_MODE" = false ]; then
    echo "  Extended analysis:  $OUTPUT_DIR/extended_analysis/"
fi
echo "  Logs:               $OUTPUT_DIR/logs/"
echo ""
echo "To view the publication analysis, open:"
echo "  $OUTPUT_DIR/test_results/publication_analysis.html"
echo ""

# Verify outputs
echo "Verifying outputs..."
MISSING=0

check_file() {
    if [ ! -f "$1" ]; then
        echo "  MISSING: $1"
        MISSING=$((MISSING + 1))
    else
        echo "  OK: $1"
    fi
}

check_file "$OUTPUT_DIR/test_results/test_results.json"
check_file "$OUTPUT_DIR/test_results/publication_analysis.html"
check_file "$OUTPUT_DIR/test_results/raw_data.csv"

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "All expected outputs present."
    exit 0
else
    echo ""
    echo "WARNING: $MISSING expected output(s) missing."
    exit 1
fi
