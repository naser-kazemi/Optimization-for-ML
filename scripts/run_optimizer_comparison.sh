#!/bin/bash
# =============================================================================
# Optimizer Comparison Experiment
# =============================================================================
# Runs training with multiple optimizers and generates comparison plots.
#
# Optimizers tested:
#   - Adam (baseline)
#   - AdamW
#   - Muon
#   - AdamNS (momentum mode - recommended)
#   - AdamNS (grad mode)
#   - AdamNS (update mode)
#
# Usage:
#   ./run_optimizer_comparison.sh [--quick] [--optimizers "adam,muon"]
#
# Options:
#   --quick         Run with fewer steps (500 instead of 2000)
#   --optimizers    Comma-separated list of optimizers to run
#   --skip-train    Skip training, only generate plots from existing results
#   --output-dir    Output directory (default: optimizer_comparison)
# =============================================================================

set -e

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Default configuration
MAX_STEPS=3000
EVAL_INTERVAL=50
OUTPUT_DIR="optimizer_comparison_long"
SKIP_TRAIN=true

# All available optimizers
ALL_OPTIMIZERS="adam,adamw,muon,adam_ns,adam_ns_grad,adam_ns_update"
OPTIMIZERS="$ALL_OPTIMIZERS"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_STEPS=500
            EVAL_INTERVAL=50
            shift
            ;;
        --optimizers)
            OPTIMIZERS="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Optimizer Comparison Experiment"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Optimizers: $OPTIMIZERS"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert comma-separated optimizers to array
IFS=',' read -ra OPT_ARRAY <<< "$OPTIMIZERS"

# Training phase
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "=== TRAINING PHASE ==="
    echo ""

    for opt in "${OPT_ARRAY[@]}"; do
        opt=$(echo "$opt" | xargs)  # Trim whitespace
        echo "----------------------------------------------"
        echo "Training with optimizer: $opt"
        echo "----------------------------------------------"

        # Run training (unbuffered output for real-time logging)
        python -u train_spectral.py \
            optimizer.type="$opt" \
            training.max_steps=$MAX_STEPS \
            training.eval_interval=$EVAL_INTERVAL \
            logging.output_dir="$OUTPUT_DIR" \
            2>&1 | stdbuf -oL tee "$OUTPUT_DIR/train_${opt}.log"

        echo ""
        echo "Completed: $opt"
        echo ""
    done
fi

# Visualization phase
echo ""
echo "=== VISUALIZATION PHASE ==="
echo ""

# Generate comparison plots
python -u reports/visualize_optimizer_comparison.py \
    --results-dir "$OUTPUT_DIR" \
    --save-dir "$OUTPUT_DIR/plots" \
    --optimizers "$OPTIMIZERS"

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || true
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || true
echo ""
echo "Generated plots:"
ls -la "$OUTPUT_DIR/plots/"*.png 2>/dev/null || true
echo ""
