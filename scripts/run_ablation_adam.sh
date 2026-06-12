#!/bin/bash
#
# Adam Ablation Experiments
#
# Runs ablation experiments for Adam optimizer with:
# - Different learning rates
# - Different model depths
# - With and without spectral regularization
#
# Usage: ./run_ablation_adam.sh [OPTIONS]
#
# Options:
#   --device DEVICE       Device to use (auto, cuda, cuda:0, cuda:1, etc.). Default: auto
#   --max-steps N         Maximum training steps. Default: 5000
#   --output-dir DIR      Base output directory. Default: ablation_results/adam
#   --quick               Quick run with minimal settings for testing
#   --skip-training       Skip training, only generate visualizations
#   --lr-only             Only run learning rate ablations
#   --depth-only          Only run depth ablations
#   --reg-only            Only run regularization comparison
#
# Examples:
#   ./run_ablation_adam.sh                           # Full ablation suite
#   ./run_ablation_adam.sh --device cuda:0           # Run on specific GPU
#   ./run_ablation_adam.sh --quick                   # Quick test run
#   ./run_ablation_adam.sh --lr-only                 # Only LR ablations

set -e

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEVICE="auto"
MAX_STEPS=3000
OUTPUT_BASE="ablation_results/adam"
QUICK_MODE=false
SKIP_TRAINING=false
LR_ONLY=false
DEPTH_ONLY=false
REG_ONLY=false

# Ablation parameters
LEARNING_RATES="1e-4 5e-4 1e-3 5e-3"
DEPTHS="2 4 8 16"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            MAX_STEPS=500
            LEARNING_RATES="1e-3 5e-3"
            DEPTHS="2 4"
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --lr-only)
            LR_ONLY=true
            shift
            ;;
        --depth-only)
            DEPTH_ONLY=true
            shift
            ;;
        --reg-only)
            REG_ONLY=true
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

print_separator() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

print_banner() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                     ADAM ABLATION EXPERIMENTS                             ║"
    echo "║                                                                           ║"
    echo "║  Ablations:                                                               ║"
    echo "║    • Learning Rate: $LEARNING_RATES"
    echo "║    • Model Depth: $DEPTHS"
    echo "║                                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

run_experiment() {
    local name="$1"
    local output_dir="$2"
    shift 2
    local extra_args="$@"

    log "Running: $name"
    log "Output: $output_dir"

    mkdir -p "$output_dir"

    python train_ablation.py \
        optimizer.type=adam \
        training.max_steps="$MAX_STEPS" \
        training.device="$DEVICE" \
        logging.output_dir="$output_dir" \
        $extra_args

    log "Completed: $name"
    echo ""
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
print_banner

echo "Configuration:"
echo "  Device:         $DEVICE"
echo "  Max Steps:      $MAX_STEPS"
echo "  Output Base:    $OUTPUT_BASE"
echo "  Quick Mode:     $QUICK_MODE"
echo ""

if [ "$SKIP_TRAINING" = true ]; then
    log "Skipping training, only generating visualizations..."
else
    # Create base output directory
    mkdir -p "$OUTPUT_BASE"

    # =========================================================================
    # LEARNING RATE ABLATIONS
    # =========================================================================
    if [ "$LR_ONLY" = true ] || ([ "$DEPTH_ONLY" = false ] && [ "$REG_ONLY" = false ]); then
        print_separator "Learning Rate Ablations"

        for LR in $LEARNING_RATES; do
            lr_dir="$OUTPUT_BASE/lr_$LR"
            run_experiment "Adam LR=$LR" "$lr_dir" \
                optimizer.lr="$LR" \
                model.depth=4 \
                spectral_reg.enabled=false
        done
    fi

    # =========================================================================
    # MODEL DEPTH ABLATIONS
    # =========================================================================
    if [ "$DEPTH_ONLY" = true ] || ([ "$LR_ONLY" = false ] && [ "$REG_ONLY" = false ]); then
        print_separator "Model Depth Ablations"

        for DEPTH in $DEPTHS; do
            depth_dir="$OUTPUT_BASE/depth_$DEPTH"
            run_experiment "Adam Depth=$DEPTH" "$depth_dir" \
                optimizer.lr=1e-3 \
                model.depth="$DEPTH" \
                spectral_reg.enabled=false
        done
    fi

fi

# =============================================================================
# VISUALIZATION
# =============================================================================
print_separator "Generating Visualizations"

log "Running ablation visualization..."
python reports/visualize_ablation.py \
    --results-dir "$OUTPUT_BASE" \
    --optimizer adam \
    --save-dir "$OUTPUT_BASE/plots"

log "Visualization complete!"

# =============================================================================
# SUMMARY
# =============================================================================
print_separator "EXPERIMENT SUMMARY"

echo "Results saved to: $OUTPUT_BASE"
echo ""

echo "Experiment directories:"
for dir in "$OUTPUT_BASE"/*/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        n_files=$(ls -1 "$dir"/*.jsonl 2>/dev/null | wc -l || echo "0")
        echo "  ✓ $name ($n_files JSONL files)"
    fi
done

echo ""
echo "Visualizations:"
for f in "$OUTPUT_BASE/plots"/*.png; do
    if [ -f "$f" ]; then
        echo "  ✓ $(basename "$f")"
    fi
done

echo ""
log "Adam ablation experiments completed!"
echo ""
