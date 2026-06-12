#!/bin/bash
#
# Optimization Geometry Experiments
#
# This script runs training with different optimizers (SGD, Adam, AdamW)
# and collects comprehensive geometry metrics for comparison.
#
# Usage: ./run_geometry_experiments.sh [OPTIONS]
#
# Options:
#   --device DEVICE      Device to use (auto, cuda, mps, cpu). Default: auto
#   --time-budget SECS   Training time budget in seconds. Default: 120
#   --optimizers "a b c" Space-separated list of optimizers. Default: "sgd adam adamw"
#   --skip-training      Skip training, only generate plots from existing data
#   --quick              Quick run with minimal settings for testing
#
# Examples:
#   ./run_geometry_experiments.sh                           # Full run with defaults
#   ./run_geometry_experiments.sh --quick                   # Quick test run
#   ./run_geometry_experiments.sh --device cuda --time-budget 300
#   ./run_geometry_experiments.sh --skip-training           # Only generate plots

set -e

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEVICE="auto"
TIME_BUDGET=120
OPTIMIZERS="sgd adam adamw"
SKIP_TRAINING=true
QUICK_MODE=false

OUTPUT_DIR="geometry_results"
REPORT_DIR="reports/geometry"

# Geometry tracking intervals
GEOMETRY_INTERVAL=10
HESSIAN_INTERVAL=50
EVAL_INTERVAL=20

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --time-budget)
            TIME_BUDGET="$2"
            shift 2
            ;;
        --optimizers)
            OPTIMIZERS="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            TIME_BUDGET=60
            GEOMETRY_INTERVAL=5
            HESSIAN_INTERVAL=20
            EVAL_INTERVAL=10
            shift
            ;;
        --help)
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

# =============================================================================
# MAIN EXECUTION
# =============================================================================
print_separator "OPTIMIZATION GEOMETRY EXPERIMENTS"

echo "Configuration:"
echo "  Device:         $DEVICE"
echo "  Time Budget:    ${TIME_BUDGET}s"
echo "  Optimizers:     $OPTIMIZERS"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Report Dir:     $REPORT_DIR"
echo "  Geometry Log:   Every $GEOMETRY_INTERVAL steps"
echo "  Hessian Log:    Every $HESSIAN_INTERVAL steps"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORT_DIR"

# =============================================================================
# TRAINING PHASE
# =============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    for OPT in $OPTIMIZERS; do
        print_separator "Training with $OPT optimizer"

        log "Starting $OPT training run..."

        # Run training with geometry tracking
        python train_geometry.py \
            optimizer.type="$OPT" \
            training.device="$DEVICE" \
            training.time_budget="$TIME_BUDGET" \
            training.eval_interval="$EVAL_INTERVAL" \
            geometry.log_interval="$GEOMETRY_INTERVAL" \
            geometry.hessian_interval="$HESSIAN_INTERVAL" \
            logging.output_dir="$OUTPUT_DIR" \
            logging.use_wandb=false

        log "Completed $OPT training run"

        # Small pause between runs to let GPU cool down
        if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "auto" ]; then
            log "Pausing 5s before next run..."
            sleep 5
        fi
    done

    print_separator "TRAINING COMPLETE"
fi

# =============================================================================
# VISUALIZATION PHASE
# =============================================================================
print_separator "Generating Visualizations"

log "Running visualization pipeline..."

python reports/visualize_geometry.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$REPORT_DIR" \
    --optimizers $OPTIMIZERS

log "Visualization complete"

# =============================================================================
# SUMMARY
# =============================================================================
print_separator "EXPERIMENT SUMMARY"

echo "Generated files:"
echo ""

echo "Metrics (per optimizer):"
for OPT in $OPTIMIZERS; do
    if [ -f "$OUTPUT_DIR/metrics_${OPT}.csv" ]; then
        LINES=$(wc -l < "$OUTPUT_DIR/metrics_${OPT}.csv")
        echo "  - $OUTPUT_DIR/metrics_${OPT}.csv ($LINES rows)"
    fi
done

echo ""
echo "Per-layer data:"
for OPT in $OPTIMIZERS; do
    if [ -f "$OUTPUT_DIR/layer_metrics_${OPT}.json" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/layer_metrics_${OPT}.json" | cut -f1)
        echo "  - $OUTPUT_DIR/layer_metrics_${OPT}.json ($SIZE)"
    fi
done

echo ""
echo "Visualizations:"
for f in "$REPORT_DIR"/*.png; do
    if [ -f "$f" ]; then
        echo "  - $f"
    fi
done

echo ""
log "All experiments completed successfully!"
echo ""
echo "To view results:"
echo "  - Open $REPORT_DIR/comprehensive_dashboard.png for overview"
echo "  - Open $REPORT_DIR/training_dynamics.png for detailed metrics"
echo "  - See $REPORT_DIR/sharpness_generalization.png for hypothesis testing"
echo ""
