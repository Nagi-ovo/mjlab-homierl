#!/bin/bash
# Nightly training script for MJLab benchmarks
#
# This script runs the tracking benchmark and generates a report.
# It is designed to be called by a systemd timer or cron job.
#
# Usage:
#   ./scripts/benchmarks/nightly_train.sh
#
# Environment variables:
#   MJLAB_DIR: Path to mjlab repository (default: script directory)
#   CUDA_DEVICE: GPU device to use (default: 0)
#   WANDB_TAGS: Comma-separated tags for the run (default: nightly)
#   SKIP_TRAINING: Set to "1" to skip training and only generate report

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJLAB_DIR="${MJLAB_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_TAGS="${WANDB_TAGS:-nightly}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"

# Training configuration
TASK="Mjlab-Tracking-Flat-Unitree-G1"
NUM_ENVS=4096
MAX_ITERATIONS=6000
REGISTRY_NAME="rll_humanoid/wandb-registry-Motions/side_kick_test"

# Report output
REPORT_DIR="${MJLAB_DIR}/benchmark_results"
GH_PAGES_BRANCH="gh-pages"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

cd "$MJLAB_DIR" || error "Failed to cd to $MJLAB_DIR"

log "Starting nightly benchmark run"
log "Task: $TASK"
log "GPU: $CUDA_DEVICE"

# Run training
if [[ "$SKIP_TRAINING" != "1" ]]; then
    log "Starting training..."

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run train "$TASK" \
        --env.scene.num-envs "$NUM_ENVS" \
        --agent.max-iterations "$MAX_ITERATIONS" \
        --registry-name "$REGISTRY_NAME" \
        --agent.wandb-tags "[$WANDB_TAGS]"

    log "Training completed"
else
    log "Skipping training (SKIP_TRAINING=1)"
fi

# Generate report
log "Generating benchmark report..."
uv run python scripts/benchmarks/generate_report.py \
    --tag nightly \
    --output-dir "$REPORT_DIR"

log "Report generated at $REPORT_DIR"

# Deploy to GitHub Pages (if on a machine with git configured)
if git rev-parse --git-dir > /dev/null 2>&1; then
    log "Deploying to GitHub Pages..."

    # Save current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # Stash any uncommitted changes
    git stash push -m "nightly-benchmark-stash" --quiet 2>/dev/null || true

    # Create or switch to gh-pages branch
    if git show-ref --verify --quiet "refs/heads/$GH_PAGES_BRANCH"; then
        git checkout "$GH_PAGES_BRANCH"
        git pull origin "$GH_PAGES_BRANCH" --rebase 2>/dev/null || true
    else
        git checkout --orphan "$GH_PAGES_BRANCH"
        git rm -rf . 2>/dev/null || true
    fi

    # Copy report files
    cp -r "$REPORT_DIR"/* .

    # Commit and push
    git add -A
    git commit -m "Update nightly benchmarks $(date '+%Y-%m-%d')" || log "No changes to commit"
    git push origin "$GH_PAGES_BRANCH" || log "Failed to push (may need manual push)"

    # Return to original branch
    git checkout "$CURRENT_BRANCH"
    git stash pop --quiet 2>/dev/null || true

    log "Deployed to GitHub Pages"
else
    log "Not in a git repository, skipping GitHub Pages deployment"
fi

log "Nightly benchmark complete"
