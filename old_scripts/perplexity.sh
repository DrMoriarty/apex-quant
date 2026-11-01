#!/usr/bin/env bash
#
# perplexity.sh - Measure perplexity of a GGUF model on wikitext
#
# Part of the APEX (Adaptive Precision for EXpert Models) project.
# https://github.com/mudler/apex-quant
#
# Usage:
#   ./scripts/perplexity.sh <model.gguf> <wikitext-2-raw/wiki.test.raw>
#
# Environment variables:
#   LLAMA_PERPLEXITY  Path to llama-perplexity binary (auto-detected if unset)
#   LLAMA_CPP_DIR     Path to llama.cpp root (auto-detected)
#   PPL_CONTEXT       Context size (default: 2048)
#   PPL_THREADS       Number of threads (default: auto)
#   PPL_GPU_LAYERS    GPU layers to offload (default: 99)
#

set -euo pipefail

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

# ---------- locate llama-perplexity ----------
find_perplexity() {
    if [[ -n "${LLAMA_PERPLEXITY:-}" ]]; then
        echo "$LLAMA_PERPLEXITY"
        return
    fi

    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"

    local candidates=(
        "llama-perplexity"
        "./llama-perplexity"
        "./build/bin/llama-perplexity"
    )

    # Add llama.cpp-relative paths
    if [[ -n "${LLAMA_CPP_DIR:-}" ]]; then
        candidates+=("$LLAMA_CPP_DIR/build/bin/llama-perplexity")
    fi
    candidates+=(
        "$script_dir/../llama.cpp/build/bin/llama-perplexity"
        "../llama.cpp/build/bin/llama-perplexity"
    )

    for c in "${candidates[@]}"; do
        if command -v "$c" &>/dev/null || [[ -x "$c" ]]; then
            echo "$c"
            return
        fi
    done

    die "Could not find llama-perplexity. Set LLAMA_PERPLEXITY or LLAMA_CPP_DIR env var, or add llama-perplexity to PATH."
}

# ---------- parse args ----------
if [[ $# -lt 2 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat <<'HELPEOF'
APEX perplexity.sh - Measure perplexity of a GGUF model

Usage:
  ./scripts/perplexity.sh <model.gguf> <wikitext-2-raw/wiki.test.raw>

Runs llama-perplexity on the given model and wikitext data file, then
extracts and displays the final perplexity value.

Environment variables:
  LLAMA_PERPLEXITY  Path to llama-perplexity binary (auto-detected)
  LLAMA_CPP_DIR     Path to llama.cpp root (auto-detected)
  PPL_CONTEXT       Context size (default: 2048)
  PPL_GPU_LAYERS    GPU layers to offload (default: 99)
  PPL_THREADS       Number of threads (default: auto)

Examples:
  ./scripts/perplexity.sh model-apex.gguf wikitext-2-raw/wiki.test.raw
  LLAMA_CPP_DIR=~/llama.cpp ./scripts/perplexity.sh model.gguf wiki.test.raw
HELPEOF
    exit 0
fi

MODEL="$1"
WIKITEXT="$2"

[[ -f "$MODEL" ]]   || die "Model file not found: $MODEL"
[[ -f "$WIKITEXT" ]] || die "Wikitext file not found: $WIKITEXT"

PERPLEXITY_BIN="$(find_perplexity)"
CTX="${PPL_CONTEXT:-2048}"
NGL="${PPL_GPU_LAYERS:-99}"

info "=== APEX Perplexity Benchmark ==="
info "Binary:     $PERPLEXITY_BIN"
info "Model:      $MODEL"
info "Wikitext:   $WIKITEXT"
info "Context:    $CTX"
info "GPU layers: $NGL"
echo ""

# ---------- build command ----------
CMD=("$PERPLEXITY_BIN"
    --model "$MODEL"
    --file "$WIKITEXT"
    --ctx-size "$CTX"
    --n-gpu-layers "$NGL"
)

if [[ -n "${PPL_THREADS:-}" ]]; then
    CMD+=(--threads "$PPL_THREADS")
fi

# ---------- run and capture output ----------
TMPLOG="$(mktemp /tmp/apex_ppl_log.XXXXXX)"
trap "rm -f '$TMPLOG'" EXIT

info "Running perplexity measurement..."
"${CMD[@]}" 2>&1 | tee "$TMPLOG"

# ---------- extract final PPL ----------
FINAL_PPL="$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$TMPLOG" || true)"

if [[ -z "$FINAL_PPL" ]]; then
    # Try alternative output format
    FINAL_PPL="$(grep -oP 'perplexity = \K[0-9.]+' "$TMPLOG" | tail -1 || true)"
fi

if [[ -n "$FINAL_PPL" ]]; then
    echo ""
    info "==========================="
    info "Final Perplexity: $FINAL_PPL"
    info "==========================="
else
    echo ""
    info "WARNING: Could not extract perplexity value from output."
    info "Check the log above for results."
fi
