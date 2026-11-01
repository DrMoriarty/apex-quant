#!/usr/bin/env bash
#
# run_all_benchmarks.sh — Run eval.sh on all models, sequentially
#
# Generates F16 reference logits once, then calls eval.sh for each model.
# Results saved as JSON files in benchmark_results/final/.
#
# Usage:
#   ./scripts/run_all_benchmarks.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

EVAL_SCRIPT=./scripts/eval.sh
LLAMA_PPL=./llama.cpp/build/bin/llama-perplexity
CACHE=~/.cache/autoresearch-quant
WIKI=$CACHE/wikitext-2-raw/wiki.test.raw
REF_LOGITS=$CACHE/reference_logits.bin
RESULTS_DIR=./benchmark_results/final

export LLAMA_CPP_DIR=./llama.cpp/build/bin
export EVAL_DATA_DIR=$CACHE/eval-data
export NGL=99

mkdir -p "$RESULTS_DIR"

echo "=== APEX Full Benchmark Suite ==="
echo "Date: $(date -Iseconds)"
echo ""

# --- Step 1: Generate F16 reference logits (for KL divergence) ---
if [ -f "$REF_LOGITS" ]; then
    echo "F16 reference logits exist: $REF_LOGITS ($(du -h "$REF_LOGITS" | cut -f1))"
else
    echo "Generating F16 reference logits (one-time, ~15 min)..."
    $LLAMA_PPL -m "$CACHE/reference-f16.gguf" -f "$WIKI" -ngl $NGL \
        --save-all-logits "$REF_LOGITS" > "$RESULTS_DIR/f16_logits.log" 2>&1
    echo "Done: $REF_LOGITS"
fi
echo ""

# --- Step 2: Eval each model sequentially using eval.sh ---

# F16 (no KL — it IS the reference)
echo "================================================================"
$EVAL_SCRIPT "$CACHE/reference-f16.gguf" \
    --skip kl \
    -o "$RESULTS_DIR/f16.json"
echo ""

# Q8_0
echo "================================================================"
$EVAL_SCRIPT "$CACHE/Qwen3.5-35B-A3B-Q8_0.gguf" \
    --kl-reference "$REF_LOGITS" \
    -o "$RESULTS_DIR/q8_0.json"
echo ""

# APEX Quality
echo "================================================================"
$EVAL_SCRIPT "$CACHE/Qwen3.5-35B-A3B-APEX-Quality.gguf" \
    --kl-reference "$REF_LOGITS" \
    -o "$RESULTS_DIR/apex_quality.json"
echo ""

# APEX Balanced
echo "================================================================"
$EVAL_SCRIPT "$CACHE/Qwen3.5-35B-A3B-APEX-Balanced.gguf" \
    --kl-reference "$REF_LOGITS" \
    -o "$RESULTS_DIR/apex_balanced.json"
echo ""

# APEX Compact
echo "================================================================"
$EVAL_SCRIPT "$CACHE/Qwen3.5-35B-A3B-APEX-Compact.gguf" \
    --kl-reference "$REF_LOGITS" \
    -o "$RESULTS_DIR/apex_compact.json"
echo ""

# Unsloth Q8_K_XL (download if needed)
UNSLOTH="$CACHE/unsloth/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
if [ ! -f "$UNSLOTH" ]; then
    echo "Downloading Unsloth Q8_K_XL..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
            Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf --local-dir "$CACHE/unsloth"
    elif [ -f ".venv/bin/huggingface-cli" ]; then
        .venv/bin/huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
            Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf --local-dir "$CACHE/unsloth"
    else
        echo "ERROR: huggingface-cli not found. Skipping Unsloth."
    fi
fi

if [ -f "$UNSLOTH" ]; then
    echo "================================================================"
    $EVAL_SCRIPT "$UNSLOTH" \
        --kl-reference "$REF_LOGITS" \
        -o "$RESULTS_DIR/unsloth_q8_k_xl.json"
    echo ""
fi

# --- Step 3: Summary ---
echo ""
echo "================================================================"
echo "=== ALL BENCHMARKS COMPLETE ==="
echo "================================================================"
echo ""
echo "Results:"
for f in "$RESULTS_DIR"/*.json; do
    name=$(python3 -c "import json; print(json.load(open('$f'))['model'])")
    size=$(python3 -c "import json; print(json.load(open('$f'))['size_gb'])")
    ppl=$(python3 -c "import json; print(json.load(open('$f'))['perplexity'])")
    printf "  %-25s %5s GB  PPL %s\n" "$name" "$size" "$ppl"
done
