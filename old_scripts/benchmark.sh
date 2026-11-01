#!/usr/bin/env bash
#
# benchmark.sh — Full benchmark suite for APEX quantized models
#
# Benchmarks one or more GGUF models with:
#   1. Wikitext-2 perplexity
#   2. KL divergence vs F16 reference (requires --reference)
#   3. Inference speed (pp512 + tg128)
#
# Results are collected into a single TSV and plots are generated.
#
# Usage:
#   # Benchmark a single model
#   ./benchmark.sh model.gguf
#
#   # Benchmark multiple models with KL divergence
#   ./benchmark.sh --reference f16.gguf model1.gguf model2.gguf model3.gguf
#
#   # Run everything: all APEX quants + baselines
#   ./benchmark.sh --run-all --reference ~/.cache/autoresearch-quant/reference-f16.gguf
#
# Options:
#   --reference <f16.gguf>   F16 model for KL divergence (generates logits once, reuses)
#   --wikitext <path>        Path to wiki.test.raw
#   --output <dir>           Output directory (default: ./benchmark_results)
#   --run-all                Auto-discover APEX quants in CACHE_DIR and benchmark all
#   --skip-ppl               Skip perplexity measurement
#   --skip-bench             Skip inference speed measurement
#   --skip-kl                Skip KL divergence (even if --reference given)
#   --plots                  Generate comparison plots after benchmarking
#   --help                   Show this help
#
# Environment:
#   LLAMA_CPP_DIR            Path to llama.cpp build/bin directory
#   CACHE_DIR                Model cache (default: ~/.cache/autoresearch-quant)
#
set -euo pipefail

# --- Defaults ---
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/autoresearch-quant}"
WIKITEXT="${CACHE_DIR}/wikitext-2-raw/wiki.test.raw"
REFERENCE=""
OUTPUT_DIR="./benchmark_results"
MODELS=()
RUN_ALL=false
DO_PPL=true
DO_BENCH=true
DO_KL=true
DO_PLOTS=false

# --- Find llama.cpp ---
find_llama_cpp() {
    local dirs=(
        "${LLAMA_CPP_DIR:-}"
        "./llama.cpp/build/bin"
        "$(dirname "$0")/../llama.cpp/build/bin"
    )
    for d in "${dirs[@]}"; do
        [ -n "$d" ] && [ -f "$d/llama-perplexity" ] && echo "$d" && return 0
    done
    command -v llama-perplexity &>/dev/null && dirname "$(command -v llama-perplexity)" && return 0
    return 1
}

# --- Parse args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --reference)    REFERENCE="$2"; shift 2 ;;
        --wikitext)     WIKITEXT="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --run-all)      RUN_ALL=true; shift ;;
        --skip-ppl)     DO_PPL=false; shift ;;
        --skip-bench)   DO_BENCH=false; shift ;;
        --skip-kl)      DO_KL=false; shift ;;
        --plots)        DO_PLOTS=true; shift ;;
        --help|-h)
            sed -n '3,36p' "$0"
            exit 0
            ;;
        -*)             echo "Unknown option: $1" >&2; exit 1 ;;
        *)              MODELS+=("$1"); shift ;;
    esac
done

LLAMA_DIR=$(find_llama_cpp) || { echo "Error: llama.cpp not found. Set LLAMA_CPP_DIR." >&2; exit 1; }
echo "llama.cpp: $LLAMA_DIR"

# --- Auto-discover models in --run-all mode ---
if $RUN_ALL; then
    echo "Discovering GGUF models in $CACHE_DIR ..."
    while IFS= read -r f; do
        MODELS+=("$f")
    done < <(find "$CACHE_DIR" -maxdepth 1 -name "*.gguf" ! -name "reference-f16.gguf" -type f 2>/dev/null | sort)
    echo "Found ${#MODELS[@]} model(s)"
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models specified. Provide GGUF files or use --run-all." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RESULTS_TSV="$OUTPUT_DIR/benchmark_results.tsv"

# --- Write TSV header ---
echo -e "model\tsize_gb\tperplexity\tppl_error\tkl_mean\tkl_max\tpp512_ts\ttg128_ts" > "$RESULTS_TSV"

# --- Generate F16 reference logits (once) for KL divergence ---
KL_BASE_LOGITS=""
if $DO_KL && [ -n "$REFERENCE" ] && [ -f "$REFERENCE" ]; then
    KL_BASE_LOGITS="$OUTPUT_DIR/reference_logits.bin"
    if [ -f "$KL_BASE_LOGITS" ]; then
        echo "F16 reference logits already exist: $KL_BASE_LOGITS"
    else
        echo ""
        echo "=== Generating F16 reference logits (one-time) ==="
        echo "This runs perplexity on the F16 model and saves logits for KL comparison."
        echo "Model: $REFERENCE"
        "$LLAMA_DIR/llama-perplexity" \
            -m "$REFERENCE" \
            -f "$WIKITEXT" \
            --save-all-logits "$KL_BASE_LOGITS" \
            > "$OUTPUT_DIR/reference_logits.log" 2>&1
        echo "F16 logits saved to: $KL_BASE_LOGITS"

        # Also extract F16 perplexity
        F16_PPL=$(grep "Final estimate" "$OUTPUT_DIR/reference_logits.log" | grep -oP 'PPL = \K[0-9.]+' || echo "N/A")
        echo "F16 perplexity: $F16_PPL"
    fi
elif $DO_KL; then
    echo "Warning: KL divergence requested but no --reference provided. Skipping KL."
    DO_KL=false
fi

# --- Benchmark each model ---
for MODEL in "${MODELS[@]}"; do
    if [ ! -f "$MODEL" ]; then
        echo "Warning: $MODEL not found, skipping"
        continue
    fi

    MODEL_NAME=$(basename "$MODEL" .gguf)
    SIZE_BYTES=$(stat --format=%s "$MODEL")
    SIZE_GB=$(python3 -c "print(f'{$SIZE_BYTES / 1073741824:.1f}')")

    echo ""
    echo "========================================"
    echo "Benchmarking: $MODEL_NAME ($SIZE_GB GB)"
    echo "========================================"

    PPL_VAL="N/A"
    PPL_ERR="N/A"
    KL_MEAN="N/A"
    KL_MAX="N/A"
    PP_SPEED="N/A"
    TG_SPEED="N/A"

    # --- Perplexity ---
    if $DO_PPL; then
        echo "  [1/3] Wikitext-2 perplexity..."
        PPL_LOG="$OUTPUT_DIR/${MODEL_NAME}_ppl.log"
        if "$LLAMA_DIR/llama-perplexity" \
            -m "$MODEL" \
            -f "$WIKITEXT" \
            > "$PPL_LOG" 2>&1; then
            PPL_VAL=$(grep "Final estimate" "$PPL_LOG" | grep -oP 'PPL = \K[0-9.]+' || echo "N/A")
            PPL_ERR=$(grep "Final estimate" "$PPL_LOG" | grep -oP '\+/- \K[0-9.]+' || echo "N/A")
            echo "        PPL = $PPL_VAL +/- $PPL_ERR"
        else
            echo "        FAILED (see $PPL_LOG)"
        fi
    fi

    # --- KL Divergence ---
    if $DO_KL && [ -n "$KL_BASE_LOGITS" ]; then
        echo "  [2/3] KL divergence vs F16..."
        KL_LOG="$OUTPUT_DIR/${MODEL_NAME}_kl.log"
        if "$LLAMA_DIR/llama-perplexity" \
            -m "$MODEL" \
            -f "$WIKITEXT" \
            --kl-divergence \
            --kl-divergence-base "$KL_BASE_LOGITS" \
            > "$KL_LOG" 2>&1; then
            # Extract KL stats — llama-perplexity outputs "Mean    KLD:   0.011433 ± ..."
            KL_MEAN=$(grep -P '^Mean\s+KLD:' "$KL_LOG" 2>/dev/null | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "N/A")
            KL_MAX=$(grep -P '^Maximum KLD:' "$KL_LOG" 2>/dev/null | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "N/A")
            echo "        KL mean = $KL_MEAN, max = $KL_MAX"
        else
            echo "        FAILED (see $KL_LOG)"
        fi
    fi

    # --- Inference Speed ---
    if $DO_BENCH; then
        echo "  [3/3] Inference speed..."
        BENCH_LOG="$OUTPUT_DIR/${MODEL_NAME}_bench.log"
        if "$LLAMA_DIR/llama-bench" \
            -m "$MODEL" \
            -p 512 -n 128 \
            > "$BENCH_LOG" 2>&1; then
            PP_SPEED=$(grep "pp512" "$BENCH_LOG" | grep -oP '[\d.]+(?= \±)' | head -1 || echo "N/A")
            TG_SPEED=$(grep "tg128" "$BENCH_LOG" | grep -oP '[\d.]+(?= \±)' | head -1 || echo "N/A")
            echo "        pp512 = $PP_SPEED t/s, tg128 = $TG_SPEED t/s"
        else
            echo "        FAILED (see $BENCH_LOG)"
        fi
    fi

    # --- Append to TSV ---
    echo -e "${MODEL_NAME}\t${SIZE_GB}\t${PPL_VAL}\t${PPL_ERR}\t${KL_MEAN}\t${KL_MAX}\t${PP_SPEED}\t${TG_SPEED}" >> "$RESULTS_TSV"

    echo "  Done: $MODEL_NAME"
done

echo ""
echo "=== All Benchmarks Complete ==="
echo "Results: $RESULTS_TSV"
echo ""
column -t -s $'\t' "$RESULTS_TSV"

# --- Generate plots ---
if $DO_PLOTS; then
    PLOT_SCRIPT="$(dirname "$0")/plot_results.py"
    if [ -f "$PLOT_SCRIPT" ]; then
        echo ""
        echo "Generating plots..."
        python3 "$PLOT_SCRIPT" --tsv "$RESULTS_TSV" --output-dir "$OUTPUT_DIR/plots"
    else
        echo "Plot script not found at $PLOT_SCRIPT"
    fi
fi
