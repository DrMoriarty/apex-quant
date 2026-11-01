#!/usr/bin/env bash
#
# eval.sh — Run evaluation benchmarks on a GGUF model, output JSON
#
# Runs: perplexity, KL divergence, HellaSwag, Winogrande, MMLU, ARC-Challenge,
#       TruthfulQA, inference speed (pp512 + tg128)
#
# Usage:
#   ./eval.sh model.gguf                                    # all benchmarks, JSON to stdout
#   ./eval.sh model.gguf -o results.json                    # save to file
#   ./eval.sh model.gguf --kl-reference ref_logits.bin      # include KL divergence
#   ./eval.sh model.gguf --only ppl,hellaswag               # specific benchmarks
#   ./eval.sh model.gguf --hellaswag-tasks 1000             # custom task count
#
# Environment:
#   LLAMA_CPP_DIR     Path to llama.cpp build/bin directory
#   EVAL_DATA_DIR     Path to eval datasets (default: ~/.cache/apex-quant/eval-data)
#   NGL               GPU layers (default: 99)
#
set -euo pipefail

# --- Defaults ---
MODEL=""
OUTPUT=""
NGL="${NGL:-99}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${HOME}/.cache/apex-quant/eval-data}"
HELLASWAG_TASKS=400
WINOGRANDE_TASKS=400
KL_REFERENCE=""
ONLY=""
SKIP=""

# --- Find llama.cpp ---
find_bin() {
    local name="$1"
    local dirs=(
        "${LLAMA_CPP_DIR:-}"
        "./llama.cpp/build/bin"
        "$(dirname "$0")/../llama.cpp/build/bin"
    )
    for d in "${dirs[@]}"; do
        [ -n "$d" ] && [ -f "$d/$name" ] && echo "$d/$name" && return 0
    done
    command -v "$name" 2>/dev/null && return 0
    return 1
}

# --- Parse args ---
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output)          OUTPUT="$2"; shift 2 ;;
        --only)               ONLY="$2"; shift 2 ;;
        --skip)               SKIP="$2"; shift 2 ;;
        --hellaswag-tasks)    HELLASWAG_TASKS="$2"; shift 2 ;;
        --winogrande-tasks)   WINOGRANDE_TASKS="$2"; shift 2 ;;
        --kl-reference)       KL_REFERENCE="$2"; shift 2 ;;
        --eval-data)          EVAL_DATA_DIR="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,17p' "$0"
            exit 0
            ;;
        -*)                   echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [ -z "$MODEL" ]; then MODEL="$1"; else echo "Unexpected: $1" >&2; exit 1; fi
            shift ;;
    esac
done

[ -z "$MODEL" ] && { echo "Usage: $0 model.gguf [-o output.json]" >&2; exit 1; }
[ ! -f "$MODEL" ] && { echo "Error: $MODEL not found" >&2; exit 1; }

LLAMA_PPL=$(find_bin llama-perplexity) || { echo "Error: llama-perplexity not found" >&2; exit 1; }
LLAMA_BENCH=$(find_bin llama-bench) || { echo "Error: llama-bench not found" >&2; exit 1; }

should_run() {
    local bench="$1"
    if [ -n "$ONLY" ]; then
        echo ",$ONLY," | grep -q ",$bench," && return 0 || return 1
    fi
    if [ -n "$SKIP" ]; then
        echo ",$SKIP," | grep -q ",$bench," && return 1 || return 0
    fi
    return 0
}

# --- Download eval data if needed ---
download_data() {
    mkdir -p "$EVAL_DATA_DIR"
    local files=(
        "hellaswag_val_full.txt|https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt"
        "winogrande-debiased-eval.csv|https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/raw/main/winogrande-debiased-eval.csv"
        "mmlu-validation.bin|https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/mmlu-validation.bin"
        "arc-challenge-validation.bin|https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/arc-challenge-validation.bin"
        "truthful-qa-validation.bin|https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/truthful-qa-validation.bin"
    )
    for entry in "${files[@]}"; do
        local fname="${entry%%|*}"
        local url="${entry##*|}"
        if [ ! -f "$EVAL_DATA_DIR/$fname" ]; then
            echo "Downloading $fname..." >&2
            curl -sL -o "$EVAL_DATA_DIR/$fname" "$url"
        fi
    done
}

download_data

# --- Model info ---
MODEL_NAME=$(basename "$MODEL" .gguf)
SIZE_BYTES=$(stat --format=%s "$MODEL")
SIZE_GB=$(python3 -c "print(round($SIZE_BYTES / 1073741824, 1))")
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Evaluating: $MODEL_NAME ($SIZE_GB GB)" >&2

# --- Run benchmarks ---
declare -A RESULTS
RESULTS[model]="$MODEL_NAME"
RESULTS[size_gb]="$SIZE_GB"

# Perplexity
if should_run ppl; then
    echo "  [ppl] Wikitext-2 perplexity..." >&2
    WIKITEXT="${EVAL_DATA_DIR}/../wikitext-2-raw/wiki.test.raw"
    [ ! -f "$WIKITEXT" ] && WIKITEXT="${EVAL_DATA_DIR}/wiki.test.raw"
    if [ -f "$WIKITEXT" ]; then
        $LLAMA_PPL -m "$MODEL" -f "$WIKITEXT" -ngl $NGL > "$TMPDIR/ppl.log" 2>&1
        RESULTS[perplexity]=$(grep "Final estimate" "$TMPDIR/ppl.log" | grep -oP 'PPL = \K[0-9.]+' || echo "null")
        RESULTS[ppl_error]=$(grep "Final estimate" "$TMPDIR/ppl.log" | grep -oP '\+/- \K[0-9.]+' || echo "null")
        echo "        PPL = ${RESULTS[perplexity]}" >&2
    else
        echo "        Skipped (wikitext not found)" >&2
        RESULTS[perplexity]="null"
        RESULTS[ppl_error]="null"
    fi
fi

# KL Divergence
if should_run kl && [ -n "$KL_REFERENCE" ]; then
    echo "  [kl] KL divergence vs reference..." >&2
    WIKITEXT="${EVAL_DATA_DIR}/../wikitext-2-raw/wiki.test.raw"
    [ ! -f "$WIKITEXT" ] && WIKITEXT="${EVAL_DATA_DIR}/wiki.test.raw"
    if [ -f "$WIKITEXT" ] && [ -f "$KL_REFERENCE" ]; then
        $LLAMA_PPL -m "$MODEL" -f "$WIKITEXT" -ngl $NGL \
            --kl-divergence --kl-divergence-base "$KL_REFERENCE" \
            > "$TMPDIR/kl.log" 2>&1
        RESULTS[kl_mean]=$(grep -P '^Mean\s+KLD:' "$TMPDIR/kl.log" | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "null")
        RESULTS[kl_max]=$(grep -P '^Maximum KLD:' "$TMPDIR/kl.log" | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "null")
        RESULTS[kl_99_9]=$(grep -P '^99\.9%\s+KLD:' "$TMPDIR/kl.log" | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "null")
        RESULTS[kl_median]=$(grep -P '^Median\s+KLD:' "$TMPDIR/kl.log" | grep -oP ':\s+\K[0-9.]+' | head -1 || echo "null")
        echo "        KL mean=${RESULTS[kl_mean]}, max=${RESULTS[kl_max]}, 99.9%=${RESULTS[kl_99_9]}" >&2
    else
        echo "        Skipped (wikitext or reference not found)" >&2
    fi
elif should_run kl; then
    echo "  [kl] Skipped (use --kl-reference <logits.bin>)" >&2
fi

# HellaSwag
if should_run hellaswag; then
    echo "  [hellaswag] ${HELLASWAG_TASKS} tasks..." >&2
    $LLAMA_PPL -m "$MODEL" -f "$EVAL_DATA_DIR/hellaswag_val_full.txt" \
        --hellaswag --hellaswag-tasks "$HELLASWAG_TASKS" -ngl $NGL \
        > "$TMPDIR/hellaswag.log" 2>&1
    RESULTS[hellaswag]=$(grep "^${HELLASWAG_TASKS}" "$TMPDIR/hellaswag.log" | awk '{print $2}' | tr -d '%' || echo "null")
    echo "        HellaSwag = ${RESULTS[hellaswag]}%" >&2
fi

# Winogrande
if should_run winogrande; then
    echo "  [winogrande] ${WINOGRANDE_TASKS} tasks..." >&2
    $LLAMA_PPL -m "$MODEL" -f "$EVAL_DATA_DIR/winogrande-debiased-eval.csv" \
        --winogrande --winogrande-tasks "$WINOGRANDE_TASKS" -ngl $NGL \
        > "$TMPDIR/winogrande.log" 2>&1
    RESULTS[winogrande]=$(grep "Final Winogrande" "$TMPDIR/winogrande.log" | grep -oP ':\s+\K[0-9.]+' || echo "null")
    echo "        Winogrande = ${RESULTS[winogrande]}" >&2
fi

# MMLU
if should_run mmlu; then
    echo "  [mmlu] Multiple choice..." >&2
    $LLAMA_PPL -m "$MODEL" -bf "$EVAL_DATA_DIR/mmlu-validation.bin" \
        --multiple-choice -c 2048 -ngl $NGL \
        > "$TMPDIR/mmlu.log" 2>&1
    RESULTS[mmlu]=$(grep "Final result" "$TMPDIR/mmlu.log" | grep -oP ':\s+\K[0-9.]+' || echo "null")
    echo "        MMLU = ${RESULTS[mmlu]}" >&2
fi

# ARC-Challenge
if should_run arc; then
    echo "  [arc] ARC-Challenge..." >&2
    $LLAMA_PPL -m "$MODEL" -bf "$EVAL_DATA_DIR/arc-challenge-validation.bin" \
        --multiple-choice -np 8 -c 2048 -ngl $NGL \
        > "$TMPDIR/arc.log" 2>&1
    RESULTS[arc_challenge]=$(grep "Final result" "$TMPDIR/arc.log" | grep -oP ':\s+\K[0-9.]+' || echo "null")
    echo "        ARC = ${RESULTS[arc_challenge]}" >&2
fi

# TruthfulQA
if should_run truthfulqa; then
    echo "  [truthfulqa] TruthfulQA..." >&2
    $LLAMA_PPL -m "$MODEL" -bf "$EVAL_DATA_DIR/truthful-qa-validation.bin" \
        --multiple-choice -np 16 -c 2048 -ngl $NGL \
        > "$TMPDIR/truthfulqa.log" 2>&1
    RESULTS[truthfulqa]=$(grep "Final result" "$TMPDIR/truthfulqa.log" | grep -oP ':\s+\K[0-9.]+' || echo "null")
    echo "        TruthfulQA = ${RESULTS[truthfulqa]}" >&2
fi

# Speed
if should_run speed; then
    echo "  [speed] Inference benchmark..." >&2
    $LLAMA_BENCH -m "$MODEL" -p 512 -n 128 -ngl $NGL > "$TMPDIR/bench.log" 2>&1
    RESULTS[pp512_ts]=$(grep "pp512" "$TMPDIR/bench.log" | grep -oP '[\d.]+(?= \±)' | head -1 || echo "null")
    RESULTS[tg128_ts]=$(grep "tg128" "$TMPDIR/bench.log" | grep -oP '[\d.]+(?= \±)' | head -1 || echo "null")
    echo "        pp512=${RESULTS[pp512_ts]} t/s, tg128=${RESULTS[tg128_ts]} t/s" >&2
fi

# --- Output JSON ---
json_val() {
    local v="${RESULTS[$1]:-null}"
    if [ "$v" = "null" ] || [ -z "$v" ]; then
        echo "null"
    elif echo "$v" | grep -qP '^[0-9.]+$'; then
        echo "$v"
    else
        echo "\"$v\""
    fi
}

JSON=$(cat <<ENDJSON
{
  "model": "${RESULTS[model]:-}",
  "size_gb": $(json_val size_gb),
  "perplexity": $(json_val perplexity),
  "ppl_error": $(json_val ppl_error),
  "kl_mean": $(json_val kl_mean),
  "kl_max": $(json_val kl_max),
  "kl_99_9": $(json_val kl_99_9),
  "kl_median": $(json_val kl_median),
  "hellaswag": $(json_val hellaswag),
  "winogrande": $(json_val winogrande),
  "mmlu": $(json_val mmlu),
  "arc_challenge": $(json_val arc_challenge),
  "truthfulqa": $(json_val truthfulqa),
  "pp512_ts": $(json_val pp512_ts),
  "tg128_ts": $(json_val tg128_ts)
}
ENDJSON
)

if [ -n "$OUTPUT" ]; then
    echo "$JSON" > "$OUTPUT"
    echo "Results saved to: $OUTPUT" >&2
else
    echo "$JSON"
fi
