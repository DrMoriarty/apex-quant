#!/usr/bin/env bash
#
# quantize.sh — APEX quantization for llama.cpp
#
# Usage:
#   # Using a built-in profile
#   ./scripts/quantize.sh --profile balanced input.gguf output.gguf
#
#   # Using a custom tensor-type file
#   ./scripts/quantize.sh --config configs/my_config.txt input.gguf output.gguf
#
#   # With imatrix (for I-variants and Mini)
#   ./scripts/quantize.sh --profile mini --imatrix imatrix.dat input.gguf output.gguf
#
#   # Generate config only (no quantization)
#   ./scripts/quantize.sh --profile quality --generate-config -o config.txt
#
# Profiles: quality, i-quality, balanced, i-balanced, compact, i-compact, mini, custom
#
# Environment:
#   LLAMA_QUANTIZE    Path to llama-quantize binary (auto-detected)
#   LLAMA_CPP_DIR     Path to llama.cpp build/bin directory
#   NUM_LAYERS        Number of transformer layers (default: 40)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

# --- Defaults ---
PROFILE="balanced"
CONFIG_FILE=""
IMATRIX=""
BASE_TYPE="Q6_K"
NUM_LAYERS="${NUM_LAYERS:-40}"
GENERATE_ONLY=false
CONFIG_OUTPUT=""
POSITIONAL=()

# --- Find llama-quantize ---
find_quantize() {
    if [ -n "${LLAMA_QUANTIZE:-}" ] && [ -f "$LLAMA_QUANTIZE" ]; then
        echo "$LLAMA_QUANTIZE"; return 0
    fi
    local dirs=(
        "${LLAMA_CPP_DIR:-}"
        "./llama.cpp/build/bin"
        "$SCRIPT_DIR/../llama.cpp/build/bin"
    )
    for d in "${dirs[@]}"; do
        [ -n "$d" ] && [ -f "$d/llama-quantize" ] && echo "$d/llama-quantize" && return 0
    done
    command -v llama-quantize 2>/dev/null && return 0
    return 1
}

# --- Parse args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --profile|-p)          PROFILE="$2"; shift 2 ;;
        --config|-c)           CONFIG_FILE="$2"; shift 2 ;;
        --imatrix|-i)          IMATRIX="$2"; shift 2 ;;
        --base-type|-b)        BASE_TYPE="$2"; shift 2 ;;
        --layers|-l)           NUM_LAYERS="$2"; shift 2 ;;
        --generate-config)     GENERATE_ONLY=true; shift ;;
        -o)                    CONFIG_OUTPUT="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,22p' "$0"
            echo ""
            echo "Profiles:"
            echo "  quality      Q6_K/Q5_K/IQ4_XS experts, Q8_0 shared, Q6_K attn (21.3 GB)"
            echo "  i-quality    Same + imatrix. Best accuracy (83.5% HellaSwag)"
            echo "  balanced     Q6_K/Q5_K experts, Q8_0 shared, Q6_K attn (23.6 GB)"
            echo "  i-balanced   Same + imatrix. Lowest KL divergence"
            echo "  compact      Q4_K/Q3_K experts, Q6_K shared, Q4_K attn (16.1 GB)"
            echo "  i-compact    Same + imatrix. Best quality at 16 GB"
            echo "  mini         Q3_K/IQ2_S experts + imatrix. Consumer 16GB VRAM (12.2 GB)"
            exit 0
            ;;
        -*)                    die "Unknown option: $1" ;;
        *)                     POSITIONAL+=("$1"); shift ;;
    esac
done

# --- Determine base quant type for each profile ---
case "$PROFILE" in
    quality|i-quality)   BASE_TYPE="Q6_K" ;;
    balanced|i-balanced) BASE_TYPE="Q6_K" ;;
    compact|i-compact)   BASE_TYPE="Q4_K_M" ;;
    mini)                BASE_TYPE="Q3_K_M" ;;
esac

# --- I-profiles require imatrix ---
case "$PROFILE" in
    i-quality|i-balanced|i-compact|mini)
        if [ -z "$IMATRIX" ]; then
            warn() { echo "WARNING: $*" >&2; }
            warn "Profile '$PROFILE' benefits from --imatrix. Continuing without it."
        fi
        ;;
esac

# --- Generate or use config ---
if [ -n "$CONFIG_FILE" ]; then
    # User provided a config file
    [ -f "$CONFIG_FILE" ] || die "Config file not found: $CONFIG_FILE"
    TTFILE="$CONFIG_FILE"
    info "Using config: $CONFIG_FILE"
elif $GENERATE_ONLY; then
    # Just generate the config and exit
    if [ -n "$CONFIG_OUTPUT" ]; then
        "$SCRIPT_DIR/generate_config.sh" --profile "$PROFILE" --layers "$NUM_LAYERS" -o "$CONFIG_OUTPUT"
    else
        "$SCRIPT_DIR/generate_config.sh" --profile "$PROFILE" --layers "$NUM_LAYERS"
    fi
    exit 0
else
    # Generate a temp config
    TTFILE="$(mktemp)"
    trap 'rm -f "$TTFILE"' EXIT
    "$SCRIPT_DIR/generate_config.sh" --profile "$PROFILE" --layers "$NUM_LAYERS" -o "$TTFILE"
    info "Generated config for profile '$PROFILE' ($NUM_LAYERS layers)"
fi

# --- If generate-only, we're done ---
$GENERATE_ONLY && exit 0

# --- Need input and output ---
[ ${#POSITIONAL[@]} -ge 2 ] || die "Usage: $0 --profile <profile> <input.gguf> <output.gguf>"
INPUT="${POSITIONAL[0]}"
OUTPUT="${POSITIONAL[1]}"
[ -f "$INPUT" ] || die "Input file not found: $INPUT"

# --- Find llama-quantize ---
QUANTIZE=$(find_quantize) || die "llama-quantize not found. Set LLAMA_QUANTIZE or LLAMA_CPP_DIR."

# --- Build quantize command ---
QUANT_ARGS=("--tensor-type-file" "$TTFILE")
[ -n "$IMATRIX" ] && QUANT_ARGS+=("--imatrix" "$IMATRIX")

info "=== APEX Quantize ==="
info "Profile:    $PROFILE"
info "Base type:  $BASE_TYPE"
info "Input:      $INPUT"
info "Output:     $OUTPUT"
[ -n "$IMATRIX" ] && info "Imatrix:    $IMATRIX"
info "Config:     $TTFILE ($(wc -l < "$TTFILE") lines)"
info ""

"$QUANTIZE" "${QUANT_ARGS[@]}" "$INPUT" "$OUTPUT" "$BASE_TYPE"

info ""
info "Done: $(ls -lh "$OUTPUT" | awk '{print $5}') -> $OUTPUT"
