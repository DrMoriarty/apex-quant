#!/usr/bin/env bash
#
# push_to_hf.sh - Upload APEX quantized GGUF files to HuggingFace
#
# Part of the APEX (Adaptive Precision for EXpert Models) project.
# https://github.com/mudler/apex-quant
#
# Usage:
#   ./scripts/push_to_hf.sh [OPTIONS] <REPO_ID> <GGUF_FILE> [GGUF_FILE...]
#
# Options:
#   --readme FILE    Path to README.md to upload as model card
#   --private        Create private repository
#   --no-create      Do not create repo (fail if it doesn't exist)
#   -h, --help       Show this help message
#
# Examples:
#   # Upload APEX Quality quant with its model card
#   ./scripts/push_to_hf.sh \
#       --readme hf/README_3TIER_IQ.md \
#       myuser/Qwen3.5-35B-A3B-APEX-Quality \
#       Qwen3.5-35B-A3B-APEX-Quality.gguf
#

set -euo pipefail

# ---------- helpers ----------
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }
warn() { echo "WARNING: $*" >&2; }

# ---------- find huggingface CLI ----------
find_hf_cli() {
    if command -v huggingface-cli &>/dev/null; then
        echo "huggingface-cli"
    elif command -v hf &>/dev/null; then
        echo "hf"
    else
        die "Neither 'huggingface-cli' nor 'hf' found. Install with: pip install huggingface_hub"
    fi
}

# ---------- parse args ----------
README_FILE=""
PRIVATE=false
CREATE_REPO=true
POSITIONAL=()

show_help() {
    cat <<'HELPEOF'
APEX push_to_hf.sh - Upload GGUF files to HuggingFace Hub

Usage:
  ./scripts/push_to_hf.sh [OPTIONS] <REPO_ID> <GGUF_FILE> [GGUF_FILE...]

This script uploads APEX quantized GGUF model files to a HuggingFace
repository. It will create the repository if it does not already exist,
upload all specified GGUF files, and optionally upload a README.md model card.

Options:
  --readme FILE    Path to README.md to upload as the model card.
                   This file will be uploaded as README.md in the repo root.
  --private        Create the repository as private (default: public)
  --no-create      Do not create the repo; fail if it doesn't exist
  -h, --help       Show this help message

Arguments:
  REPO_ID          HuggingFace repository ID (e.g., myuser/my-model-gguf)
  GGUF_FILE        One or more GGUF files to upload

Examples:
  # Upload APEX Quality with its model card
  ./scripts/push_to_hf.sh \
      --readme hf/README_3TIER_IQ.md \
      myuser/Qwen3.5-35B-A3B-APEX-Quality \
      Qwen3.5-35B-A3B-APEX-Quality.gguf

  # Upload APEX Balanced with its model card
  ./scripts/push_to_hf.sh \
      --readme hf/README_LAYER.md \
      myuser/Qwen3.5-35B-A3B-APEX-Balanced \
      Qwen3.5-35B-A3B-APEX-Balanced.gguf

  # Upload multiple files to an existing repo
  ./scripts/push_to_hf.sh --no-create myuser/my-quants \
      model-quality.gguf model-balanced.gguf

  # Upload as private repo
  ./scripts/push_to_hf.sh --private myuser/my-private-quant model.gguf
HELPEOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --readme)
            [[ $# -ge 2 ]] || die "--readme requires a file path argument"
            README_FILE="$2"
            shift 2
            ;;
        --readme=*)
            README_FILE="${1#*=}"
            shift
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --no-create)
            CREATE_REPO=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        -*)
            die "Unknown option: $1 (use --help for usage)"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 2 ]] || die "Usage: $0 [OPTIONS] <REPO_ID> <GGUF_FILE> [GGUF_FILE...]"

REPO_ID="${POSITIONAL[0]}"
GGUF_FILES=("${POSITIONAL[@]:1}")

# Validate inputs
if [[ -n "$README_FILE" ]] && [[ ! -f "$README_FILE" ]]; then
    die "README file not found: $README_FILE"
fi

for f in "${GGUF_FILES[@]}"; do
    [[ -f "$f" ]] || die "GGUF file not found: $f"
done

HF_CLI="$(find_hf_cli)"
info "Using HuggingFace CLI: $HF_CLI"

# ---------- check authentication ----------
info "Checking HuggingFace authentication..."
if ! "$HF_CLI" whoami &>/dev/null; then
    die "Not logged in to HuggingFace. Run: huggingface-cli login"
fi

HF_USER="$("$HF_CLI" whoami 2>/dev/null | head -1 || echo "unknown")"
info "Authenticated as: $HF_USER"
echo ""

# ---------- create repo if needed ----------
if [[ "$CREATE_REPO" == true ]]; then
    info "Creating repository $REPO_ID (if not exists)..."

    create_args=("repo" "create" "${REPO_ID##*/}" "--type" "model")

    if [[ "$REPO_ID" == */* ]]; then
        org_name="${REPO_ID%%/*}"
        if [[ "$org_name" != "$HF_USER" ]]; then
            create_args+=("--organization" "$org_name")
        fi
    fi

    if [[ "$PRIVATE" == true ]]; then
        create_args+=("--private")
    fi

    "$HF_CLI" "${create_args[@]}" 2>/dev/null || true
    info "Repository ready: https://huggingface.co/$REPO_ID"
fi
echo ""

# ---------- upload README if provided ----------
if [[ -n "$README_FILE" ]]; then
    info "Uploading model card: $README_FILE -> README.md"
    "$HF_CLI" upload "$REPO_ID" "$README_FILE" README.md --repo-type model
    info "Model card uploaded."
    echo ""
fi

# ---------- upload GGUF files ----------
for gguf_file in "${GGUF_FILES[@]}"; do
    local_basename="$(basename "$gguf_file")"
    local_size="$(du -h "$gguf_file" | cut -f1)"
    info "Uploading $local_basename ($local_size)..."

    "$HF_CLI" upload "$REPO_ID" "$gguf_file" "$local_basename" --repo-type model

    info "  Uploaded: $local_basename"
done
echo ""

# ---------- summary ----------
info "=== Upload Summary ==="
info "Repository: https://huggingface.co/$REPO_ID"
info "Files uploaded:"
for gguf_file in "${GGUF_FILES[@]}"; do
    info "  - $(basename "$gguf_file")"
done
if [[ -n "$README_FILE" ]]; then
    info "  - README.md (from $README_FILE)"
fi
echo ""
info "Done."
