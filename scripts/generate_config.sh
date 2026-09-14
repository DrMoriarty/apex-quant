#!/usr/bin/env bash
#
# generate_config.sh — Generate APEX tensor-type configuration files
#
# Creates a tensor-type file for llama-quantize's --tensor-type-file flag.
# Supports any number of layers and all APEX profiles.
#
# Usage:
#   ./scripts/generate_config.sh --profile balanced --layers 40 > config.txt
#   ./scripts/generate_config.sh --profile mini --layers 40 -o configs/my_config.txt
#   ./scripts/generate_config.sh --custom --edge-exp Q6_K --mid-exp Q4_K \
#       --shared Q8_0 --attn Q6_K --layers 40 > config.txt
#
# Profiles:
#   quality     Q6_K/Q5_K/IQ4_XS experts, Q8_0 shared, Q6_K attn
#   i-quality   Same as quality (use with --imatrix at quantize time)
#   balanced    Q6_K/Q5_K experts, Q8_0 shared, Q6_K attn
#   i-balanced  Same as balanced (use with --imatrix at quantize time)
#   compact     Q4_K/Q3_K experts, Q6_K shared, Q4_K attn
#   i-compact   Same as compact (use with --imatrix at quantize time)
#   mini        Q3_K edge / IQ2_S mid experts, Q5_K/Q4_K shared, Q4_K/Q3_K attn
#   nano        Q3_K edge / IQ2_S near / IQ2_XXS mid experts (2.06 bpw mid) — needs imatrix
#   micro       Q3_K edge / IQ2_XS near / IQ1_M mid experts (1.75 bpw mid) — needs imatrix, experimental
#   tq-quality  quality + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
#   tq-balanced balanced + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
#   tq-compact  compact + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
#   tq-mini     mini + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
#   tq-nano     nano + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
#   tq-micro    micro + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
#   custom      Specify each type manually via flags
#
# Dense/hybrid profiles (--arch dense is implied; for models whose FFN is dense,
# e.g. Qwen3.8-27B: 64 layers, 48 linear-attention + 16 full-attention):
#   dense-flat            Control. Flat per-role allocation, NO layer gradient —
#                         replicates what the shelf dynamic quants actually do.
#   dense-grad            dense-flat + FFN layer-position gradient. Isolates that
#                         one lever; attention is left identical to the control.
#   dense-hybrid          dense-grad + full-attn pinned up / linear-attn cut.
#                         Full-attn is only ~6% of params, linear-attn ~20%.
#   dense-hybrid-quality  dense-hybrid rebuilt in the Q5/Q6 band, to check the
#                         winning allocation still wins away from the Q4 band.
#
# The three Q4-band profiles are deliberately size-matched: an A/B between
# allocations is only interpretable if the arms are the same size.
#
set -euo pipefail

# Defaults
PROFILE=""
LAYERS=40
OUTPUT=""
DENSE_LAYERS=0                    # leading dense (non-MoE) FFN layers, e.g. LFM2-MoE
ARCH=moe                          # moe | dense — selects which emitter runs
ATTN_WIDE_BOUNDS=0                # 1 = attention uses expert-zone boundaries (TQ profiles)
# Custom mode overrides
EDGE_EXP="" NEAR_EXP="" MID_EXP=""
EDGE_SHARED="" MID_SHARED=""
EDGE_ATTN="" MID_ATTN=""
# Dense/hybrid types
FFN_EDGE_GATE="" FFN_NEAR_GATE="" FFN_MID_GATE=""
FFN_EDGE_UP="" FFN_NEAR_UP="" FFN_MID_UP=""
FFN_EDGE_DOWN="" FFN_NEAR_DOWN="" FFN_MID_DOWN=""
LINATTN="" LINATTN_SMALL=""       # attn_qkv/attn_gate/ssm_out ; ssm_alpha/ssm_beta
FULLATTN="" FULLATTN_V=""         # attn_q/attn_k/attn_output ; attn_v
EMBD_TYPE="" OUTPUT_TYPE=""

show_help() {
    sed -n '3,25p' "$0"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --profile|-p)      PROFILE="$2"; shift 2 ;;
        --layers|-l)       LAYERS="$2"; shift 2 ;;
        --dense-layers)    DENSE_LAYERS="$2"; shift 2 ;;
        --arch)            ARCH="$2"; shift 2 ;;
        --linattn)         LINATTN="$2"; shift 2 ;;
        --fullattn)        FULLATTN="$2"; shift 2 ;;
        --embd)            EMBD_TYPE="$2"; shift 2 ;;
        --output-type)     OUTPUT_TYPE="$2"; shift 2 ;;
        --output|-o)       OUTPUT="$2"; shift 2 ;;
        --custom)          PROFILE="custom"; shift ;;
        --edge-exp)        EDGE_EXP="$2"; shift 2 ;;
        --near-exp)        NEAR_EXP="$2"; shift 2 ;;
        --mid-exp)         MID_EXP="$2"; shift 2 ;;
        --edge-shared)     EDGE_SHARED="$2"; shift 2 ;;
        --mid-shared)      MID_SHARED="$2"; shift 2 ;;
        --edge-attn)       EDGE_ATTN="$2"; shift 2 ;;
        --mid-attn)        MID_ATTN="$2"; shift 2 ;;
        --help|-h)         show_help ;;
        *)                 echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -z "$PROFILE" ] && { echo "Error: --profile required" >&2; exit 1; }

# Edge boundaries (first/last N layers get higher precision)
EDGE_HI=4                         # L0..EDGE_HI
EDGE_LO=$(( LAYERS - 5 ))         # EDGE_LO..LAYERS-1
NEAR_HI=9                         # EDGE_HI+1..NEAR_HI
NEAR_LO=$(( LAYERS - 10 ))        # NEAR_LO..EDGE_LO-1

# Set types per profile
case "$PROFILE" in
    quality|i-quality)
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-iq4_xs}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-Q6_K}"
        ;;
    balanced|i-balanced)
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-Q5_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-Q6_K}"
        ;;
    compact|i-compact)
        EDGE_EXP="${EDGE_EXP:-Q4_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-Q3_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q6_K}"
        MID_SHARED="${MID_SHARED:-Q6_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q4_K}"
        ;;
    mini)
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-iq2_s}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    nano|i-nano)
        # APEX Nano — aggressive mid-layer routed experts at IQ2_XXS (2.06 bpw)
        # Target: ~25-30% smaller than Mini at modest quality cost. Requires imatrix.
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_s}"
        MID_EXP="${MID_EXP:-iq2_xxs}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    micro|i-micro)
        # APEX Micro — extreme mid-layer routed experts at IQ1_M (1.75 bpw)
        # Only viable on MoE: sparse expert activation + shared expert kept high-precision
        # softens per-token error. Quality drop expected — experimental tier. Requires imatrix.
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_xs}"
        MID_EXP="${MID_EXP:-iq1_m}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-Q3_K}"
        ;;
    tq-quality|i-tq-quality)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-iq4_xs}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-tq4_1s}"
        ;;
    tq-balanced|i-tq-balanced)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q6_K}"
        NEAR_EXP="${NEAR_EXP:-Q5_K}"
        MID_EXP="${MID_EXP:-Q5_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q8_0}"
        MID_SHARED="${MID_SHARED:-Q8_0}"
        EDGE_ATTN="${EDGE_ATTN:-Q6_K}"
        MID_ATTN="${MID_ATTN:-tq4_1s}"
        ;;
    tq-compact|i-tq-compact)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q4_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-Q3_K}"
        EDGE_SHARED="${EDGE_SHARED:-Q6_K}"
        MID_SHARED="${MID_SHARED:-Q6_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-tq4_1s}"
        ;;
    tq-mini)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-Q3_K}"
        MID_EXP="${MID_EXP:-iq2_s}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-TQ3_1S}"
        ;;
    tq-nano)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_s}"
        MID_EXP="${MID_EXP:-iq2_xxs}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-TQ3_1S}"
        ;;
    tq-micro)
        ATTN_WIDE_BOUNDS=1
        EDGE_EXP="${EDGE_EXP:-Q3_K}"
        NEAR_EXP="${NEAR_EXP:-iq2_xs}"
        MID_EXP="${MID_EXP:-iq1_m}"
        EDGE_SHARED="${EDGE_SHARED:-Q5_K}"
        MID_SHARED="${MID_SHARED:-Q4_K}"
        EDGE_ATTN="${EDGE_ATTN:-Q4_K}"
        MID_ATTN="${MID_ATTN:-TQ3_1S}"
        ;;
    dense-flat|dense-grad|dense-hybrid|dense-hybrid-quality)
        ARCH=dense
        # Shared across the three Q4-band arms: attention/embedding allocation
        # matching what the shelf quants do, so only the lever under test moves.
        LINATTN="${LINATTN:-Q5_K}"       ; LINATTN_SMALL="${LINATTN_SMALL:-Q4_K}"
        FULLATTN="${FULLATTN:-Q5_K}"     ; FULLATTN_V="${FULLATTN_V:-Q6_K}"
        EMBD_TYPE="${EMBD_TYPE:-Q4_K}"   ; OUTPUT_TYPE="${OUTPUT_TYPE:-Q6_K}"

        case "$PROFILE" in
            dense-flat)
                # Control: flat per-role, no layer gradient (what UD quants do).
                FFN_EDGE_GATE=IQ4_XS ; FFN_NEAR_GATE=IQ4_XS ; FFN_MID_GATE=IQ4_XS
                FFN_EDGE_UP=Q5_K     ; FFN_NEAR_UP=Q5_K     ; FFN_MID_UP=Q5_K
                FFN_EDGE_DOWN=Q5_K   ; FFN_NEAR_DOWN=Q5_K   ; FFN_MID_DOWN=Q5_K
                ;;
            dense-grad|dense-hybrid)
                # Edge layers up, middle layers down — bits moved, not added, so
                # this stays size-matched to dense-flat. Monotonic edge>=near>=mid.
                FFN_EDGE_GATE=Q5_K   ; FFN_NEAR_GATE=Q4_K   ; FFN_MID_GATE=IQ4_XS
                FFN_EDGE_UP=Q6_K     ; FFN_NEAR_UP=Q5_K     ; FFN_MID_UP=Q4_K
                FFN_EDGE_DOWN=Q6_K   ; FFN_NEAR_DOWN=Q5_K   ; FFN_MID_DOWN=Q5_K
                ;;
            dense-hybrid-quality)
                FFN_EDGE_GATE=Q6_K   ; FFN_NEAR_GATE=Q5_K   ; FFN_MID_GATE=Q5_K
                FFN_EDGE_UP=Q6_K     ; FFN_NEAR_UP=Q6_K     ; FFN_MID_UP=Q5_K
                FFN_EDGE_DOWN=Q8_0   ; FFN_NEAR_DOWN=Q6_K   ; FFN_MID_DOWN=Q6_K
                EMBD_TYPE="${EMBD_TYPE:-Q5_K}"
                ;;
        esac

        if [ "$PROFILE" = "dense-hybrid" ] || [ "$PROFILE" = "dense-hybrid-quality" ]; then
            # The structural lever: the 16 full-attention layers carry the global
            # mixing but are only ~6% of params, so pinning them is cheap. The 48
            # linear-attention layers are ~20%, so cutting them pays for it.
            FULLATTN=Q8_0 ; FULLATTN_V=Q8_0
            LINATTN=Q4_K  ; LINATTN_SMALL=Q4_K
        fi
        ;;
    custom)
        [ -z "$EDGE_EXP" ] && { echo "Error: --custom requires --edge-exp" >&2; exit 1; }
        [ -z "$MID_EXP" ] && MID_EXP="$EDGE_EXP"
        [ -z "$NEAR_EXP" ] && NEAR_EXP="$EDGE_EXP"
        [ -z "$EDGE_SHARED" ] && EDGE_SHARED="Q8_0"
        [ -z "$MID_SHARED" ] && MID_SHARED="$EDGE_SHARED"
        [ -z "$EDGE_ATTN" ] && EDGE_ATTN="Q6_K"
        [ -z "$MID_ATTN" ] && MID_ATTN="$EDGE_ATTN"
        ;;
    *)
        echo "Error: unknown profile '$PROFILE'" >&2
        echo "Available: quality, i-quality, balanced, i-balanced, compact, i-compact, mini, nano, i-nano, micro, i-micro, tq-quality, i-tq-quality, tq-balanced, i-tq-balanced, tq-compact, i-tq-compact, tq-mini, tq-nano, tq-micro, custom" >&2
        echo "Dense:     dense-flat, dense-grad, dense-hybrid, dense-hybrid-quality" >&2
        exit 1
        ;;
esac

case "$ARCH" in
    moe|dense) ;;
    *) echo "Error: --arch must be 'moe' or 'dense' (got '$ARCH')" >&2; exit 1 ;;
esac

# "dense" means two different things and they must not be confused: --dense-layers
# is the count of LEADING dense FFN layers inside a MoE (LFM2, Step-3.x, Laguna),
# whereas --arch dense means the whole model has no experts at all.
if [ "$ARCH" = "dense" ] && [ "$DENSE_LAYERS" -ne 0 ]; then
    echo "Error: --arch dense is incompatible with --dense-layers ($DENSE_LAYERS)." >&2
    echo "       --dense-layers counts leading dense FFN layers inside a MoE model;" >&2
    echo "       --arch dense means the model has no expert layers at all." >&2
    exit 1
fi

# Generate config
generate() {
    for (( i=0; i<LAYERS; i++ )); do
        # Expert type based on layer position
        if (( i <= EDGE_HI || i >= EDGE_LO )); then
            exp_type="$EDGE_EXP"
        elif (( i <= NEAR_HI || i >= NEAR_LO )); then
            exp_type="$NEAR_EXP"
        else
            exp_type="$MID_EXP"
        fi

        # Shared type based on layer position
        if (( i <= EDGE_HI || i >= EDGE_LO )); then
            shared_type="$EDGE_SHARED"
        else
            shared_type="$MID_SHARED"
        fi

        # Attention type based on layer position
        if [ "$ATTN_WIDE_BOUNDS" -eq 1 ]; then
            if (( i <= EDGE_HI || i >= EDGE_LO )); then
                attn_type="$EDGE_ATTN"
            else
                attn_type="$MID_ATTN"
            fi
        else
            if (( i <= 2 || i >= LAYERS - 3 )); then
                attn_type="$EDGE_ATTN"
            else
                attn_type="$MID_ATTN"
            fi
        fi

        if (( i < DENSE_LAYERS )); then
            # Leading dense (non-MoE) FFN layers — keep at shared (edge) precision.
            # ".weight" suffix prevents the regex from also matching ffn_*_exps/ffn_gate_inp.
            echo "blk.${i}.ffn_gate.weight=${shared_type}"
            echo "blk.${i}.ffn_up.weight=${shared_type}"
            echo "blk.${i}.ffn_down.weight=${shared_type}"
        else
            # Routed expert tensors (dominant cost in MoE)
            echo "blk.${i}.ffn_gate_exps=${exp_type}"
            echo "blk.${i}.ffn_up_exps=${exp_type}"
            echo "blk.${i}.ffn_down_exps=${exp_type}"
        fi

        # Shared expert tensors (archs with shared experts, e.g. Qwen3-MoE)
        echo "blk.${i}.ffn_gate_shexp=${shared_type}"
        echo "blk.${i}.ffn_up_shexp=${shared_type}"
        echo "blk.${i}.ffn_down_shexp=${shared_type}"

        # Attention tensors (attention layers)
        echo "blk.${i}.attn_q=${attn_type}"
        echo "blk.${i}.attn_k=${attn_type}"
        echo "blk.${i}.attn_v=${attn_type}"
        echo "blk.${i}.attn_output=${attn_type}"
        echo "blk.${i}.attn_gate=${attn_type}"
        echo "blk.${i}.attn_qkv=${attn_type}"

        # Short-convolution mixing tensors (LFM2 conv layers — attention-equivalent)
        echo "blk.${i}.shortconv.in_proj=${attn_type}"
        echo "blk.${i}.shortconv.out_proj=${attn_type}"

        # SSM tensors (Mamba/hybrid archs)
        echo "blk.${i}.ssm_alpha=${attn_type}"
        echo "blk.${i}.ssm_beta=${attn_type}"
        echo "blk.${i}.ssm_out=${attn_type}"
    done
}

# Dense / hybrid-attention emitter.
#
# Unlike the MoE path there are no expert tensors: the FFN is the dominant cost
# (~63% on Qwen3.8-27B) and every weight is on the critical path for every token,
# so the compression has to come from allocation rather than from sparsity.
#
# Linear-attention and full-attention layers are distinguished purely by tensor
# name — attn_qkv/attn_gate/ssm_* exist only on linear layers, attn_q/k/v/output
# only on full ones — so both sets are emitted for every layer and llama-quantize
# ignores the lines that match nothing. (The MoE path already relies on this for
# ssm_*/shortconv on architectures that have neither.)
#
# All names carry an explicit ".weight" suffix: entries are regex-searched, and
# without it "ffn_gate" would also match "ffn_gate_inp".
generate_dense() {
    # Patterns are anchored. llama-quantize matches these with std::regex_search
    # (llama-quant.cpp: unanchored, first match wins), so an unanchored
    # "output.weight" would also capture every "blk.N.attn_output.weight" and
    # silently mis-quantize all 16 attention output projections.
    echo "^token_embd\\.weight$=${EMBD_TYPE}"
    echo "^output\\.weight$=${OUTPUT_TYPE}"

    for (( i=0; i<LAYERS; i++ )); do
        if (( i <= EDGE_HI || i >= EDGE_LO )); then
            gate="$FFN_EDGE_GATE" ; up="$FFN_EDGE_UP" ; down="$FFN_EDGE_DOWN"
        elif (( i <= NEAR_HI || i >= NEAR_LO )); then
            gate="$FFN_NEAR_GATE" ; up="$FFN_NEAR_UP" ; down="$FFN_NEAR_DOWN"
        else
            gate="$FFN_MID_GATE"  ; up="$FFN_MID_UP"  ; down="$FFN_MID_DOWN"
        fi

        echo "^blk\\.${i}\\.ffn_gate\\.weight$=${gate}"
        echo "^blk\\.${i}\\.ffn_up\\.weight$=${up}"
        echo "^blk\\.${i}\\.ffn_down\\.weight$=${down}"

        # Full-attention layers (Qwen3.8: i%4==3). ~6% of params.
        echo "^blk\\.${i}\\.attn_q\\.weight$=${FULLATTN}"
        echo "^blk\\.${i}\\.attn_k\\.weight$=${FULLATTN}"
        echo "^blk\\.${i}\\.attn_v\\.weight$=${FULLATTN_V}"
        echo "^blk\\.${i}\\.attn_output\\.weight$=${FULLATTN}"

        # Linear-attention layers. ~20% of params.
        echo "^blk\\.${i}\\.attn_qkv\\.weight$=${LINATTN}"
        echo "^blk\\.${i}\\.attn_gate\\.weight$=${LINATTN}"
        echo "^blk\\.${i}\\.ssm_out\\.weight$=${LINATTN}"
        echo "^blk\\.${i}\\.ssm_alpha\\.weight$=${LINATTN_SMALL}"
        echo "^blk\\.${i}\\.ssm_beta\\.weight$=${LINATTN_SMALL}"
    done
}

emit() {
    if [ "$ARCH" = "dense" ]; then
        generate_dense
    else
        generate
    fi
}

if [ -n "$OUTPUT" ]; then
    emit > "$OUTPUT"
    echo "Config written to: $OUTPUT ($(wc -l < "$OUTPUT") lines, $LAYERS layers, arch=$ARCH)" >&2
else
    emit
fi
