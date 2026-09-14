#!/usr/bin/env bash
#
# Tests for scripts/generate_config.sh
#
# Two kinds of test live here:
#   * REGRESSION GUARDS  — assert the MoE path still reproduces the configs that
#     every published APEX repo was built from. These pass today by construction;
#     their job is to fail the day someone changes shared code.
#   * BEHAVIOUR TESTS    — assert the dense/hybrid mode emits what the Qwen3.8-27B
#     experiment arms need.
#
# Usage: ./tests/test_generate_config.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="${REPO_ROOT}/scripts/generate_config.sh"
EST="${REPO_ROOT}/scripts/estimate_config_size.py"
INVENTORY="${REPO_ROOT}/models/qwen38_27b_tensors.json"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; [ $# -gt 1 ] && printf '        %s\n' "$2"; FAIL=$((FAIL+1)); }

# Types assigned to a given tensor suffix across all layers, deduplicated.
# Dense configs use anchored, escaped patterns (^blk\.0\.ffn_gate\.weight$=Q5_K),
# so strip the regex metacharacters before matching.
types_for() {
    # $1 = config file, $2 = tensor suffix (e.g. ffn_gate.weight)
    sed 's/[\^$]//g; s/\\//g' "$1" | grep -E "^blk\.[0-9]+\.$2=" | sed 's/.*=//' | sort -u
}

# Does the config assign this exact global tensor?
# NB: no `grep -q` here — it exits on first match, sed then takes SIGPIPE, and
# `set -o pipefail` would turn a successful match into a failed pipeline.
assigns_global() {
    local out
    out=$(sed 's/[\^$]//g; s/\\//g' "$1" | grep -E "^$2=" || true)
    [ -n "$out" ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Regression guards: the MoE path must reproduce committed configs byte-for-byte
# ─────────────────────────────────────────────────────────────────────────────
echo "MoE regression guards (published configs must not drift)"

# prefix:layers:dense_layers — configs known to be current with the generator.
for spec in laguna_xs21:40:1 step37_flash:45:3; do
    prefix="${spec%%:*}"; rest="${spec#*:}"; layers="${rest%%:*}"; dense="${rest##*:}"
    for profile in quality balanced compact mini; do
        committed="${REPO_ROOT}/configs/${prefix}_${profile}.txt"
        [ -f "$committed" ] || continue
        "$GEN" --profile "$profile" --layers "$layers" --dense-layers "$dense" \
               -o "$TMP/regen.txt" 2>/dev/null
        if diff -q "$committed" "$TMP/regen.txt" >/dev/null 2>&1; then
            ok "${prefix}_${profile} reproduced byte-for-byte"
        else
            bad "${prefix}_${profile} drifted" "$(diff "$committed" "$TMP/regen.txt" | head -3 | tr '\n' ' ')"
        fi
    done
done

# The qwen36_35b configs predate shortconv support (added with LFM2, 92f1013), so
# the generator now emits two extra lines per layer that those files do not have.
# Harmless in practice — qwen36 has no shortconv tensors, so llama-quantize ignores
# them — but it means these files cannot serve as a byte-for-byte oracle. Assert
# the drift is EXACTLY that and nothing more, so a real regression still trips.
for profile in quality balanced compact mini; do
    committed="${REPO_ROOT}/configs/qwen36_35b_${profile}.txt"
    [ -f "$committed" ] || continue
    "$GEN" --profile "$profile" --layers 40 --dense-layers 0 -o "$TMP/regen.txt" 2>/dev/null
    if diff <(grep -v 'shortconv' "$committed") <(grep -v 'shortconv' "$TMP/regen.txt") >/dev/null 2>&1; then
        ok "qwen36_35b_${profile} matches apart from known shortconv drift"
    else
        bad "qwen36_35b_${profile} drifted beyond shortconv" \
            "$(diff <(grep -v shortconv "$committed") <(grep -v shortconv "$TMP/regen.txt") | head -3 | tr '\n' ' ')"
    fi
done

# apex configs (pre-date shortconv support; committed files use mixed-case type
# names, e.g. q6_K, while the generator now emits Q6_K). Treat them like qwen36.
# NOTE: apex_mini.txt and apex_tq_mini.txt were hand-tuned (NEAR experts set to
# iq2_s instead of generator's Q3_K) so they cannot serve as strict oracles.
for profile in quality balanced compact; do
    committed="${REPO_ROOT}/configs/apex_${profile}.txt"
    [ -f "$committed" ] || continue
    "$GEN" --profile "$profile" --layers 40 --dense-layers 0 -o "$TMP/regen.txt" 2>/dev/null
    if diff <(grep -v 'shortconv' "$committed" | tr '[:upper:]' '[:lower:]') \
            <(grep -v 'shortconv' "$TMP/regen.txt" | tr '[:upper:]' '[:lower:]') >/dev/null 2>&1; then
        ok "apex_${profile} matches apart from known shortconv/case drift"
    else
        bad "apex_${profile} drifted beyond shortconv/case" \
            "$(diff <(grep -v 'shortconv' "$committed" | tr '[:upper:]' '[:lower:]') <(grep -v 'shortconv' "$TMP/regen.txt" | tr '[:upper:]' '[:lower:]') | head -3 | tr '\n' ' ')"
    fi
done

for profile in tq-quality tq-balanced tq-compact; do
    committed="${REPO_ROOT}/configs/apex_${profile}.txt"
    [ -f "$committed" ] || continue
    "$GEN" --profile "$profile" --layers 40 -o "$TMP/regen.txt" 2>/dev/null
    if diff <(grep -v 'shortconv' "$committed" | tr '[:upper:]' '[:lower:]') \
            <(grep -v 'shortconv' "$TMP/regen.txt" | tr '[:upper:]' '[:lower:]') >/dev/null 2>&1; then
        ok "apex_${profile} matches apart from known shortconv/case drift"
    else
        bad "apex_${profile} drifted beyond shortconv/case" \
            "$(diff <(grep -v 'shortconv' "$committed" | tr '[:upper:]' '[:lower:]') <(grep -v 'shortconv' "$TMP/regen.txt" | tr '[:upper:]' '[:lower:]') | head -3 | tr '\n' ' ')"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# TQ profile behavioural tests
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "TQ profile behaviour (TurboQuant attention types)"

L40=40

# tq-quality: mid-layer attention must be tq4_1s, edge must be Q6_K
"$GEN" --profile tq-quality --layers "$L40" -o "$TMP/tq_quality.txt" 2>/dev/null
mid_attn=$(grep 'blk\.10\.attn_q=' "$TMP/tq_quality.txt" | sed 's/.*=//')
edge_attn=$(grep 'blk\.0\.attn_q=' "$TMP/tq_quality.txt" | sed 's/.*=//')
if [ "$mid_attn" = tq4_1s ] && [ "$edge_attn" = Q6_K ]; then
    ok "tq-quality: mid attn=tq4_1s, edge attn=Q6_K"
else
    bad "tq-quality: mid attn=$mid_attn, edge attn=$edge_attn"
fi

# tq-balanced: same TQ type pattern
"$GEN" --profile tq-balanced --layers "$L40" -o "$TMP/tq_balanced.txt" 2>/dev/null
mid_attn=$(grep 'blk\.15\.attn_q=' "$TMP/tq_balanced.txt" | sed 's/.*=//')
if [ "$mid_attn" = tq4_1s ]; then
    ok "tq-balanced: mid attn=tq4_1s"
else
    bad "tq-balanced: mid attn=$mid_attn (expected tq4_1s)"
fi

# tq-compact: mid-layer attention must be tq4_1s
"$GEN" --profile tq-compact --layers "$L40" -o "$TMP/tq_compact.txt" 2>/dev/null
mid_attn=$(grep 'blk\.20\.attn_q=' "$TMP/tq_compact.txt" | sed 's/.*=//')
if [ "$mid_attn" = tq4_1s ]; then
    ok "tq-compact: mid attn=tq4_1s"
else
    bad "tq-compact: mid attn=$mid_attn (expected tq4_1s)"
fi

# tq-mini: mid-layer attention must be TQ3_1S
"$GEN" --profile tq-mini --layers "$L40" -o "$TMP/tq_mini.txt" 2>/dev/null
mid_attn=$(grep 'blk\.20\.attn_q=' "$TMP/tq_mini.txt" | sed 's/.*=//')
if [ "$mid_attn" = TQ3_1S ]; then
    ok "tq-mini: mid attn=TQ3_1S"
else
    bad "tq-mini: mid attn=$mid_attn (expected TQ3_1S)"
fi

# tq-nano: mid-layer attention must be TQ3_1S, mid experts must be iq2_xxs
"$GEN" --profile tq-nano --layers "$L40" -o "$TMP/tq_nano.txt" 2>/dev/null
mid_attn=$(grep 'blk\.20\.attn_q=' "$TMP/tq_nano.txt" | sed 's/.*=//')
mid_exp=$(grep 'blk\.20\.ffn_gate_exps=' "$TMP/tq_nano.txt" | sed 's/.*=//')
if [ "$mid_attn" = TQ3_1S ] && [ "$mid_exp" = iq2_xxs ]; then
    ok "tq-nano: mid attn=TQ3_1S, mid exp=iq2_xxs"
else
    bad "tq-nano: mid attn=$mid_attn, mid exp=$mid_exp (expected TQ3_1S, iq2_xxs)"
fi

# tq-micro: mid-layer attention must be TQ3_1S, mid experts must be iq1_m
"$GEN" --profile tq-micro --layers "$L40" -o "$TMP/tq_micro.txt" 2>/dev/null
mid_attn=$(grep 'blk\.20\.attn_q=' "$TMP/tq_micro.txt" | sed 's/.*=//')
mid_exp=$(grep 'blk\.20\.ffn_gate_exps=' "$TMP/tq_micro.txt" | sed 's/.*=//')
if [ "$mid_attn" = TQ3_1S ] && [ "$mid_exp" = iq1_m ]; then
    ok "tq-micro: mid attn=TQ3_1S, mid exp=iq1_m"
else
    bad "tq-micro: mid attn=$mid_attn, mid exp=$mid_exp (expected TQ3_1S, iq1_m)"
fi

# ATTN_WIDE_BOUNDS: edge layers 3,4 must use EDGE_ATTN (not MID_ATTN)
for profile in tq-quality tq-balanced tq-compact tq-mini tq-nano tq-micro; do
    "$GEN" --profile "$profile" --layers "$L40" -o "$TMP/tt.txt" 2>/dev/null
    l3_attn=$(grep 'blk\.3\.attn_q=' "$TMP/tt.txt" | sed 's/.*=//')
    l3_exp=$(grep 'blk\.3\.ffn_gate_exps=' "$TMP/tt.txt" | sed 's/.*=//')
    # L3 is in EDGE zone (0..4), so attn should equal EDGE_ATTN, not MID_ATTN
    l5_attn=$(grep 'blk\.5\.attn_q=' "$TMP/tt.txt" | sed 's/.*=//')
    if [ "$l3_attn" != "$l5_attn" ]; then
        ok "wide bounds: ${profile} layer 3 attn ($l3_attn) differs from layer 5 ($l5_attn)"
    else
        bad "wide bounds: ${profile} layer 3 and 5 both use $l3_attn — ATTN_WIDE_BOUNDS not applied"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Dense/hybrid mode
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "Dense/hybrid mode (Qwen3.8-27B: 64 layers, 48 linear-attn + 16 full-attn)"

DL=64

# --- the dense FFN tensors are the ones that actually exist on a dense model ---
"$GEN" --profile dense-grad --layers "$DL" -o "$TMP/grad.txt" 2>/dev/null
if [ -s "$TMP/grad.txt" ] && [ -n "$(types_for "$TMP/grad.txt" 'ffn_gate\.weight')" ]; then
    ok "dense mode emits blk.N.ffn_gate.weight"
else
    bad "dense mode emits blk.N.ffn_gate.weight" "profile dense-grad produced nothing usable"
fi

if [ -s "$TMP/grad.txt" ] && ! grep -qE 'ffn_(gate|up|down)_exps' "$TMP/grad.txt"; then
    ok "dense mode emits no _exps tensors (they do not exist on a dense model)"
else
    bad "dense mode emits no _exps tensors"
fi

# --- embeddings and output head: 9.3% of this model, currently never emitted ---
if assigns_global "$TMP/grad.txt" 'token_embd\.weight' && \
   assigns_global "$TMP/grad.txt" 'output\.weight'; then
    ok "dense mode pins token_embd.weight and output.weight"
else
    bad "dense mode pins token_embd.weight and output.weight" "left to llama-quantize defaults"
fi

# --- arm B: the layer-position gradient, the lever no shelf quant applies here ---
n_types=$(types_for "$TMP/grad.txt" 'ffn_gate\.weight' | wc -l)
if [ "$n_types" -gt 1 ]; then
    ok "dense-grad applies a layer-position gradient to FFN ($n_types distinct types)"
else
    bad "dense-grad applies a layer-position gradient to FFN" "found $n_types distinct type(s), expected >1"
fi

# --- arm A: the flat control must have no gradient, or it is not a control ---
"$GEN" --profile dense-flat --layers "$DL" -o "$TMP/flat.txt" 2>/dev/null
n_flat=$(types_for "$TMP/flat.txt" 'ffn_gate\.weight' | wc -l)
if [ "$n_flat" -eq 1 ]; then
    ok "dense-flat control has no FFN layer gradient (1 type)"
else
    bad "dense-flat control has no FFN layer gradient" "found $n_flat distinct type(s), expected exactly 1"
fi

# --- arm C: full-attention and linear-attention must be separately assignable ---
"$GEN" --profile dense-hybrid --layers "$DL" -o "$TMP/hybrid.txt" 2>/dev/null
full=$(types_for "$TMP/hybrid.txt" 'attn_q\.weight' | head -1)
lin=$(types_for  "$TMP/hybrid.txt" 'attn_qkv\.weight' | head -1)
if [ -n "$full" ] && [ -n "$lin" ] && [ "$full" != "$lin" ]; then
    ok "dense-hybrid separates full-attn ($full) from linear-attn ($lin)"
else
    bad "dense-hybrid separates full-attn from linear-attn" "full='$full' linear='$lin'"
fi

# dense-grad must NOT yet split them, otherwise arm C tests two changes at once.
# Both must be non-empty: two missing tensors would otherwise compare equal and
# give a vacuous pass.
gfull=$(types_for "$TMP/grad.txt" 'attn_q\.weight' | head -1)
glin=$(types_for  "$TMP/grad.txt" 'attn_qkv\.weight' | head -1)
if [ -n "$gfull" ] && [ -n "$glin" ] && [ "$gfull" = "$glin" ]; then
    ok "dense-grad leaves attention untouched (isolates the FFN lever)"
else
    bad "dense-grad leaves attention untouched" "full='$gfull' linear='$glin' — arm B would confound two levers"
fi

# --- regression: the output-head rule must not swallow attn_output ---
# llama-quantize matches patterns with std::regex_search (unanchored, first match
# wins, llama-quant.cpp:694), so an unanchored "output.weight" rule also matches
# "blk.N.attn_output.weight" and silently mis-quantizes all 16 attention output
# projections. Patterns are anchored to prevent this; assert it stays that way.
aout=$(types_for "$TMP/hybrid.txt" 'attn_output\.weight' | head -1)
ohead=$(sed 's/[\^$]//g; s/\\//g' "$TMP/hybrid.txt" | grep -E '^output\.weight=' | sed 's/.*=//')
if [ -n "$aout" ] && [ "$aout" = "$full" ] && [ "$aout" != "$ohead" ]; then
    ok "attn_output takes the full-attn type ($aout), not the output-head type ($ohead)"
else
    bad "attn_output takes the full-attn type, not the output-head type" \
        "attn_output='$aout' full-attn='$full' output-head='$ohead' — unanchored pattern leak"
fi

# --- the two meanings of "dense" must not collide silently ---
# Guard against passing for the wrong reason: the same invocation *without*
# --dense-layers must succeed, so the rejection is provably about the combination.
if ! "$GEN" --profile dense-grad --layers "$DL" -o /dev/null 2>/dev/null; then
    bad "--arch dense + --dense-layers is rejected" "baseline dense-grad invocation does not even succeed"
elif "$GEN" --profile dense-grad --layers "$DL" --dense-layers 3 -o /dev/null 2>/dev/null; then
    bad "--arch dense + --dense-layers is rejected" "accepted a contradictory combination"
else
    ok "--arch dense + --dense-layers is rejected"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Matched size: the experiment is only interpretable if the arms are same-size
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "Matched-size constraint (arms must be comparable)"

if [ ! -f "$EST" ] || [ ! -f "$INVENTORY" ]; then
    bad "size estimator available" "missing $EST or $INVENTORY"
else
    sizes=()
    for arm in flat grad hybrid; do
        s=$(python3 "$EST" --inventory "$INVENTORY" --config "$TMP/$arm.txt" --quiet 2>/dev/null)
        sizes+=("$s")
    done
    spread=$(python3 -c "
v=[float(x) for x in '${sizes[*]}'.split()]
print(f'{max(v)-min(v):.3f} {min(v):.2f} {max(v):.2f}')" 2>/dev/null)
    delta=$(echo "$spread" | cut -d' ' -f1)
    lo=$(echo "$spread" | cut -d' ' -f2)
    hi=$(echo "$spread" | cut -d' ' -f3)
    if python3 -c "import sys; sys.exit(0 if float('$delta') <= 0.20 else 1)" 2>/dev/null; then
        ok "arms are size-matched (spread ${delta} GB, ${lo}-${hi} GB)"
    else
        bad "arms are size-matched" "spread ${delta} GB exceeds 0.20 GB tolerance (${lo}-${hi} GB)"
    fi

    # The arms must also land in the size class they will be judged against on
    # HuggingFace: Q4_K_M is 17.11-17.77 GB across publishers and unsloth's
    # UD-Q4_K_XL is 17.92 GB, so 17.5-18.0 is that shelf.
    if python3 -c "import sys; sys.exit(0 if 17.5 <= float('$lo') and float('$hi') <= 18.0 else 1)" 2>/dev/null; then
        ok "arms land in the Q4_K_M / UD-Q4_K_XL band (17.5-18.0 GB)"
    else
        bad "arms land in the Q4_K_M / UD-Q4_K_XL band (17.5-18.0 GB)" "got ${lo}-${hi} GB"
    fi
fi

echo
echo "─────────────────────────────────────────"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
