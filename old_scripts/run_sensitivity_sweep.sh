#!/usr/bin/env bash
#
# Per-group precision sensitivity sweep — Qwen3.8-27B on thor.
# Runs INSIDE one `flock /tmp/gpu` held for the whole sweep.
#
# For each arm: quantize -> verify size against prediction -> KL vs the BF16
# reference logits -> DELETE the quant. 12 arms x ~22 GB will not fit on disk,
# so only one arm exists at a time.
#
# PPL is measured for the baseline only: the previous experiment established
# that PPL cannot resolve differences at this quality level (all arms within
# +/-0.074), so spending ~3 min/arm on it buys nothing. KL is the metric.
#
# Results are appended to results.tsv as each arm finishes, so a crash loses at
# most the arm in flight.
#
set -uo pipefail
W="$HOME/apex_q38"
IMG=$(cat "$W/image.txt" 2>/dev/null || echo nvidia/cuda:13.0.1-devel-ubuntu24.04)
BIN=/w/llama.cpp/build-cuda/bin
L4T=/opt/nvidia/l4t-gpu-libs/nvgpu
BF16=/w/Qwen3.8-27B-BF16.gguf
QMEM=${QMEM:-40g}
CHUNKS=200
TSV="$W/sens_results.tsv"

log() { echo "[$(date -Is)] $*"; }
gpu() { sudo docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
          -v "$W:/w" -w /w -e LD_LIBRARY_PATH="$BIN:$L4T" "$IMG" "$@"; }

log "=== sensitivity sweep: holding /tmp/gpu ==="
free -g | awk '/^Mem:/{print "    available: "$7" GB"}'
df -h /home/mudler | tail -1

for f in "$W/Qwen3.8-27B-BF16.gguf" "$W/imatrix.gguf" "$W/reference-logits.bin" "$W/wiki.test.raw"; do
    [ -s "$f" ] || { log "MISSING $f"; exit 1; }
done
mkdir -p "$W/sens"
[ -f "$TSV" ] || printf "arm\tsize_bytes\tsize_gb\tkl_mean\tkl_mean_err\tkl_median\tkl_99_9\n" > "$TSV"

# free the previous experiment's quants — their results are already recorded
for old in "$W"/Qwen3.8-27B-APEX-dense_*.gguf; do
    [ -e "$old" ] && { log "removing old quant $(basename "$old")"; rm -f "$old"; }
done
df -h /home/mudler | tail -1

# arm -> predicted bytes (from scripts/estimate_config_size.py, trunk-only)
declare -A EXPECT=(
  [baseline]=22072000000
  [ffn_gate]=19843000000  [ffn_up]=19843000000   [ffn_down]=19843000000
  [linattn_qkv]=21088000000 [linattn_gate]=21482000000 [linattn_out]=21482000000
  [fullattn]=21416000000  [token_embd]=21575000000 [output]=21575000000
  [ffn_mid_only]=17476000000 [ffn_edge_only]=21027000000
)
ARMS="baseline ffn_gate ffn_up ffn_down linattn_qkv linattn_gate linattn_out fullattn token_embd output ffn_mid_only ffn_edge_only"

for arm in $ARMS; do
    if cut -f1 "$TSV" | grep -qx "$arm"; then log "$arm already measured, skipping"; continue; fi

    cfg="/w/sens_cfg/qwen38_27b_sens_${arm}.txt"
    out="$W/sens_${arm}.gguf"
    log "--- $arm ---"

    log "  quantizing"
    sudo docker run --rm --memory="$QMEM" --memory-swap="$QMEM" \
        -v "$W:/w" -w /w "$IMG" /w/q_inner.sh \
        --tensor-type-file "$cfg" --imatrix /w/imatrix.gguf \
        $BF16 "/w/sens_${arm}.gguf" Q6_K > "$W/sens/quant_${arm}.log" 2>&1
    rc=$?
    sudo chown "$(id -u):$(id -g)" "$out" 2>/dev/null || true
    sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
    want=${EXPECT[$arm]}
    lo=$(( want * 97 / 100 )); hi=$(( want * 103 / 100 ))
    log "  rc=$rc size=$sz (want ~$want)"
    if [ "$sz" -lt "$lo" ] || [ "$sz" -gt "$hi" ]; then
        log "  SIZE OUT OF RANGE — skipping this arm, not measuring a bad quant"
        tail -3 "$W/sens/quant_${arm}.log"
        rm -f "$out"; continue
    fi

    if [ "$arm" = "baseline" ]; then
        log "  PPL (baseline only)"
        gpu $BIN/llama-perplexity -m "/w/sens_${arm}.gguf" -f /w/wiki.test.raw \
            -ngl 99 --chunks $CHUNKS > "$W/sens/ppl_${arm}.log" 2>&1
        grep "Final estimate" "$W/sens/ppl_${arm}.log" | tail -1
    fi

    log "  KL"
    gpu $BIN/llama-perplexity -m "/w/sens_${arm}.gguf" -f /w/wiki.test.raw \
        -ngl 99 --chunks $CHUNKS --kl-divergence \
        --kl-divergence-base /w/reference-logits.bin > "$W/sens/kl_${arm}.log" 2>&1
    log "  kl rc=$?"

    klm=$(grep -oP 'Mean\s+KLD:\s+\K[0-9.]+'   "$W/sens/kl_${arm}.log" | tail -1)
    kle=$(grep -oP 'Mean\s+KLD:\s+[0-9.]+\s+±\s+\K[0-9.]+' "$W/sens/kl_${arm}.log" | tail -1)
    kmd=$(grep -oP 'Median\s+KLD:\s+\K[0-9.]+' "$W/sens/kl_${arm}.log" | tail -1)
    k99=$(grep -oP '99\.9%\s+KLD:\s+\K[0-9.]+' "$W/sens/kl_${arm}.log" | tail -1)
    printf "%s\t%s\t%.3f\t%s\t%s\t%s\t%s\n" "$arm" "$sz" \
        "$(awk "BEGIN{print $sz/1000000000}")" "${klm:-NA}" "${kle:-NA}" "${kmd:-NA}" "${k99:-NA}" >> "$TSV"
    log "  KL mean=${klm:-NA}"

    rm -f "$out"           # keep only one arm on disk at a time
    df -h /home/mudler | tail -1
done

log "=== SWEEP TABLE ==="
cat "$TSV"
echo "THOR_SWEEP_DONE"
