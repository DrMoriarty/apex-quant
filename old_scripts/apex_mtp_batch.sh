#!/bin/bash
# Re-quantize Qwen 3.6 MoE models with bundled MTP head (llama.cpp PR #22673).
# Pattern: imatrix on dgx.casa, quantize+upload on jumphost. APEX_VARIANT=MTP
# infixes "-MTP" into output filenames; new HF repos are <orig>-APEX-MTP-GGUF.
#
# Usage:
#   ./scripts/apex_mtp_batch.sh                 # do all queued models
#   ./scripts/apex_mtp_batch.sh qwen36_35b_mtp  # do one model (yaml basename)
#
# Requires: hf-cli authed on dgx + jumphost; llama.cpp at >= 255582687 on both.

set -uo pipefail

# ─── Queue (in execution order) ──────────────────────────────────────────────
QUEUE=(
  qwen36_opus_distill_mtp
  qwen36_opus47_distill_mtp
  carnice_qwen36_mtp
  qwopus36_mtp
)
# Skipped (config declares mtp_num_hidden_layers=1 but safetensors have no MTP weights):
#   - qwen36_heretic_mtp  (llmfan46 — trained trunk only, 1026 tensors, 0 MTP)
#   - darwin_36b_opus_mtp (FINAL-Bench — evo merge dropped MTP, 693 tensors, 0 MTP)
# qwen36_35b_mtp already completed in earlier run.

# ─── Hosts ───────────────────────────────────────────────────────────────────
DGX_HOST="dgx.casa"
DGX_WORK="/home/mudler/work"
DGX_REPO="/home/mudler/autoresearch-quant"
JH_SSH="ssh ubuntu@57.131.21.202 -p 2233 -o ConnectTimeout=30 -o ServerAliveInterval=15"
JH_WORK="/home/ubuntu/work/apex"
JH_REPO="/home/ubuntu/autoresearch-quant"
LOCAL_REPO="/home/mudler/autoresearch-quant"

# ─── Helpers ──────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo ""; echo "[$(ts)] ═══ $* ═══"; }
info() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ✗ $*" >&2; exit 1; }

parse_yaml() { grep "^$2:" "$1" | head -1 | sed "s/^$2:[[:space:]]*//" | awk '{print $1}'; }

# ─── Per-model orchestration ─────────────────────────────────────────────────
run_one() {
  local yaml_base="$1"
  local yaml="$LOCAL_REPO/models/${yaml_base}.yaml"
  [ -f "$yaml" ] || die "yaml not found: $yaml"

  local NAME MODEL_ID PREFIX HF_REPO
  NAME=$(parse_yaml "$yaml" name)
  MODEL_ID=$(parse_yaml "$yaml" model_id)
  PREFIX=$(parse_yaml "$yaml" config_prefix)
  HF_REPO=$(parse_yaml "$yaml" hf_repo)

  log "Model: ${NAME}  (prefix=${PREFIX}  repo=${HF_REPO})"

  local DGX_DIR="${DGX_WORK}/${PREFIX}"
  local JH_DIR="${JH_WORK}/${PREFIX}"

  # ───── Phase A: dgx — download safetensors ─────
  log "[${NAME}] A: download safetensors on dgx"
  ssh "$DGX_HOST" "
    mkdir -p ${DGX_DIR}/safetensors
    if find ${DGX_DIR}/safetensors -name '*.safetensors' 2>/dev/null | grep -q .; then
      echo '  safetensors already present, skipping download'
    else
      export PATH=/home/mudler/.local/bin:\$PATH
      cd ${DGX_DIR}/safetensors && hf download ${MODEL_ID} --local-dir . 2>&1 | tail -5
    fi
  " || die "dgx download failed for $NAME"

  # ───── Phase B: dgx — patch tokenizer if Darwin-style ─────
  ssh "$DGX_HOST" "
    cfg=${DGX_DIR}/safetensors/tokenizer_config.json
    if [ -f \"\$cfg\" ] && grep -q 'TokenizersBackend' \"\$cfg\"; then
      echo '  patching TokenizersBackend → Qwen2Tokenizer'
      python3 -c \"import json; p='\$cfg'; d=json.load(open(p)); d['tokenizer_class']='Qwen2Tokenizer'; json.dump(d, open(p,'w'), indent=2)\"
    fi
  "

  # ───── Phase C: dgx — convert to BF16 GGUF (bundled MTP by default w/ PR 22673) ─────
  log "[${NAME}] C: convert safetensors → f16.gguf (bundled MTP)"
  ssh "$DGX_HOST" "
    if [ -f ${DGX_DIR}/f16.gguf ] && [ \$(stat -c%s ${DGX_DIR}/f16.gguf) -gt 50000000000 ]; then
      echo '  f16.gguf exists ('\$(du -h ${DGX_DIR}/f16.gguf | cut -f1)'), skipping convert'
    else
      docker stop local-ai-worker 2>/dev/null | tail -1 || true
      source ${DGX_REPO}/.venv/bin/activate
      python3 ${DGX_REPO}/llama.cpp/convert_hf_to_gguf.py ${DGX_DIR}/safetensors \\
        --outfile ${DGX_DIR}/f16.gguf --outtype bf16 2>&1 | tail -5
    fi
    ls -lh ${DGX_DIR}/f16.gguf
  " || die "dgx convert failed for $NAME"

  # ───── Phase D: dgx — imatrix ─────
  log "[${NAME}] D: llama-imatrix on dgx"
  ssh "$DGX_HOST" "
    if [ -f ${DGX_DIR}/imatrix.dat ] && [ \$(stat -c%s ${DGX_DIR}/imatrix.dat) -gt 50000000 ]; then
      echo '  imatrix.dat exists, skipping'
    else
      docker stop local-ai-worker 2>/dev/null | tail -1 || true
      ${DGX_REPO}/llama.cpp/build/bin/llama-imatrix \\
        -m ${DGX_DIR}/f16.gguf \\
        -f ${DGX_REPO}/calibration/calibration_v1.3.txt \\
        -ngl 99 --save-frequency 100 \\
        -o ${DGX_DIR}/imatrix.dat 2>&1 | tail -5
    fi
    ls -lh ${DGX_DIR}/imatrix.dat
  " || die "dgx imatrix failed for $NAME"

  # ───── Phase E: dgx — upload F16 to new MTP repo ─────
  log "[${NAME}] E: upload F16 to ${HF_REPO}"
  ssh "$DGX_HOST" "
    export PATH=/home/mudler/.local/bin:\$PATH
    cd ${DGX_DIR}
    # Check if already uploaded by querying remote
    if hf download ${HF_REPO} ${NAME}-F16.gguf --local-dir /tmp/.f16probe 2>/dev/null >/dev/null; then
      echo '  F16 already on HF, skipping upload'; rm -rf /tmp/.f16probe
    else
      hf upload --repo-type model ${HF_REPO} f16.gguf ${NAME}-F16.gguf \\
        --commit-message 'Upload F16 reference (with MTP)' 2>&1 | tail -3
    fi
  " || die "F16 upload failed for $NAME"

  # ───── Phase F: prep jumphost dir + sync imatrix ─────
  log "[${NAME}] F: sync imatrix dgx → jumphost"
  $JH_SSH "mkdir -p ${JH_DIR}" || die "jumphost mkdir failed"
  ssh "$DGX_HOST" "rsync -avh ${DGX_DIR}/imatrix.dat ubuntu@57.131.21.202:${JH_DIR}/imatrix.dat -e 'ssh -p 2233' 2>&1 | tail -3" \
    || die "imatrix rsync failed"

  # ───── Phase G: jumphost — download F16 from HF ─────
  log "[${NAME}] G: download F16 on jumphost"
  $JH_SSH "
    if [ -f ${JH_DIR}/f16.gguf ] && [ \$(stat -c%s ${JH_DIR}/f16.gguf) -gt 50000000000 ]; then
      echo '  f16.gguf already on jumphost'
    else
      export PATH=/home/ubuntu/.local/bin:\$PATH
      cd ${JH_DIR} && hf download ${HF_REPO} ${NAME}-F16.gguf --local-dir . 2>&1 | tail -3
      [ -f ${NAME}-F16.gguf ] && mv ${NAME}-F16.gguf f16.gguf
    fi
    ls -lh ${JH_DIR}/f16.gguf
  " || die "F16 download on jumphost failed for $NAME"

  # ───── Phase H: jumphost — sync yaml + start uploader watcher ─────
  log "[${NAME}] H: sync yaml + start uploader watcher"
  rsync -av "$LOCAL_REPO/models/${yaml_base}.yaml" "ubuntu@57.131.21.202:${JH_REPO}/models/" \
    -e "ssh -p 2233" 2>&1 | tail -2

  # Write per-model uploader inline (matches APEX-MTP filename pattern)
  $JH_SSH "cat > /tmp/uploader_${PREFIX}.sh << 'WATCHER'
#!/bin/bash
set -uo pipefail
export PATH=/home/ubuntu/.local/bin:\$PATH
MODEL_DIR=${JH_DIR}
REPO=${HF_REPO}
PATTERN='${NAME}-APEX-MTP-*.gguf'
MIN_BYTES=6000000000
declare -A UPLOADED=()
ts() { date '+%Y-%m-%d %H:%M:%S'; }
while true; do
  sleep 60
  for f in \$MODEL_DIR/\$PATTERN; do
    [ -f \"\$f\" ] || continue
    base=\$(basename \"\$f\")
    [ \"\${UPLOADED[\$base]:-}\" = done ] && continue
    if pgrep -af 'llama-quantize' | grep -qF \"\$base\"; then
      echo \"[\$(ts)] skip \$base (llama-quantize writing)\"; continue
    fi
    s1=\$(stat -c%s \"\$f\"); sleep 30; s2=\$(stat -c%s \"\$f\" 2>/dev/null || echo 0)
    [ \"\$s1\" != \"\$s2\" ] && { echo \"[\$(ts)] skip \$base (growing)\"; continue; }
    [ \"\$s1\" -lt \"\$MIN_BYTES\" ] && { echo \"[\$(ts)] skip \$base (small)\"; continue; }
    echo \"[\$(ts)] uploading \$base (\$((s1/1024/1024/1024)) GB)\"
    if hf upload --repo-type model \"\$REPO\" \"\$f\" \"\$base\" --commit-message \"Add \$base\"; then
      echo \"[\$(ts)] uploaded \$base, deleting local\"; rm -f \"\$f\"; UPLOADED[\$base]=done
    else
      echo \"[\$(ts)] FAILED upload (\$?), will retry\"
    fi
  done
done
WATCHER
chmod +x /tmp/uploader_${PREFIX}.sh
tmux kill-session -t up-${PREFIX} 2>/dev/null
tmux new-session -d -s up-${PREFIX} 'stdbuf -oL /tmp/uploader_${PREFIX}.sh > /tmp/up-${PREFIX}.log 2>&1'
tmux ls | grep up-${PREFIX}
"

  # ───── Phase I: jumphost — run apex_pipeline.sh phases ─────
  log "[${NAME}] I: run pipeline (config, quantize, ivariants) with APEX_VARIANT=MTP"
  $JH_SSH "
    tmux kill-session -t pipe-${PREFIX} 2>/dev/null
    tmux new-session -d -s pipe-${PREFIX} -c ${JH_REPO} \\
      'export PATH=/home/ubuntu/.local/bin:\$PATH; \\
       export LLAMA_CPP_DIR=/home/ubuntu/llama.cpp/build/bin; \\
       WORK_DIR=${JH_WORK} APEX_VARIANT=MTP SKIP_TIERS=micro \\
       bash scripts/apex_pipeline.sh --config models/${yaml_base}.yaml \\
         --only config,quantize,ivariants > /tmp/pipe-${PREFIX}.log 2>&1'
    sleep 3
    tmux ls | grep pipe-${PREFIX}
  "

  # ───── Phase J: wait for pipeline to finish ─────
  log "[${NAME}] J: waiting for pipeline (poll every 60s)"
  while $JH_SSH "tmux ls 2>&1 | grep -q pipe-${PREFIX}"; do
    sleep 60
    $JH_SSH "tail -1 /tmp/pipe-${PREFIX}.log | head -c 200" || true
    echo
  done
  info "  pipeline tmux ended"

  # ───── Phase K: wait for uploader to drain ─────
  log "[${NAME}] K: waiting for uploader to drain remaining quants"
  while $JH_SSH "ls ${JH_DIR}/${NAME}-APEX-MTP-*.gguf 2>/dev/null | head -1" > /tmp/.drain_check 2>/dev/null; do
    [ -s /tmp/.drain_check ] || break
    sleep 60
    info "  still draining: $(cat /tmp/.drain_check | wc -l) file(s)"
  done
  $JH_SSH "tmux kill-session -t up-${PREFIX} 2>/dev/null"
  info "  uploader stopped"

  # ───── Phase L: sync generated configs back to local + commit ─────
  log "[${NAME}] L: sync configs back to local"
  rsync -av "ubuntu@57.131.21.202:${JH_REPO}/configs/${PREFIX}_*.txt" "${LOCAL_REPO}/configs/" \
    -e "ssh -p 2233" 2>&1 | tail -3

  # ───── Phase M: model card upload ─────
  log "[${NAME}] M: write + upload model card"
  local CARD="${LOCAL_REPO}/model_cards/${yaml_base}_modelcard.md"
  if [ ! -f "$CARD" ]; then
    write_mtp_card "$NAME" "$MODEL_ID" "$HF_REPO" "$CARD"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  hf upload --repo-type model "$HF_REPO" "$CARD" README.md \
    --commit-message "Add APEX-MTP model card" 2>&1 | tail -3

  # ───── Phase N: cleanup ─────
  log "[${NAME}] N: cleanup dgx + jumphost work dirs"
  ssh "$DGX_HOST" "rm -rf ${DGX_DIR}" || true
  $JH_SSH "rm -rf ${JH_DIR}" || true

  info "[${NAME}] DONE ✓"
}

# ─── Model card generator ────────────────────────────────────────────────────
write_mtp_card() {
  local name="$1" model_id="$2" repo="$3" out="$4"
  cat > "$out" << CARD
---
license: apache-2.0
base_model: ${model_id}
tags:
  - gguf
  - quantized
  - apex
  - apex-mtp
  - moe
  - mixture-of-experts
  - qwen3
  - qwen3.6
  - speculative-decoding
  - self-speculative
  - mtp
---

<!-- apex-banner-v2 -->
<div style="background-color: #f59e0b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
<h2 style="color: white; margin: 0 0 10px 0;">⚡ Each donation = another big MoE quantized</h2>
<p style="font-size: 18px; margin: 0 0 15px 0;">I host <b>30+ free APEX MoE quantizations</b> as independent research. My only local hardware is an <b>NVIDIA DGX Spark</b> (122 GB unified memory) — enough for ~30-50B-class MoEs, but <b>bigger ones (200B+) require rented compute</b> on H100/H200/Blackwell, typically \$20-100 per quant.<br>If APEX quants are useful to you, your support directly funds those bigger runs.</p>
<p style="font-size: 20px; margin: 0;">
<a href="https://www.patreon.com/cw/mudler" style="color: white; text-decoration: underline;">🎉 Patreon (Monthly)</a> &nbsp;|&nbsp;
<a href="https://www.buymeacoffee.com/mudler" style="color: white; text-decoration: underline;">☕ Buy Me a Coffee</a> &nbsp;|&nbsp;
<a href="https://github.com/sponsors/mudler" style="color: white; text-decoration: underline;">⭐ GitHub Sponsors</a>
</p>
</div>

# ${name} — APEX-MTP GGUF

**APEX (Adaptive Precision for EXpert Models)** quantizations of [${model_id}](https://huggingface.co/${model_id}), with the **MTP (multi-token prediction) head bundled** for in-the-box self-speculative decoding.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/mudler/apex-quant) | [Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.pdf)

## What's different from the plain APEX repo?

These GGUFs bundle the model's **MTP (multi-token prediction) head** alongside the trunk in a single file, courtesy of [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). With a recent llama.cpp (>= commit 255582687) you can enable self-speculative decoding using just this one file — no separate draft model needed:

\`\`\`bash
llama-server -m ${name}-APEX-MTP-I-Balanced.gguf --draft-mtp
\`\`\`

The non-MTP version is still available at [${repo%-MTP-GGUF}-GGUF](https://huggingface.co/${repo%-MTP-GGUF}-GGUF) — slightly smaller, but no self-spec.

## File sizes

Each quant is ~2.5% larger than its non-MTP counterpart (one extra transformer-block worth of weights, no embedding duplication since MTP shares the trunk's embed_tokens).

## What is APEX?

APEX is a MoE-aware mixed-precision quantization strategy. Per-tensor-role gradient: routed experts compress hardest, shared experts kept high (always active), attention/Mamba uniform; 5+5 symmetric edge gradient across the 40 trunk layers + MTP layer 40 at edge precision. I-variants use diverse imatrix calibration (chat, code, reasoning, tool-calling, agentic traces, Wikipedia).

See the [APEX project](https://github.com/mudler/apex-quant) for full details.

## Architecture

- **Base**: Qwen 3.6 35B-A3B family (Qwen3_5MoeForCausalLM)
- **Layers**: 40 trunk + 1 MTP (bundled)
- **Experts**: 256 routed + 1 shared (8 active per token)
- **Hidden size**: 2048
- **Calibration**: v1.3 diverse dataset

## Credits

- **APEX quantization**: [LocalAI](https://github.com/mudler/LocalAI) team
- **MTP support**: llama.cpp PR #22673 by Aman Gupta + ggerganov
- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp)
CARD
  info "  wrote model card: $out"
}

# ─── Main ────────────────────────────────────────────────────────────────────
if [ $# -gt 0 ]; then
  run_one "$1"
else
  for m in "${QUEUE[@]}"; do
    run_one "$m"
  done
  log "All ${#QUEUE[@]} models complete"
fi
