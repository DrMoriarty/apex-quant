#!/bin/bash
# Re-do the APEX-MTP batch with bumped MTP (blk.40.* → F16) on every tier
# except I-Nano. Skips safetensors download + convert (F16 already on HF
# from the original run). Per-model: dgx imatrix → jumphost quantize all
# 8 tiers → upload, with hardened verification to prevent the I-Nano
# zero-padded-upload bug from recurring.
#
# Bug history: original batch's IQ2_XXS quantize for I-Nano produced a file
# whose size stabilized before page-cache flush. Uploader saw stable size +
# dead llama-quantize and uploaded zero-filled content (hf-cli reported
# "Processing Files (0/0)" but exit 0 → file deleted locally, HF copy is
# zero-padded). All 5 I-Nano uploads were corrupt.
#
# Fix in this re-do:
#  - sync + 10s sleep after every llama-quantize before file is eligible
#  - GGUF magic verify (first 4 bytes == 'GGUF') on local before upload
#  - GGUF magic re-verify on remote after upload; abort if mismatch
#
# Usage:
#   ./scripts/apex_mtp_redo_batch.sh           # all 5 models
#   ./scripts/apex_mtp_redo_batch.sh qwen36_35b_mtp

set -uo pipefail

QUEUE=(
  # qwen36_35b_mtp          # done (recovery)
  # qwen36_opus_distill_mtp # done (recovery — manual i-variants)
  qwen36_opus47_distill_mtp
  carnice_qwen36_mtp
  qwopus36_mtp
)

DGX_HOST="dgx.casa"
DGX_WORK="/home/mudler/work"
DGX_REPO="/home/mudler/autoresearch-quant"
JH_SSH="ssh ubuntu@57.131.21.202 -p 2233 -o ConnectTimeout=30 -o ServerAliveInterval=15"
JH_WORK="/home/ubuntu/work/apex"
JH_REPO="/home/ubuntu/autoresearch-quant"
LOCAL_REPO="/home/mudler/autoresearch-quant"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo ""; echo "[$(ts)] ═══ $* ═══"; }
info() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ✗ $*" >&2; exit 1; }

parse_yaml() { grep "^$2:" "$1" | head -1 | sed "s/^$2:[[:space:]]*//" | awk '{print $1}'; }

verify_gguf_magic_remote() {
  local url=$(curl -sI -L -o /dev/null -w "%{url_effective}" "$1?download=true")
  local hdr=$(curl -s -r 0-3 "$url" | xxd | awk 'NR==1{print $2}')
  [ "$hdr" = "4747" ]
}

redo_one() {
  local yaml_base="$1"
  local yaml="$LOCAL_REPO/models/${yaml_base}.yaml"
  local NAME=$(parse_yaml "$yaml" name)
  local PREFIX=$(parse_yaml "$yaml" config_prefix)
  local HF_REPO=$(parse_yaml "$yaml" hf_repo)

  log "REDO: ${NAME}  (prefix=${PREFIX}  repo=${HF_REPO})"

  # NB: must match what apex_pipeline.sh expects: WORK_DIR/CONFIG_PREFIX (no suffix)
  local DGX_DIR="${DGX_WORK}/${PREFIX}"
  local JH_DIR="${JH_WORK}/${PREFIX}"

  # ── dgx: download F16 + regen imatrix (decoupled via nohup script so
  # long-running phase isn't blocked behind a single fragile ssh) ──
  log "[${NAME}] dgx: write phase script + launch via nohup"
  local DGX_SCRIPT="/tmp/dgx_phase_${PREFIX}.sh"
  local DGX_LOG="/tmp/dgx_phase_${PREFIX}.log"
  local DGX_DONE="${DGX_DIR}/.dgx_done"
  ssh -o ConnectTimeout=30 "$DGX_HOST" "cat > $DGX_SCRIPT" << SCRIPTEOF
#!/bin/bash
set -uo pipefail
mkdir -p ${DGX_DIR}
rm -f ${DGX_DONE}
export PATH=/home/mudler/.local/bin:\$PATH
if [ ! -f ${DGX_DIR}/f16.gguf ] || [ \$(stat -c%s ${DGX_DIR}/f16.gguf) -lt 50000000000 ]; then
  cd ${DGX_DIR}
  hf download ${HF_REPO} ${NAME}-F16.gguf --local-dir . 2>&1 | tail -3
  [ -f ${NAME}-F16.gguf ] && mv ${NAME}-F16.gguf f16.gguf
fi
ls -lh ${DGX_DIR}/f16.gguf
if [ ! -f ${DGX_DIR}/imatrix.dat ] || [ \$(stat -c%s ${DGX_DIR}/imatrix.dat) -lt 50000000 ]; then
  docker stop local-ai-worker 2>/dev/null | tail -1 || true
  ${DGX_REPO}/llama.cpp/build/bin/llama-imatrix \\
    -m ${DGX_DIR}/f16.gguf \\
    -f ${DGX_REPO}/calibration/calibration_v1.3.txt \\
    -ngl 99 --save-frequency 100 \\
    -o ${DGX_DIR}/imatrix.dat 2>&1 | tail -3
fi
ls -lh ${DGX_DIR}/imatrix.dat
echo done > ${DGX_DONE}
SCRIPTEOF
  ssh -o ConnectTimeout=30 "$DGX_HOST" "chmod +x $DGX_SCRIPT; tmux kill-session -t dgx_phase_${PREFIX} 2>/dev/null; tmux new-session -d -s dgx_phase_${PREFIX} '$DGX_SCRIPT > $DGX_LOG 2>&1'" \
    || die "failed to launch dgx phase script for $NAME"

  log "[${NAME}] dgx: polling for completion (.dgx_done flag)"
  while ! ssh -o ConnectTimeout=30 "$DGX_HOST" "test -f $DGX_DONE" 2>/dev/null; do
    sleep 60
    info "  $(ssh -o ConnectTimeout=30 "$DGX_HOST" "ls -lh ${DGX_DIR}/imatrix.dat 2>/dev/null | awk '{print \$5}' || echo 'no imatrix yet'")"
  done
  info "  dgx phase done"

  # ── jumphost: prep, sync imatrix + configs, download F16 ──
  $JH_SSH "mkdir -p ${JH_DIR}" || die "jumphost mkdir failed for $NAME"
  ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 "$DGX_HOST" \
    "set -e; rsync -avh ${DGX_DIR}/imatrix.dat ubuntu@57.131.21.202:${JH_DIR}/imatrix.dat -e 'ssh -p 2233' 2>&1 | tail -3" \
    || die "imatrix rsync failed for $NAME"
  $JH_SSH "[ -f ${JH_DIR}/imatrix.dat ] && [ \$(stat -c%s ${JH_DIR}/imatrix.dat) -gt 50000000 ]" \
    || die "imatrix.dat missing or too small on jumphost for $NAME"

  log "[${NAME}] sync patched configs to jumphost"
  rsync -av "$LOCAL_REPO/configs/${PREFIX}_"*.txt "ubuntu@57.131.21.202:${JH_REPO}/configs/" \
    -e "ssh -p 2233" 2>&1 | tail -2
  rsync -av "$LOCAL_REPO/models/${yaml_base}.yaml" "ubuntu@57.131.21.202:${JH_REPO}/models/" \
    -e "ssh -p 2233" 2>&1 | tail -2

  log "[${NAME}] jumphost: F16 download"
  $JH_SSH "
    if [ ! -f ${JH_DIR}/f16.gguf ] || [ \$(stat -c%s ${JH_DIR}/f16.gguf) -lt 50000000000 ]; then
      export PATH=/home/ubuntu/.local/bin:\$PATH
      cd ${JH_DIR} && hf download ${HF_REPO} ${NAME}-F16.gguf --local-dir . 2>&1 | tail -3
      [ -f ${NAME}-F16.gguf ] && mv ${NAME}-F16.gguf f16.gguf
    fi
    ls -lh ${JH_DIR}/f16.gguf
  " || die "F16 download on jumphost failed for $NAME"

  # ── start hardened uploader watcher ──
  log "[${NAME}] start hardened uploader"
  $JH_SSH "cat > /tmp/uploader_redo_${PREFIX}.sh << WATCHEREOF
#!/bin/bash
set -uo pipefail
export PATH=/home/ubuntu/.local/bin:\\\$PATH
MODEL_DIR=${JH_DIR}
REPO=${HF_REPO}
PATTERN='${NAME}-APEX-MTP-*.gguf'
MIN_BYTES=6000000000
declare -A UPLOADED=()
ts() { date '+%Y-%m-%d %H:%M:%S'; }
verify_local() {
  # \\\$1 = file path. Returns 0 if GGUF magic present. Reads only first 4 bytes.
  local hdr=\\\$(head -c 4 \"\\\$1\" 2>/dev/null | xxd | awk 'NR==1{print \\\$2}')
  [ \"\\\$hdr\" = '4747' ]
}
# (verify_remote dropped: xet-served URLs don't honor range requests so it
#  always returned false. Trust hf-cli exit code + local verify instead.)
while true; do
  sleep 60
  for f in \\\$MODEL_DIR/\\\$PATTERN; do
    [ -f \"\\\$f\" ] || continue
    base=\\\$(basename \"\\\$f\")
    [ \"\\\${UPLOADED[\\\$base]:-}\" = done ] && continue
    if pgrep -af 'llama-quantize' | grep -qF \"\\\$base\"; then
      echo \"[\\\$(ts)] skip \\\$base (llama-quantize writing)\"; continue
    fi
    # Force-flush page cache before stability check
    sync; sleep 5
    s1=\\\$(stat -c%s \"\\\$f\"); sleep 30; s2=\\\$(stat -c%s \"\\\$f\" 2>/dev/null || echo 0)
    [ \"\\\$s1\" != \"\\\$s2\" ] && { echo \"[\\\$(ts)] skip \\\$base (growing)\"; continue; }
    [ \"\\\$s1\" -lt \"\\\$MIN_BYTES\" ] && { echo \"[\\\$(ts)] skip \\\$base (small)\"; continue; }
    # HARDENED: verify content is real GGUF, not zero-padded
    if ! verify_local \"\\\$f\"; then
      echo \"[\\\$(ts)] ✗ LOCAL CORRUPT \\\$base (GGUF magic missing) — re-syncing + waiting\"
      sync; sleep 30; continue
    fi
    echo \"[\\\$(ts)] uploading \\\$base (\\\$((s1/1024/1024/1024)) GB)\"
    if hf upload --repo-type model \"\\\$REPO\" \"\\\$f\" \"\\\$base\" --commit-message \"Add \\\$base (MTP-bumped)\"; then
      echo \"[\\\$(ts)] uploaded \\\$base, deleting local\"
      rm -f \"\\\$f\"
      UPLOADED[\\\$base]=done
    else
      echo \"[\\\$(ts)] FAILED upload (\\\$?), will retry\"
    fi
  done
done
WATCHEREOF
chmod +x /tmp/uploader_redo_${PREFIX}.sh
tmux kill-session -t up-redo-${PREFIX} 2>/dev/null
tmux new-session -d -s up-redo-${PREFIX} 'stdbuf -oL /tmp/uploader_redo_${PREFIX}.sh > /tmp/up-redo-${PREFIX}.log 2>&1'
tmux ls | grep up-redo-${PREFIX}
"

  # ── jumphost: run pipeline phase 5+7 ──
  log "[${NAME}] jumphost: pipeline quantize + ivariants (APEX_VARIANT=MTP SKIP_TIERS=micro)"
  $JH_SSH "
    tmux kill-session -t pipe-redo-${PREFIX} 2>/dev/null
    tmux new-session -d -s pipe-redo-${PREFIX} -c ${JH_REPO} \\
      'export PATH=/home/ubuntu/.local/bin:\$PATH; \\
       export LLAMA_CPP_DIR=/home/ubuntu/llama.cpp/build/bin; \\
       WORK_DIR=${JH_WORK} APEX_VARIANT=MTP SKIP_TIERS=micro \\
       bash scripts/apex_pipeline.sh --config models/${yaml_base}.yaml \\
         --only quantize,ivariants > /tmp/pipe-redo-${PREFIX}.log 2>&1'
    sleep 3
    tmux ls | grep pipe-redo-${PREFIX}
  "

  log "[${NAME}] waiting for pipeline to finish (poll every 60s)"
  while $JH_SSH "tmux ls 2>&1 | grep -q pipe-redo-${PREFIX}"; do
    sleep 60
    $JH_SSH "tail -1 /tmp/pipe-redo-${PREFIX}.log | head -c 200" || true
    echo
  done
  info "  pipeline ended"

  log "[${NAME}] waiting for uploader to drain"
  local stable=0
  while [ $stable -lt 3 ]; do
    sleep 60
    # Correctly check file count — ssh always returns 0 even when no matches,
    # so we need to compare the count.
    remaining=$($JH_SSH "ls ${JH_DIR}/${NAME}-APEX-MTP-*.gguf 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    if [ "${remaining:-0}" -gt 0 ]; then
      info "  still draining: $remaining file(s)"
      stable=0
    else
      stable=$((stable + 1))
      info "  drain stable: $stable/3"
    fi
  done
  $JH_SSH "tmux kill-session -t up-redo-${PREFIX} 2>/dev/null"

  # ── verify all 8 expected files have GGUF magic on HF ──
  log "[${NAME}] verify all 8 tiers on HF"
  local bad=0
  for tier in Quality Balanced Compact I-Quality I-Balanced I-Compact I-Mini I-Nano; do
    if verify_gguf_magic_remote "https://huggingface.co/${HF_REPO}/resolve/main/${NAME}-APEX-MTP-${tier}.gguf"; then
      info "  ✓ ${tier}"
    else
      info "  ✗ ${tier} CORRUPT"; bad=$((bad+1))
    fi
  done
  [ $bad -gt 0 ] && die "$bad/8 tiers corrupt — investigate"

  # cleanup
  ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 "$DGX_HOST" "rm -rf ${DGX_DIR}" || true
  $JH_SSH "rm -rf ${JH_DIR}" || true

  info "[${NAME}] REDO DONE ✓"
}

if [ $# -gt 0 ]; then
  redo_one "$1"
else
  for m in "${QUEUE[@]}"; do
    redo_one "$m"
  done
  log "All ${#QUEUE[@]} redos complete"
fi
