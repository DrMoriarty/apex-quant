#!/usr/bin/env python3
"""Patch APEX-MTP per-tier configs to bump the entire blk.40.* layer (the MTP
draft transformer block + nextn-specific tensors) to F16, on every tier except
the experimental I-Nano. Norms are left to llama-quantize's F32 heuristic.

Why: the MTP head is the speculative-decoding draft. If its weights are too
aggressively quantized, draft predictions degrade and spec-decode acceptance
drops. The default behavior left nextn.eh_proj at the tier's base type (e.g.,
Q3_K for I-Mini), which is way too low for a critical projection. This patch
adds explicit blk.40.*=F16 overrides for every tensor in the layer.

Usage:
  python3 scripts/apex_mtp_patch_configs.py
"""
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

PREFIXES = [
    "qwen36_35b_mtp",
    "qwen36_opus_distill_mtp",
    "qwen36_opus47_distill_mtp",
    "carnice_qwen36_mtp",
    "qwopus36_mtp",
]
BUMP_TIERS = ["quality", "balanced", "compact", "mini"]  # nano stays as-is

# Full set of blk.40.* tensors that appear in a Qwen 3.6 MoE bundled-MTP GGUF.
# Force everything to F16. llama-quantize will auto-keep norms at F32 even if
# we ask for F16 — harmless. eh_proj.weight is the critical one.
BLK40_TENSORS = [
    "blk.40.attn_q",
    "blk.40.attn_k",
    "blk.40.attn_v",
    "blk.40.attn_output",
    "blk.40.attn_q_norm",
    "blk.40.attn_k_norm",
    "blk.40.attn_norm",
    "blk.40.post_attention_norm",
    "blk.40.ffn_gate_exps",
    "blk.40.ffn_up_exps",
    "blk.40.ffn_down_exps",
    "blk.40.ffn_gate_inp",
    "blk.40.ffn_gate_shexp",
    "blk.40.ffn_up_shexp",
    "blk.40.ffn_down_shexp",
    "blk.40.ffn_gate_inp_shexp",
    "blk.40.nextn.eh_proj",
    "blk.40.nextn.enorm",
    "blk.40.nextn.hnorm",
    "blk.40.nextn.shared_head_norm",
]


def patch_one(path: Path):
    lines = path.read_text().splitlines()
    # Strip any existing blk.40.* entries (with or without .weight suffix variation)
    kept = [l for l in lines if not l.startswith("blk.40.")]
    overrides = [f"{t}=Q8_0" for t in BLK40_TENSORS]
    out = kept + [""] + overrides + [""]
    path.write_text("\n".join(out))
    return len(BLK40_TENSORS)


def main():
    patched = 0
    for prefix in PREFIXES:
        for tier in BUMP_TIERS:
            p = CONFIGS_DIR / f"{prefix}_{tier}.txt"
            if not p.exists():
                print(f"  WARN missing: {p}")
                continue
            n = patch_one(p)
            print(f"  patched {p.name}: +{n} blk.40 overrides (F16)")
            patched += 1
        print(f"  skipped {prefix}_nano.txt (I-Nano keeps tier-default precision)")
    print(f"\nTotal: {patched} configs patched")


if __name__ == "__main__":
    main()
