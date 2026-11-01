#!/usr/bin/env python3
"""
generate_sensitivity_configs.py — per-group precision sensitivity sweep configs

Emits a Q6_K baseline plus one arm per tensor group, where that group alone is
dropped to Q3_K. The sensitivity of a group is then dKL(arm) - dKL(baseline),
and the quantity that governs allocation is dKL per GB saved.

See benchmark_results/qwen38_27b/NEXT_EXPERIMENT.md.

Patterns are ANCHORED. llama-quantize matches with std::regex_search — unanchored
and first-match-wins (llama-quant.cpp:694) — so a bare "output.weight" rule would
also capture every "blk.N.attn_output.weight".

Usage:
    ./scripts/generate_sensitivity_configs.py --layers 64 --outdir configs
"""

import argparse
import os

BASE = "Q6_K"      # near-lossless reference level
PERTURB = "Q3_K"   # deliberately large, so each group's effect is resolvable

# per-layer tensors, grouped. ssm_alpha/ssm_beta are ~0.04% each and are pinned
# at BASE in every arm rather than being probed.
LAYER_TENSORS = {
    "ffn_gate":     ["ffn_gate.weight"],
    "ffn_up":       ["ffn_up.weight"],
    "ffn_down":     ["ffn_down.weight"],
    "linattn_qkv":  ["attn_qkv.weight"],
    "linattn_gate": ["attn_gate.weight"],
    "linattn_out":  ["ssm_out.weight"],
    "fullattn":     ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"],
    "_fixed":       ["ssm_alpha.weight", "ssm_beta.weight"],
}
GLOBAL_TENSORS = {"token_embd": "token_embd.weight", "output": "output.weight"}

FFN = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]


def bands(n):
    """edge / near / mid layer index sets, matching the APEX convention."""
    edge = set(range(0, 5)) | set(range(n - 5, n))
    near = set(range(5, 10)) | set(range(n - 10, n - 5))
    mid = set(range(n)) - edge - near
    return edge, near, mid


def emit(layers, group=None, layer_subset=None):
    """One config. `group` is dropped to PERTURB; if layer_subset is given the
    drop applies only on those layers (used for the edge-vs-mid arms)."""
    out = []
    for name, tensor in GLOBAL_TENSORS.items():
        t = PERTURB if group == name else BASE
        out.append(f"^{tensor.replace('.', chr(92) + '.')}$={t}")

    for i in range(layers):
        for g, tensors in LAYER_TENSORS.items():
            for tn in tensors:
                if g == "_fixed":
                    t = BASE
                elif group == "ffn_layers" and tn in FFN:
                    t = PERTURB if (layer_subset and i in layer_subset) else BASE
                elif group == g:
                    t = PERTURB
                else:
                    t = BASE
                esc = tn.replace(".", chr(92) + ".")
                out.append(f"^blk\\.{i}\\.{esc}$={t}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--outdir", default="configs")
    ap.add_argument("--prefix", default="qwen38_27b_sens")
    args = ap.parse_args()

    edge, _, mid = bands(args.layers)

    arms = {"baseline": (None, None)}
    for g in LAYER_TENSORS:
        if g != "_fixed":
            arms[g] = (g, None)
    for g in GLOBAL_TENSORS:
        arms[g] = (g, None)
    arms["ffn_mid_only"] = ("ffn_layers", mid)
    arms["ffn_edge_only"] = ("ffn_layers", edge)

    os.makedirs(args.outdir, exist_ok=True)
    for arm, (group, subset) in arms.items():
        lines = emit(args.layers, group, subset)
        path = os.path.join(args.outdir, f"{args.prefix}_{arm}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{arm:16} -> {path}  ({len(lines)} rules)")
    print(f"\n{len(arms)} arms (1 baseline + {len(arms)-1} perturbations)")


if __name__ == "__main__":
    main()
