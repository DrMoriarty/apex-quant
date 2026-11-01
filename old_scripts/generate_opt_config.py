#!/usr/bin/env python3
"""
generate_opt_config.py — allocation built from the measured sensitivity curve

The sensitivity sweep (benchmark_results/qwen38_27b/SENSITIVITY.md) measured
dKL per GB saved for every tensor group. Bits belong where they are dearest:
the correct allocation equalises marginal cost per byte, so groups with a low
exchange rate get compressed hard and groups with a high one get protected.

Measured exchange rates (dKL per GB, Q6_K -> Q3_K perturbation):

    token_embd     0.00231   <- cheapest, compress hard
    ffn (mid)      0.00565
    ffn_gate       0.00596
    ffn_up         0.00806
    linattn_gate   0.00880
    ffn_down       0.00920
    linattn_qkv    0.01077
    linattn_out    0.01474
    ffn (edge)     0.01483
    fullattn       0.01583
    output         0.03526   <- dearest, protect

Two facts drive the design:
  * output is 15.3x dearer than token_embd at identical shape (5120 x 248320)
  * FFN edge layers are 2.63x dearer per byte than middle layers

Patterns are anchored (llama-quant.cpp:694 matches unanchored, first-match-wins).

Usage:
    ./scripts/generate_opt_config.py --layers 64 --level balanced -o out.txt
"""

import argparse

# Levels tuned against scripts/estimate_config_size.py to hit a target band.
# Ordering within each level follows the measured exchange rates above.
LEVELS = {
    # ~17.6 GB — matched to the dense_flat control (17.64 GB) for a direct rematch
    "balanced": dict(
        token_embd="Q3_K", output="Q8_0",
        fullattn="Q6_K", linattn_out="Q6_K", linattn_qkv="Q5_K",
        linattn_gate="Q4_K", ssm_small="Q4_K",
        ffn_edge=("Q6_K", "Q6_K", "Q6_K"),      # gate, up, down
        ffn_near=("Q5_K", "Q5_K", "Q5_K"),
        ffn_mid=("IQ4_XS", "Q4_K", "Q4_K"),
    ),
    # ~12.5 GB — the Mini band, where the first experiment suggests the headroom is
    "mini": dict(
        token_embd="IQ2_S", output="Q6_K",
        fullattn="Q5_K", linattn_out="Q5_K", linattn_qkv="Q4_K",
        linattn_gate="Q3_K", ssm_small="Q3_K",
        ffn_edge=("Q4_K", "Q4_K", "Q5_K"),
        ffn_near=("Q3_K", "Q3_K", "Q4_K"),
        ffn_mid=("IQ3_XXS", "Q3_K", "Q3_K"),
    ),
    # ~13.4 GB — mini refined: protect `output` (dearest group, 0.03526) with the
    # bits freed from token_embd (cheapest, 0.00231) and the FFN middle (0.00565)
    "mini-v2": dict(
        token_embd="IQ2_S", output="Q8_0",   # IQ2_XXS aborts: no imatrix for token_embd
        fullattn="Q5_K", linattn_out="Q5_K", linattn_qkv="Q4_K",
        linattn_gate="Q3_K", ssm_small="Q3_K",
        ffn_edge=("Q4_K", "Q4_K", "Q5_K"),
        ffn_near=("Q3_K", "Q3_K", "Q4_K"),
        ffn_mid=("IQ3_XXS", "IQ3_XXS", "IQ3_XXS"),
    ),
    # ~11 GB — nano band. If allocation matters more as damage grows, the gap
    # against flat should widen here relative to mini.
    "nano": dict(
        token_embd="IQ2_S", output="Q6_K",   # IQ2_XXS aborts: no imatrix for token_embd
        fullattn="Q4_K", linattn_out="Q4_K", linattn_qkv="Q3_K",
        linattn_gate="IQ3_XXS", ssm_small="IQ3_XXS",
        ffn_edge=("Q3_K", "Q3_K", "Q4_K"),
        ffn_near=("IQ3_XXS", "IQ3_XXS", "Q3_K"),
        ffn_mid=("IQ2_XXS", "IQ2_S", "IQ2_S"),
    ),
    "nano-flat": dict(
        token_embd="IQ2_S", output="Q6_K",
        fullattn="Q4_K", linattn_out="Q4_K", linattn_qkv="Q4_K",
        linattn_gate="Q4_K", ssm_small="Q4_K",
        ffn_edge=("IQ2_S", "IQ2_S", "IQ2_S"),
        ffn_near=("IQ2_S", "IQ2_S", "IQ2_S"),
        ffn_mid=("IQ2_S", "IQ2_S", "IQ2_S"),
    ),
    # ~14.5 GB — Compact band. Sits between Mini (opt wins, 20.5%) and the Q4
    # band (opt loses), so this locates the crossover where it actually matters:
    # it is the lowest tier we would ship above Mini.
    "compact": dict(
        token_embd="IQ2_S", output="Q6_K",
        fullattn="Q5_K", linattn_out="Q5_K", linattn_qkv="Q5_K",
        linattn_gate="Q4_K", ssm_small="Q4_K",
        ffn_edge=("Q5_K", "Q5_K", "Q5_K"),
        ffn_near=("Q4_K", "Q4_K", "Q4_K"),
        ffn_mid=("Q3_K", "Q3_K", "Q4_K"),
    ),
    "compact-flat": dict(
        token_embd="Q3_K", output="Q6_K",
        fullattn="Q5_K", linattn_out="Q5_K", linattn_qkv="Q5_K",
        linattn_gate="Q5_K", ssm_small="Q5_K",
        ffn_edge=("Q3_K", "Q4_K", "Q4_K"),
        ffn_near=("Q3_K", "Q4_K", "Q4_K"),
        ffn_mid=("Q3_K", "Q4_K", "Q4_K"),
    ),
    # flat control at the mini band: one type per role, no gradient, shelf-style
    "mini-flat": dict(
        token_embd="Q3_K", output="Q6_K",
        fullattn="Q5_K", linattn_out="Q5_K", linattn_qkv="Q5_K",
        linattn_gate="Q5_K", ssm_small="Q5_K",
        ffn_edge=("IQ3_XXS", "Q3_K", "Q3_K"),
        ffn_near=("IQ3_XXS", "Q3_K", "Q3_K"),
        ffn_mid=("IQ3_XXS", "Q3_K", "Q3_K"),
    ),
}


def emit(layers, L):
    e = lambda s: s.replace(".", chr(92) + ".")
    out = [f"^{e('token_embd.weight')}$={L['token_embd']}",
           f"^{e('output.weight')}$={L['output']}"]
    edge = set(range(0, 5)) | set(range(layers - 5, layers))
    near = set(range(5, 10)) | set(range(layers - 10, layers - 5))
    for i in range(layers):
        band = L["ffn_edge"] if i in edge else (L["ffn_near"] if i in near else L["ffn_mid"])
        for tn, t in zip(("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"), band):
            out.append(f"^blk\\.{i}\\.{e(tn)}$={t}")
        for tn in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"):
            out.append(f"^blk\\.{i}\\.{e(tn)}$={L['fullattn']}")
        out.append(f"^blk\\.{i}\\.{e('attn_qkv.weight')}$={L['linattn_qkv']}")
        out.append(f"^blk\\.{i}\\.{e('attn_gate.weight')}$={L['linattn_gate']}")
        out.append(f"^blk\\.{i}\\.{e('ssm_out.weight')}$={L['linattn_out']}")
        for tn in ("ssm_alpha.weight", "ssm_beta.weight"):
            out.append(f"^blk\\.{i}\\.{e(tn)}$={L['ssm_small']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--level", choices=sorted(LEVELS), default="balanced")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    lines = emit(a.layers, LEVELS[a.level])
    open(a.output, "w").write("\n".join(lines) + "\n")
    print(f"{a.level}: {len(lines)} rules -> {a.output}")


if __name__ == "__main__":
    main()
