#!/usr/bin/env python3
"""
estimate_config_size.py — predict the on-disk size of a tensor-type config

Reads a tensor inventory (name -> [numel, stays_f32]) plus a --tensor-type-file
config and reports the resulting GGUF size, without quantizing anything.

Its purpose is the matched-size constraint: an A/B between two allocations is
only interpretable if both land at the same size, and finding that out after a
6-hour quantize run on OVH is expensive.

Accuracy: validated against three published Qwen3.8-27B GGUFs (unsloth
UD-Q4_K_XL, bartowski Q4_K_M, ggml-org Q4_K_M) to within 0.1%.

Usage:
    ./scripts/estimate_config_size.py --inventory models/qwen38_27b_tensors.json \
        --config configs/qwen38_27b_dense_hybrid.txt --base Q5_K
    ./scripts/estimate_config_size.py ... --quiet     # just the number, in GB
"""

import argparse
import json
import re
import sys
from collections import defaultdict

# Effective bits-per-weight including super-block scale/min overhead.
BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5,
    "Q5_0": 5.5, "Q4_0": 4.5,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q3_K": 3.4375, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625,
    "Q2_K": 2.625, "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625,
    "IQ1_M": 1.75, "IQ1_S": 1.5625,
}

# Tensors below this many elements are norms/biases that llama.cpp keeps at F32.
F32_FLOOR = 100_000


def load_inventory(path):
    with open(path) as f:
        blob = json.load(f)
    return blob["tensors"] if "tensors" in blob else blob


def load_config(path):
    """Parse a --tensor-type-file into an ordered list of (pattern, type)."""
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            pat, qtype = line.rsplit("=", 1)
            rules.append((pat.strip(), qtype.strip().upper()))
    return rules


def match_type(name, rules):
    """llama-quantize semantics: entries are regex-matched, first match wins."""
    for pat, qtype in rules:
        try:
            if re.search(pat, name):
                return qtype
        except re.error:
            if pat == name:
                return qtype
    return None


def estimate(inventory, rules, base):
    total_bits = 0.0
    by_type = defaultdict(int)
    uncovered = 0

    for name, (numel, stays_f32) in inventory.items():
        if stays_f32 or numel < F32_FLOOR:
            total_bits += numel * 32.0
            by_type["F32 (norms)"] += numel
            continue

        qtype = match_type(name, rules)
        if qtype is None:
            qtype = base
            uncovered += 1

        if qtype not in BPW:
            raise SystemExit(f"unknown quant type {qtype!r} (tensor {name})")

        total_bits += numel * BPW[qtype]
        by_type[qtype] += numel

    return total_bits / 8 / 1e9, by_type, uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="Q5_K",
                    help="ftype for tensors the config does not cover (default: Q5_K)")
    ap.add_argument("--quiet", action="store_true", help="print only the size in GB")
    args = ap.parse_args()

    inventory = load_inventory(args.inventory)
    rules = load_config(args.config)
    size_gb, by_type, uncovered = estimate(inventory, rules, args.base.upper())

    if args.quiet:
        print(f"{size_gb:.3f}")
        return

    total_params = sum(v[0] for v in inventory.values())
    print(f"config:    {args.config}")
    print(f"inventory: {len(inventory)} tensors, {total_params/1e9:.2f} B params")
    print(f"rules:     {len(rules)}  (uncovered tensors fall back to {args.base.upper()}: {uncovered})")
    print(f"\nestimated size: {size_gb:.2f} GB\n")
    print(f"{'type':<14}{'params':>12}{'share':>9}")
    for qtype, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"{qtype:<14}{n/1e9:>10.3f} B{100*n/total_params:>8.1f}%")


if __name__ == "__main__":
    sys.exit(main())
