#!/usr/bin/env python3
"""Detect architecture parameters from a GGUF file.

Reads GGUF tensor list and determines:
  - architecture (moe or dense)
  - number of dense layers (for MoE models, leading layers without experts)
  - total number of layers

Usage:
  ./scripts/detect_gguf_params.py model.gguf
  # Output: {"arch": "moe", "dense_layers": 1, "layers": 40}
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Detect GGUF architecture parameters")
    parser.add_argument("gguf", help="GGUF file to inspect")
    args = parser.parse_args()

    if not os.path.isfile(args.gguf):
        print(f"Error: file not found: {args.gguf}", file=sys.stderr)
        sys.exit(1)

    # Reuse tensor reader from estimate_size.py
    sys.path.append(SCRIPT_DIR)
    from estimate_size import read_gguf_tensor_list, ALWAYS_F32

    tensors = read_gguf_tensor_list(args.gguf)

    if not tensors:
        print("Error: no tensors found in GGUF", file=sys.stderr)
        sys.exit(1)

    has_experts = False
    expert_layers = set()
    all_layers = set()
    mtp_layers = set()

    pattern_ffn = re.compile(r"blk\.(\d+)\.ffn_")
    # Gate-less MoE (Nemotron Lightning etc.) ship only ffn_up_exps/ffn_down_exps.
    pattern_expert = re.compile(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps")
    pattern_mtp = re.compile(r"blk\.(\d+)\.(nextn|mtl|mtp|future)\.")
    pattern_moe_mtp = re.compile(r"blk\.(\d+)\.moe\.")

    # Pass 1: identify which layers belong to the MTP head.
    for name, _, _ in tensors:
        m = pattern_mtp.search(name)
        if m:
            mtp_layers.add(int(m.group(1)))
            continue
        m = pattern_moe_mtp.search(name)
        if m:
            mtp_layers.add(int(m.group(1)))
            continue
        m = pattern_ffn.search(name)
        if m:
            all_layers.add(int(m.group(1)))
        m = pattern_expert.search(name)
        if m:
            expert_layers.add(int(m.group(1)))
            has_experts = True

    # Pass 2: within MTP head layers, take every tensor except those pinned to F32.
    mtp_tensor_names = []
    for name, _, _ in tensors:
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m.group(1)) in mtp_layers and not any(p.fullmatch(name) for p in ALWAYS_F32):
            mtp_tensor_names.append(name)

    all_layers -= mtp_layers
    expert_layers -= mtp_layers
    layers = max(all_layers) + 1 if all_layers else 0

    if has_experts:
        # Count leading dense layers: layers 0..N-1 that have no experts
        dense_layers = 0
        for i in range(layers):
            if i in expert_layers:
                break
            dense_layers += 1
        arch = "moe"
    else:
        arch = "dense"
        dense_layers = 0

    mtp_depth = len(mtp_layers)
    result = {
        "arch": arch,
        "dense_layers": dense_layers,
        "layers": layers,
        "mtp_depth": mtp_depth,
        "mtp_layers": sorted(mtp_layers),
        "mtp_tensors": sorted(mtp_tensor_names),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()