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
    from estimate_size import read_gguf_tensor_list

    tensors = read_gguf_tensor_list(args.gguf)

    if not tensors:
        print("Error: no tensors found in GGUF", file=sys.stderr)
        sys.exit(1)

    has_experts = False
    expert_layers = set()
    all_layers = set()
    mtp_layers = set()

    pattern_ffn = re.compile(r"blk\.(\d+)\.ffn_")
    pattern_expert = re.compile(r"blk\.(\d+)\.ffn_gate_exps")
    pattern_mtp = re.compile(r"blk\.(\d+)\.(nextn|mtl|mtp|future)\.")
    pattern_moe_mtp = re.compile(r"blk\.(\d+)\.moe\.")

    for name, _, _ in tensors:
        # Detect MTP head layers
        m = pattern_mtp.search(name)
        if m:
            mtp_layers.add(int(m.group(1)))
            continue

        # DeepSeek-style: blk.{N}.moe.* at high indices = MTP head
        m = pattern_moe_mtp.search(name)
        if m:
            mtp_layers.add(int(m.group(1)))
            continue

        m = pattern_ffn.search(name)
        if m:
            layer = int(m.group(1))
            all_layers.add(layer)

        m = pattern_expert.search(name)
        if m:
            layer = int(m.group(1))
            expert_layers.add(layer)
            has_experts = True

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
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()