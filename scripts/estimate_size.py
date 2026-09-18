#!/usr/bin/env python3
"""Estimate the on-disk size of a quantized GGUF without quantizing.

Accepts the same consumer-facing parameters as quantize.py (--profile, --config,
--layers, --base-type, input.gguf) and reports the estimated output size.

Usage:
    ./scripts/estimate_size.py --profile balanced model.gguf
    ./scripts/estimate_size.py --config configs/my_config.txt model.gguf
    ./scripts/estimate_size.py --profile compact --layers 64 model.gguf
    ./scripts/estimate_size.py --profile balanced --quiet model.gguf

Environment:
    NUM_LAYERS   Default layer count (default: 40)
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── GGUF header parser ───────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" in LE


def _read_gguf_string(f):
    slen = struct.unpack("<Q", f.read(8))[0]
    return f.read(slen).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 8:  # STRING
        _read_gguf_string(f)
    elif vtype == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        for _ in range(arr_len):
            _read_gguf_value(f, arr_type)
    else:
        raise SystemExit(f"Unknown GGUF value type {vtype}")


def read_gguf_tensor_list(path):
    """Return list of (name, numel) from a GGUF file header."""
    with open(path, "rb") as f:
        raw = f.read(24)
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", raw)
        if struct.unpack("<I", magic)[0] != GGUF_MAGIC:
            raise SystemExit(f"Not a GGUF file: {path}")
        if version not in (2, 3):
            raise SystemExit(f"Unsupported GGUF version {version}")

        # Skip metadata KV pairs
        for _ in range(n_kv):
            _read_gguf_string(f)  # key
            vtype = struct.unpack("<I", f.read(4))[0]
            _read_gguf_value(f, vtype)

        # Read tensor info
        tensors = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            f.read(4)   # type
            f.read(8)   # offset
            numel = 1
            for d in dims:
                numel *= d
            tensors.append((name, numel))
    return tensors


# ── Size estimation (mirrors estimate_config_size.py logic) ──────────────────

BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5,
    "Q5_0": 5.5, "Q4_0": 4.5,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q3_K": 3.4375, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625,
    "Q2_K": 2.625, "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625,
    "IQ1_M": 1.75, "IQ1_S": 1.5625,
    "TQ4_1S": 4.5, "TQ3_1S": 3.4375,
}

F32_FLOOR = 100_000


def load_config(path):
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
    for pat, qtype in rules:
        try:
            if re.search(pat, name):
                return qtype
        except re.error:
            if pat == name:
                return qtype
    return None


def estimate(tensors, rules, base):
    total_bits = 0.0
    by_type = defaultdict(int)
    uncovered = 0
    uncovered_names = []

    for name, numel in tensors:
        if numel < F32_FLOOR:
            total_bits += numel * 32.0
            by_type["F32 (norms)"] += numel
            continue

        qtype = match_type(name, rules)
        if qtype is None:
            qtype = base
            uncovered += 1
            uncovered_names.append(name)

        if qtype not in BPW:
            raise SystemExit(f"Unknown quant type {qtype!r} (tensor {name})")

        total_bits += numel * BPW[qtype]
        by_type[qtype] += numel

    size_gb = total_bits / 8 / 1e9
    return size_gb, by_type, uncovered, uncovered_names


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate quantized GGUF size without quantizing",
    )
    parser.add_argument("--profile", "-p", default="balanced",
                        help="Profile name (default: balanced)")
    parser.add_argument("--config", "-c",
                        help="Custom tensor-type file")
    parser.add_argument("--base-type", "-b", default="Q6_K",
                        help="Base quant type (default: Q6_K)")
    parser.add_argument("--layers", "-l", type=int,
                        default=int(os.environ.get("NUM_LAYERS", 40)),
                        help="Number of transformer layers (default: 40)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Print size in GB only")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print list of tensors not covered by rules")
    parser.add_argument("input", help="Input GGUF file")

    args = parser.parse_args()

    base_type_map = {
        "quality": "Q6_K", "i-quality": "Q6_K",
        "balanced": "Q6_K", "i-balanced": "Q6_K",
        "compact": "Q4_K", "i-compact": "Q4_K_M",
        "mini": "Q3_K",
    }
    base = base_type_map.get(args.profile, args.base_type).upper()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    config_file = args.config
    tmpfile = None

    if config_file:
        if not os.path.isfile(config_file):
            print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
            sys.exit(1)
    else:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmpfile.close()
        config_file = tmpfile.name
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", args.profile, "--layers", str(args.layers),
               "-o", config_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR generating config: {e.stderr.decode()}", file=sys.stderr)
            if tmpfile:
                os.unlink(tmpfile.name)
            sys.exit(1)

    try:
        tensors = read_gguf_tensor_list(args.input)
        rules = load_config(config_file)
        size_gb, by_type, uncovered, uncovered_names = estimate(tensors, rules, base)

        total_params = sum(n for _, n in tensors)

        if args.quiet:
            print(f"{size_gb:.3f}")
            return

        file_size_gb = os.path.getsize(args.input) / 1e9
        print(f"input:     {args.input} ({file_size_gb:.2f} GB)")
        print(f"profile:   {args.profile}")
        print(f"base type: {base}")
        print(f"tensors:   {len(tensors)}, {total_params / 1e9:.2f} B params total")
        print(f"rules:     {len(rules)} (uncovered fall back to {base}: {uncovered})")
        print(f"\nestimated output size: {size_gb:.2f} GB\n")
        print(f"{'type':<14}{'params':>12}{'share':>9}{'size':>10}")
        for qtype, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
            bpw = BPW.get(qtype, 32.0)
            gb = n * bpw / 8 / 1e9
            print(f"{qtype:<14}{n / 1e9:>10.3f} B{100 * n / total_params:>8.1f}%{gb:>9.2f} GB")
        if args.verbose and uncovered_names:
            print(f"\nuncovered tensors ({len(uncovered_names)}):")
            for name in uncovered_names:
                print(f"  {name}")
    finally:
        if tmpfile:
            os.unlink(tmpfile.name)


if __name__ == "__main__":
    main()
