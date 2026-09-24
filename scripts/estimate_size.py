#!/usr/bin/env python3
"""Estimate the on-disk size of a quantized GGUF without quantizing.

Accepts the same consumer-facing parameters as quantize.py (--profile, --config,
--layers, --base-type, input.gguf) and reports the estimated output size.

Usage:
    ./scripts/estimate_size.py --profile balanced model.gguf
    ./scripts/estimate_size.py --config configs/my_config.txt model.gguf
    ./scripts/estimate_size.py --profile compact --layers 64 model.gguf
    ./scripts/estimate_size.py --profile balanced --quiet model.gguf
    ./scripts/estimate_size.py --compare model.gguf         # inspect tensor types x groups
    ./scripts/estimate_size.py --compare model.gguf --config configs/my_config.txt

Layer count is explicitly set via --layers or auto-detected from the GGUF file.
"""

import argparse
import json
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

# ggml_type enum — source of truth: /usr/local/include/ggml.h
GGML_TYPE = {
    0: "F32", 1: "F16",
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}


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
    """Return list of (name, numel, qtype) from a GGUF file header."""
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
            type_id = struct.unpack("<I", f.read(4))[0]
            f.read(8)   # offset
            numel = 1
            for d in dims:
                numel *= d
            qtype = GGML_TYPE.get(type_id, f"?{type_id}")
            tensors.append((name, numel, qtype))
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

ALWAYS_F32 = [
    re.compile(r"blk\.\d+\.ffn_norm\.weight$"),
    re.compile(r"blk\.\d+\.attn_k_norm\.weight$"),
    re.compile(r"blk\.\d+\.attn_norm\.weight$"),
    re.compile(r"blk\.\d+\.attn_q_norm\.weight$"),
    re.compile(r"blk\.\d+\.ffn_gate_inp\.weight$"),
    re.compile(r"blk\.\d+\.ffn_gate_inp_shexp\.weight$"),
    re.compile(r"blk\.\d+\.post_attention_norm\.weight$"),
    re.compile(r"blk\.\d+\.ssm_a$"),
    re.compile(r"blk\.\d+\.ssm_conv1d\.weight$"),
    re.compile(r"blk\.\d+\.ssm_dt\.bias$"),
    re.compile(r"blk\.\d+\.ssm_norm\.weight$"),
    re.compile(r"output_norm\.weight$"),
    # lfm2moe
    re.compile(r"token_embd_norm\.weight$"),
    re.compile(r"blk\.\d+\.shortconv\.conv\.weight$"),
    re.compile(r"blk\.\d+\.exp_probs_b\.bias"),
]

GROUP_EXPERTS = re.compile(r"blk\.\d+\.ffn_(gate|up|down)_exps")
GROUP_SHEXPERTS = re.compile(r"blk\.\d+\.ffn_(gate|up|down)_shexp")
GROUP_ATTENTION = re.compile(r"blk\.\d+\.attn_")
GROUP_MTP_NAME = re.compile(r"blk\.\d+\.(nextn|mtl|mtp|future)\.")


def _classify_group(name):
    if GROUP_MTP_NAME.search(name):
        return "MTP"
    if GROUP_SHEXPERTS.search(name):
        return "ShExperts"
    if GROUP_EXPERTS.search(name):
        return "Experts"
    if GROUP_ATTENTION.search(name):
        return "Attention"
    return "Other"


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
    group_params = defaultdict(int)
    group_bits = defaultdict(float)
    cat_type_bits = defaultdict(lambda: defaultdict(float))
    uncovered = 0
    uncovered_names = []

    for name, numel, _ in tensors:
        group = _classify_group(name)
        group_params[group] += numel

        if any(p.fullmatch(name) for p in ALWAYS_F32):
            bits = numel * 32.0
            total_bits += bits
            by_type["F32"] += numel
            group_bits[group] += bits
            cat_type_bits[group]["F32"] += bits
            continue

        qtype = match_type(name, rules)
        if qtype is None:
            qtype = base
            uncovered += 1
            uncovered_names.append(name)

        if qtype not in BPW:
            raise SystemExit(f"Unknown quant type {qtype!r} (tensor {name})")

        bits = numel * BPW[qtype]
        total_bits += bits
        by_type[qtype] += numel
        group_bits[group] += bits
        cat_type_bits[group][qtype] += bits

    size_gb = total_bits / 8 / 1e9
    return size_gb, by_type, uncovered, uncovered_names, group_params, group_bits, cat_type_bits


# ── Compare: show actual GGUF breakdown ──────────────────────────────────────

def print_gguf_breakdown(tensors):
    total_params = 0
    by_type = defaultdict(int)
    type_params = defaultdict(int)
    type_bits = defaultdict(float)
    group_params = defaultdict(int)
    group_bits = defaultdict(float)
    cat_type_params = defaultdict(lambda: defaultdict(int))
    cat_type_bits = defaultdict(lambda: defaultdict(float))

    for name, numel, qtype in tensors:
        total_params += numel
        group = _classify_group(name)
        bpw = BPW.get(qtype, 32.0)
        bits = numel * bpw
        by_type[qtype] += numel
        type_params[qtype] += numel
        type_bits[qtype] += bits
        group_params[group] += numel
        group_bits[group] += bits
        cat_type_params[group][qtype] += numel
        cat_type_bits[group][qtype] += bits

    print(f"\n{'type':<14}{'params':>12}{'share':>9}{'size':>10}")
    print("-" * 48)
    for qtype, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        gb = type_bits[qtype] / 8 / 1e9
        print(f"{qtype:<14}{n/1e9:>10.3f} B{100*n/total_params:>8.1f}%{gb:>9.2f} GB")
    total_gb = sum(type_bits.values()) / 8 / 1e9
    print(f"{'TOTAL':<14}{total_params/1e9:>10.3f} B{' ':>9}{total_gb:>9.2f} GB")

    print(f"\n{'group':<14}{'params':>12}{'share':>9}{'size':>10}")
    print("-" * 48)
    groups = [g for g in ("Experts", "ShExperts", "Attention", "MTP", "Other")
              if group_params.get(g, 0) > 0]
    for grp in groups:
        gp = group_params.get(grp, 0)
        gb = group_bits.get(grp, 0.0) / 8 / 1e9
        print(f"{grp:<14}{gp/1e9:>10.3f} B{100*gp/total_params:>8.1f}%{gb:>9.2f} GB")

    all_types = sorted({t for d in cat_type_bits.values() for t in d},
                       key=lambda t: -sum(cat_type_bits[c].get(t, 0) for c in groups))
    if len(all_types) > 1:
        hdr = f"\n{'group':<14}" + "".join(f"{t:>10}" for t in all_types)
        print(hdr)
        print("-" * (14 + 10 * len(all_types)))
        for grp in groups:
            row = f"{grp:<14}"
            for t in all_types:
                gb = cat_type_bits[grp].get(t, 0) / 8 / 1e9
                row += f"{gb:>9.2f} "
            print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def detect_gguf_params(gguf_path):
    """Detect architecture parameters from a GGUF file."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "detect_gguf_params.py"), gguf_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: failed to detect GGUF params: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Estimate quantized GGUF size without quantizing",
    )
    parser.add_argument("--profile", "-p",
                        help="Profile name (default: balanced)", default="tier1")
    parser.add_argument("--config", "-c",
                        help="Custom tensor-type file")
    parser.add_argument("--base-type", "-b", default="Q8_0",
                        help="Base quant type (default: Q8_0)")
    parser.add_argument("--layers", "-l", type=int, default=None,
                        help="Number of transformer layers (default: auto-detect from GGUF, fallback 40)")
    parser.add_argument("--dense-layers", type=int, default=None,
                        help="Leading dense (non-MoE) FFN layers (default: auto-detect)")
    parser.add_argument("--arch", choices=["moe", "dense"], default=None,
                        help="Architecture: moe or dense (default: auto-detect)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Print size in GB only")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print list of tensors not covered by rules")
    parser.add_argument("--compare", action="store_true",
                        help="Show actual GGUF tensor breakdown by quant type x group")
    quant_mode = parser.add_mutually_exclusive_group()
    quant_mode.add_argument("--speed", action="store_const", dest="quant_mode", const="speed",
                            help="Use QUANTS_RANKED_SPEED (Q4_K/Q3_K/Q2_K) for all tensors")
    quant_mode.add_argument("--size", action="store_const", dest="quant_mode", const="size",
                            help="Use QUANTS_RANKED_SIZE (IQ4_NL/IQ3_S/IQ2_S) for all tensors")
    quant_mode.add_argument("--mixed", action="store_const", dest="quant_mode", const="mixed",
                            help="Experts use QUANTS_RANKED_SIZE, everything else QUANTS_RANKED_SPEED")
    parser.add_argument("input", help="Input GGUF file")

    args = parser.parse_args()

    # Auto-detect parameters from GGUF if not explicitly set
    if os.path.isfile(args.input):
        detected = detect_gguf_params(args.input)
        if detected:
            if args.layers is None:
                args.layers = detected["layers"]
            if args.dense_layers is None:
                args.dense_layers = detected["dense_layers"]
            if args.arch is None:
                args.arch = detected["arch"]
            print(f">>> Detected from GGUF: arch={detected['arch']}, layers={detected['layers']}, dense_layers={detected['dense_layers']}")
    if args.layers is None:
        args.layers = 40

    profile = args.profile or "balanced"

    # Base type per profile
    base_type_map = {
        "tier1": "Q8_0", 
        "tier2": "Q8_0", 
        "tier3": "Q8_0", 
        "tier4": "Q8_0", 
        "tier5": "Q8_0", 
        "tier6": "Q8_0", 
        "tier7": "Q6_K", 
        "tier8": "Q6_K", 
        "tier9": "Q6_K", 
        "tier10": "Q5_K", 
        "tier11": "Q5_K", 
        "tier12": "Q5_K", 
        "tier13": "Q4_K", 
    }
    base = base_type_map.get(profile, args.base_type).upper()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    config_file = args.config
    tmpfile = None

    if config_file:
        if not os.path.isfile(config_file):
            print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
            sys.exit(1)
    elif args.compare and not args.profile and not args.config:
        # --compare without --config or --profile: just inspect the file
        pass
    else:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmpfile.close()
        config_file = tmpfile.name
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", profile, "--layers", str(args.layers),
               "-o", config_file]
        if args.dense_layers:
            cmd.extend(["--dense-layers", str(args.dense_layers)])
        if args.arch:
            cmd.extend(["--arch", args.arch])
        if args.quant_mode:
            cmd.append(f"--{args.quant_mode}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR generating config: {e.stderr.decode()}", file=sys.stderr)
            if tmpfile:
                os.unlink(tmpfile.name)
            sys.exit(1)

    try:
        tensors = read_gguf_tensor_list(args.input)
        total_params = sum(n for _, n, _ in tensors)

        # ── Compare-only: no estimate, just inspect the file ──
        if args.compare and not config_file:
            print(f"input:  {args.input}")
            print(f"tensors: {len(tensors)}, {total_params/1e9:.2f} B params")
            print_gguf_breakdown(tensors)
            return

        # ── Estimate from config (with optional compare) ──
        rules = load_config(config_file)
        size_gb, by_type, uncovered, uncovered_names, group_params, group_bits, cat_type_bits = estimate(tensors, rules, base)

        if args.quiet:
            print(f"{size_gb:.3f}")
            return

        file_size_gb = os.path.getsize(args.input) / 1e9
        print(f"input:     {args.input} ({file_size_gb:.2f} GB)")
        print(f"profile:   {profile}")
        print(f"base type: {base}")
        print(f"tensors:   {len(tensors)}, {total_params / 1e9:.2f} B params total")
        print(f"rules:     {len(rules)} (uncovered fall back to {base}: {uncovered})")
        print(f"\nestimated output size: {size_gb:.2f} GB\n")
        print(f"{'group':<14}{'params':>12}{'share':>9}{'size':>10}")
        print("-" * 45)
        for grp in ("Experts", "ShExperts", "Attention", "MTP", "Other"):
            gp = group_params.get(grp, 0)
            if gp == 0:
                continue
            gb = group_bits.get(grp, 0.0) / 8 / 1e9
            print(f"{grp:<14}{gp / 1e9:>10.3f} B{100 * gp / total_params:>8.1f}%{gb:>9.2f} GB")
        print(f"\n{'type':<14}{'params':>12}{'share':>9}{'size':>10}")
        print("-" * 45)
        for qtype, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
            bpw = BPW.get(qtype, 32.0)
            gb = n * bpw / 8 / 1e9
            print(f"{qtype:<14}{n / 1e9:>10.3f} B{100 * n / total_params:>8.1f}%{gb:>9.2f} GB")

        est_groups = [g for g in ("Experts", "ShExperts", "Attention", "MTP", "Other")
                      if group_params.get(g, 0) > 0]
        all_types = sorted({t for d in cat_type_bits.values() for t in d},
                          key=lambda t: -sum(cat_type_bits[c].get(t, 0) for c in est_groups))
        if len(all_types) > 1:
            hdr = f"\n{'group':<14}" + "".join(f"{t:>10}" for t in all_types)
            print(hdr)
            print("-" * (14 + 10 * len(all_types)))
            for grp in est_groups:
                row = f"{grp:<14}"
                for t in all_types:
                    gb = cat_type_bits[grp].get(t, 0) / 8 / 1e9
                    row += f"{gb:>9.2f} "
                print(row)

        if args.verbose and uncovered_names:
            print(f"\nuncovered tensors ({len(uncovered_names)}):")
            for name in uncovered_names:
                print(f"  {name}")

        if args.compare:
            print(f"\n{'=' * 64}")
            print(f"ACTUAL GGUF FILE BREAKDOWN")
            print(f"{'=' * 64}")
            print_gguf_breakdown(tensors)
    finally:
        if tmpfile:
            os.unlink(tmpfile.name)


if __name__ == "__main__":
    main()
