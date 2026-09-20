#!/usr/bin/env python3
"""Extract a tensor-type config from an existing GGUF file.

Reads the GGUF header to get each tensor's quantization type and outputs a
config file compatible with llama-quantize's --tensor-type-file.

Output format is auto-detected:
  - MoE (has ffn_*_exps tensors): unanchored patterns, no .weight suffix
  - dense/hybrid: anchored patterns with .weight suffix

Usage:
  ./scripts/grab_config.py model.gguf
  ./scripts/grab_config.py model.gguf -o grabbed_config.txt
  ./scripts/grab_config.py --format dense model.gguf
  ./scripts/grab_config.py --format moe model.gguf
"""

import argparse
import struct
import sys

GGUF_MAGIC = 0x46554747

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
    elif vtype == 8:
        _read_gguf_string(f)
    elif vtype == 9:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        for _ in range(arr_len):
            _read_gguf_value(f, arr_type)
    else:
        raise SystemExit(f"Unknown GGUF value type {vtype}")


def read_gguf_tensors(path):
    """Return list of (name, numel, qtype_str) from GGUF header."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise SystemExit(f"Not a GGUF file: {path}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise SystemExit(f"Unsupported GGUF version {version}")

        for _ in range(n_kv):
            _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            _read_gguf_value(f, vtype)

        tensors = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            type_id = struct.unpack("<I", f.read(4))[0]
            f.read(8)
            numel = 1
            for d in dims:
                numel *= d
            qtype = GGML_TYPE.get(type_id, f"?{type_id}")
            tensors.append((name, numel, qtype))
    return tensors


def format_name_moe(name):
    """Convert GGUF tensor name to MoE config pattern (unanchored, no .weight)."""
    if name.endswith(".weight"):
        name = name[:-7]
    return name


def format_name_dense(name):
    """Convert GGUF tensor name to dense config pattern (anchored)."""
    escaped = name.replace(".", r"\.")
    return f"^{escaped}$"


def build_config(tensors, fmt):
    """Build config lines from tensor list.

    Skips imatrix metadata (.in_sum2, .counts) and non-weight tensors.
    Includes F32 tensors (norms, conv1d, biases) — they are real tensors
    that the user may want to see or override.
    """
    lines = []
    format_fn = format_name_moe if fmt == "moe" else format_name_dense

    for name, numel, qtype in tensors:
        if name.endswith(".in_sum2") or name.endswith(".counts"):
            continue
        if not name.endswith(".weight"):
            continue
        pattern = format_fn(name)
        lines.append(f"{pattern}={qtype}")

    return lines


def main():
    p = argparse.ArgumentParser(
        description="Extract tensor-type config from an existing GGUF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input GGUF file")
    p.add_argument("--output", "-o", help="Write config to file instead of stdout")
    p.add_argument("--format", "-f", choices=["moe", "dense"],
                   default="moe",
                   help="Output format: moe=unanchored, dense=anchored regex (default: moe)")
    args = p.parse_args()

    tensors = read_gguf_tensors(args.input)
    fmt = args.format

    lines = build_config(tensors, fmt)

    output = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Config written to: {args.output} ({len(lines)} lines, "
              f"format={fmt})", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
