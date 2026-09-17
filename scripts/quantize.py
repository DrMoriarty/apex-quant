#!/usr/bin/env python3
"""APEX quantization for llama.cpp (Python version of quantize.sh).

Usage:
  # Using a built-in profile
  ./scripts/quantize.py --profile balanced input.gguf output.gguf

  # Using a custom tensor-type file
  ./scripts/quantize.py --config configs/my_config.txt input.gguf output.gguf

  # With imatrix (for I-variants and Mini)
  ./scripts/quantize.py --profile mini --imatrix imatrix.dat input.gguf output.gguf

  # Generate config only (no quantization)
  ./scripts/quantize.py --profile quality --generate-config -o config.txt

  # Estimate output size (no quantization)
  ./scripts/quantize.py --profile balanced --dry-run input.gguf

Profiles: quality, i-quality, balanced, i-balanced, compact, i-compact, mini, custom

Environment:
  LLAMA_QUANTIZE    Path to llama-quantize binary (auto-detected)
  LLAMA_CPP_DIR     Path to llama.cpp build/bin directory
  NUM_LAYERS        Number of transformer layers (default: 40)
"""

import argparse
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_quantize():
    """Find llama-quantize binary."""
    # Check LLAMA_QUANTIZE env
    q = os.environ.get("LLAMA_QUANTIZE", "")
    if q and os.path.isfile(q):
        return q

    # Check LLAMA_CPP_DIR
    d = os.environ.get("LLAMA_CPP_DIR", "")
    if d:
        p = os.path.join(d, "llama-quantize")
        if os.path.isfile(p):
            return p

    # Check relative paths
    candidates = [
        "./llama.cpp/build/bin",
        os.path.join(SCRIPT_DIR, "..", "llama.cpp", "build", "bin"),
    ]
    for d in candidates:
        p = os.path.join(d, "llama-quantize")
        if os.path.isfile(p):
            return p

    # Check PATH
    try:
        subprocess.check_output(["command", "-v", "llama-quantize"], shell=True)
        return "llama-quantize"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def main():
    parser = argparse.ArgumentParser(
        description="APEX quantization for llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", "-p", default="balanced",
                        help="Profile name (default: balanced)")
    parser.add_argument("--config", "-c",
                        help="Custom tensor-type file")
    parser.add_argument("--imatrix", "-i",
                        help="Importance matrix file")
    parser.add_argument("--base-type", "-b", default="Q6_K",
                        help="Base quant type (default: Q6_K)")
    parser.add_argument("--layers", "-l", type=int,
                        default=int(os.environ.get("NUM_LAYERS", 40)),
                        help="Number of transformer layers (default: 40)")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate config only (no quantization)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate output size without quantizing")
    parser.add_argument("-o", "--output",
                        help="Output config file (for --generate-config)")
    parser.add_argument("input", nargs="?", help="Input GGUF file")
    parser.add_argument("output_file", nargs="?", help="Output GGUF file")

    args = parser.parse_args()

    # Base type per profile
    base_type_map = {
        "quality": "Q6_K", "i-quality": "Q6_K",
        "balanced": "Q6_K", "i-balanced": "Q6_K",
        "compact": "Q4_K_M", "i-compact": "Q4_K_M",
        "mini": "Q3_K_M",
    }
    args.base_type = base_type_map.get(args.profile, args.base_type)

    # I-profiles warn about missing imatrix
    if args.profile in ("i-quality", "i-balanced", "i-compact", "mini"):
        if not args.imatrix:
            print(f"WARNING: Profile '{args.profile}' benefits from --imatrix. Continuing without it.",
                  file=sys.stderr)

    # Generate config
    config_file = None
    tmpfile = None

    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config_file = args.config
        print(f">>> Using config: {config_file}")
    elif args.generate_config:
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", args.profile, "--layers", str(args.layers)]
        if args.output:
            cmd.extend(["-o", args.output])
        subprocess.run(cmd, check=True)
        sys.exit(0)
    else:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmpfile.close()
        config_file = tmpfile.name
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", args.profile, "--layers", str(args.layers),
               "-o", config_file]
        subprocess.run(cmd, check=True)
        print(f">>> Generated config for profile '{args.profile}' ({args.layers} layers)")

    if args.generate_config:
        sys.exit(0)

    # Need at least input
    if not args.input:
        parser.error("Input GGUF file is required")

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Dry-run: estimate size without quantizing
    if args.dry_run:
        est_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "estimate_size.py"),
                   "--profile", args.profile, "--layers", str(args.layers),
                   "--base-type", args.base_type, args.input]
        if config_file:
            est_cmd.extend(["--config", config_file])
        try:
            subprocess.run(est_cmd, check=True)
        finally:
            if tmpfile:
                os.unlink(tmpfile.name)
        sys.exit(0)

    # Need output for actual quantization
    if not args.output_file:
        parser.error("Output GGUF file is required (or use --dry-run / --generate-config)")

    # Find llama-quantize
    quantize_bin = find_quantize()
    if not quantize_bin:
        print("ERROR: llama-quantize not found. Set LLAMA_QUANTIZE or LLAMA_CPP_DIR.",
              file=sys.stderr)
        sys.exit(1)

    # Build quantize command
    quant_cmd = [quantize_bin, "--tensor-type-file", config_file]
    if args.imatrix:
        quant_cmd.extend(["--imatrix", args.imatrix])

    print("=== APEX Quantize ===")
    print(f"Profile:    {args.profile}")
    print(f"Base type:  {args.base_type}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output_file}")
    if args.imatrix:
        print(f"Imatrix:    {args.imatrix}")
    with open(config_file) as f:
        lines = len(f.readlines())
    print(f"Config:     {config_file} ({lines} lines)")
    print()

    try:
        subprocess.run(quant_cmd + [args.input, args.output_file, args.base_type], check=True)
    finally:
        if tmpfile:
            os.unlink(tmpfile.name)

    size = os.path.getsize(args.output_file)
    print(f"\n>>> Done: {size / (1024*1024):.1f} MB -> {args.output_file}")


if __name__ == "__main__":
    main()
