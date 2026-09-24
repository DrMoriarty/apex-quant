#!/usr/bin/env python3
"""APEX quantization for llama.cpp.

Usage:
  # Using a built-in tier profile
  ./scripts/quantize.py --profile tier5 input.gguf output.gguf

  # Using a custom tensor-type file
  ./scripts/quantize.py --config configs/my_config.txt input.gguf output.gguf

  # With imatrix (for i-tierN variants)
  ./scripts/quantize.py --profile i-tier5 --imatrix imatrix.dat input.gguf output.gguf

  # Generate config only (no quantization)
  ./scripts/quantize.py --profile tier1 --generate-config -o config.txt

  # Estimate output size (no quantization)
  ./scripts/quantize.py --profile tier5 --dry-run input.gguf

Profiles:
  tier1–tier13   MoE tier profiles (tier1 = near-full precision, tier13 = IQ1/IQ2 band)
  i-tierN        Same as tierN, for use with --imatrix
  dense-*        Dense/hybrid profiles: dense-flat, dense-grad, dense-hybrid,
                 dense-hybrid-quality

Architecture (--arch moe|dense) and layer count are auto-detected from the GGUF
file; can be overridden via --arch, --layers, --dense-layers.

Environment:
  LLAMA_QUANTIZE    Path to llama-quantize binary (auto-detected)
  LLAMA_CPP_DIR     Path to llama.cpp build/bin directory
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")


def load_dotenv(path=None):
    """Load .env file into os.environ (without overwriting existing vars)."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()


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
        description="APEX quantization for llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", "-p", default="tier1",
                        help="Profile name (default: tier1)")
    parser.add_argument("--config", "-c",
                        help="Custom tensor-type file")
    parser.add_argument("--imatrix", "-i",
                        help="Importance matrix file")
    parser.add_argument("--base-type", "-b", default="Q8_0",
                        help="Base quant type (default: Q8_0)")
    parser.add_argument("--layers", "-l", type=int, default=None,
                        help="Number of transformer layers (default: auto-detect from GGUF)")
    parser.add_argument("--dense-layers", type=int, default=0,
                        help="Leading dense (non-MoE) FFN layers (default: 0)")
    parser.add_argument("--arch", choices=["moe", "dense"], default=None,
                        help="Architecture: moe or dense (default: auto-detect)")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate config only (no quantization)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate output size without quantizing")
    quant_mode = parser.add_mutually_exclusive_group()
    quant_mode.add_argument("--speed", action="store_const", dest="quant_mode", const="speed",
                            help="Use QUANTS_RANKED_SPEED (Q4_K/Q3_K/Q2_K) for all tensors")
    quant_mode.add_argument("--size", action="store_const", dest="quant_mode", const="size",
                            help="Use QUANTS_RANKED_SIZE (IQ4_NL/IQ3_S/IQ2_S) for all tensors")
    quant_mode.add_argument("--mixed", action="store_const", dest="quant_mode", const="mixed",
                            help="Experts use QUANTS_RANKED_SIZE, everything else QUANTS_RANKED_SPEED")
    parser.add_argument("-o", "--output",
                        help="Output config file (for --generate-config)")
    parser.add_argument("input", nargs="?", help="Input GGUF file")
    parser.add_argument("output_file", nargs="?", help="Output GGUF file")

    args = parser.parse_args()

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
        "tier10": "Q5_K_M", 
        "tier11": "Q5_K_M", 
        "tier12": "Q5_K_M", 
        "tier13": "Q4_K_M", 
    }
    args.base_type = base_type_map.get(args.profile, args.base_type)

    # Auto-detect parameters if not provided
    if args.input and os.path.isfile(args.input):
        detected = detect_gguf_params(args.input)
        if detected:
            if args.layers is None:
                args.layers = detected["layers"]
            if args.dense_layers == 0:
                args.dense_layers = detected["dense_layers"]
            if not args.arch:
                args.arch = detected["arch"]
            print(f">>> Detected from GGUF: arch={detected['arch']}, layers={detected['layers']}, dense_layers={detected['dense_layers']}")

    if args.layers is None:
        args.layers = 40

    # Generate config only mode
    if args.generate_config:
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", args.profile, "--layers", str(args.layers)]
        if args.dense_layers:
            cmd.extend(["--dense-layers", str(args.dense_layers)])
        if args.arch:
            cmd.extend(["--arch", args.arch])
        if args.quant_mode:
            cmd.append(f"--{args.quant_mode}")
        if args.output:
            cmd.extend(["-o", args.output])
        subprocess.run(cmd, check=True)
        sys.exit(0)

    # Need at least input for quantization / dry-run
    if not args.input:
        parser.error("Input GGUF file is required (or use --generate-config)")

    # Generate config
    config_file = None
    tmpfile = None

    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config_file = args.config
        print(f">>> Using config: {config_file}")
    else:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmpfile.close()
        config_file = tmpfile.name
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_config.py"),
               "--profile", args.profile, "--layers", str(args.layers),
               "-o", config_file]
        if args.dense_layers:
            cmd.extend(["--dense-layers", str(args.dense_layers)])
        if args.arch:
            cmd.extend(["--arch", args.arch])
        if args.quant_mode:
            cmd.append(f"--{args.quant_mode}")
        subprocess.run(cmd, check=True)
        print(f">>> Generated config for profile '{args.profile}' ({args.layers} layers)")

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
        if args.quant_mode:
            est_cmd.append(f"--{args.quant_mode}")
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
