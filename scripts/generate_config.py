#!/usr/bin/env python3
"""Generate APEX tensor-type configuration files.

Creates a tensor-type file for llama-quantize's --tensor-type-file flag.
Supports any number of layers and all APEX profiles.

Usage:
  ./scripts/generate_config.py --profile balanced --layers 40 > config.txt
  ./scripts/generate_config.py --profile mini --layers 40 -o configs/my_config.txt
  ./scripts/generate_config.py --custom --edge-exp Q6_K --mid-exp Q4_K \
      --shared Q8_0 --attn Q6_K --layers 40 > config.txt

Profiles:
  quality     Q6_K/Q5_K/IQ4_XS experts, Q8_0 shared, Q6_K attn
  i-quality   Same as quality (use with --imatrix at quantize time)
  balanced    Q6_K/Q5_K experts, Q8_0 shared, Q6_K attn
  i-balanced  Same as balanced (use with --imatrix at quantize time)
  compact     Q4_K/Q3_K experts, Q6_K shared, Q4_K attn
  i-compact   Same as compact (use with --imatrix at quantize time)
  mini        Q3_K edge / IQ2_S mid experts, Q5_K/Q4_K shared, Q4_K/Q3_K attn
  nano        Q3_K edge / IQ2_S near / IQ2_XXS mid experts (2.06 bpw mid) — needs imatrix
  micro       Q3_K edge / IQ2_XS near / IQ1_M mid experts (1.75 bpw mid) — needs imatrix, experimental
  tq-quality  quality + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
  tq-balanced balanced + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
  tq-compact  compact + TurboQuant attention (tq4_1s) in mid layers — uses wide attention bounds
  tq-mini     mini + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
  tq-nano     nano + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
  tq-micro    micro + TurboQuant attention (TQ3_1S) in mid layers — needs imatrix, no i-variant
  custom      Specify each type manually via flags

Dense/hybrid profiles (--arch dense is implied; for models whose FFN is dense,
e.g. Qwen3.8-27B: 64 layers, 48 linear-attention + 16 full-attention):
  dense-flat            Control. Flat per-role allocation, NO layer gradient —
                        replicates what the shelf dynamic quants actually do.
  dense-grad            dense-flat + FFN layer-position gradient. Isolates that
                        one lever; attention is left identical to the control.
  dense-hybrid          dense-grad + full-attn pinned up / linear-attn cut.
                        Full-attn is only ~6% of params, linear-attn ~20%.
  dense-hybrid-quality  dense-hybrid rebuilt in the Q5/Q6 band, to check the
                        winning allocation still wins away from the Q4 band.

The three Q4-band profiles are deliberately size-matched: an A/B between
allocations is only interpretable if the arms are the same size.
"""

import argparse
import math
import sys

# Profile definitions: (edge_exp, near_exp, mid_exp, edge_shared, mid_shared, edge_attn, mid_attn, embd_type)
PROFILES = {
    "balanced":     ("Q6_K",   "Q5_K",   "Q5_K",   "Q8_0", "Q8_0", "Q6_K", "Q6_K",   "Q8_0"),
    "quality":      ("Q6_K",   "Q5_K",   "iq4_xs", "Q8_0", "Q8_0", "Q6_K", "Q6_K",   "Q8_0"),
    "compact":      ("Q4_K",   "Q3_K",   "Q3_K",   "Q6_K", "Q6_K", "Q4_K", "Q4_K",   "Q8_0"),
    "mini":         ("Q3_K",   "Q3_K",   "iq2_s",  "Q5_K", "Q4_K", "Q4_K", "Q3_K",   "Q8_0"),
    "nano":         ("Q3_K",   "iq2_s",  "iq2_xxs","Q5_K", "Q4_K", "Q4_K", "Q3_K",   "Q8_0"),
    "micro":        ("Q3_K",   "iq2_xs", "iq1_m",  "Q5_K", "Q4_K", "Q4_K", "Q3_K",   "Q8_0"),

    "tier1":        ("Q8_0",   "Q6_K",   "Q5_K",   "Q8_0", "Q8_0", "Q8_0", "Q8_0",   "Q8_0"),
    "tier2":        ("Q6_K",   "Q6_K",   "Q4_K",   "Q8_0", "Q8_0", "Q8_0", "Q8_0",   "Q8_0"),
    "tier3":        ("Q6_K",   "Q5_K",   "Q4_K",   "Q8_0", "Q8_0", "Q8_0", "Q6_K",   "Q6_K"),
    "tier4":        ("Q5_K",   "Q4_K",   "Q3_K",   "Q8_K", "Q6_K", "Q6_K", "Q5_K",   "Q6_K"),

    "tier5":        ("Q4_K",   "Q3_K",   "Q2_K",   "Q8_0", "Q6_K", "Q8_0", "Q6_K",   "Q8_0"),
    "tier6":        ("Q3_K",   "Q2_K",   "Q2_K",   "Q8_0", "Q6_K", "Q8_0", "Q6_K",   "Q8_0"),


    "tq-balanced":  ("Q6_K",   "Q5_K",   "Q5_K",   "Q8_0", "Q8_0", "Q6_K", "tq4_1s", "Q8_0"),
    "tq-quality":   ("Q6_K",   "Q5_K",   "iq4_xs", "Q8_0", "Q8_0", "Q6_K", "tq4_1s", "Q8_0"),
    "tq-compact":   ("Q4_K",   "Q3_K",   "Q3_K",   "Q6_K", "Q6_K", "Q4_K", "tq4_1s", "Q8_0"),
    "tq-mini":      ("Q3_K",   "Q3_K",   "iq2_s",  "Q5_K", "Q4_K", "Q4_K", "TQ3_1S", "Q8_0"),
    "tq-nano":      ("Q3_K",   "iq2_s",  "iq2_xxs","Q5_K", "Q4_K", "Q4_K", "TQ3_1S", "Q8_0"),
    "tq-micro":     ("Q3_K",   "iq2_xs", "iq1_m",  "Q5_K", "Q4_K", "Q4_K", "TQ3_1S", "Q8_0"),
}

TQ_PROFILES = {
    "tq-quality", "tq-balanced", "tq-compact", "tq-mini", "tq-nano", "tq-micro",
}

DENSE_PROFILES = {"dense-flat", "dense-grad", "dense-hybrid", "dense-hybrid-quality"}
MOE_PROFILES = set(PROFILES.keys()) | {"custom"}
ALL_PROFILES = MOE_PROFILES | DENSE_PROFILES


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate APEX tensor-type configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--profile", "-p", required=True,
                   help="Profile name (see module docstring)")
    p.add_argument("--layers", "-l", type=int, default=40,
                   help="Number of transformer layers (default: 40)")
    p.add_argument("--dense-layers", type=int, default=0,
                   help="Leading dense (non-MoE) FFN layers (default: 0)")
    p.add_argument("--arch", default="moe", choices=["moe", "dense"],
                   help="Architecture: moe or dense (default: moe)")
    p.add_argument("--output", "-o",
                   help="Write config to file instead of stdout")
    p.add_argument("--custom", action="store_true",
                   help="Use custom mode (--edge-exp required)")

    # Custom mode types
    p.add_argument("--edge-exp", default="")
    p.add_argument("--near-exp", default="")
    p.add_argument("--mid-exp", default="")
    p.add_argument("--edge-shared", default="")
    p.add_argument("--mid-shared", default="")
    p.add_argument("--edge-attn", default="")
    p.add_argument("--mid-attn", default="")
    p.add_argument("--embd-type", dest="embd_type_moe", default="",
                   help="Embedding/output tensor type for MoE profiles (default: Q8_0)")

    # Dense/hybrid overrides
    p.add_argument("--linattn", default="")
    p.add_argument("--fullattn", default="")
    p.add_argument("--embd", default="")
    p.add_argument("--output-type", dest="output_type", default="")

    args = p.parse_args(argv)

    if args.custom:
        args.profile = "custom"

    return args


def resolve_profile(args):
    """Return (arch, layers, dense_layers, types_dict) after applying profile defaults."""
    profile = args.profile
    layers = args.layers
    dense_layers = args.dense_layers
    arch = args.arch

    # Resolve profile-level arch override
    if profile in DENSE_PROFILES:
        arch = "dense"

    # Validate arch vs dense-layers
    if arch == "dense" and dense_layers != 0:
        print("Error: --arch dense is incompatible with --dense-layers ({}).".format(dense_layers),
              file=sys.stderr)
        print("       --dense-layers counts leading dense FFN layers inside a MoE model;",
              file=sys.stderr)
        print("       --arch dense means the model has no expert layers at all.", file=sys.stderr)
        sys.exit(1)

    lookup = profile[2:] if profile.startswith("i-") else profile
    if lookup in PROFILES:
        edge_exp, near_exp, mid_exp, edge_shared, mid_shared, edge_attn, mid_attn, embd_type = PROFILES[lookup]
        types = {
            "edge_exp": args.edge_exp or edge_exp,
            "near_exp": args.near_exp or near_exp,
            "mid_exp": args.mid_exp or mid_exp,
            "edge_shared": args.edge_shared or edge_shared,
            "mid_shared": args.mid_shared or mid_shared,
            "edge_attn": args.edge_attn or edge_attn,
            "mid_attn": args.mid_attn or mid_attn,
            "embd_type": getattr(args, "embd_type_moe", "") or embd_type,
        }
        attn_wide = lookup in TQ_PROFILES
        return {
            "arch": arch,
            "layers": layers,
            "dense_layers": dense_layers,
            "attn_wide": attn_wide,
            "profile": profile,
            "types": types,
        }

    if profile in DENSE_PROFILES:
        return resolve_dense_profile(profile, layers, args)

    if profile == "custom":
        if not args.edge_exp:
            print("Error: --custom requires --edge-exp", file=sys.stderr)
            sys.exit(1)
        types = {
            "edge_exp": args.edge_exp,
            "near_exp": args.near_exp or args.edge_exp,
            "mid_exp": args.mid_exp or args.edge_exp,
            "edge_shared": args.edge_shared or "Q8_0",
            "mid_shared": args.mid_shared or (args.edge_shared or "Q8_0"),
            "edge_attn": args.edge_attn or "Q6_K",
            "mid_attn": args.mid_attn or (args.edge_attn or "Q6_K"),
            "embd_type": getattr(args, "embd_type_moe", "") or "Q8_0",
        }
        return {
            "arch": arch,
            "layers": layers,
            "dense_layers": dense_layers,
            "attn_wide": False,
            "profile": "custom",
            "types": types,
        }

    available = ", ".join(sorted(MOE_PROFILES - {"custom"}))
    dense_avail = ", ".join(sorted(DENSE_PROFILES))
    print("Error: unknown profile '{}'".format(profile), file=sys.stderr)
    print("Available: {}, custom".format(available), file=sys.stderr)
    print("Dense:     {}".format(dense_avail), file=sys.stderr)
    sys.exit(1)


def resolve_dense_profile(profile, layers, args):
    """Resolve dense/hybrid profile to a config dict."""
    # Defaults shared across the three Q4-band arms
    linattn = args.linattn or "Q5_K"
    linattn_small = "Q4_K"  # no CLI override in shell; always Q4_K
    fullattn = args.fullattn or "Q5_K"
    fullattn_v = "Q6_K"
    embd_type = args.embd or "Q4_K"
    output_type = args.output_type or "Q6_K"

    # FFN types per arm
    if profile == "dense-flat":
        ffn = {
            "edge_gate": "IQ4_XS", "near_gate": "IQ4_XS", "mid_gate": "IQ4_XS",
            "edge_up":   "Q5_K",   "near_up":   "Q5_K",   "mid_up":   "Q5_K",
            "edge_down": "Q5_K",   "near_down": "Q5_K",   "mid_down": "Q5_K",
        }
    elif profile in ("dense-grad", "dense-hybrid"):
        ffn = {
            "edge_gate": "Q5_K",   "near_gate": "Q4_K",   "mid_gate": "IQ4_XS",
            "edge_up":   "Q6_K",   "near_up":   "Q5_K",   "mid_up":   "Q4_K",
            "edge_down": "Q6_K",   "near_down": "Q5_K",   "mid_down": "Q5_K",
        }
    elif profile == "dense-hybrid-quality":
        ffn = {
            "edge_gate": "Q6_K",   "near_gate": "Q5_K",   "mid_gate": "Q5_K",
            "edge_up":   "Q6_K",   "near_up":   "Q6_K",   "mid_up":   "Q5_K",
            "edge_down": "Q8_0",   "near_down": "Q6_K",   "mid_down": "Q6_K",
        }

    if profile in ("dense-hybrid", "dense-hybrid-quality"):
        fullattn = "Q8_0"
        fullattn_v = "Q8_0"
        linattn = "Q4_K"
        linattn_small = "Q4_K"

    return {
        "arch": "dense",
        "layers": layers,
        "dense_layers": 0,
        "attn_wide": False,
        "profile": profile,
        "ffn": ffn,
        "linattn": linattn,
        "linattn_small": linattn_small,
        "fullattn": fullattn,
        "fullattn_v": fullattn_v,
        "embd_type": embd_type,
        "output_type": output_type,
    }


def get_zone(i, layers, dense_layers=0):
    """Return zone name for layer index: 'edge', 'near', or 'mid'.

    Dense layers (indices 0..dense_layers-1) are always 'edge'.
    The remaining layers are split into edge/near/mid with ceil rounding
    for edge and near per-side sizes.
    """
    non_dense = layers - dense_layers
    zone_size = max(1, math.ceil(non_dense * 5 / 40))

    # Dense layers are always edge
    if i < dense_layers:
        return "edge"

    # Virtual index within non-dense layers
    j = i - dense_layers
    edge_hi = zone_size - 1
    edge_lo = non_dense - zone_size
    near_hi = 2 * zone_size - 1
    near_lo = non_dense - 2 * zone_size

    if j <= edge_hi or j >= edge_lo:
        return "edge"
    elif j <= near_hi or j >= near_lo:
        return "near"
    else:
        return "mid"


def generate_moe(cfg):
    """Generate lines for MoE (and hybrid with dense head) architectures."""
    lines = []
    layers = cfg["layers"]
    dense_layers = cfg["dense_layers"]
    types = cfg["types"]
    attn_wide = cfg["attn_wide"]
    embd_type = types["embd_type"]

    lines.append(f"token_embd.weight={embd_type}")
    lines.append(f"output.weight={embd_type}")
    
    for i in range(layers):
        zone = get_zone(i, layers, dense_layers)

        # Expert type
        exp_type = types[f"{zone}_exp"]

        # Shared type: edge uses edge_shared, near and mid use mid_shared
        if zone == "edge":
            shared_type = types["edge_shared"]
        else:
            shared_type = types["mid_shared"]

        # Attention type
        if attn_wide:
            if zone == "edge":
                attn_type = types["edge_attn"]
            else:
                attn_type = types["mid_attn"]
        else:
            non_dense = layers - dense_layers
            attn_edge_size = max(1, math.ceil(non_dense * 3 / 40))
            if i < dense_layers + attn_edge_size or i >= layers - attn_edge_size:
                attn_type = types["edge_attn"]
            else:
                attn_type = types["mid_attn"]

        # Expert or dense FFN tensors
        if i < dense_layers:
            lines.append(f"blk.{i}.ffn_gate.weight={shared_type}")
            lines.append(f"blk.{i}.ffn_up.weight={shared_type}")
            lines.append(f"blk.{i}.ffn_down.weight={shared_type}")
        else:
            lines.append(f"blk.{i}.ffn_gate_exps={exp_type}")
            lines.append(f"blk.{i}.ffn_up_exps={exp_type}")
            lines.append(f"blk.{i}.ffn_down_exps={exp_type}")

        # Shared expert tensors
        lines.append(f"blk.{i}.ffn_gate_shexp={shared_type}")
        lines.append(f"blk.{i}.ffn_up_shexp={shared_type}")
        lines.append(f"blk.{i}.ffn_down_shexp={shared_type}")

        # Attention tensors
        lines.append(f"blk.{i}.attn_q={attn_type}")
        lines.append(f"blk.{i}.attn_k={attn_type}")
        lines.append(f"blk.{i}.attn_v={attn_type}")
        lines.append(f"blk.{i}.attn_output={attn_type}")
        lines.append(f"blk.{i}.attn_gate={attn_type}")
        lines.append(f"blk.{i}.attn_qkv={attn_type}")

        # Short-convolution mixing tensors (LFM2 conv layers)
        lines.append(f"blk.{i}.shortconv.in_proj={attn_type}")
        lines.append(f"blk.{i}.shortconv.out_proj={attn_type}")

        # SSM tensors (Mamba/hybrid archs)
        lines.append(f"blk.{i}.ssm_alpha={attn_type}")
        lines.append(f"blk.{i}.ssm_beta={attn_type}")
        lines.append(f"blk.{i}.ssm_out={attn_type}")

    return lines


def generate_dense(cfg):
    """Generate lines for dense / hybrid-attention architectures."""
    lines = []
    layers = cfg["layers"]
    ffn = cfg["ffn"]
    linattn = cfg["linattn"]
    linattn_small = cfg["linattn_small"]
    fullattn = cfg["fullattn"]
    fullattn_v = cfg["fullattn_v"]
    embd_type = cfg["embd_type"]
    output_type = cfg["output_type"]

    lines.append(f"^token_embd\\.weight$={embd_type}")
    lines.append(f"^output\\.weight$={output_type}")

    for i in range(layers):
        zone = get_zone(i, layers)

        gate = ffn[f"{zone}_gate"]
        up = ffn[f"{zone}_up"]
        down = ffn[f"{zone}_down"]

        lines.append(f"^blk\\.{i}\\.ffn_gate\\.weight$={gate}")
        lines.append(f"^blk\\.{i}\\.ffn_up\\.weight$={up}")
        lines.append(f"^blk\\.{i}\\.ffn_down\\.weight$={down}")

        # Full-attention layers
        lines.append(f"^blk\\.{i}\\.attn_q\\.weight$={fullattn}")
        lines.append(f"^blk\\.{i}\\.attn_k\\.weight$={fullattn}")
        lines.append(f"^blk\\.{i}\\.attn_v\\.weight$={fullattn_v}")
        lines.append(f"^blk\\.{i}\\.attn_output\\.weight$={fullattn}")

        # Linear-attention layers
        lines.append(f"^blk\\.{i}\\.attn_qkv\\.weight$={linattn}")
        lines.append(f"^blk\\.{i}\\.attn_gate\\.weight$={linattn}")
        lines.append(f"^blk\\.{i}\\.ssm_out\\.weight$={linattn}")
        lines.append(f"^blk\\.{i}\\.ssm_alpha\\.weight$={linattn_small}")
        lines.append(f"^blk\\.{i}\\.ssm_beta\\.weight$={linattn_small}")

    return lines


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_profile(args)

    if cfg["arch"] == "dense":
        lines = generate_dense(cfg)
    else:
        lines = generate_moe(cfg)

    output = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Config written to: {args.output} ({len(lines)} lines, "
              f"{cfg['layers']} layers, arch={cfg['arch']})", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
