#!/usr/bin/env python3
"""Generate APEX tensor-type configuration files.

Creates a tensor-type file for llama-quantize's --tensor-type-file flag.
Supports any number of layers and all APEX profiles.

Usage:
  ./scripts/generate_config.py --profile tier5 --layers 40 > config.txt
  ./scripts/generate_config.py --profile tier7 --layers 40 -o configs/my_config.txt

MoE tier profiles (--arch moe, default):
  tier1–tier15  Quality-rank indices for each tensor role (experts, shared
                experts, attention, embeddings). The ranks step down from
                tier1 (near-full precision) to tier15 (IQ1/IQ2 band); the
                concrete quant per rank depends on the quant mode (below).
  i-tierN       Same as tierN (use with --imatrix at quantize time).

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

Quant mode (default: speed):
  Profiles assign quality-rank indices; the mode selects which ranked table
  maps each index to a concrete quant type.
  --speed  QUANTS_RANKED_SPEED (Q4_K/Q3_K/Q2_K) for all tensors.
  --size   QUANTS_RANKED_SIZE (IQ4_NL/IQ3_S/IQ2_S) for all tensors.
  --mixed  Experts use QUANTS_RANKED_SIZE, everything else QUANTS_RANKED_SPEED. Default.

Profile modifiers (used with profiles, positive = lower quality, negative = higher):
  --edge-exp N      Shift edge-expert quality by N steps
  --near-exp N      Shift near-expert quality by N steps
  --mid-exp N       Shift mid-expert quality by N steps
  --edge-shared N   Shift edge-shared quality by N steps
  --mid-shared N    Shift mid-shared quality by N steps
  --edge-attn N     Shift edge-attn quality by N steps
  --mid-attn N      Shift mid-attn quality by N steps
  --embd-type N     Shift embedding quality by N steps

The three Q4-band profiles are deliberately size-matched: an A/B between
allocations is only interpretable if the arms are the same size.
"""

import argparse
import math
import sys

# Quantization types sorted by quality descending (index = quality rank).
# Use ranked_quant(index) to convert an index to a quant string with clamping.
QUANTS_RANKED_SIZE = [
    "Q8_0",       # 0
    "Q6_K",       # 1
    "Q5_K",       # 2
    "IQ4_NL",     # 3
    "IQ3_S",      # 4
    "IQ2_S",      # 5
    "IQ1_M",      # 6
]

QUANTS_RANKED_SPEED = [
    "Q8_0",       # 0
    "Q6_K",       # 1
    "Q5_K",       # 2
    "Q4_K",       # 3
    "Q3_K",       # 4
    "Q2_K",       # 5
    "IQ1_S",      # 6
]


_quant_mode = "mixed"  # "mixed" | "speed" | "size"


def ranked_quant(index, role="expert"):
    """Return quant string for a quality-rank index, clamping to valid range.

    *role* selects the quant table when mode is "mixed":
      "expert"  → QUANTS_RANKED_SIZE
      "shared"  → QUANTS_RANKED_SPEED
      "attn"    → QUANTS_RANKED_SPEED
      "embd"    → QUANTS_RANKED_SPEED
    When mode is "speed" or "size" the corresponding table is used regardless.
    """
    if _quant_mode == "speed":
        quants = QUANTS_RANKED_SPEED
    elif _quant_mode == "size":
        quants = QUANTS_RANKED_SIZE
    else: # mixed mode
        quants = QUANTS_RANKED_SIZE if role == "expert" else QUANTS_RANKED_SPEED
    if index < 0:
        return quants[0]
    if index >= len(quants):
        return quants[-1]
    return quants[index]


# Profile definitions: (edge_exp, near_exp, mid_exp, edge_shared, mid_shared, edge_attn, mid_attn, embd_type)
# Values are quality-rank indices into the ranked quant tables (0=Q8_0, 1=Q6_K, 2=Q5_K, ...);
# the table is chosen by the quant mode (see ranked_quant).
PROFILES = {
    "tier1":        (0, 0, 1, 0, 0, 0, 0, 0),
    "tier2":        (1, 1, 1, 0, 0, 0, 0, 0),
    "tier3":        (1, 1, 2, 0, 0, 0, 0, 0),
    "tier4":        (2, 2, 2, 0, 0, 1, 1, 0),
    "tier5":        (2, 2, 3, 0, 0, 1, 1, 0),
    "tier6":        (2, 3, 3, 0, 1, 1, 1, 0),
    "tier7":        (3, 3, 3, 0, 1, 2, 2, 1),
    "tier8":        (3, 3, 4, 0, 1, 2, 2, 1),
    "tier9":        (3, 4, 4, 0, 1, 2, 2, 1),
    "tier10":       (4, 4, 4, 0, 1, 3, 3, 2),
    "tier11":       (4, 4, 5, 0, 1, 3, 3, 2),
    "tier12":       (4, 5, 5, 0, 1, 3, 3, 2),
    "tier13":       (5, 5, 5, 0, 1, 4, 4, 3),
    "tier14":       (5, 5, 6, 0, 1, 4, 4, 3),
    "tier15":       (5, 6, 6, 0, 1, 4, 4, 3),
}

DENSE_PROFILES = {"dense-flat", "dense-grad", "dense-hybrid", "dense-hybrid-quality"}
MOE_PROFILES = set(PROFILES.keys())
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
    quant_mode = p.add_mutually_exclusive_group()
    quant_mode.add_argument("--speed", action="store_const", dest="quant_mode", const="speed",
                            help="Use QUANTS_RANKED_SPEED (Q4_K/Q3_K/Q2_K) for all tensors")
    quant_mode.add_argument("--size", action="store_const", dest="quant_mode", const="size",
                            help="Use QUANTS_RANKED_SIZE (IQ4_NL/IQ3_S/IQ2_S) for all tensors")
    quant_mode.add_argument("--mixed", action="store_const", dest="quant_mode", const="mixed",
                            help="Use mixed quants.")

    # Profile modifiers (integers)
    p.add_argument("--edge-exp", type=int, default=None,
                   help="Modifier for edge-expert quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--near-exp", type=int, default=None,
                   help="Modifier for near-expert quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--mid-exp", type=int, default=None,
                   help="Modifier for mid-expert quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--edge-shared", type=int, default=None,
                   help="Modifier for edge-shared quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--mid-shared", type=int, default=None,
                   help="Modifier for mid-shared quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--edge-attn", type=int, default=None,
                   help="Modifier for edge-attn quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--mid-attn", type=int, default=None,
                   help="Modifier for mid-attn quality (-N/+N shifts quality, default: 0)")
    p.add_argument("--embd-type", dest="embd_type_moe", type=int, default=None,
                   help="Modifier for embedding quality (-N/+N shifts quality, default: 0)")

    # Dense/hybrid overrides
    p.add_argument("--linattn", default="")
    p.add_argument("--fullattn", default="")
    p.add_argument("--embd", default="")
    p.add_argument("--output-type", dest="output_type", default="")

    args = p.parse_args(argv)

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
        indices = list(PROFILES[lookup])
        
        # Apply modifiers from CLI arguments (positive = lower quality, negative = higher quality)
        # Indices: 0=edge_exp, 1=near_exp, 2=mid_exp, 3=edge_shared, 4=mid_shared, 5=edge_attn, 6=mid_attn, 7=embd_type
        if args.edge_exp is not None:
            indices[0] += args.edge_exp
        if args.near_exp is not None:
            indices[1] += args.near_exp
        if args.mid_exp is not None:
            indices[2] += args.mid_exp
        if args.edge_shared is not None:
            indices[3] += args.edge_shared
        if args.mid_shared is not None:
            indices[4] += args.mid_shared
        if args.edge_attn is not None:
            indices[5] += args.edge_attn
        if args.mid_attn is not None:
            indices[6] += args.mid_attn
        if args.embd_type_moe is not None:
            indices[7] += args.embd_type_moe
        
        return {
            "arch": arch,
            "layers": layers,
            "dense_layers": dense_layers,
            "profile": profile,
            "indices": tuple(indices),
        }

    if profile in DENSE_PROFILES:
        return resolve_dense_profile(profile, layers, args)

    available = ", ".join(sorted(MOE_PROFILES))
    dense_avail = ", ".join(sorted(DENSE_PROFILES))
    print("Error: unknown profile '{}'".format(profile), file=sys.stderr)
    print("Available: {}".format(available), file=sys.stderr)
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
    zone_size = max(1, math.ceil(non_dense * 0.1))  # edge/near = 10% of layers per side

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
    ei, ni, mi, esi, msi, eai, mai, emi = cfg["indices"]

    lines.append(f"token_embd.weight={ranked_quant(emi, 'embd')}")
    lines.append(f"output.weight={ranked_quant(emi, 'embd')}")

    for i in range(layers):
        zone = get_zone(i, layers, dense_layers)

        if zone == "edge":
            exp = ei
            sh = esi
        elif zone == "near":
            exp = ni
            sh = msi
        else:
            exp = mi
            sh = msi

        # Attention index: narrower edge band (7.5% per side vs 10% for zones)
        non_dense = layers - dense_layers
        attn_edge_size = max(1, math.ceil(non_dense * 3 / 40))
        if i < dense_layers + attn_edge_size or i >= layers - attn_edge_size:
            ai = eai
        else:
            ai = mai

        # Expert or dense FFN tensors
        if i < dense_layers:
            lines.append(f"blk.{i}.ffn_gate.weight={ranked_quant(sh, 'shared')}")
            lines.append(f"blk.{i}.ffn_up.weight={ranked_quant(sh, 'shared')}")
            lines.append(f"blk.{i}.ffn_down.weight={ranked_quant(sh - 1, 'shared')}")  # -1
        else:
            lines.append(f"blk.{i}.ffn_gate_exps={ranked_quant(exp, 'expert')}")
            lines.append(f"blk.{i}.ffn_up_exps={ranked_quant(exp, 'expert')}")
            lines.append(f"blk.{i}.ffn_down_exps={ranked_quant(exp - 1, 'expert')}")  # -1

        # Shared expert tensors
        lines.append(f"blk.{i}.ffn_gate_shexp={ranked_quant(sh, 'shared')}")
        lines.append(f"blk.{i}.ffn_up_shexp={ranked_quant(sh, 'shared')}")
        lines.append(f"blk.{i}.ffn_down_shexp={ranked_quant(sh - 1, 'shared')}")  # -1

        # Attention tensors
        lines.append(f"blk.{i}.attn_q={ranked_quant(ai, 'attn')}")
        lines.append(f"blk.{i}.attn_k={ranked_quant(ai, 'attn')}")
        lines.append(f"blk.{i}.attn_v={ranked_quant(ai - 2, 'attn')}")
        lines.append(f"blk.{i}.attn_output={ranked_quant(ai - 1, 'attn')}")
        lines.append(f"blk.{i}.attn_gate={ranked_quant(ai - 2, 'attn')}")
        lines.append(f"blk.{i}.attn_qkv={ranked_quant(ai - 2, 'attn')}")

        # Short-convolution mixing tensors (LFM2 conv layers)
        lines.append(f"blk.{i}.shortconv.in_proj={ranked_quant(ai, 'attn')}")
        lines.append(f"blk.{i}.shortconv.out_proj={ranked_quant(ai - 1, 'attn')}")

        # SSM tensors (Mamba/hybrid archs)
        lines.append(f"blk.{i}.ssm_alpha={ranked_quant(ai, 'attn')}")
        lines.append(f"blk.{i}.ssm_beta={ranked_quant(ai, 'attn')}")
        lines.append(f"blk.{i}.ssm_out={ranked_quant(ai, 'attn')}")

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
    global _quant_mode
    args = parse_args(argv)
    _quant_mode = args.quant_mode or "mixed"
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
