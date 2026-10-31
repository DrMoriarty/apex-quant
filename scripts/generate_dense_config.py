#!/usr/bin/env python3
"""generate_dense_config.py — imatrix-importance-driven quant allocator for
dense / hybrid-attention models.

Produces a per-tensor quant assignment from the model's own imatrix and a
user-specified tier (tier1–tier15), replacing the hand-tuned static dense-*
profiles (scripts/generate_config.py --arch dense) as the primary dense path.
The static profiles remain as A/B controls.

Usage:
    ./scripts/generate_dense_config.py --model model.gguf --imatrix imatrix.gguf --tier tier13
    ./scripts/generate_dense_config.py --model model.gguf --imatrix imatrix.gguf --tier tier13 -o configs/my.txt
    ./scripts/generate_dense_config.py --inventory models/qwen38_27b_tensors.json \
        --imatrix imatrix.gguf --tier tier13 --dry-run
    # --inventory (name -> [numel, stays_f32]) replaces --model when no GGUF
    # is at hand; sizes are identical since only the header is read.

Algorithm
---------
1.  Read per-tensor importance from the imatrix GGUF. Both llama.cpp formats
    are supported: legacy ``imatrix.<tensor>`` values entries (``.counts``
    companions normalize, ``.in_sum2`` companions ignored) and the newer
    raw-sums format storing only ``<tensor>.in_sum2`` + ``<tensor>.counts``
    (no prefix; importance reconstructed as RMS = sqrt(in_sum2/counts)).
    Score = mean (or --stat p95) of the per-column values, aggregated
    numel-weighted per group.
2.  Group model tensors (source GGUF header or --inventory) into
    role x zone: ffn_gate/ffn_up/ffn_down x edge/near/mid (get_zone()
    convention, 10% per side edge, next 10% near), fullattn (q/k/output),
    attn_v, linattn_qkv/attn_gate, linattn_out (ssm_out), ssm_in, ssm_small
    (alpha/beta), token_embd, output.weight. MTP/NextN tensors are pinned F16
    (spec-decode acceptance collapses if the draft head is quantized hard —
    see scripts/apex_mtp_patch_configs.py).
3.  Map tier1–tier15 to a target bits-per-weight budget, calibrated to the
    MoE tier ladder (tier1 ≈ near-full precision … tier15 ≈ IQ2 band).
    --target-bpw overrides the table.
4.  Greedy knapsack: start every covered group at its floor, rank candidate
    promotions by importance per GB saved, and repeatedly promote the highest
    ranked group one step up the discrete quant ladder while the estimated
    size stays under budget. Stepping the ladder (rather than pricing
    arbitrary bit moves) respects measured convexity: exchange rates do not
    extrapolate across precision (benchmark_results/qwen38_27b/SENSITIVITY.md,
    "Two limits found the hard way").
5.  Emit anchored ``^blk\\.N\\.<tensor>$=<quant>`` lines — llama-quantize
    matches with std::regex_search, unanchored first-match-wins
    (llama-quant.cpp:694; a bare ``output.weight`` would also capture
    ``blk.N.attn_output.weight``).

Hard constraints (measured, SENSITIVITY.md):
  * token_embd has ZERO imatrix coverage (get_rows, not matmul): very-low-bit
    types requiring imatrix abort at quantize time (rc=133). token_embd floor:
    IQ2_S; it is never assigned IQ2_XXS or below, and no-coverage groups are
    only ever stepped along a K-quant-only "safe ladder".
  * no-coverage groups are promoted last (undefined importance), so
    token_embd bottoms out at IQ2_S in the bands where allocation matters.

Output is directly consumable by:
    ./scripts/quantize.py --config <out.txt> model.gguf out.gguf
    ./scripts/estimate_size.py --config <out.txt> model.gguf
    ./scripts/estimate_config_size.py --inventory ... --config <out.txt>
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from estimate_size import BPW, ALWAYS_F32  # noqa: E402
from generate_config import get_zone  # noqa: E402

F32_FLOOR = 100_000  # norms/biases below this many elements stay F32

# Discrete quant ladder, ascending quality. Effective BPW incl. super-block
# overhead is in estimate_size.BPW.
LADDER = ["IQ2_S", "IQ3_XXS", "Q3_K", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

# Ladder for tensors WITHOUT imatrix coverage: only types known to quantize
# without an importance entry. IQ3_XXS/IQ4_XS are skipped out of caution —
# IQ2_XXS is measured to abort (rc=133) on the uncovered token_embd, and the
# IQ3/IQ4 XS variants share the importance-dependent code path.
SAFE_LADDER = ["IQ2_S", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

# Target average bits-per-weight (over all params) per tier. Calibrated so the
# dense ladder brackets the measured bands on Qwen3.8-27B: tier13 ≈ Mini
# (~13.5 GB), tier10 ≈ Compact/Q4_K band, tier1 ≈ uniform Q8_0.
TIER_BPW = {
    "tier1": 8.5,
    "tier2": 8.0,
    "tier3": 7.5,
    "tier4": 7.0,
    "tier5": 6.6,
    "tier6": 6.2,
    "tier7": 5.9,
    "tier8": 5.6,
    "tier9": 5.3,
    "tier10": 5.0,
    "tier11": 4.6,
    "tier12": 4.3,
    "tier13": 4.0,
    "tier14": 3.7,
    "tier15": 3.4,
}

# Groups whose first ladder step is above IQ2_S (output.weight is the dearest
# group on the measured curve — 15.3x token_embd — never park it at the very
# bottom even in the deepest band).
GROUP_FLOORS = {"output": "Q3_K"}

# role → tensor-name pattern (fullmatch against the GGUF tensor name).
# Order matters only for exotic names; all patterns are exact-match anchored.
ROLE_PATTERNS = [
    (re.compile(r"token_embd(\.weight)?$"), "token_embd", None),
    (re.compile(r"^output\.weight$"), "output", None),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"), "ffn_gate", "layer"),
    (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"), "ffn_up", "layer"),
    (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"), "ffn_down", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"), "fullattn", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"), "fullattn", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"), "fullattn", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"), "attn_v", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_qkv\.weight$"), "linattn_qkv", "layer"),
    (re.compile(r"^blk\.(\d+)\.attn_gate\.weight$"), "linattn_gate", "layer"),
    (re.compile(r"^blk\.(\d+)\.ssm_out\.weight$"), "linattn_out", "layer"),
    (re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"), "ssm_in", "layer"),
    (re.compile(r"^blk\.(\d+)\.ssm_(alpha|beta)\.weight$"), "ssm_small", "layer"),
    (re.compile(r"^blk\.(\d+)\.shortconv\.in_proj\.weight$"), "linattn_qkv", "layer"),
    (re.compile(r"^blk\.(\d+)\.shortconv\.out_proj\.weight$"), "linattn_out", "layer"),
]

# MTP/NextN detection, mirroring detect_gguf_params.py: a blk.N that contains
# ANY nextn/mtp/mtl/future-prefixed tensor is an MTP block — the draft layer
# reuses standard tensor names (blk.64.attn_q.weight on Qwen3.8-27B), so the
# whole block must be pinned f16, not allocated by role.
MTP_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(nextn|mtl|mtp|future)\.")
BLK_RE = re.compile(r"^blk\.(\d+)\.")


def find_mtp_blocks(tensors):
    """Set of blk indices that belong to the MTP/NextN draft head."""
    blocks = set()
    for name, _, _ in tensors:
        m = MTP_BLOCK_RE.match(name)
        if m:
            blocks.add(int(m.group(1)))
    return blocks


def classify(name, mtp_blocks=frozenset()):
    """Return (role, blk_index_or_None) for a tensor name, or (None, None)."""
    m = BLK_RE.match(name)
    if m and int(m.group(1)) in mtp_blocks:
        return "mtp", int(m.group(1))
    for rx, role, scope in ROLE_PATTERNS:
        m = rx.fullmatch(name)
        if m:
            blk = int(m.group(1)) if scope == "layer" else None
            return role, blk
    return None, None


# ── Model inventory ──────────────────────────────────────────────────────────

def load_inventory_json(path):
    """Load a tensor inventory JSON (name -> [numel, stays_f32])."""
    import json
    with open(path) as f:
        blob = json.load(f)
    tensors = blob["tensors"] if isinstance(blob, dict) and "tensors" in blob else blob
    out = []
    for name, v in tensors.items():
        numel, stays = int(v[0]), bool(v[1])
        out.append((name, numel, stays))
    return out


def detect_layers(tensors, mtp_blocks):
    """Number of transformer blocks = max non-MTP blk index + 1."""
    mx = -1
    for name, _, _ in tensors:
        role, blk = classify(name, mtp_blocks)
        if role == "mtp" or blk is None:
            continue
        mx = max(mx, blk)
    return mx + 1 if mx >= 0 else 0


# ── Imatrix GGUF reading ─────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" LE


def _read_gguf_string(f):
    slen = struct.unpack("<Q", f.read(8))[0]
    return f.read(slen).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype):
    """Read one metadata value; return it for scalars we care about."""
    if vtype == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 5: 4, 6: 4, 7: 1, 11: 8, 12: 8}
    if vtype in sizes:
        f.read(sizes[vtype])
        return None
    if vtype == 8:  # STRING
        _read_gguf_string(f)
        return None
    if vtype == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        vals = [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
        return vals
    raise SystemExit(f"generate_dense_config: unknown GGUF value type {vtype}")


def read_imatrix(path):
    """Read an imatrix GGUF (llama-imatrix -o output).

    Returns (entries, chunk_count) where entries maps tensor name ->
    (values ndarray F32, counts ndarray F32 or None).

    Handles two on-disk formats:

    * legacy: ``imatrix.<name>`` F32 per-column values entry, with optional
      ``imatrix.<name>.counts`` companion (the ``imatrix.<name>.in_sum2``
      companion is ignored — the plain entry is already normalized).
    * raw-sums (newer llama.cpp builds): per base tensor exactly two entries
      with NO "imatrix." prefix — ``<name>.in_sum2`` (F32, one sum-of-squares
      value per output column) and ``<name>.counts`` (call counter, scalar or
      per-column). The legacy per-column statistic is reconstructed as
      RMS = sqrt(in_sum2_col / counts).
    """
    entries = {}
    sum2 = {}        # raw-sums format: base name -> per-column sum of squares
    sum_counts = {}  # raw-sums format: base name -> call counter(s)
    chunk_count = None
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if struct.unpack("<I", magic)[0] != GGUF_MAGIC:
            raise SystemExit(f"generate_dense_config: not a GGUF file: {path}")
        if version not in (2, 3):
            raise SystemExit(f"generate_dense_config: unsupported GGUF version {version}")
        alignment = 32
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _read_gguf_value(f, vtype)
            if key == "general.alignment" and val:
                alignment = int(val)
            if key == "imatrix.chunk_count" and val is not None:
                chunk_count = int(val) if not isinstance(val, list) else None

        infos = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            type_id = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            infos.append((name, type_id, int(np.prod(dims)) if dims else 0, offset))

        def read_entry(name, type_id, numel, offset):
            if type_id != 0:  # expect F32
                print(f"warning: imatrix entry {name!r} is not F32, skipped", file=sys.stderr)
                return None
            f.seek(data_start + offset)
            return np.frombuffer(f.read(numel * 4), dtype="<f4").astype(np.float64)

        data_start = (f.tell() + alignment - 1) // alignment * alignment
        for name, type_id, numel, offset in infos:
            if name.startswith("imatrix."):
                rest = name[len("imatrix."):]
                if rest.endswith(".in_sum2"):
                    continue  # legacy companion; the plain entry is authoritative
                vals = read_entry(name, type_id, numel, offset)
                if vals is None:
                    continue
                if rest.endswith(".counts"):
                    base = rest[:-len(".counts")]
                    prev = entries.get(base, (None, None))
                    entries[base] = (prev[0], vals)
                else:
                    prev = entries.get(rest, (None, None))
                    entries[rest] = (vals, prev[1])
            elif name.endswith(".in_sum2"):
                vals = read_entry(name, type_id, numel, offset)
                if vals is not None:
                    sum2[name[:-len(".in_sum2")]] = vals
            elif name.endswith(".counts"):
                vals = read_entry(name, type_id, numel, offset)
                if vals is not None:
                    sum_counts[name[:-len(".counts")]] = vals
            # anything else is not an imatrix entry — skip

    # Reconstruct the legacy statistic for raw-sums format bases. A plain
    # legacy entry, when present, wins over the reconstruction.
    for base, s2 in sum2.items():
        if entries.get(base, (None, None))[0] is not None:
            continue
        cnt = sum_counts.get(base)
        if cnt is None:
            print(f"warning: no counts entry for imatrix tensor {base!r}; "
                  f"using unnormalized in_sum2", file=sys.stderr)
            vals = s2
        elif cnt.size == 1:
            c = float(cnt[0])
            vals = s2 / c if c > 0 else s2
        elif cnt.size == s2.size:
            vals = np.where(cnt > 0, s2 / np.maximum(cnt, 1), s2)
        else:
            print(f"warning: counts shape mismatch for imatrix tensor {base!r}; "
                  f"using unnormalized in_sum2", file=sys.stderr)
            vals = s2
        entries[base] = (np.sqrt(vals), None)
    return entries, chunk_count


def tensor_score(vals, counts, stat):
    """Importance score for one imatrix entry."""
    if counts is not None and len(counts) == len(vals):
        vals = np.where(counts > 0, vals / np.maximum(counts, 1), vals)
    if stat == "p95":
        return float(np.quantile(vals, 0.95))
    return float(np.mean(vals))


def group_importance(groups, imx, stat):
    """numel-weighted mean importance per group; returns {group_key: score}.

    Groups with no imatrix entry at all are omitted (no-coverage).
    """
    out = {}
    for key, members in groups.items():
        num = den = 0.0
        covered = False
        for name, numel in members:
            entry = imx.get(name)
            if entry is None or entry[0] is None:
                continue
            covered = True
            num += tensor_score(entry[0], entry[1], stat) * numel
            den += numel
        if covered and den > 0:
            out[key] = num / den
    return out


# ── Grouping ─────────────────────────────────────────────────────────────────

def build_groups(tensors, layers, mtp_blocks):
    """Split quantizable tensors into role x zone groups.

    Returns (groups, mtp_names, unclassified) where groups maps
    (role,) or (role, zone) -> [(name, numel), ...].
    """
    groups = {}
    mtp_names = []
    unclassified = []
    for name, numel, stays in tensors:
        if stays or numel < F32_FLOOR:
            continue
        if any(p.fullmatch(name) for p in ALWAYS_F32):
            continue
        role, blk = classify(name, mtp_blocks)
        if role is None:
            unclassified.append(name)
            continue
        if role == "mtp":
            mtp_names.append(name)
            continue
        key = (role,) if blk is None else (role, get_zone(blk, layers))
        groups.setdefault(key, []).append((name, numel))
    return groups, mtp_names, unclassified


# ── Greedy knapsack allocation ───────────────────────────────────────────────

def total_bits(tensors, assignment, base_type, mtp_blocks=frozenset()):
    """Estimated output bits given {tensor_name: quant} and a fallback type."""
    bits = 0.0
    for name, numel, stays in tensors:
        if stays or numel < F32_FLOOR or any(p.fullmatch(name) for p in ALWAYS_F32):
            bits += numel * 32.0
            continue
        role, _ = classify(name, mtp_blocks)
        if role == "mtp":
            bits += numel * BPW["F16"]  # MTP draft head pinned f16
            continue
        q = assignment.get(name, base_type)
        bits += numel * BPW[q]
    return bits


def allocate(groups, importance, tensors, base_type, budget_bits, cap_q="Q6_K",
             mtp_blocks=frozenset()):
    """Greedy ladder allocation under a bit budget.

    Returns {(group_key): ladder_index} plus diagnostics. Every group starts
    at its floor; promotions are ranked by importance per GB saved and applied
    one ladder step at a time while the budget holds. The MoE -1 convention
    (ffn_down never below ffn_gate/ffn_up of the same zone) is enforced
    during the greedy walk.

    *cap_q* caps the ladder: measured allocations never buy Q8_0 away from
    the near-full tiers — output at Q8_0 funded by cutting an already-Q3 FFN
    lost by 6.1σ (SENSITIVITY.md, mini-v2). Only tiers with target >= 7.5 bpw
    get the full ladder; everyone else caps at Q6_K.
    """
    cap_idx = LADDER.index(cap_q) if cap_q in LADDER else len(LADDER) - 1
    ladders = {}
    floors = {}
    for key in groups:
        role = key[0]
        covered = key in importance
        ladders[key] = (LADDER if covered else SAFE_LADDER)[:cap_idx + 1]
        floor_q = GROUP_FLOORS.get(role, "IQ2_S")
        if floor_q not in ladders[key]:
            floor_q = ladders[key][0]
        floors[key] = ladders[key].index(floor_q)

    levels = dict(floors)
    assignment = {}
    for key, members in groups.items():
        for name, _ in members:
            assignment[name] = ladders[key][levels[key]]

    cur_bits = total_bits(tensors, assignment, base_type, mtp_blocks)
    floor_bits = cur_bits
    if floor_bits > budget_bits:
        print(f"error: floor allocation needs {floor_bits / 8 / 1e9:.2f} GB, "
              f"budget is {budget_bits / 8 / 1e9:.2f} GB — tier too deep for "
              f"this model's floors", file=sys.stderr)
        sys.exit(1)

    def group_numel(key):
        return sum(n for _, n in groups[key])

    def step_bits(key, from_idx, to_idx):
        """Bits ADDED by promoting key from ladder index from_idx to to_idx."""
        ladder = ladders[key]
        return (BPW[ladder[to_idx]] - BPW[ladder[from_idx]]) * group_numel(key)

    while True:
        best = None  # (ratio, is_down, numel, key, delta_bits)
        for key in groups:
            if key not in importance:
                continue  # no-coverage groups are promoted last, see below
            i = levels[key]
            ladder = ladders[key]
            if i + 1 >= len(ladder):
                continue
            delta_bits = step_bits(key, i, i + 1)
            if delta_bits <= 0:
                continue
            if cur_bits + delta_bits > budget_bits:
                continue
            # Rank by importance mass per GB: (mu * numel) / delta_gb. numel
            # cancels, so this is mu / delta_bpw — size-independent, matching
            # the measured per-GB exchange rates (SENSITIVITY.md). Ranking raw
            # mu per GB instead would flood tiny groups (attn_v, ssm_small)
            # with bits.
            ratio = importance[key] * group_numel(key) / (delta_bits / 8e9)
            cand = (ratio, 1 if key[0] == "ffn_down" else 0, group_numel(key),
                    key, delta_bits)
            if best is None or cand[:3] > best[:3]:
                best = cand

        # No-coverage groups: promote only once covered groups can no longer
        # absorb budget (importance is undefined for them).
        if best is None:
            for key in groups:
                if key in importance:
                    continue
                i = levels[key]
                ladder = ladders[key]
                if i + 1 >= len(ladder):
                    continue
                delta_bits = step_bits(key, i, i + 1)
                if delta_bits <= 0 or cur_bits + delta_bits > budget_bits:
                    continue
                cand = (0.0, 1 if key[0] == "ffn_down" else 0, group_numel(key),
                        key, delta_bits)
                if best is None or cand[:3] > best[:3]:
                    best = cand

        if best is None:
            break

        _, _, _, key, _ = best
        levels[key] += 1
        for name, _ in groups[key]:
            assignment[name] = ladders[key][levels[key]]
        cur_bits += best[4]

        # ffn_down sits one step above ffn_gate/ffn_up at equal importance
        # (mirrors generate_config.py's -1 convention).
        if key[0] in ("ffn_gate", "ffn_up"):
            dkey = ("ffn_down", key[1])
            if dkey in groups and levels[dkey] < levels[key]:
                ladder = ladders[dkey]
                while levels[dkey] < levels[key] and levels[dkey] + 1 < len(ladder):
                    delta = step_bits(dkey, levels[dkey], levels[dkey] + 1)
                    levels[dkey] += 1
                    for name, _ in groups[dkey]:
                        assignment[name] = ladder[levels[dkey]]
                    cur_bits += delta

    return levels, ladders, assignment, cur_bits, floor_bits


# ── Emission ─────────────────────────────────────────────────────────────────

def _esc(name):
    return name.replace(".", r"\.")


def emit_config(layers, levels, ladders, mtp_names):
    """Emit anchored tensor-type lines, same format as generate_dense()."""
    lines = []
    if ("token_embd",) in levels:
        lines.append(f"^{_esc('token_embd.weight')}$={ladders[('token_embd',)][levels[('token_embd',)]]}")
    if ("output",) in levels:
        lines.append(f"^{_esc('output.weight')}$={ladders[('output',)][levels[('output',)]]}")

    layer_roles = [
        ("ffn_gate", "ffn_gate.weight"),
        ("ffn_up", "ffn_up.weight"),
        ("ffn_down", "ffn_down.weight"),
        ("fullattn", "attn_q.weight"),
        ("fullattn", "attn_k.weight"),
        ("attn_v", "attn_v.weight"),
        ("fullattn", "attn_output.weight"),
        ("linattn_qkv", "attn_qkv.weight"),
        ("linattn_gate", "attn_gate.weight"),
        ("linattn_out", "ssm_out.weight"),
        ("ssm_in", "ssm_in.weight"),
        ("ssm_small", "ssm_alpha.weight"),
        ("ssm_small", "ssm_beta.weight"),
    ]
    for i in range(layers):
        zone = get_zone(i, layers)
        for role, tn in layer_roles:
            key = (role, zone)
            if key not in levels:
                continue
            lines.append(f"^blk\\.{i}\\.{_esc(tn)}$={ladders[key][levels[key]]}")

    for name in mtp_names:
        lines.append(f"^{_esc(name)}$=f16")

    return lines


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Imatrix-importance-driven dense quant allocator (tier1–tier15)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Source GGUF file (header is read)")
    src.add_argument("--inventory",
                     help="Tensor inventory JSON (name -> [numel, stays_f32])")
    p.add_argument("--imatrix", "-i", required=True,
                   help="Imatrix GGUF produced by llama-imatrix -o")
    p.add_argument("--tier", "-t", required=True, choices=sorted(TIER_BPW),
                   help="Tier profile (tier1 = near-full precision … tier15 = IQ2 band)")
    p.add_argument("--target-bpw", type=float, default=None,
                   help="Override the tier's target bits-per-weight budget")
    p.add_argument("--stat", choices=["mean", "p95"], default="mean",
                   help="Importance statistic: mean or 95th percentile of imatrix values")
    p.add_argument("--base", default="Q8_0",
                   help="Quant for tensors the config does not cover (default: Q8_0)")
    p.add_argument("--layers", "-l", type=int, default=None,
                   help="Transformer layer count (default: auto-detect)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the allocation table instead of writing a config")
    p.add_argument("--output", "-o", help="Write config to file instead of stdout")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base_type = args.base.upper()

    if args.model:
        if not os.path.isfile(args.model):
            print(f"error: model GGUF not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        from estimate_size import read_gguf_tensor_list
        tensors = [(n, n_el, False) for n, n_el, _ in read_gguf_tensor_list(args.model)]
    else:
        if not os.path.isfile(args.inventory):
            print(f"error: inventory not found: {args.inventory}", file=sys.stderr)
            sys.exit(1)
        tensors = load_inventory_json(args.inventory)

    mtp_blocks = find_mtp_blocks(tensors)
    layers = args.layers if args.layers else detect_layers(tensors, mtp_blocks)
    if layers <= 0:
        print("error: could not detect layer count; pass --layers", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.imatrix):
        print(f"error: imatrix not found: {args.imatrix}", file=sys.stderr)
        sys.exit(1)
    imx, chunk_count = read_imatrix(args.imatrix)

    groups, mtp_names, unclassified = build_groups(tensors, layers, mtp_blocks)
    importance = group_importance(groups, imx, args.stat)

    covered_keys = {k for k in groups if k in importance}
    print(f"imatrix: {len(imx)} entries"
          + (f", {chunk_count} chunks" if chunk_count else "")
          + f"; groups: {len(groups)} ({len(covered_keys)} covered)", file=sys.stderr)
    for key in sorted(groups):
        if key not in importance:
            label = key[0] if len(key) == 1 else f"{key[0]}[{key[1]}]"
            print(f"  no imatrix coverage: {label} -> safe ladder, floor priority",
                  file=sys.stderr)
    if unclassified:
        print(f"warning: {len(unclassified)} tensors match no role "
              f"(fall back to {base_type}):", file=sys.stderr)
        for name in unclassified[:10]:
            print(f"  {name}", file=sys.stderr)
        if len(unclassified) > 10:
            print(f"  … and {len(unclassified) - 10} more", file=sys.stderr)

    total_params = sum(n for _, n, _ in tensors)
    target_bpw = args.target_bpw if args.target_bpw else TIER_BPW[args.tier]
    budget_bits = target_bpw * total_params

    cap_q = "Q8_0" if target_bpw >= 7.5 else "Q6_K"
    levels, ladders, assignment, cur_bits, floor_bits = allocate(
        groups, importance, tensors, base_type, budget_bits, cap_q=cap_q,
        mtp_blocks=mtp_blocks)

    est_gb = cur_bits / 8 / 1e9
    target_gb = budget_bits / 8 / 1e9
    print(f"tier:      {args.tier} (target {target_bpw} bpw ≈ {target_gb:.2f} GB)",
          file=sys.stderr)
    print(f"estimated: {est_gb:.3f} GB  (floor {floor_bits / 8 / 1e9:.3f} GB, "
          f"headroom used {(cur_bits - floor_bits) / 8 / 1e9:.3f} GB)", file=sys.stderr)

    if args.dry_run:
        rows = []
        for key in sorted(groups, key=lambda k: (k[0], k[1] if len(k) > 1 else "")):
            numel = sum(n for _, n in groups[key])
            imp = importance.get(key)
            rows.append((
                key[0],
                key[1] if len(key) > 1 else "-",
                f"{imp:.5g}" if imp is not None else "no-cov",
                ladders[key][levels[key]],
                numel,
                numel * BPW[ladders[key][levels[key]]] / 8 / 1e9,
            ))
        print(f"\n{'group':<14}{'zone':<6}{'importance':>12}{'quant':>9}"
              f"{'params':>14}{'est GB':>10}")
        print("-" * 66)
        for r in rows:
            print(f"{r[0]:<14}{r[1]:<6}{r[2]:>12}{r[3]:>9}{r[4] / 1e6:>13.1f}M{r[5]:>10.3f}")
        print("-" * 66)
        print(f"{'TOTAL':<41}{sum(r[4] for r in rows) / 1e9:>13.2f} B{est_gb:>10.3f}")
        return

    lines = emit_config(layers, levels, ladders, mtp_names)
    output = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"config written: {args.output} ({len(lines)} lines, {layers} layers)",
              file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
