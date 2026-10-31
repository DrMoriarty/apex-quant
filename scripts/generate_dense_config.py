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
3.  Map tier1–tier15 to a target bits-per-weight budget. TIER_BPW is derived
    from the MoE tier ladder (generate_config.PROFILES) translated through the
    dense quant ladder and smoothed to a convex decreasing curve; it brackets
    the MoE reference calibration (Mellum2-12B-A2.5B tier sizes) at every
    tier. --target-bpw overrides the table.
    If the requested tier's budget is below the model's floor allocation
    (every group at its floor + base_type for unclassified tensors), the
    script does not fail: it walks from the requested tier towards tier1 and
    uses the nearest (deepest) feasible tier, printing a warning to stderr
    (exit code stays 0).
4.  Greedy knapsack: start every covered group at its floor, rank candidate
    promotions by importance per GB saved, and repeatedly promote the highest
    ranked group one step up the discrete quant ladder while the estimated
    size stays under budget. Stepping the ladder (rather than pricing
    arbitrary bit moves) respects measured convexity: exchange rates do not
    extrapolate across precision (benchmark_results/qwen38_27b/SENSITIVITY.md,
    "Two limits found the hard way").
    Then an exchange pass makes the bpw spread endogenous in tier depth:
    while a valuable promotion is blocked by the budget, the least valuable
    covered groups are demoted down the deep rungs (IQ1_S … IQ2_XS, covered
    groups only) to fund it — one ladder step at a time, cheapest
    importance-per-GB first. Deep tiers therefore demote more (bottom drops,
    top climbs); shallow tiers see no blocked promotions and are unchanged.
5.  Emit anchored ``^blk\\.N\\.<tensor>$=<quant>`` lines — llama-quantize
    matches with std::regex_search, unanchored first-match-wins
    (llama-quant.cpp:694; a bare ``output.weight`` would also capture
    ``blk.N.attn_output.weight``).

Hard constraints (measured, SENSITIVITY.md):
  * token_embd has ZERO imatrix coverage (get_rows, not matmul): very-low-bit
    types requiring imatrix abort at quantize time (rc=133). token_embd floor:
    IQ2_S; it is never assigned IQ2_XXS or below, and no-coverage groups are
    only ever stepped along a K-quant-only "safe ladder".
  * The deep rungs (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS) exist only on the covered
    ladder: no-coverage groups are never demoted into them and never promoted
    through them (rc=133 touches only tensors without an imatrix entry).
  * output keeps its Q3_K floor on every tier (dearest group on the measured
    curve — never parked below it, never demoted).
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

# Deep rungs below IQ2_S, ascending quality (effective BPW in
# estimate_size.BPW: IQ1_S 1.5625 … IQ2_XS 2.3125). Sub-IQ2_S types are read
# through the importance matrix at quantize time — quantizing them WITHOUT an
# imatrix entry aborts (rc=133) — so they are legal only for groups WITH full
# imatrix coverage. token_embd (zero coverage) and every other no-coverage
# group stays on SAFE_LADDER above its IQ2_S floor; only covered groups may
# be demoted into this band, and only on deep tiers (see allocate()).
DEEP_RUNGS = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS"]
COVERED_LADDER = DEEP_RUNGS + LADDER

# Target average bits-per-weight (over all params) per tier.
#
# Derived from the MoE tier structure (generate_config.PROFILES), not
# hand-picked: each MoE role rank is translated through the dense LADDER
# (mixed mode: expert ranks via QUANTS_RANKED_SIZE, other roles via
# QUANTS_RANKED_SPEED; each quant snapped to its nearest dense-ladder rung by
# estimate_size.BPW) and averaged over a typical dense param distribution —
# FFN 70% (edge/near/mid zones 10/10/80), attention 26% (edge/mid 10/90),
# token_embd + output.weight 4%; shared-expert slots have no dense counterpart
# and get zero weight. The raw per-tier averages oscillate around their trend
# (tier triplets differ by a single rank) and tiers 13–15 collapse (rank 6 =
# IQ1_M clamps to the dense ladder's IQ2_S floor), so the points are smoothed
# with a least-squares quadratic — a convex decreasing parabola whose steps
# flatten with depth, matching the MoE reference calibration (Mellum2-12B-
# A2.5B, generate_config.py tiers): converted to GB at 12 B params the curve
# brackets the measured tier1..13 sizes 11.35 … 4.79 GB within ±7% at every
# tier. Monotone tier1 > … > tier15, convex (bpw deltas 0.58 → 0.13 shrink
# with depth — late ladder steps save fewer bits per weight, exactly the
# flattening the MoE reference shows).
TIER_BPW = {
    "tier1": 7.65,
    "tier2": 7.07,
    "tier3": 6.53,
    "tier4": 6.02,
    "tier5": 5.55,
    "tier6": 5.11,
    "tier7": 4.71,
    "tier8": 4.34,
    "tier9": 4.00,
    "tier10": 3.70,
    "tier11": 3.43,
    "tier12": 3.20,
    "tier13": 3.00,
    "tier14": 2.84,
    "tier15": 2.71,
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


def build_ladders(groups, importance, cap_q="Q6_K"):
    """Per-group ladder and floor index for a ladder capped at *cap_q*.

    Returns (ladders, floors): ladders maps group key -> quant list
    (COVERED_LADDER — incl. the sub-IQ2_S deep rungs — for imatrix-covered
    groups, SAFE_LADDER otherwise, both truncated at the cap), floors maps
    group key -> floor ladder index. Floors are tier-independent: the deep
    rungs are reachable only through demotion (allocate), never by parking.
    """
    cap_idx_cov = (COVERED_LADDER.index(cap_q) if cap_q in COVERED_LADDER
                   else len(COVERED_LADDER) - 1)
    cap_idx_safe = (SAFE_LADDER.index(cap_q) if cap_q in SAFE_LADDER
                    else len(SAFE_LADDER) - 1)
    ladders = {}
    floors = {}
    for key in groups:
        role = key[0]
        covered = key in importance
        ladder = COVERED_LADDER if covered else SAFE_LADDER
        ladders[key] = ladder[:cap_idx_cov + 1 if covered else cap_idx_safe + 1]
        floor_q = GROUP_FLOORS.get(role, "IQ2_S")
        if floor_q not in ladders[key]:
            floor_q = ladders[key][0]
        floors[key] = ladders[key].index(floor_q)
    return ladders, floors


def floor_bits(groups, importance, tensors, base_type, cap_q="Q6_K",
               mtp_blocks=frozenset()):
    """Estimated bits with every group parked at its floor (no greedy walk).

    This is the feasibility pre-pass: if it exceeds the budget, no allocation
    under that budget (and that ladder cap) exists.
    """
    ladders, floors = build_ladders(groups, importance, cap_q)
    assignment = {}
    for key, members in groups.items():
        for name, _ in members:
            assignment[name] = ladders[key][floors[key]]
    return total_bits(tensors, assignment, base_type, mtp_blocks)


def allocate(groups, importance, tensors, base_type, budget_bits, cap_q="Q6_K",
             mtp_blocks=frozenset(), demote_depth=0):
    """Greedy ladder allocation under a bit budget.

    Returns {(group_key): ladder_index} plus diagnostics. Every group starts
    at its floor; promotions are ranked by importance per GB saved and applied
    one ladder step at a time while the budget holds. The MoE -1 convention
    (ffn_down never below ffn_gate/ffn_up of the same zone) is enforced
    throughout.

    A second, exchange-shaped pass makes the bpw spread endogenous in tier
    depth. Parking everything at its floor leaves the least important groups
    at IQ2_S even when the budget is far too tight to lift the important ones
    out of the floor band (tier15 on a 4B dense model: ~0.05 GB of headroom
    over the floors → a ~1.4x bpw spread, vs ~3.7x on the MoE ladder). While
    a valuable promotion is blocked by the budget, the least valuable covered
    groups are demoted down COVERED_LADDER's deep rungs (IQ1_S … IQ2_XS) to
    fund it. Each exchange moves bits from a lower importance-per-GB group to
    a higher one — a strict improvement of the importance-weighted objective —
    so the deeper the tier, the more exchanges pay off: the bottom drops
    towards IQ1_M/IQ2_XXS while the top climbs to Q5_K/Q6_K, and shallow
    tiers (no blocked promotions) are left untouched. How far below IQ2_S a
    group may fall is itself tier-dependent (*demote_depth*, derived from the
    tier number in main): deep tiers may demote into the IQ1_M band, shallow
    tiers keep the sub-IQ2_S rungs out of reach entirely, so the bpw spread
    grows monotonically with tier depth. Groups with an explicit
    GROUP_FLOORS entry (output) and all no-coverage groups are never demoted:
    sub-IQ2_S types need the importance matrix at quantize time (rc=133
    otherwise), which only covered tensors can legally use.

    *cap_q* caps the ladder: measured allocations never buy Q8_0 away from
    the near-full tiers — output at Q8_0 funded by cutting an already-Q3 FFN
    lost by 6.1σ (SENSITIVITY.md, mini-v2). Only tiers with target >= 7.5 bpw
    get the full ladder; everyone else caps at Q6_K.

    Raises ValueError when the budget is below the floor allocation — the
    caller (main) runs the tier-fallback walk; this function no longer
    exits on infeasible tiers.
    """
    ladders, floors = build_ladders(groups, importance, cap_q)

    levels = dict(floors)
    assignment = {}
    for key, members in groups.items():
        for name, _ in members:
            assignment[name] = ladders[key][levels[key]]

    cur_bits = total_bits(tensors, assignment, base_type, mtp_blocks)
    floor_bits_ = cur_bits
    if floor_bits_ > budget_bits:
        raise ValueError(
            f"floor allocation needs {floor_bits_ / 8 / 1e9:.2f} GB, "
            f"budget is {budget_bits / 8 / 1e9:.2f} GB — tier too deep for "
            f"this model's floors")

    def group_numel(key):
        return sum(n for _, n in groups[key])

    def step_bits(key, from_idx, to_idx):
        """Bits ADDED by moving key from ladder index from_idx to to_idx."""
        ladder = ladders[key]
        return (BPW[ladder[to_idx]] - BPW[ladder[from_idx]]) * group_numel(key)

    def set_level(key, idx):
        levels[key] = idx
        for name, _ in groups[key]:
            assignment[name] = ladders[key][idx]

    def follow_convention(key):
        """Keep ffn_down at/above a promoted ffn_gate/ffn_up (MoE -1 convention)."""
        nonlocal cur_bits
        if key[0] not in ("ffn_gate", "ffn_up"):
            return
        dkey = ("ffn_down", key[1])
        if dkey not in groups:
            return
        ladder = ladders[dkey]
        while levels[dkey] < levels[key] and levels[dkey] + 1 < len(ladder):
            delta = step_bits(dkey, levels[dkey], levels[dkey] + 1)
            if cur_bits + delta > budget_bits:
                break  # size gate wins over the soft -1 convention
            set_level(dkey, levels[dkey] + 1)
            cur_bits += delta

    def promote_pass(covered, skip=frozenset()):
        """Greedy one-step-up walk ranked by importance per GB saved.

        *covered* selects the class: imatrix-covered groups (importance-
        ranked) or the no-coverage tail (undefined importance, promoted
        only once the covered class can no longer absorb budget).
        """
        nonlocal cur_bits
        while True:
            best = None  # (ratio, is_down, numel, key, delta_bits)
            for key in groups:
                if covered:
                    if key not in importance or key in skip:
                        continue
                elif key in importance:
                    continue
                i = levels[key]
                ladder = ladders[key]
                if i + 1 >= len(ladder):
                    continue
                delta_bits = step_bits(key, i, i + 1)
                if delta_bits <= 0:
                    continue
                if cur_bits + delta_bits > budget_bits:
                    continue
                if covered:
                    # Rank by importance mass per GB: (mu * numel) / delta_gb.
                    # numel cancels, so this is mu / delta_bpw —
                    # size-independent, matching the measured per-GB exchange
                    # rates (SENSITIVITY.md). Ranking raw mu per GB instead
                    # would flood tiny groups (attn_v, ssm_small) with bits.
                    ratio = importance[key] * group_numel(key) / (delta_bits / 8e9)
                else:
                    ratio = 0.0
                cand = (ratio, 1 if key[0] == "ffn_down" else 0,
                        group_numel(key), key, delta_bits)
                if best is None or cand[:3] > best[:3]:
                    best = cand
            if best is None:
                break
            _, _, _, key, _ = best
            set_level(key, levels[key] + 1)
            cur_bits += best[4]
            follow_convention(key)

    promote_pass(covered=True)

    # ── Exchange pass: tier-depth demotions ─────────────────────────────────
    # Rung moves are ranked here by raw importance per weight (mu), not by
    # the mass-per-GB of the promotion passes: the value/cost ratio of ANY
    # single-rung move of a group is mu (both objective delta and bit cost
    # scale with Δbpw × numel), so mu is the honest exchange rate between a
    # sacrifice and a climb. Mass-per-GB would veto demoting the large
    # low-mu mid-zone groups — their importance *mass* is huge even though
    # their importance *per weight* is the lowest — which is exactly the
    # deep-tier transfer the spread needs (demote big mid groups a few rungs
    # to fund multi-step climbs of small high-mu edge groups).
    #
    # Sacrificed groups are frozen below their floor: they never climb back
    # during the exchange (that would undo the transfer) nor in the final
    # absorb pass (the budget that funded their demotion is spent on better
    # groups).
    demoted = set()

    def demote_floor_idx(key):
        # Explicit GROUP_FLOORS entries (output) keep their floor even under
        # demotion; every other covered group may slide down to IQ1_S.
        return (floors[key] if key[0] in GROUP_FLOORS
                else max(0, floors[key] - demote_depth))

    def can_demote(key):
        i = levels[key]
        if i <= demote_floor_idx(key):
            return False
        if key[0] == "ffn_down":
            for peer in ("ffn_gate", "ffn_up"):
                p = levels.get((peer, key[1]))
                if p is not None and i - 1 < p:
                    return False
        return True

    def freeable_bits(exclude):
        """Bits freed by demoting every legal candidate to its demote floor."""
        total = 0.0
        for key in groups:
            if (key not in importance or key[0] in GROUP_FLOORS
                    or key == exclude):
                continue
            i = levels[key]
            while i - 1 >= demote_floor_idx(key):
                if key[0] == "ffn_down" and not can_demote(key):
                    break
                total += step_bits(key, i - 1, i)
                i -= 1
        return total

    while True:
        # Every promotion the budget currently blocks, highest mu first.
        proms = []
        for key in groups:
            if key not in importance or key in demoted:
                continue
            i = levels[key]
            ladder = ladders[key]
            if i + 1 >= len(ladder):
                continue
            delta_bits = step_bits(key, i, i + 1)
            if delta_bits <= 0 or cur_bits + delta_bits <= budget_bits:
                continue
            proms.append((importance[key], key, delta_bits))
        proms.sort(reverse=True)
        if not proms:
            break  # nothing blocked: floors + headroom already spread it

        # Fund the best blocked promotion that cheaper sacrifices (lower mu)
        # can cover; an unfundable one (step larger than everything cheaper
        # can free) must not trigger sacrifice for nothing.
        chosen = None
        for mu, pkey, pdelta in proms:
            dems = sorted(
                (importance[key], key) for key in groups
                if key in importance and key[0] not in GROUP_FLOORS
                and key != pkey and can_demote(key)
                and importance[key] < mu)
            if dems and pdelta <= freeable_bits(exclude=pkey):
                chosen = (pkey, pdelta, dems[0][1])
                break
        if chosen is None:
            break

        pkey, pdelta, dkey = chosen
        freed = step_bits(dkey, levels[dkey] - 1, levels[dkey])
        set_level(dkey, levels[dkey] - 1)
        cur_bits -= freed
        demoted.add(dkey)
        if cur_bits + pdelta <= budget_bits:
            set_level(pkey, levels[pkey] + 1)
            cur_bits += pdelta
            follow_convention(pkey)

    # Absorb the leftover headroom: covered groups that were not sacrificed
    # first, then the no-coverage tail (undefined importance).
    promote_pass(covered=True, skip=demoted)
    promote_pass(covered=False)

    return levels, ladders, assignment, cur_bits, floor_bits_


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

    # Feasibility walk: cap_q (Q8_0 vs Q6_K ladder) depends on the tier's
    # target bpw, so the floor pre-pass must run per candidate tier rather
    # than comparing raw budgets. Walk from the requested tier towards tier1
    # and take the first (deepest) tier whose budget fits the floors.
    tier_num = int(args.tier[len("tier"):])
    selected_tier = None
    req_target_bpw = args.target_bpw if args.target_bpw else TIER_BPW[args.tier]
    req_floor_bits = None
    last_fb = last_bpw = None
    for cand in range(tier_num, 0, -1):
        cand_tier = f"tier{cand}"
        # --target-bpw overrides the requested tier's budget; once the walk
        # falls back, shallower candidates use their own table budget (the
        # warning reports a tier, not a custom target).
        cand_bpw = (args.target_bpw if cand == tier_num and args.target_bpw
                    else TIER_BPW[cand_tier])
        budget_bits = cand_bpw * total_params
        cap_q = "Q8_0" if cand_bpw >= 7.5 else "Q6_K"
        fb = floor_bits(groups, importance, tensors, base_type, cap_q=cap_q,
                        mtp_blocks=mtp_blocks)
        if req_floor_bits is None:
            req_floor_bits = fb
        last_fb, last_bpw = fb, cand_bpw
        if fb <= budget_bits:
            selected_tier = cand_tier
            target_bpw = cand_bpw
            break

    if selected_tier is None:
        # Even tier1 cannot host the floors — nothing shallower exists.
        print(f"error: floor allocation needs {last_fb / 8 / 1e9:.2f} GB, "
              f"budget is {last_bpw * total_params / 8 / 1e9:.2f} GB — tier "
              "too deep for this model's floors even at tier1", file=sys.stderr)
        sys.exit(1)

    if selected_tier != args.tier:
        req_budget = req_target_bpw * total_params
        fbw_bpw = TIER_BPW[selected_tier]
        print(f"warning: {args.tier} infeasible for this model "
              f"(floor {req_floor_bits / 8 / 1e9:.2f} GB > budget "
              f"{req_budget / 8 / 1e9:.2f} GB), falling back to "
              f"{selected_tier} ({fbw_bpw} bpw ≈ "
              f"{fbw_bpw * total_params / 8 / 1e9:.2f} GB)", file=sys.stderr)

    budget_bits = target_bpw * total_params
    cap_q = "Q8_0" if target_bpw >= 7.5 else "Q6_K"
    # Tier-dependent demotion depth: how many rungs below the IQ2_S floor a
    # covered group may be demoted into (DEEP_RUNGS: IQ2_XS, IQ2_XXS, IQ1_M,
    # IQ1_S). Spread between the most and least compressed groups must grow
    # with tier depth — deep tiers slide low-importance groups into the
    # IQ1_M/IQ2_XXS band to fund climbs to the Q5_K/Q6_K ceiling, shallow
    # tiers never leave the floor band. Linear in the tier number: tier1 → 0
    # (no demotion), tier5 → 0, tier10 → IQ2_XS, tier15 → IQ1_M.
    demote_depth = (int(selected_tier[len("tier"):]) - 1) * 3 // 14
    levels, ladders, assignment, cur_bits, alloc_floor_bits = allocate(
        groups, importance, tensors, base_type, budget_bits, cap_q=cap_q,
        mtp_blocks=mtp_blocks, demote_depth=demote_depth)

    est_gb = cur_bits / 8 / 1e9
    target_gb = budget_bits / 8 / 1e9
    fallback_note = (f" (fallback from {args.tier})"
                     if selected_tier != args.tier else "")
    print(f"tier:      {selected_tier}{fallback_note} "
          f"(target {target_bpw} bpw ≈ {target_gb:.2f} GB)",
          file=sys.stderr)
    print(f"estimated: {est_gb:.3f} GB  (floor {alloc_floor_bits / 8 / 1e9:.3f} GB, "
          f"headroom used {(cur_bits - alloc_floor_bits) / 8 / 1e9:.3f} GB)", file=sys.stderr)

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
