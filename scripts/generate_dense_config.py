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
3.  Rank the imatrix-covered groups by importance (mu) in ascending order —
    from the most quantizable (lowest mu) to the least quantizable (highest).
    The tier no longer maps to a size budget: it sets the LOWER BOUND of a
    quality continuum [0..100], linear in the tier number
    (tier t → 100*(15-t)/14: tier15 spans the full range with floor 0,
    tier7 starts at 50, tier1 collapses the range to the single top point).
4.  Rank-scale-bin: the ordered groups are stretched linearly onto
    [tier_floor .. 100] — the most quantizable group lands exactly on the
    tier floor, the least quantizable one on 100 (the ladder cap), the rest
    spread evenly between by rank. Ranking is ordinal (group ranks, not raw
    mu values), so imatrix outliers do not distort the spread, and groups
    are never clamped: the whole ordered set scales. The continuum is then
    binned into equal-width bands over the discrete quant ladder
    (IQ1_S … cap); a group takes the rung its position falls into. The
    mapping is monotone in rank, so adjacent-by-importance groups cannot
    jump apart — the budget knapsack produced non-physical jumps (Nemotron-3
    Nano-4B tier15: blk.9 attn Q5_K → blk.10 attn IQ1_M → blk.32 attn
    Q5_K); rank-scale-bin excludes them by construction.
    The TOP anchor is tier-dependent: near-full tiers (reference budget
    >= 7.5 bpw — tier1, tier2) anchor at Q8_0, every deeper tier at Q6_K
    (Q8_0 away from near-full tiers measured worse, 6.1σ —
    benchmark_results/qwen38_27b/SENSITIVITY.md, mini-v2 arm). The bottom
    of the continuum slides into the deep rungs (IQ1_S … IQ2_XS) as the
    tier deepens — the tier-depth demotion of the removed exchange pass
    falls out of the scaling for free.
    Size in GB is no longer an input at any point: it is the honest output
    of estimate_size.py, not an echo of a budget.
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
    ladder: no-coverage groups are never binned into them (rc=133 touches
    only tensors without an imatrix entry).
  * output keeps its Q3_K floor on every tier (dearest group on the measured
    curve — never parked below it). The floor clamp can locally override the
    continuum for floored groups; everywhere else rank monotonicity holds.
  * no-coverage groups have no position on the continuum (undefined
    importance) and stay parked at their SAFE_LADDER floors, so token_embd
    bottoms out at IQ2_S on every tier.

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

# Tier → quality-continuum lower bound. A tier no longer encodes a size
# budget (the old TIER_BPW table is gone — see allocate()): it sets how far
# down the quality scale the most quantizable group may be pushed. Linear in
# the tier number: tier t → 100*(15-t)/14 (tier15 → 0 = full continuum,
# tier1 → 100 = everything at the cap).

def tier_quality_floor(tier):
    """Lower bound of the quality continuum for a tier1–tier15 profile."""
    return 100.0 * (15 - int(tier[len("tier"):])) / 14


# Top anchor of the continuum per tier. From the old measured budget
# calibration (TIER_BPW: tier1 7.65, tier2 7.07, tier3+ <= 6.53 bpw) only
# the anchor fork survives: tiers whose reference budget was >= 7.5 bpw
# anchor at Q8_0, every deeper tier at Q6_K — Q8_0 away from near-full
# tiers measured worse by 6.1σ (benchmark_results/qwen38_27b/SENSITIVITY.md,
# mini-v2 arm: output at Q8_0 funded by cutting an already-Q3 FFN lost).
TIER_CAP = {f"tier{t}": ("Q8_0" if t <= 2 else "Q6_K") for t in range(1, 16)}

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
    group key -> floor ladder index. Floors are tier-independent; on deep
    tiers the continuum itself positions covered groups onto the deep rungs,
    while no-coverage groups are never binned below their IQ2_S floor.
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


def allocate(groups, importance, cap_q, tier_floor):
    """Rank-scale-bin allocation: importance rank → quality continuum → ladder.

    Covered groups (with an imatrix entry) are ranked by importance mu in
    ascending order — most quantizable first — and stretched linearly onto
    the quality continuum [tier_floor .. 100]: the lowest-mu group lands
    exactly on the tier floor, the highest-mu one on 100 (the ladder cap),
    the rest spread evenly in between by rank. Ranking is ordinal (group
    ranks, not raw mu values), so imatrix outliers do not distort the spread,
    and groups are never clamped to the bounds — the whole ordered set
    scales. The continuum is binned into equal-width bands over the group's
    ladder; a group takes the rung its position falls into. Binning is
    monotone in position, so adjacent-by-importance groups never land more
    than adjacent rungs apart — the non-physical quant jumps of the removed
    budget knapsack (importance determined by budget remainder, not mu) are
    excluded by construction.

    No-coverage groups have no position on the continuum (undefined
    importance) and stay parked at their SAFE_LADDER floors. Explicit
    GROUP_FLOORS entries (output → Q3_K) are enforced as a lower clamp after
    binning — a measured constraint that may locally override the continuum
    for floored groups. The MoE -1 convention (ffn_down never below
    ffn_gate/ffn_up of the same zone) is enforced last, by raising ffn_down;
    with no size budget to defend, the convention always wins.

    *cap_q* truncates the ladder from the top: measured allocations never buy
    Q8_0 away from the near-full tiers (TIER_CAP). There is no size budget
    anywhere in this function — the tier only positions the continuum, and
    the resulting size is whatever estimate_size.py reports afterwards.

    Returns (levels, ladders, assignment, positions) where positions maps
    every group key to its continuum position in [tier_floor..100] (None for
    no-coverage groups, which have none).
    """
    ladders, floors = build_ladders(groups, importance, cap_q)

    covered = sorted((k for k in groups if k in importance),
                     key=lambda k: importance[k])
    n = len(covered)
    positions = {k: None for k in groups}
    levels = {}
    for i, key in enumerate(covered):
        # Even spread: lowest mu → tier floor, highest mu → 100. A single
        # covered group is both ends of the scale; it takes the top.
        pos = (100.0 if n == 1
               else tier_floor + (100.0 - tier_floor) * i / (n - 1))
        positions[key] = pos
        ladder = ladders[key]
        rung = min(int(pos * len(ladder) / 100.0), len(ladder) - 1)
        # Only explicit GROUP_FLOORS entries (output) clamp the continuum;
        # the IQ2_S default floor in `floors` is a no-coverage constraint —
        # covered groups may bin into the sub-I2_S deep rungs (they are the
        # successors of the removed demote_depth mechanism).
        if key[0] in GROUP_FLOORS:
            rung = max(rung, floors[key])
        levels[key] = rung

    # No-coverage tail: no importance, no continuum position — parked.
    for key in groups:
        if key not in importance:
            levels[key] = floors[key]

    # MoE -1 convention: ffn_down at/above ffn_gate/ffn_up of its zone.
    for key in list(levels):
        if key[0] != "ffn_down" or len(key) < 2:
            continue
        peers = [levels.get((p, key[1])) for p in ("ffn_gate", "ffn_up")]
        need = max((p for p in peers if p is not None), default=0)
        levels[key] = min(max(levels[key], need), len(ladders[key]) - 1)

    assignment = {}
    for key, members in groups.items():
        for name, _ in members:
            assignment[name] = ladders[key][levels[key]]
    return levels, ladders, assignment, positions


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
    p.add_argument("--tier", "-t", required=True, choices=sorted(TIER_CAP),
                   help="Tier profile (tier1 = near-full precision … "
                        "tier15 = full IQ1_S–cap continuum; sets the quality "
                        "floor, not a size budget)")
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

    # Rank-scale-bin: the tier positions the quality continuum and picks the
    # ladder cap; no size budget exists anywhere in the pipeline. Size is a
    # post-hoc estimate, not a target.
    cap_q = TIER_CAP[args.tier]
    q_floor = tier_quality_floor(args.tier)
    levels, ladders, assignment, positions = allocate(
        groups, importance, cap_q, q_floor)
    cur_bits = total_bits(tensors, assignment, base_type, mtp_blocks)

    est_gb = cur_bits / 8 / 1e9
    print(f"tier:      {args.tier} (quality band {q_floor:.0f}–100, "
          f"ladder cap {cap_q})", file=sys.stderr)
    print(f"estimated: {est_gb:.3f} GB  (not budget-constrained: the tier "
          "sets a quality floor, estimate_size reports the size)",
          file=sys.stderr)

    if args.dry_run:
        rows = []
        for key in sorted(groups, key=lambda k: (k[0], k[1] if len(k) > 1 else "")):
            numel = sum(n for _, n in groups[key])
            imp = importance.get(key)
            pos = positions.get(key)
            rows.append((
                key[0],
                key[1] if len(key) > 1 else "-",
                f"{imp:.5g}" if imp is not None else "no-cov",
                f"{pos:.1f}" if pos is not None else "-",
                ladders[key][levels[key]],
                numel,
                numel * BPW[ladders[key][levels[key]]] / 8 / 1e9,
            ))
        print(f"\n{'group':<14}{'zone':<6}{'importance':>12}{'pos':>6}"
              f"{'quant':>9}{'params':>14}{'est GB':>10}")
        print("-" * 72)
        for r in rows:
            print(f"{r[0]:<14}{r[1]:<6}{r[2]:>12}{r[3]:>6}{r[4]:>9}"
                  f"{r[5] / 1e6:>13.1f}M{r[6]:>10.3f}")
        print("-" * 72)
        print(f"{'TOTAL':<47}{sum(r[5] for r in rows) / 1e9:>13.2f} B{est_gb:>10.3f}")
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
