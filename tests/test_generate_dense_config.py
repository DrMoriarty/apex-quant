#!/usr/bin/env python3
"""Behavioural tests for scripts/generate_dense_config.py (imatrix allocator).

Runs the allocator through --inventory (name -> [numel, stays_f32]) so no
model GGUF is needed, and feeds it a synthetic imatrix GGUF written by
write_imatrix() — the same minimal GGUF v3 layout the script's reader parses
(legacy "imatrix.<tensor>" F32 entries).

Asserts the documented measured constraints (AGENTS.md, "MoE vs dense/hybrid"):
  * every emitted pattern is anchored (^…$) — llama-quantize matches with
    std::regex_search, unanchored first-match-wins (llama-quant.cpp:694);
  * every quantizable inventory tensor is covered (no silent base-type leaks);
  * token_embd has zero imatrix coverage -> IQ2_S floor on the imatrix-free
    SAFE_LADDER, never a sub-IQ2_S deep rung;
  * the ladder is capped at Q6_K except tiers 1–2 (Q8_0 anchor);
  * output.weight never parks below its Q3_K floor;
  * importance ranking is monotone: the more important group never gets a
    worse type, and flipping the importance ranking flips the allocation.

Usage: python3 tests/run_all.py
  (or: python3 -m unittest tests.test_generate_dense_config)
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOC = os.path.join(REPO_ROOT, "scripts", "generate_dense_config.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import generate_dense_config as gdc  # noqa: E402
from estimate_size import ALWAYS_F32  # noqa: E402

NLAYERS = 16  # zone_size=ceil(1.6)=2: edge 0-1,14-15; near 2-3,12-13; mid 4-11

QUALITY = gdc.DEEP_RUNGS + gdc.LADDER  # ascending quality


def qrank(qtype):
    return QUALITY.index(qtype)


ROLES = ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v",
         "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ssm_in",
         "ssm_alpha", "ssm_beta")

BASE_IMPORTANCE = 1.0     # every tensor gets at least this
IMPORTANT_BONUS = 5.0     # added to the tensors of the "important" zone
OUTPUT_IMPORTANCE = 10.0  # output.weight: dearest group


def write_inventory(path):
    tensors = {
        "token_embd.weight": [4096 * 128, False],
        "output.weight": [4096 * 128, False],
        "output_norm.weight": [4096, True],  # stays f32, never allocated
    }
    for i in range(NLAYERS):
        for role in ROLES:
            tensors[f"blk.{i}.{role}.weight"] = [100_000 + i, False]
        tensors[f"blk.{i}.ffn_norm.weight"] = [4096, True]  # below F32_FLOOR
    with open(path, "w") as f:
        json.dump(tensors, f)


def write_imatrix(path, entries):
    """Minimal GGUF v3 with legacy `imatrix.<name>` F32 entries."""
    infos = [(f"imatrix.{name}", arr) for name, arr in entries.items()]
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(infos), 0)
    info_bytes = b""
    offset = 0
    for name, arr in infos:
        nb = name.encode()
        info_bytes += struct.pack("<Q", len(nb)) + nb
        info_bytes += struct.pack("<I", 1)          # n_dims
        info_bytes += struct.pack("<Q", len(arr))   # dim0
        info_bytes += struct.pack("<I", 0)          # ggml type F32
        info_bytes += struct.pack("<Q", offset)
        offset += len(arr) * 4
    pad = (-(len(header) + len(info_bytes))) % 32
    with open(path, "wb") as f:
        f.write(header + info_bytes + b"\x00" * pad)
        for _, arr in infos:
            f.write(struct.pack(f"<{len(arr)}f", *arr))


def make_importance(important_zone):
    """Importance map with one zone boosted; token_embd left uncovered.

    important_zone: "mid" (layers 4-11) or "edge" (layers 0-1,14-15).
    """
    entries = {}
    for i in range(NLAYERS):
        in_zone = (4 <= i <= 11) if important_zone == "mid" else (i < 2 or i >= 14)
        bonus = IMPORTANT_BONUS if in_zone else 0.0
        for role in ROLES:
            entries[f"blk.{i}.{role}.weight"] = [BASE_IMPORTANCE + bonus] * 8
    entries["output.weight"] = [OUTPUT_IMPORTANCE] * 8
    # token_embd.weight intentionally absent: zero imatrix coverage (get_rows)
    return entries


class DenseAllocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="apex_dense_alloc_")
        tmp = cls._tmp.name
        cls.inventory = os.path.join(tmp, "inventory.json")
        write_inventory(cls.inventory)
        cls.imatrix_mid = os.path.join(tmp, "imx_mid.gguf")
        write_imatrix(cls.imatrix_mid, make_importance("mid"))
        cls.imatrix_edge = os.path.join(tmp, "imx_edge.gguf")
        write_imatrix(cls.imatrix_edge, make_importance("edge"))
        cls.configs = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def run_allocator(cls, tier, imatrix):
        key = (tier, imatrix)
        if key not in cls.configs:
            proc = subprocess.run(
                [sys.executable, ALLOC, "--inventory", cls.inventory,
                 "--imatrix", imatrix, "--tier", tier],
                capture_output=True, text=True)
            if proc.returncode != 0:
                raise AssertionError(
                    f"allocator tier={tier} failed rc={proc.returncode}:\n{proc.stderr}")
            cls.configs[key] = proc.stdout
        return cls.configs[key]

    def rules(self, tier, imatrix=None):
        cfg = self.run_allocator(tier, imatrix or self.imatrix_mid)
        rules = []
        for line in cfg.splitlines():
            line = line.strip()
            if line:
                pat, qtype = line.rsplit("=", 1)
                rules.append((pat, qtype))
        return rules

    def assigned(self, tier, name, imatrix=None):
        """First-match (std::regex_search) type for a tensor name."""
        for pat, qtype in self.rules(tier, imatrix):
            if re.search(pat, name):
                return qtype
        return None

    # ── anchoring & coverage ─────────────────────────────────────────────

    def test_every_pattern_is_anchored(self):
        cfg = self.run_allocator("tier13", self.imatrix_mid)
        for line in cfg.splitlines():
            self.assertRegex(line, r"^\^[^\s=]+\$=(f16|[A-Za-z0-9_]+)$",
                             f"unanchored or malformed line: {line!r}")

    def test_covers_every_quantizable_inventory_tensor(self):
        with open(self.inventory) as f:
            tensors = json.load(f)
        rules = self.rules("tier13")
        uncovered = []
        for name, (numel, stays) in tensors.items():
            if stays or numel < gdc.F32_FLOOR:
                continue
            if any(p.fullmatch(name) for p in ALWAYS_F32):
                continue
            if not any(re.search(pat, name) for pat, _ in rules):
                uncovered.append(name)
        self.assertEqual(uncovered, [],
                         "tensors falling through to the base type (silent Q8_0)")

    # ── measured constraints ─────────────────────────────────────────────

    def test_token_embd_parked_at_iq2_s_floor(self):
        # Zero imatrix coverage -> safe ladder, floor priority, on every tier.
        for tier in ("tier1", "tier7", "tier13"):
            with self.subTest(tier=tier):
                t = self.assigned(tier, "token_embd.weight")
                self.assertEqual(t, "IQ2_S")
                self.assertIn(t, gdc.SAFE_LADDER,
                              "token_embd left the imatrix-free ladder")
                self.assertNotIn(t, gdc.DEEP_RUNGS,
                                 "token_embd must never take a sub-IQ2_S rung")

    def test_ladder_capped_at_q6_k_except_tier1_2(self):
        for tier in ("tier3", "tier7", "tier13"):
            with self.subTest(tier=tier):
                types = {q for _, q in self.rules(tier)}
                self.assertNotIn("Q8_0", types,
                                 f"{tier}: Q8_0 above the measured Q6_K cap")
                allowed = set(gdc.DEEP_RUNGS) | set(
                    gdc.LADDER[:gdc.LADDER.index("Q6_K") + 1])
                self.assertLessEqual(types, allowed)

    def test_tier1_tier2_anchor_at_q8_0(self):
        types = {q for _, q in self.rules("tier1")}
        self.assertIn("Q8_0", types,
                      "tier1 anchors the continuum at the Q8_0 ladder cap")

    def test_output_weight_never_below_q3_k(self):
        for tier in ("tier1", "tier7", "tier13"):
            with self.subTest(tier=tier):
                t = self.assigned(tier, "output.weight")
                self.assertGreaterEqual(qrank(t), qrank("Q3_K"),
                                        f"{tier}: output.weight parked at {t}")

    # ── importance ranking ───────────────────────────────────────────────

    def test_more_important_group_gets_at_least_as_good_type(self):
        for imatrix, important in [(self.imatrix_mid, "mid"),
                                   (self.imatrix_edge, "edge")]:
            with self.subTest(important=important):
                for tier in ("tier7", "tier13"):
                    edge = self.assigned(tier, "blk.0.ffn_gate.weight", imatrix)
                    mid = self.assigned(tier, "blk.8.ffn_gate.weight", imatrix)
                    better, worse = ((mid, edge) if important == "mid"
                                     else (edge, mid))
                    self.assertGreaterEqual(
                        qrank(better), qrank(worse),
                        f"{tier}: more important {important} group got {better} "
                        f"vs {worse} for the less important one")

    def test_flipping_importance_flips_allocation(self):
        # The lever must actually respond to the imatrix, not to zone position:
        # with mid important mid wins, with edge important edge wins — both
        # strictly, so the test cannot pass on two equal types.
        for tier in ("tier7", "tier13"):
            with self.subTest(tier=tier):
                edge_mid_imp = self.assigned(tier, "blk.0.ffn_gate.weight",
                                             self.imatrix_mid)
                mid_mid_imp = self.assigned(tier, "blk.8.ffn_gate.weight",
                                            self.imatrix_mid)
                self.assertLess(qrank(edge_mid_imp), qrank(mid_mid_imp),
                                f"{tier}: mid-important run did not favour mid")
                edge_edge_imp = self.assigned(tier, "blk.0.ffn_gate.weight",
                                              self.imatrix_edge)
                mid_edge_imp = self.assigned(tier, "blk.8.ffn_gate.weight",
                                             self.imatrix_edge)
                self.assertLess(qrank(mid_edge_imp), qrank(edge_edge_imp),
                                f"{tier}: edge-important run did not favour edge")

    def test_tier_deepens_the_floor(self):
        # tier1 collapses the continuum to the top; tier13 slides the most
        # quantizable covered group into the deep rungs.
        t1 = self.assigned("tier1", "blk.0.ffn_gate.weight")
        t13 = self.assigned("tier13", "blk.0.ffn_gate.weight")
        # qrank is ascending quality, so "deeper" means a strictly lower index.
        self.assertLess(qrank(t13), qrank(t1),
                        "tier13 must allocate deeper than tier1")
        self.assertIn(t13, gdc.COVERED_LADDER)


if __name__ == "__main__":
    unittest.main()
