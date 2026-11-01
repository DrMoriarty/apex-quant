#!/usr/bin/env python3
"""Tests for scripts/estimate_size.py — the pre-quantize size gate.

Unit level (imported module, synthetic tensors only — no real GGUF files):
  * load_config — config parsing (comments, blanks, whitespace, case);
  * match_type   — regex_search first-match semantics, anchoring, the
    re.error exact-equality fallback;
  * estimate     — BPW arithmetic, ALWAYS_F32 classification, base-type
    fallback counting, unknown-type abort;
  * estimate_breakdown — by_type/by_group/group_types structure.

CLI level (subprocess on a synthetic minimal GGUF header):
  * --json schema of the --compare (no config) branch: mode/input/tensors/
    by_type/by_group;
  * per-profile base type resolution (base_type_map: tier7 → Q6_K,
    tier13 → Q4_K, tier1–6 → Q8_0; non-tier profiles take --base-type).

Usage: python3 tests/run_all.py   (or: python3 -m unittest tests.test_estimate_size)
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EST = os.path.join(REPO_ROOT, "scripts", "estimate_size.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import estimate_size as est  # noqa: E402


def write_gguf(path, tensors):
    """Minimal GGUF v3 header (no KV pairs): tensors = [(name, dims, type_id)]."""
    buf = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 0)
    for name, dims, tid in tensors:
        nb = name.encode()
        buf += struct.pack("<Q", len(nb)) + nb
        buf += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        buf += struct.pack("<I", tid) + struct.pack("<Q", 0)
    with open(path, "wb") as f:
        f.write(buf)


# ── load_config ──────────────────────────────────────────────────────────────

class LoadConfigTests(unittest.TestCase):
    def test_parses_rules_skipping_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.txt")
            with open(path, "w") as f:
                f.write("# comment line\n"
                        "\n"
                        "  ^token_embd\\.weight$ = q4_k  \n"
                        "blk.0.attn_q=Q5_K\n"
                        "no-equals-line\n")
            rules = est.load_config(path)
        self.assertEqual(rules, [
            ("^token_embd\\.weight$", "Q4_K"),
            ("blk.0.attn_q", "Q5_K"),
        ])

    def test_types_are_uppercased(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.txt")
            with open(path, "w") as f:
                f.write("x=q6_k\n")
            rules = est.load_config(path)
        self.assertEqual(rules, [("x", "Q6_K")])


# ── match_type ───────────────────────────────────────────────────────────────

class MatchTypeTests(unittest.TestCase):
    def test_anchored_pattern_does_not_leak(self):
        # An anchored output-head rule must not capture attn_output.
        rules = [(r"^output\.weight$", "Q6_K")]
        self.assertIsNone(est.match_type("blk.3.attn_output.weight", rules))
        self.assertEqual(est.match_type("output.weight", rules), "Q6_K")

    def test_unanchored_first_match_wins(self):
        # std::regex_search replay: "blk.0.attn_q" also matches
        # "blk.0.attn_qkv.weight" — and the FIRST rule wins.
        rules = [(r"blk\.0\.attn_q", "Q4_K"),
                 (r"blk\.0\.attn_qkv\.weight", "Q6_K")]
        self.assertEqual(est.match_type("blk.0.attn_qkv.weight", rules), "Q4_K")
        self.assertEqual(est.match_type("blk.0.attn_q.weight", rules), "Q4_K")

    def test_no_match_returns_none(self):
        self.assertIsNone(est.match_type("blk.1.attn_q", [(r"blk\.0\.", "Q4_K")]))

    def test_invalid_regex_falls_back_to_exact_equality(self):
        rules = [("(bad-regex", "Q4_K")]
        self.assertEqual(est.match_type("(bad-regex", rules), "Q4_K")
        self.assertIsNone(est.match_type("x(bad-regex", rules))


# ── estimate / estimate_breakdown ────────────────────────────────────────────

class EstimateTests(unittest.TestCase):
    def test_bpw_arithmetic(self):
        # 0.8e9 params at Q4_K (4.5 bpw) = 0.45 GB
        tensors = [("blk.0.attn_q.weight", 800_000_000, "")]
        size_gb, by_type, uncovered, _, _, _, _ = est.estimate(
            tensors, [(r"attn_q", "Q4_K")], "Q8_0")
        self.assertAlmostEqual(size_gb, 0.45, places=9)
        self.assertEqual(by_type, {"Q4_K": 800_000_000})
        self.assertEqual(uncovered, 0)

    def test_always_f32_classification_beats_rules(self):
        tensors = [("blk.0.ffn_norm.weight", 4096, "")]
        size_gb, by_type, uncovered, _, _, _, _ = est.estimate(
            tensors, [(r"ffn_norm", "Q4_K")], "Q8_0")
        self.assertEqual(by_type, {"F32": 4096})
        self.assertAlmostEqual(size_gb, 4096 * 32.0 / 8 / 1e9, places=15)
        self.assertEqual(uncovered, 0)

    def test_uncovered_tensors_fall_back_to_base(self):
        tensors = [("blk.0.attn_q.weight", 1_000, ""),
                   ("blk.0.attn_k.weight", 1_000, "")]
        size_gb, by_type, uncovered, names, _, _, _ = est.estimate(
            tensors, [], "Q6_K")
        self.assertEqual(uncovered, 2)
        self.assertEqual(names, ["blk.0.attn_q.weight", "blk.0.attn_k.weight"])
        self.assertEqual(by_type, {"Q6_K": 2_000})
        self.assertAlmostEqual(size_gb, 2000 * est.BPW["Q6_K"] / 8 / 1e9, places=15)

    def test_unknown_type_aborts(self):
        tensors = [("blk.0.attn_q.weight", 1_000, "")]
        with self.assertRaises(SystemExit):
            est.estimate(tensors, [(r"attn_q", "BOGUS")], "Q8_0")

    def test_group_classification(self):
        tensors = [
            ("blk.0.ffn_gate_exps", 100, ""),
            ("blk.0.ffn_gate_shexp", 100, ""),
            ("blk.0.attn_q.weight", 100, ""),
            ("blk.0.nextn.enorm.weight", 100, ""),
            ("output_norm.weight", 100, ""),
        ]
        _, _, _, _, group_params, group_bits, cat_type_bits = est.estimate(
            tensors, [], "Q8_0")
        self.assertEqual(group_params["Experts"], 100)
        self.assertEqual(group_params["ShExperts"], 100)
        self.assertEqual(group_params["Attention"], 100)
        self.assertEqual(group_params["MTP"], 100)
        self.assertEqual(group_params["Other"], 100)

    def test_breakdown_structure(self):
        tensors = [("blk.0.attn_q.weight", 500, ""),
                   ("blk.0.ffn_gate_exps", 1500, "")]
        size_gb, by_type, uncovered, _, group_params, group_bits, cat_type_bits = \
            est.estimate(tensors, [], "Q4_K")
        bd = est.estimate_breakdown(by_type, group_params, group_bits,
                                    cat_type_bits, 2000)
        self.assertEqual(set(bd), {"by_type", "by_group", "group_types"})
        self.assertEqual(set(bd["by_group"]), {"Experts", "Attention"})
        self.assertEqual(bd["by_group"]["Experts"]["params"], 1500)
        self.assertAlmostEqual(bd["by_group"]["Experts"]["share"], 0.75)
        self.assertAlmostEqual(
            sum(v["params"] for v in bd["by_type"].values()), 2000)
        self.assertAlmostEqual(
            sum(v["size_gb"] for v in bd["by_type"].values()), size_gb, places=15)


# ── CLI: --json schema and per-profile base types ────────────────────────────

class CliJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="apex_est_size_")
        cls.gguf = os.path.join(cls._tmp.name, "synth.gguf")
        write_gguf(cls.gguf, [
            ("token_embd.weight", [1000, 4], 1),      # F16
            ("blk.0.attn_q.weight", [1000, 4], 1),
            ("blk.0.attn_qkv.weight", [1000, 4], 1),
            ("blk.0.ffn_gate.weight", [1000, 4], 1),
        ])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, EST, *args],
                              capture_output=True, text=True)

    def test_compare_json_schema(self):
        # --compare without a config or profile: pure inspection of the file.
        proc = self.run_cli("--profile", "", "--compare", "--json", self.gguf)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["mode"], "compare")
        self.assertEqual(data["input"], self.gguf)
        self.assertEqual(data["tensors"], 4)
        self.assertEqual(data["total_params"], 16000)
        self.assertIn("by_type", data)
        self.assertIn("by_group", data)
        self.assertIn("group_types", data)
        self.assertEqual(data["by_type"]["F16"]["params"], 16000)
        self.assertEqual(set(data["by_group"]), {"Attention", "Other"})

    def test_estimate_json_schema(self):
        proc = self.run_cli("--profile", "tier7", "--json",
                            "--layers", "2", self.gguf)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["mode"], "estimate")
        self.assertEqual(data["tensors"], 4)
        self.assertIn("estimated_size_gb", data)
        self.assertIn("by_type", data)
        self.assertIn("by_group", data)
        self.assertGreater(data["estimated_size_gb"], 0.0)

    def test_base_type_from_profile_map(self):
        cases = {"tier5": "Q8_0", "tier7": "Q6_K", "tier13": "Q4_K"}
        for profile, expected in cases.items():
            with self.subTest(profile=profile):
                proc = self.run_cli("--profile", profile, "--json",
                                    "--layers", "2", self.gguf)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["base_type"], expected)

    def test_base_type_override_for_non_tier_profile(self):
        # Profiles outside base_type_map (e.g. dense-flat) take the
        # --base-type CLI value.
        proc = self.run_cli("--profile", "dense-flat", "--base-type", "Q4_K",
                            "--json", "--layers", "2", self.gguf)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["base_type"], "Q4_K")


if __name__ == "__main__":
    unittest.main()
