#!/usr/bin/env python3
"""Behavioural tests for scripts/generate_config.py (the active generator).

Dense/hybrid arms (the Qwen3.8-27B experiment): the A/B arms must each change
exactly one lever, the output-head rule must not swallow attn_output, and the
two meanings of "dense" must be rejected loudly instead of colliding silently.

MoE tier profiles: tier1–15 emit a valid config whose types follow PROFILES
through the ranked quant tables of the selected quant mode; role-shift
modifiers move ranks; zones split edge/near/mid; --f16-tensors prepends
anchored f16 lines.

llama-quantize matches --tensor-type-file entries with std::regex_search —
unanchored, first match wins (src/llama-quant.cpp:694). All name->type checks
below replay that exact semantics via first_match().

Usage: python3 tests/run_all.py   (or: python3 -m unittest tests.test_generate_config)
"""
import os
import re
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO_ROOT, "scripts", "generate_config.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import generate_config as gc  # noqa: E402

DL = 64  # Qwen3.8-27B: 64 layers, 48 linear-attn + 16 full-attn

# All concrete quant types the ranked tables can emit.
KNOWN_TYPES = set(gc.QUANTS_RANKED_SPEED) | set(gc.QUANTS_RANKED_SIZE)


def run_gen(*args):
    return subprocess.run([sys.executable, GEN, *args],
                          capture_output=True, text=True)


def gen_config(*args):
    """Run the generator, assert success, return the config text."""
    proc = run_gen(*args)
    if proc.returncode != 0:
        raise AssertionError(
            f"generate_config {' '.join(args)} failed rc={proc.returncode}:\n"
            f"{proc.stderr}")
    return proc.stdout


def parse_rules(config_text):
    """Config text -> [(pattern, qtype)] exactly as a consumer would read it."""
    rules = []
    for line in config_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        pat, qtype = line.rsplit("=", 1)
        rules.append((pat.strip(), qtype.strip()))
    return rules


def first_match(name, rules):
    """std::regex_search replay: unanchored, first match wins."""
    for pat, qtype in rules:
        if re.search(pat, name):
            return qtype
    return None


def types_for(config_text, suffix, layers=DL):
    """Set of types assigned to blk.N.<suffix> across all layers."""
    rules = parse_rules(config_text)
    types = set()
    for i in range(layers):
        t = first_match(f"blk.{i}.{suffix}", rules)
        if t is not None:
            types.add(t)
    return types


def expected_quant(index, role, mode):
    """Mirror gc.ranked_quant's table selection for a rank index."""
    if mode == "speed":
        table = gc.QUANTS_RANKED_SPEED
    elif mode == "size":
        table = gc.QUANTS_RANKED_SIZE
    else:  # mixed
        table = gc.QUANTS_RANKED_SIZE if role == "expert" else gc.QUANTS_RANKED_SPEED
    index = max(0, min(index, len(table) - 1))
    return table[index]


# ── Dense/hybrid mode ────────────────────────────────────────────────────────

class DenseArmsTests(unittest.TestCase):
    """The dense-*, flat/grad/hybrid experiment arms on scripts/generate_config.py."""

    def test_dense_flat_control_has_no_ffn_gradient(self):
        cfg = gen_config("--profile", "dense-flat", "--layers", str(DL))
        types = types_for(cfg, "ffn_gate.weight")
        self.assertEqual(types, {"IQ4_XS"},
                         f"dense-flat must be a flat control, got {types}")

    def test_dense_grad_applies_ffn_gradient(self):
        cfg = gen_config("--profile", "dense-grad", "--layers", str(DL))
        self.assertTrue(cfg.strip(), "dense-grad produced nothing usable")
        types = types_for(cfg, "ffn_gate.weight")
        self.assertGreater(len(types), 1,
                           f"dense-grad has no layer-position gradient: {types}")

    def test_dense_grad_leaves_attention_untouched(self):
        # Arm B isolates the FFN lever: if it also split attention, arm C would
        # test two changes at once. Both lookups must be non-empty, otherwise
        # two missing tensors would compare equal and pass vacuously.
        cfg = gen_config("--profile", "dense-grad", "--layers", str(DL))
        full = types_for(cfg, "attn_q.weight")
        lin = types_for(cfg, "attn_qkv.weight")
        self.assertTrue(full and lin, f"attention types missing: {full} / {lin}")
        self.assertEqual(full, lin,
                         "dense-grad must not split full-attn from linear-attn")

    def test_dense_hybrid_separates_attention(self):
        cfg = gen_config("--profile", "dense-hybrid", "--layers", str(DL))
        full = types_for(cfg, "attn_q.weight")
        lin = types_for(cfg, "attn_qkv.weight")
        self.assertTrue(full and lin, f"attention types missing: {full} / {lin}")
        self.assertNotEqual(full, lin,
                            "dense-hybrid must separate full-attn from linear-attn")

    def test_attn_output_takes_fullattn_type_not_output_head(self):
        # Regression: an unanchored "output.weight" rule also matches
        # "blk.N.attn_output.weight" (regex_search, first match wins) and would
        # silently mis-quantize every attention output projection. Dense
        # configs anchor their patterns; assert it stays that way.
        cfg = gen_config("--profile", "dense-hybrid", "--layers", str(DL))
        rules = parse_rules(cfg)
        aout = first_match("blk.0.attn_output.weight", rules)
        full = first_match("blk.0.attn_q.weight", rules)
        ohead = first_match("output.weight", rules)
        self.assertTrue(aout, "attn_output.weight not covered")
        self.assertEqual(aout, full, "attn_output must take the full-attn type")
        self.assertNotEqual(aout, ohead,
                            "attn_output leaked into the output-head type")

    def test_dense_pins_token_embd_and_output_with_anchors(self):
        cfg = gen_config("--profile", "dense-grad", "--layers", str(DL))
        self.assertIn("^token_embd\\.weight$=", cfg)
        self.assertIn("^output\\.weight$=", cfg)

    def test_dense_emits_no_exps_tensors(self):
        cfg = gen_config("--profile", "dense-grad", "--layers", str(DL))
        self.assertIsNone(re.search(r"ffn_(gate|up|down)_exps", cfg),
                          "expert tensors do not exist on a dense model")

    def test_arch_dense_with_dense_layers_is_rejected(self):
        # Guard against passing for the wrong reason: the same invocation
        # WITHOUT --dense-layers must succeed, so the rejection is provably
        # about the contradictory combination.
        baseline = run_gen("--profile", "dense-grad", "--layers", str(DL))
        if baseline.returncode != 0:
            self.fail("baseline dense-grad invocation does not even succeed")
        proc = run_gen("--profile", "dense-grad", "--layers", str(DL),
                       "--dense-layers", "3")
        self.assertNotEqual(proc.returncode, 0,
                            "--arch dense + --dense-layers accepted")
        self.assertIn("dense-layers", proc.stderr)

    def test_moe_profile_with_arch_dense_is_rejected(self):
        baseline = run_gen("--profile", "tier5", "--layers", "12")
        self.assertEqual(baseline.returncode, 0,
                         "tier5 alone must succeed (guard for the check below)")
        proc = run_gen("--profile", "tier5", "--layers", "12", "--arch", "dense")
        self.assertNotEqual(proc.returncode, 0,
                            "MoE profile accepted with --arch dense")
        self.assertIn("dense", proc.stderr.lower())

    def test_dense_profile_accepts_explicit_arch_dense(self):
        # dense-* implies --arch dense; passing it explicitly must not reject.
        proc = run_gen("--profile", "dense-grad", "--layers", str(DL),
                       "--arch", "dense")
        self.assertEqual(proc.returncode, 0, proc.stderr)


# ── Tier profiles, quant modes, modifiers, zoning, f16 ───────────────────────

class TierProfileTests(unittest.TestCase):
    LAYERS = 12

    def mid_layer(self):
        # layers=12 -> zone_size=2: edge 0-1,10-11; near 2-3,8-9; mid 4-7.
        return 5

    def test_all_tiers_emit_valid_config(self):
        for t in range(1, 16):
            with self.subTest(tier=t):
                cfg = gen_config("--profile", f"tier{t}",
                                 "--layers", str(self.LAYERS))
                rules = parse_rules(cfg)
                self.assertTrue(rules)
                for pat, qtype in rules:
                    self.assertIn(qtype, KNOWN_TYPES,
                                  f"tier{t}: unknown type {qtype}")
                self.assertEqual(rules[0][0], "token_embd.weight")
                self.assertEqual(rules[1][0], "output.weight")
                names = [p for p, _ in rules]
                self.assertIn("blk.0.ffn_gate_exps", names)
                self.assertIn("blk.0.attn_q", names)
                # per-layer blocks in layer order
                self.assertTrue(names[2].startswith("blk.0."))
                self.assertTrue(names[-1].startswith(f"blk.{self.LAYERS - 1}."))

    def test_tier_types_follow_profiles_table(self):
        # Mid-layer spot checks: experts, shared experts and attention of tier t
        # must equal PROFILES[t] mapped through the mode's ranked tables.
        i = self.mid_layer()
        for t in range(1, 16):
            profile = gc.PROFILES[f"tier{t}"]
            ei, ni, mi, esi, msi, eai, mai, emi = profile
            with self.subTest(tier=t):
                cfg = gen_config("--profile", f"tier{t}",
                                 "--layers", str(self.LAYERS))
                rules = parse_rules(cfg)
                self.assertEqual(
                    first_match(f"blk.{i}.ffn_gate_exps", rules),
                    expected_quant(mi, "expert", "mixed"))
                self.assertEqual(
                    first_match(f"blk.{i}.ffn_gate_shexp", rules),
                    expected_quant(msi, "shared", "mixed"))
                self.assertEqual(
                    first_match(f"blk.{i}.attn_q", rules),
                    expected_quant(mai, "attn", "mixed"))
                self.assertEqual(
                    first_match("token_embd.weight", rules),
                    expected_quant(emi, "embd", "mixed"))

    def test_quant_modes_switch_ranked_tables(self):
        i = self.mid_layer()
        t = 10  # tier10: mid_exp=4, mid_attn=3 — tables diverge at index >= 3
        mi, mai = gc.PROFILES["tier10"][2], gc.PROFILES["tier10"][6]
        for mode, flag in [("speed", "--speed"), ("size", "--size"),
                           ("mixed", "--mixed")]:
            with self.subTest(mode=mode):
                cfg = gen_config("--profile", "tier10",
                                 "--layers", str(self.LAYERS), flag)
                rules = parse_rules(cfg)
                self.assertEqual(
                    first_match(f"blk.{i}.ffn_gate_exps", rules),
                    expected_quant(mi, "expert", mode))
                self.assertEqual(
                    first_match(f"blk.{i}.attn_q", rules),
                    expected_quant(mai, "attn", mode))

    def test_role_shift_modifiers_move_ranks(self):
        # resolve_profile applies CLI modifiers to the PROFILES indices.
        cases = [
            (["--mid-exp", "-1"], 2, -1),
            (["--edge-exp", "1"], 0, 1),
            (["--near-exp", "2"], 1, 2),
            (["--edge-shared", "-1"], 3, -1),
            (["--mid-shared", "1"], 4, 1),
            (["--edge-attn", "1"], 5, 1),
            (["--mid-attn", "-1"], 6, -1),
            (["--embd-type", "1"], 7, 1),
        ]
        for extra, slot, delta in cases:
            with self.subTest(modifier=extra[0]):
                args = gc.parse_args(["--profile", "tier5", "--layers",
                                      str(self.LAYERS), *extra])
                cfg = gc.resolve_profile(args)
                base = gc.PROFILES["tier5"][slot]
                self.assertEqual(cfg["indices"][slot], base + delta)

    def test_mid_exp_modifier_changes_emitted_type(self):
        i = self.mid_layer()
        default = gen_config("--profile", "tier5", "--layers", str(self.LAYERS))
        shifted = gen_config("--profile", "tier5", "--layers", str(self.LAYERS),
                             "--mid-exp", "-1")
        t_default = first_match(f"blk.{i}.ffn_gate_exps", parse_rules(default))
        t_shifted = first_match(f"blk.{i}.ffn_gate_exps", parse_rules(shifted))
        # tier5 mid_exp=3; -1 lands on rank 2 (both through the SIZE table in
        # mixed mode): IQ4_NL -> Q5_K, i.e. the modifier moved the rank.
        self.assertEqual(t_default, expected_quant(3, "expert", "mixed"))
        self.assertEqual(t_shifted, expected_quant(2, "expert", "mixed"))

    def test_embd_type_modifier_changes_embedding(self):
        default = gen_config("--profile", "tier5", "--layers", str(self.LAYERS))
        shifted = gen_config("--profile", "tier5", "--layers", str(self.LAYERS),
                             "--embd-type", "1")
        t_default = first_match("token_embd.weight", parse_rules(default))
        t_shifted = first_match("token_embd.weight", parse_rules(shifted))
        self.assertEqual(t_default, "Q8_0")
        self.assertEqual(t_shifted, "Q6_K")

    def test_edge_attn_modifier_changes_edge_attention(self):
        cfg = gen_config("--profile", "tier5", "--layers", str(self.LAYERS),
                         "--edge-attn", "1")
        # tier5 edge_attn=1 (+1 -> 2); layer 0 is in the narrow attention edge band.
        self.assertEqual(first_match("blk.0.attn_q", parse_rules(cfg)),
                         expected_quant(2, "attn", "mixed"))

    def test_zone_boundaries(self):
        # layers=64, no dense layers: zone_size=ceil(6.4)=7 — edge 0-6 and
        # 57-63, near 7-13 and 50-56, mid everywhere between.
        zones = {i: gc.get_zone(i, 64) for i in range(64)}
        self.assertEqual({i for i, z in zones.items() if z == "edge"},
                         set(range(0, 7)) | set(range(57, 64)))
        self.assertEqual({i for i, z in zones.items() if z == "near"},
                         set(range(7, 14)) | set(range(50, 57)))
        self.assertEqual({i for i, z in zones.items() if z == "mid"},
                         set(range(14, 50)))

    def test_zone_with_dense_layers(self):
        # Leading dense layers are always edge; the remaining layers keep the
        # edge/near bands of 10% per side (zone_size=ceil(61*0.1)=7), shifted
        # by dense_layers: virtual j = i - 3.
        self.assertEqual(gc.get_zone(2, 64, 3), "edge")
        self.assertEqual(gc.get_zone(3, 64, 3), "edge")   # j=0, first non-dense
        self.assertEqual(gc.get_zone(9, 64, 3), "edge")   # j=6, last edge
        self.assertEqual(gc.get_zone(10, 64, 3), "near")  # j=7, first near
        self.assertEqual(gc.get_zone(16, 64, 3), "near")  # j=13, last near
        self.assertEqual(gc.get_zone(17, 64, 3), "mid")
        self.assertEqual(gc.get_zone(20, 64, 3), "mid")

    def test_f16_tensors_prepended_anchored(self):
        n1 = "blk.12.nextn.enorm.weight"
        n2 = "blk.12.nextn.hnorm.weight"
        cfg = gen_config("--profile", "tier5", "--layers", "2",
                         "--f16-tensors", f"{n1}, {n2}")
        lines = cfg.splitlines()
        # insert(0) per name -> the last name ends up first
        self.assertEqual(lines[0], f"^{re.escape(n2)}$=f16")
        self.assertEqual(lines[1], f"^{re.escape(n1)}$=f16")
        self.assertEqual(lines[2], "token_embd.weight=Q8_0")


if __name__ == "__main__":
    unittest.main()
