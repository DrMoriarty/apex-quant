#!/usr/bin/env python3
"""Tests for scripts/reduce_calibration.py (calibration text reduction).

The script shrinks an imatrix calibration text into -light (~50%) and
-ultralight (~25%) variants while preserving text diversity: chunks are
selected with an even stride across the whole document (no head-biased
truncation), and oversized paragraphs are split on sentence boundaries so no
chunk exceeds MAX_CHUNK.

Runs the script as a subprocess on a synthetic calibration file in a temp dir
(it is a module-level script driven by sys.argv, not importable).

Usage: python3 tests/run_all.py   (or: python3 -m unittest tests.test_reduce_calibration)
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUCE = os.path.join(REPO_ROOT, "scripts", "reduce_calibration.py")


def make_calibration_text():
    """~19 KB of distinct text: 24 short paragraphs + one 6 KB monster.

    Every paragraph is unique (numbered sentences), so "output is a subset of
    the input" and "indices span the document" are both checkable.
    """
    paragraphs = []
    for p in range(24):
        sentences = [f"Paragraph {p:02d} sentence {s:02d} on quantization "
                     f"error dynamics of expert {p * 10 + s}." for s in range(5)]
        paragraphs.append(" ".join(sentences))
    # one paragraph far beyond MAX_CHUNK (2000) -> must be split by sentences
    long_sentences = [f"Monster paragraph sentence {s:03d} discusses the "
                      f"per-layer sensitivity of the {s}-th FFN block in "
                      f"detail, with enough words to fill space evenly."
                      for s in range(60)]
    paragraphs.append(" ".join(long_sentences))
    return "\n\n".join(paragraphs) + "\n"


class ReduceCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="apex_reduce_cal_")
        cls.src = os.path.join(cls._tmp.name, "calibration.txt")
        cls.text = make_calibration_text()
        with open(cls.src, "w") as f:
            f.write(cls.text)
        cls.total = len(cls.text)
        proc = subprocess.run([sys.executable, REDUCE, cls.src],
                              capture_output=True, text=True)
        cls.proc = proc
        cls.light = os.path.join(cls._tmp.name, "calibration-light.txt")
        cls.ultra = os.path.join(cls._tmp.name, "calibration-ultralight.txt")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def read(self, path):
        with open(path) as f:
            return f.read()

    def paragraphs(self, text):
        # strip(): the trailing part before EOF keeps the file's final newline
        return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    def original_paragraphs(self):
        return self.paragraphs(self.text)

    def test_runs_successfully_and_writes_both_variants(self):
        self.assertEqual(self.proc.returncode, 0,
                         f"rc={self.proc.returncode}\n{self.proc.stderr}")
        self.assertTrue(os.path.isfile(self.light), "-light variant missing")
        self.assertTrue(os.path.isfile(self.ultra), "-ultralight variant missing")

    def test_variant_sizes_hit_the_targets(self):
        light = len(self.read(self.light))
        ultra = len(self.read(self.ultra))
        self.assertTrue(0.40 <= light / self.total <= 0.62,
                        f"light at {light / self.total:.0%}, expected ~50%")
        self.assertTrue(0.15 <= ultra / self.total <= 0.35,
                        f"ultralight at {ultra / self.total:.0%}, expected ~25%")
        self.assertGreater(light, ultra,
                           "-light must be larger than -ultralight")

    def test_diversity_preserved_subset_of_original(self):
        originals = self.original_paragraphs()
        for path in (self.light, self.ultra):
            with self.subTest(variant=os.path.basename(path)):
                paras = self.paragraphs(self.read(path))
                self.assertGreater(len(paras), 1,
                                   "single-chunk output means no stride spread")
                for p in paras:
                    self.assertTrue(any(p in o for o in originals),
                                    f"chunk not from the original text: {p[:60]!r}")

    def test_chunks_span_the_whole_document(self):
        # Even-stride selection: the first selected chunk must come from the
        # first third of the document and the last one from the last third.
        originals = self.original_paragraphs()
        for path in (self.light, self.ultra):
            with self.subTest(variant=os.path.basename(path)):
                paras = self.paragraphs(self.read(path))
                indices = [next(i for i, o in enumerate(originals) if p in o)
                           for p in paras]
                self.assertLess(min(indices), len(originals) / 3,
                                "selection misses the head of the document")
                self.assertGreater(max(indices), 2 * len(originals) / 3,
                                   "selection misses the tail of the document")

    def test_no_chunk_exceeds_max_chunk(self):
        # The 6 KB monster paragraph must have been split on sentence
        # boundaries; every emitted chunk stays within MAX_CHUNK (2000).
        for path in (self.light, self.ultra):
            with self.subTest(variant=os.path.basename(path)):
                for p in self.paragraphs(self.read(path)):
                    self.assertLessEqual(len(p), 2000,
                                         f"oversized chunk ({len(p)} bytes)")

    def test_missing_argument_is_rejected(self):
        proc = subprocess.run([sys.executable, REDUCE],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Usage", proc.stdout)


if __name__ == "__main__":
    unittest.main()
