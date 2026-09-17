#!/usr/bin/env python3
"""Create reduced imatrix calibration files preserving text diversity."""

import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input.txt>")
    sys.exit(1)

SRC = Path(sys.argv[1])
MAX_CHUNK = 2000  # max bytes per chunk before splitting

stem = SRC.stem
suffix = SRC.suffix
out_50 = SRC.with_name(f"{stem}-light{suffix}")
out_25 = SRC.with_name(f"{stem}-ultralight{suffix}")

with open(SRC, "r") as f:
    text = f.read()

# Split into paragraphs, then further split large paragraphs by sentences
paragraphs = re.split(r'\n\n+', text)
paragraphs = [p for p in paragraphs if p.strip()]

chunks = []
for p in paragraphs:
    if len(p) <= MAX_CHUNK:
        chunks.append(p)
    else:
        # Split by sentences (keep sentence boundaries)
        sents = re.split(r'(?<=[.!?])\s+', p)
        buf = ""
        for s in sents:
            if len(buf) + len(s) + 1 > MAX_CHUNK and buf:
                chunks.append(buf.strip())
                buf = s
            else:
                buf = buf + " " + s if buf else s
        if buf.strip():
            chunks.append(buf.strip())

total = len(text)
n = len(chunks)
print(f"Original: {len(paragraphs)} paragraphs -> {n} chunks, {total} bytes")

def select_chunks(chunks, target_bytes, label):
    """Select chunks with even stride to hit target_bytes."""
    best = None
    for stride in range(2, 50):
        for start in range(stride):
            indices = list(range(start, n, stride))
            size = sum(len(chunks[i]) for i in indices)
            # Must not exceed target by more than 5%
            if size > target_bytes * 1.05:
                continue
            ratio = size / target_bytes
            if best is None or ratio > best[0]:
                best = (ratio, stride, start, indices, size)
            if 0.95 <= ratio <= 1.05:
                return best
        if best and best[0] >= 0.92:
            return best
    return best

target_50 = total // 2
target_25 = total // 4

r50 = select_chunks(chunks, target_50, "50%")
r25 = select_chunks(chunks, target_25, "25%")

for label, r, fname in [
    ("50%", r50, out_50),
    ("25%", r25, out_25),
]:
    ratio, stride, start, indices, size = r
    print(f"{label}: stride={stride}, start={start}, "
          f"{len(indices)}/{n} chunks, {size} bytes ({size*100//total}%)")
    selected = [chunks[i] for i in indices]
    output = "\n\n".join(selected) + "\n"
    with open(fname, "w") as f:
        f.write(output)
    actual = len(output)
    print(f"  Written {fname}: {actual} bytes ({actual*100//total}%)")

print("\nDone.")
