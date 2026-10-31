#!/usr/bin/env python3
"""Regression guard: mid-body stream drops must resume, not kill the run.

with_retries wraps only the request open; the body is read lazily inside
generators, so a ReadTimeout mid-stream used to escape every retry filter
and abort the whole `cp`.  stream_with_retries owns the body iteration now;
these tests prove the resume path works for every download route:

  - S3Client.get_range_iter   (Range GET resumes from the received offset)
  - S3Client.get_full_iter    (full GET re-issued, prefix discarded)
  - _fetch_concurrent parts   (--threads>1 local copies)
  - HFClient.get_range_iter   (hf:// -> local)

Run: python3 tests/test_cloudcp_stream_retry.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cloudcp  # noqa: E402
import httpx  # noqa: E402

DATA = bytes(range(256)) * 8192  # 2 MiB
# the fake body must deliver MORE than READ_CHUNK before dropping: httpx
# buffers up to READ_CHUNK before handing bytes upstream, so a smaller drop
# loses everything inside httpx (true for real servers too — the resume
# granularity is READ_CHUNK)
DROP_AT = cloudcp.READ_CHUNK + 65536
log: list[str] = []


def make_transport(drop_at: int = DROP_AT, total: int = len(DATA)):
    """Range-aware fake server; the body drops after `drop_at` bytes
    delivered per connection (first two connections only)."""
    state = {"drops": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range", "")
        start, end = 0, total - 1
        if rng.startswith("bytes="):
            lo, hi = rng[6:].split("-")
            start, end = int(lo), int(hi)
        status = 206 if start > 0 else 200
        log.append(f"{request.url.path} {rng} -> {status}")

        def body():
            if state["drops"] < 2:
                state["drops"] += 1
                cut = min(start + drop_at, end + 1)
                # one big chunk (as a real server streams), then the
                # connection stalls: httpx.ReadTimeout mid-body
                yield DATA[start:cut]
                raise httpx.ReadTimeout("The read operation timed out")
            yield DATA[start:end + 1]

        return httpx.Response(status, headers={
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(end + 1 - start),
        }, content=body())

    return httpx.MockTransport(handler)


def make_s3(transport) -> cloudcp.S3Client:
    s3 = cloudcp.S3Client.__new__(cloudcp.S3Client)
    s3.base = "https://fake.s3"
    s3.default_bucket = "bkt"
    s3.region = "ru-central1"
    s3.auth = ("bearer", "tok")
    s3.client = httpx.Client(timeout=cloudcp.HTTP_TIMEOUT,
                             transport=transport)
    return s3


def test_range_iter_resumes_from_offset():
    log.clear()
    s3 = make_s3(make_transport())
    got = b"".join(s3.get_range_iter("bkt", "k", 0, len(DATA)))
    assert got == DATA, f"payload mismatch: {len(got)}/{len(DATA)}"
    # retry must re-open with a Range starting at the received offset
    # (delivery granularity is READ_CHUNK, so the resume point is
    # READ_CHUNK-aligned)
    assert log[1].split(" ", 1)[1].startswith(
        f"bytes={cloudcp.READ_CHUNK}-"), log
    assert "-> 206" in log[1], log
    print("ok: S3 get_range_iter resumes from offset after ReadTimeout")


def test_full_iter_discards_prefix_and_redownloads():
    log.clear()
    s3 = make_s3(make_transport())
    got = b"".join(s3.get_full_iter("bkt", "k"))
    assert got == DATA, f"payload mismatch: {len(got)}/{len(DATA)}"
    print("ok: S3 get_full_iter re-issues full GET after mid-body drop")


def test_fetch_concurrent_survives_part_drop():
    log.clear()
    cloudcp._S3 = make_s3(make_transport())  # inject fake for _range_body
    part = 512 * 1024
    tasks = []
    off = 0
    while off < len(DATA):
        ln = min(part, len(DATA) - off)
        tasks.append((off, ln,
                      cloudcp._range_body(
                          cloudcp.URI("s3", bucket="bkt", path="k",
                                      repo_id="", repo_type="model",
                                      revision=""),
                          off, ln, None)))
        off += ln
    got = cloudcp._fetch_concurrent(tasks, threads=4)
    assert b"".join(got) == DATA, "concurrent fetch payload mismatch"
    print("ok: _fetch_concurrent parts recover from mid-body ReadTimeout")


def test_hf_range_iter_resumes():
    log.clear()
    hf = cloudcp.HFClient.__new__(cloudcp.HFClient)
    hf.endpoint = "https://hf.fake"
    hf.token = None
    hf._api = None
    hf.client = httpx.Client(timeout=cloudcp.HTTP_TIMEOUT,
                             transport=make_transport(),
                             follow_redirects=True)
    uri = cloudcp.URI("hf", repo_id="r/m", repo_type="model",
                      revision="main", path="f.gguf")
    got = b"".join(hf.get_range_iter(uri, 0, len(DATA)))
    assert got == DATA, f"payload mismatch: {len(got)}/{len(DATA)}"
    print("ok: HF get_range_iter resumes from offset after ReadTimeout")


if __name__ == "__main__":
    test_range_iter_resumes_from_offset()
    test_full_iter_discards_prefix_and_redownloads()
    test_fetch_concurrent_survives_part_drop()
    test_hf_range_iter_resumes()
    print("ALL PASS")
