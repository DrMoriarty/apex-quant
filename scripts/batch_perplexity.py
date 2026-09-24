#!/usr/bin/env python3
"""APEX Batch Perplexity / KL-Divergence Evaluation Pipeline.

Batch-evaluates quantized GGUF models with llama-perplexity.

Models are taken from S3 (primary backend) or from a HuggingFace repo.
Credentials are resolved exactly like batch_quantize.py (env / .env:
HF_TOKEN, S3_ENDPOINT, S3_KEY_ID + S3_SECRET or S3_TOKEN, S3_REGION).

Flow:
  1. List .gguf files in the S3 bucket (or HF repo).  When the storage
     holds files for more than one model, narrow the selection with
     --model-pattern (fnmatch wildcard on the file basename).
  2. Pick the source model (BF16 > F16 > F32), download it and run
     llama-perplexity with --save-all-logits to produce reference logits.
  3. Every remaining file (the quants) is evaluated one by one with
     --kl-divergence --kl-divergence-base against those logits:
     perplexity plus Mean / Maximum / Median / 99.9% KLD.
  4. Models are downloaded sequentially.  While llama-perplexity runs for
     one quant, the next one is downloaded in a background thread —
     exactly one file of lookahead, nothing more, to keep disk usage low.
     A quant's local file is deleted once its report is uploaded.
  5. Every llama-perplexity run is saved as a text report named after the
     model file (<model-name>-perplexity-report.txt) and uploaded back to
     S3 (next to the model, same prefix) or into the source HF repo.

Split GGUFs (Model-00001-of-000NN.gguf) are downloaded and merged
automatically; the report is named after the merged model.

Supports **resumable** execution: state is persisted to a JSON file in the
workspace, re-running with identical arguments only performs remaining
work (already-uploaded reports are skipped).

Usage:
  python3 scripts/batch_perplexity.py --repo user/model-GGUF
  python3 scripts/batch_perplexity.py --backend s3 --prefix models/Qwen35/ \
      --model-pattern "*Tier*"

S3 endpoint (bucket embedded in the URL, same rules as batch_quantize.py):
      https://storage.yandexcloud.net/<bucket>/
      https://<bucket>.storage.yandexcloud.net/
"""

import argparse
import fnmatch
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse shared infrastructure from the batch quantization pipeline:
# dotenv loading, network retry, HF helpers, resumable download,
# shard merge and S3 (SigV4 / IAM) request machinery.
from batch_quantize import (  # noqa: E402
    SHARD_RE,
    _allow_hard_interrupt,
    _check_hf_import,
    _download_single_file,
    _expand_shard_files,
    _format_elapsed,
    _get_hf_download_url,
    _get_hf_token,
    _get_s3_settings,
    _list_repo_gguf_files,
    _pick_source_gguf,
    _retry_on_network_error,
    _s3_auth_headers,
    _s3_check,
    _SPINNER_FRAMES,
    _uri_encode,
    merge_gguf_shards,
)

import httpx  # noqa: E402
import batch_quantize as _bq  # noqa: E402

# ---------------------------------------------------------------------------
# Rich live display  (optional — falls back to plain text logging)
# ---------------------------------------------------------------------------

_HAS_RICH = False
try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    _HAS_RICH = True
except ImportError:
    pass

_LOG_RING_SIZE = 8


class _LiveState:
    """Mutable namespace shared between the pipeline and the live display."""

    def __init__(self):
        self.ctx = None             # pipeline Context
        self.state = None           # PplState
        self.lines: list[str] = []  # log ring buffer
        self.lock = threading.Lock()
        self.live: Optional["_PerpLive"] = None
        self._last_refresh = 0.0


_ls = _LiveState() if _HAS_RICH else None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _refresh_live():
    """Throttled explicit refresh of the live display."""
    if _ls is not None and _ls.live is not None:
        now = time.time()
        if now - _ls._last_refresh >= 0.2:
            _ls._last_refresh = now
            try:
                _ls.live._live.refresh()
            except Exception:
                pass


def log(msg: str):
    line = f"[{_ts()}] {msg}"
    if _ls is not None and _ls.live is not None:
        with _ls.lock:
            _ls.lines.append(line)
            if len(_ls.lines) > _LOG_RING_SIZE * 3:
                _ls.lines = _ls.lines[-_LOG_RING_SIZE:]
        _refresh_live()
    else:
        print(line, flush=True)


def log_err(msg: str):
    line = f"[{_ts()}] \u274c {msg}"
    print(line, file=sys.stderr, flush=True)
    if _ls is not None and _ls.live is not None:
        with _ls.lock:
            _ls.lines.append(line)
            if len(_ls.lines) > _LOG_RING_SIZE * 3:
                _ls.lines = _ls.lines[-_LOG_RING_SIZE:]
        _refresh_live()


def _elide_middle(text: str, width: int) -> str:
    """Truncate *text* to *width* chars, cutting out the middle.

    Long model filenames share a long common prefix (and a .gguf suffix),
    so head+tail keeps the distinguishing parts visible:
    Qwen3-Coder-30B-A3B-Instruct-Tier12.gguf → Qwen3-Coder-30B…ier12.gguf
    """
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    head = (width - 1) // 2
    tail = width - 1 - head
    return text[:head] + "…" + text[len(text) - tail:]


_STATUS_STYLE = {
    "pending":    "dim",
    "evaluating": "yellow",
    "done":       "bold green",
    "error":      "bold red",
}

_STATUS_ICON = {
    "pending":    "○",
    "evaluating": "◉",
    "done":       "✓",
    "error":      "✗",
}


def display_status() -> "Panel":
    """Build the Rich renderable for the live display."""
    from rich.text import Text

    ctx = _ls.ctx
    state = _ls.state
    table = Table(show_lines=False, padding=(0, 1), expand=True)
    table.add_column("Task", justify="left", style="bold", width=8)
    table.add_column("Model", width=28, overflow="fold")
    table.add_column("Status", width=52)
    table.add_column("Info", ratio=1, style="dim", overflow="fold")

    # ── SRC row: reference logits run ──
    src = ctx.source if ctx else None
    if src is None:
        table.add_row("SRC", "—", Text(" ○ pending", style="dim"), "")
    elif src["status"] == "running":
        frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
        elapsed = _format_elapsed(time.time() - src.get("start", time.time()))
        table.add_row("SRC", "base logits",
                      Text(f" {frame} llama-perplexity --save-all-logits… "
                           f"{elapsed}", style="yellow"), "")
    elif src["status"] == "error":
        elapsed = _format_elapsed(src.get("elapsed", 0))
        table.add_row("SRC", "base logits",
                      Text(f" ✗ failed {elapsed}", style="bold red"), "")
    else:
        ppl = src.get("ppl", "?")
        table.add_row("SRC", "base logits",
                      Text(f" ✓ done", style="green"),
                      f"PPL = {ppl}")

    # ── DL row: current download (main thread or prefetch) ──
    dl = ctx.active_download if ctx else None
    if dl is not None:
        part = dl["dest"].with_suffix(dl["dest"].suffix + ".part")
        cur = part.stat().st_size if part.exists() else 0
        total = dl.get("total", 0)
        start = dl.get("start", time.time())
        elapsed = time.time() - start
        elapsed_str = _format_elapsed(elapsed)
        frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
        if total > 0 and cur > 0:
            pct = min(cur / total * 100, 100.0)
            bar_len = 20
            filled = min(int(pct / 100 * bar_len), bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            eta_sec = elapsed / cur * (total - cur) if cur else 0
            eta_str = _format_elapsed(eta_sec)
            status_col = Text(
                f" {frame} {bar} {pct:.1f}% {elapsed_str} ETA {eta_str}",
                style="cyan")
            # Instantaneous speed, EMA-smoothed across renders (session-based:
            # resumed bytes from a previous attempt would skew the average).
            now = time.time()
            prev_b = dl.get("_spd_bytes")
            prev_t = dl.get("_spd_ts")
            if prev_t is not None and now > prev_t:
                inst = max(0.0, (cur - prev_b) / (now - prev_t) / (1024**2))
                speed = 0.7 * dl.get("_spd", inst) + 0.3 * inst
            else:
                speed = dl.get("_spd", 0.0)
            dl["_spd_bytes"] = cur
            dl["_spd_ts"] = now
            dl["_spd"] = speed
            info = (f"{cur / (1024**3):.2f}/{total / (1024**3):.2f} GB"
                    f" · {speed:.1f} MB/s")
        elif total > 0:
            status_col = Text(f" {frame} starting… {elapsed_str}",
                              style="cyan")
            info = f"{total / (1024**3):.2f} GB"
        else:
            status_col = Text(f" {frame} downloading… {elapsed_str}",
                              style="cyan")
            info = ""
        table.add_row("DL", _elide_middle(dl["label"], 28), status_col, info)

    # ── Quant rows ──
    for group in (ctx.quant_groups if ctx else []):
        key = group["name"]
        st = state.status(key) if state else "pending"
        ev = ctx.active_eval if ctx else None
        if ev is not None and ev["name"] == key:
            frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
            elapsed = _format_elapsed(time.time() - ev["start"])
            status_col = Text(f" {frame} evaluating… {elapsed}",
                              style="yellow")
            info = "llama-perplexity --kl-divergence"
        else:
            icon = _STATUS_ICON.get(st, "?")
            style = _STATUS_STYLE.get(st, "")
            status_col = Text(f" {icon} {st}", style=style)
            info = ""
            e = state.entry(key) if state else {}
            if st == "done":
                info = (f"PPL={e.get('ppl', '?')} "
                        f"KL={e.get('kl_mean', '?')}")
            elif st == "error":
                info = Text(e.get("error", "")[:80], style="red")
        table.add_row("QUANT", _elide_middle(Path(key).name, 28),
                      status_col, info)

    subtitle = f"{ctx.backend_label} · {len(ctx.quant_groups)} quants" if ctx else ""
    return Panel(table, title="[bold]APEX Batch Perplexity[/bold]",
                 subtitle=subtitle, border_style="blue")


class _PerpLive:
    """Thin wrapper around rich.live.Live with a log ring buffer.

    Rich's ``get_renderable`` callback drives rendering from Rich's own
    daemon thread; pipeline threads only mutate shared state and call
    ``refresh()`` (throttled).
    """

    def __init__(self):
        self._last_good = None
        self._live = Live(
            console=Console(file=sys.stdout, force_terminal=True),
            auto_refresh=True,
            refresh_per_second=5,
            transient=True,
            get_renderable=self._composite,
        )

    def start(self):
        self._live.start()

    def stop(self):
        self._live.stop()

    def _composite(self):
        try:
            with _ls.lock:
                logs = list(_ls.lines[-_LOG_RING_SIZE:])
            parts = [display_status()]
            padded = logs + [""] * (_LOG_RING_SIZE - len(logs))
            parts.append(Text("\n".join(padded)))
            result = Group(*parts)
            self._last_good = result
            return result
        except Exception:
            if self._last_good is not None:
                return self._last_good
            return Text("… refreshing …")


class _ShimLive:
    """Adapter so batch_quantize's internal log() lines land in our ring.

    batch_quantize's helpers check ``_lc.live is not None`` and call
    ``append_log()`` (logging) / ``update()`` (their quantize display,
    unused here).  Installing the shim routes their log lines into our
    live display without starting their table.
    """

    def append_log(self, msg: str):
        if _ls is not None and _ls.live is not None:
            with _ls.lock:
                _ls.lines.append(msg)
                if len(_ls.lines) > _LOG_RING_SIZE * 3:
                    _ls.lines = _ls.lines[-_LOG_RING_SIZE:]
            _refresh_live()

    def update(self, *_args, **_kwargs):
        _refresh_live()


def _init_live():
    if not _HAS_RICH or _ls is None or _ls.live is not None:
        return
    _ls.live = _PerpLive()
    _ls.live.start()
    # Route batch_quantize's internal logs (merge, HF download, …) to us.
    if _bq._lc is not None:
        _bq._lc.live = _ShimLive()


def _stop_live():
    if not _HAS_RICH or _ls is None or _ls.live is None:
        return
    _ls.live.stop()
    _ls.live = None
    if _bq._lc is not None:
        _bq._lc.live = None
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISK_MARGIN = 1.25          # 25% headroom over the remote file size

_PPL_RE = re.compile(r"PPL\s*=\s*([0-9.]+)")
_KLD_RE = re.compile(r"(Mean|Maximum|Median|99\.9%)\s+KLD:\s+([0-9.]+)")
_KLD_FIELDS = {
    "Mean": "kl_mean",
    "Maximum": "kl_max",
    "Median": "kl_median",
    "99.9%": "kl_99_9",
}


def report_name_for(model_name: str) -> str:
    """Model-Tier3.gguf → Model-Tier3-perplexity-report.txt"""
    stem = Path(model_name).name
    if stem.endswith(".gguf"):
        stem = stem[:-len(".gguf")]
    return f"{stem}-perplexity-report.txt"


def parse_metrics(text: str) -> dict:
    """Extract PPL / KLD metrics from llama-perplexity output."""
    m: dict = {}
    ppls = _PPL_RE.findall(text)
    if ppls:
        m["ppl"] = ppls[-1]
    for label, value in _KLD_RE.findall(text):
        m.setdefault(_KLD_FIELDS[label], value)
    return m


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class PplState:
    """JSON-backed resumable state for the perplexity batch."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self.save()

    def entry(self, key: str) -> dict:
        return self.data.setdefault("files", {}).setdefault(key, {})

    def status(self, key: str) -> str:
        return self.entry(key).get("status", "pending")

    def set_file(self, key: str, status: str, **extra):
        e = self.entry(key)
        e["status"] = status
        e["updated"] = datetime.now().isoformat()
        e.update(extra)
        self.save()


# ---------------------------------------------------------------------------
# Binary & data discovery
# ---------------------------------------------------------------------------

def find_perplexity() -> Optional[str]:
    """Locate the llama-perplexity binary."""
    q = os.environ.get("LLAMA_PERPLEXITY", "")
    if q and os.path.isfile(q):
        return q

    d = os.environ.get("LLAMA_CPP_DIR", "")
    if d:
        for sub in ("llama-perplexity", "build/bin/llama-perplexity",
                    "bin/llama-perplexity"):
            p = os.path.join(d, sub)
            if os.path.isfile(p):
                return p

    candidates = [
        "./llama.cpp/build/bin/llama-perplexity",
        str(SCRIPT_DIR.parent / "llama.cpp" / "build" / "bin" / "llama-perplexity"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    return shutil.which("llama-perplexity")


def resolve_data_file(args, workspace: Path) -> Path:
    """Locate the perplexity corpus.

    Default: dataset/combined_all_micro.txt in the repo.  Override with
    --data /path/to/corpus.txt.
    """
    if args.data:
        p = Path(args.data)
        if not p.is_file():
            log_err(f"Data file not found: {p}")
            sys.exit(1)
        return p
    candidates = [
        SCRIPT_DIR.parent / "dataset" / "combined_all_micro.txt",
        workspace / "wiki.test.raw",
        workspace / "wikitext-2-raw" / "wiki.test.raw",
        Path("wiki.test.raw"),
        Path("wikitext-2-raw") / "wiki.test.raw",
        SCRIPT_DIR.parent / "llama.cpp" / "wiki.test.raw",
    ]
    for c in candidates:
        if c.is_file():
            return c
    log_err("Perplexity corpus not found "
            f"(default: {SCRIPT_DIR.parent / 'dataset' / 'combined_all_micro.txt'}). "
            "Pass --data /path/to/corpus.txt.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# S3 listing & download (batch_quantize.py only uploads — download is here)
# ---------------------------------------------------------------------------

def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@_retry_on_network_error
def _s3_list_objects(s3: dict, prefix: str = "") -> list:
    """List all objects in the bucket under *prefix* (paginated ListObjectsV2)."""
    out: list = []
    token = None
    with httpx.Client(timeout=httpx.Timeout(120, connect=30)) as client:
        while True:
            url = f"{s3['base_url']}/{s3['bucket']}/?list-type=2"
            if prefix:
                url += f"&prefix={_uri_encode(prefix)}"
            if token:
                url += f"&continuation-token={_uri_encode(token)}"
            resp = client.get(url, headers=_s3_auth_headers(s3, "GET", url))
            _s3_check(resp, "list objects")
            root = ET.fromstring(resp.text)
            for el in root:
                if _xml_local(el.tag) != "Contents":
                    continue
                key, size = None, 0
                for c in el:
                    lt = _xml_local(c.tag)
                    if lt == "Key":
                        key = c.text
                    elif lt == "Size":
                        size = int(c.text or 0)
                if key:
                    out.append({"key": key, "size": size})
            truncated, next_token = False, None
            for el in root:
                lt = _xml_local(el.tag)
                if lt == "IsTruncated":
                    truncated = (el.text or "").strip().lower() == "true"
                elif lt == "NextContinuationToken":
                    next_token = el.text
            if not truncated or not next_token:
                break
            token = next_token
    return out


@_retry_on_network_error
def _s3_download_file(s3: dict, key: str, dest: Path, label: str) -> Path:
    """Download an S3 object to *dest* with resume support (.part file)."""
    url = f"{s3['base_url']}/{s3['bucket']}/{quote(key)}"
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _allow_hard_interrupt():
        existing = part.stat().st_size if part.exists() else 0
        with httpx.Client(follow_redirects=True,
                          timeout=httpx.Timeout(300, connect=30)) as client:
            head = client.head(url, headers=_s3_auth_headers(s3, "HEAD", url))
            head.raise_for_status()
            total = int(head.headers.get("content-length", 0))

            if existing >= total > 0:
                if dest.exists():
                    dest.unlink()
                part.rename(dest)
                log(f"{label} complete (cached): {dest.name}  "
                    f"({total / (1024**3):.2f} GB)")
                return dest

            headers = _s3_auth_headers(s3, "GET", url)
            if 0 < existing < total:
                log(f"Resuming {label} from {existing / (1024**2):.1f} MB  "
                    f"({existing / total * 100:.1f}%)")
                headers["Range"] = f"bytes={existing}-"

            mode = "ab" if existing > 0 else "wb"
            bytes_done = existing
            last_log = time.time()
            with open(part, mode) as f:
                with client.stream("GET", url, headers=headers) as resp:
                    if existing > 0 and resp.status_code == 200:
                        # Server ignored Range — restart from scratch.
                        f.seek(0)
                        f.truncate()
                        bytes_done = 0
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        bytes_done += len(chunk)
                        now = time.time()
                        if now - last_log >= 30:
                            pct = bytes_done / total * 100 if total else 0
                            log(f"  {label}: {bytes_done / (1024**3):.2f}/"
                                f"{total / (1024**3):.2f} GB  ({pct:.1f}%)")
                            last_log = now

    actual = part.stat().st_size
    if total and actual != total:
        raise ValueError(
            f"Download size mismatch: expected {total}, got {actual} ({label})")
    if dest.exists():
        dest.unlink()
    part.rename(dest)
    log(f"{label} complete: {dest.name}  ({actual / (1024**3):.2f} GB)")
    return dest


@_retry_on_network_error
def _s3_put_file(s3: dict, key: str, path: Path, what: str = "file"):
    """PUT a small file to S3 (reports are plain text, simple PUT is fine)."""
    url = f"{s3['base_url']}/{s3['bucket']}/{quote(key)}"
    data = path.read_bytes()
    headers = _s3_auth_headers(s3, "PUT", url, data)
    headers["Content-Length"] = str(len(data))
    with httpx.Client(timeout=httpx.Timeout(300, connect=30)) as client:
        resp = client.put(url, content=data, headers=headers)
        _s3_check(resp, f"{what} upload")


@_retry_on_network_error
def _hf_file_size(repo_id: str, filename: str, token: Optional[str]) -> int:
    """Content-length of a file in an HF repo (0 if unknown)."""
    url = _get_hf_download_url(repo_id, filename, token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        r = client.head(url, headers=headers)
        r.raise_for_status()
        return int(r.headers.get("content-length", 0))


@_retry_on_network_error
def _upload_report_hf(report_path: Path, repo_id: str, path_in_repo: str,
                      token: Optional[str]):
    """Upload a perplexity report into the source HF repo."""
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(report_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add perplexity report: {Path(path_in_repo).name}",
    )


# ---------------------------------------------------------------------------
# Remote file model
# ---------------------------------------------------------------------------

class Context:
    """Shared pipeline context."""

    def __init__(self, args, backend, s3, token, repo, workspace):
        self.args = args
        self.backend = backend          # "s3" | "hf"
        self.s3 = s3
        self.token = token
        self.repo = repo
        self.workspace = workspace
        self.models_dir = workspace / "models"
        self.reports_dir = workspace / "reports"
        self.sizes: dict = {}           # remote path → size in bytes
        self.downloaded: dict = {}      # group name → local path (prefetch cache)
        self.executor: Optional[ThreadPoolExecutor] = None
        # live-display state
        self.backend_label = ""
        self.quant_groups: list = []
        self.source: Optional[dict] = None   # {"status", "start"/"elapsed", "ppl"}
        self.active_download: Optional[dict] = None  # {"label","dest","total","start"}
        self.active_eval: Optional[dict] = None      # {"name", "start"}


def group_shard_files(paths: list) -> list:
    """Group shard files into model groups.

    Returns an ordered list of {"name": display_name, "files": [remote paths]}.
    Single files become one-element groups; split GGUF shards are collapsed
    into a single group named after the shard prefix.
    """
    paths = sorted(paths)
    consumed = set()
    groups = []
    for p in paths:
        if p in consumed:
            continue
        shards = _expand_shard_files(p, paths)
        consumed.update(shards)
        if len(shards) > 1:
            m = SHARD_RE.match(p)
            name = f"{m.group(1)}.gguf" if m else p
        else:
            name = p
        groups.append({"name": name, "files": shards})
    return groups


def _download_entry(ctx: Context, group: dict) -> Path:
    """Download all files of a model group, merge shards, return local path."""
    _entry_size(ctx, group)   # resolve sizes for the download progress bar
    local = []
    for f in group["files"]:
        dest = ctx.models_dir / f
        ctx.active_download = {
            "label": Path(f).name, "dest": dest,
            "total": ctx.sizes.get(f, 0), "start": time.time(),
        }
        _refresh_live()
        try:
            if ctx.backend == "hf":
                _download_single_file(ctx.repo, f, ctx.models_dir, ctx.token,
                                      label=Path(f).name)
            else:
                _s3_download_file(ctx.s3, f, dest, label=Path(f).name)
        finally:
            ctx.active_download = None
            _refresh_live()
        if not dest.exists() or dest.stat().st_size == 0:
            raise FileNotFoundError(f"Download did not produce a file: {dest}")
        local.append(dest)

    if len(local) > 1:
        return merge_gguf_shards(local, keep_shards=ctx.args.keep_files)
    return local[0]


def _prefetch_task(ctx: Context, group: dict):
    """Background download of the next quant while the current one is being
    evaluated.  Never raises — errors are logged."""
    name = group["name"]
    try:
        path = _download_entry(ctx, group)
        ctx.downloaded[name] = path
    except Exception as exc:
        log_err(f"Prefetch of {name} failed: {str(exc)[:300]}")


def _entry_size(ctx: Context, group: dict) -> int:
    """Total remote size of a model group (lazily resolves HF sizes)."""
    total = 0
    for f in group["files"]:
        size = ctx.sizes.get(f, 0)
        if not size and ctx.backend == "hf":
            try:
                size = _hf_file_size(ctx.repo, f, ctx.token)
            except Exception:
                size = 0
            ctx.sizes[f] = size
        total += size
    return total


def _maybe_prefetch(ctx: Context, group: Optional[dict], held_bytes: int):
    """Start a background download of *group* if disk allows one extra file."""
    if group is None or ctx.executor is None:
        return None
    size = _entry_size(ctx, group)
    need = int(size * DISK_MARGIN) + held_bytes
    free = shutil.disk_usage(ctx.workspace).free
    if free < need:
        log(f"⏳ Skipping prefetch of {group['name']} — "
            f"need {need / (1024**3):.1f} GB, free {free / (1024**3):.1f} GB")
        return None
    return ctx.executor.submit(_prefetch_task, ctx, group)


# ---------------------------------------------------------------------------
# llama-perplexity runner
# ---------------------------------------------------------------------------

def run_llama_ppl(ppl_bin: str, model: Path, data: Path, report_path: Path, *,
                  save_logits: Optional[Path] = None,
                  kl_base: Optional[Path] = None,
                  ngl: int = 99, chunks: Optional[int] = None) -> float:
    """Run llama-perplexity, teeing full output into *report_path*.

    *save_logits* → --save-all-logits (reference/base logits run);
    *kl_base*     → --kl-divergence --kl-divergence-base (quant run).
    Returns elapsed seconds.  Raises RuntimeError on non-zero exit.
    """
    cmd = [str(ppl_bin), "-m", str(model), "-f", str(data), "-ngl", str(ngl)]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    if save_logits:
        cmd += ["--save-all-logits", str(save_logits)]
    if kl_base:
        cmd += ["--kl-divergence", "--kl-divergence-base", str(kl_base)]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"llama-perplexity: {Path(model).name}")
    log(f"  → report: {report_path.name}")
    t0 = time.time()
    with open(report_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if _bq._interruption_requested:
                proc.kill()
                proc.wait()
                raise KeyboardInterrupt("llama-perplexity interrupted by user")
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity exited with code {proc.returncode} "
            f"(see {report_path.name})")
    log(f"  ✓ done in {elapsed / 60:.1f} min")
    return elapsed


# ---------------------------------------------------------------------------
# Report upload
# ---------------------------------------------------------------------------

def upload_report(ctx: Context, group: dict, report_path: Path) -> bool:
    """Upload the report to the same place the model came from."""
    model_file = group["files"][0]
    remote_dir = posixpath.dirname(model_file)
    name = report_name_for(group["name"])
    if remote_dir:
        remote_path = f"{remote_dir}/{name}"
    else:
        remote_path = name

    if ctx.backend == "s3":
        _s3_put_file(ctx.s3, remote_path, report_path, what="report")
        log(f"  ✓ Report → s3://{ctx.s3['bucket']}/{remote_path}")
    else:
        _upload_report_hf(report_path, ctx.repo, remote_path, ctx.token)
        log(f"  ✓ Report → {ctx.repo}/{remote_path}")
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    ppl_bin = args.ppl_bin or find_perplexity()
    if not ppl_bin:
        log_err("llama-perplexity not found. Set LLAMA_PERPLEXITY or "
                "LLAMA_CPP_DIR, or pass --ppl-bin.")
        sys.exit(1)
    if not os.path.isfile(ppl_bin) and not shutil.which(ppl_bin):
        log_err(f"llama-perplexity binary not found at: {ppl_bin}")
        sys.exit(1)

    s3 = _get_s3_settings(args)
    backend = args.backend
    if backend == "auto":
        backend = "s3" if s3 else "hf"
    if backend == "s3" and not s3:
        log_err("Backend 's3' requested but S3 endpoint/credentials are not "
                "configured (set S3_ENDPOINT + keys in .env).")
        sys.exit(1)
    if backend == "hf":
        _check_hf_import()

    token = args.token or _get_hf_token()
    if backend == "hf" and not token:
        log_err("No HF token found. Set HF_TOKEN in .env or pass --token.")
        sys.exit(1)

    if backend == "hf" and (not args.repo or "/" not in args.repo):
        log_err("--repo must be an HF repo id in the form org/name.")
        sys.exit(1)

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    state = PplState(workspace / ".batch_ppl_state.json")
    data_file = resolve_data_file(args, workspace)

    ctx = Context(args, backend, s3, token, args.repo, workspace)
    if _ls is not None:
        _ls.ctx = ctx
        _ls.state = state
    if backend == "s3":
        ctx.backend_label = f"s3://{s3['bucket']}"
    else:
        ctx.backend_label = f"hf:{args.repo}"

    # ── header ──
    log("=" * 60)
    if args.dry_run:
        log("  *** DRY RUN MODE ***")
    log("  APEX Batch Perplexity Pipeline")
    log("=" * 60)
    if backend == "s3":
        log(f"  Backend:  s3://{s3['bucket']}  (prefix: {args.prefix or '/'})")
    else:
        log(f"  Backend:  hf:{args.repo}")
    log(f"  Pattern:  {args.model_pattern or '(all .gguf files)'}")
    log(f"  Data:     {data_file}")
    log(f"  NGL:      {args.ngl}  Chunks: {args.chunks or '(default)'}")
    log(f"  Workspace: {workspace}")
    log("=" * 60)

    _init_live()
    try:
        _run_pipeline_body(args, ctx, state, ppl_bin, data_file, backend,
                           token, s3)
    finally:
        _stop_live()


def _run_pipeline_body(args, ctx: Context, state: PplState, ppl_bin,
                       data_file: Path, backend, token, s3):
    # ── 1. Discover model files ──
    if backend == "s3":
        objects = _s3_list_objects(s3, args.prefix or "")
        remote_paths = [o["key"] for o in objects if o["key"].endswith(".gguf")]
        for o in objects:
            ctx.sizes[o["key"]] = o["size"]
    else:
        remote_paths = _list_repo_gguf_files(args.repo, token)

    if args.model_pattern:
        remote_paths = [p for p in remote_paths
                        if fnmatch.fnmatch(Path(p).name, args.model_pattern)]
    remote_paths = sorted(remote_paths)

    if not remote_paths:
        log_err("No .gguf files found matching the selection criteria.")
        sys.exit(1)
    log(f"Found {len(remote_paths)} .gguf file(s).")

    # ── 2. Pick source model (BF16 > F16 > F32) ──
    by_base = {}
    for p in remote_paths:
        by_base.setdefault(Path(p).name, p)
    if args.source_file:
        source_path = None
        for p in remote_paths:
            if p == args.source_file or Path(p).name == args.source_file:
                source_path = p
                break
        if not source_path:
            log_err(f"--source-file {args.source_file} not found among the "
                    f"listed .gguf files.")
            sys.exit(1)
    else:
        source_base = _pick_source_gguf(sorted(by_base))
        if not source_base:
            log_err("Could not determine the source model "
                    "(no BF16/F16/F32 file).")
            sys.exit(1)
        source_path = by_base[source_base]

    source_files = _expand_shard_files(source_path, remote_paths)
    if len(source_files) > 1:
        log(f"Source model is split into {len(source_files)} shards: "
            f"{source_files[0]} … {source_files[-1]}")

    groups = group_shard_files(remote_paths)
    source_group = next((g for g in groups
                         if source_files[0] in g["files"]), None)
    if source_group is None:
        log_err("Internal error: could not map source file to a model group.")
        sys.exit(1)
    source_name = source_group["name"]
    quant_groups = [g for g in groups if g["name"] != source_name]

    log(f"Source model: {source_name}")
    log(f"Quants to evaluate ({len(quant_groups)}):")
    for g in quant_groups:
        log(f"  - {g['name']}")
    ctx.quant_groups = quant_groups
    _refresh_live()

    if args.dry_run:
        log("\nDRY RUN: nothing downloaded or evaluated.")
        return

    if not quant_groups:
        log_err("No quantized variants found — only the source model is present.")
        sys.exit(1)

    ctx.models_dir.mkdir(parents=True, exist_ok=True)
    ctx.reports_dir.mkdir(parents=True, exist_ok=True)

    with _allow_hard_interrupt():
        # ── 3. Download source & generate base logits ──
        src_rec = state.get("source", {})
        logits_path = ctx.workspace / (Path(source_name).stem
                                       + "-reference-logits.bin")
        src_report = ctx.reports_dir / report_name_for(source_name)
        need_logits = not logits_path.exists()
        need_run = need_logits or src_rec.get("status") not in ("done",
                                                                "evaluated")
        if need_run:
            src_local = _download_entry(ctx, source_group)
            state.set("source", {"status": "downloaded",
                                 "path": str(src_local)})
            ctx.source = {"status": "running", "start": time.time()}
            _refresh_live()
            try:
                run_llama_ppl(ppl_bin, src_local, data_file, src_report,
                              save_logits=logits_path, ngl=args.ngl,
                              chunks=args.chunks)
            except Exception as exc:
                ctx.source = {"status": "error",
                              "elapsed": time.time()
                              - ctx.source.get("start", time.time())}
                _refresh_live()
                log_err(f"Source model evaluation failed: {str(exc)[:300]}")
                sys.exit(1)
            metrics = parse_metrics(src_report.read_text())
            ctx.source = {"status": "done",
                          "elapsed": time.time()
                          - ctx.source["start"],
                          "ppl": metrics.get("ppl")}
            _refresh_live()
            log(f"  Source PPL = {metrics.get('ppl', '?')}")
            state.set("source", {"status": "evaluated",
                                 "path": str(src_local),
                                 "logits": str(logits_path),
                                 "ppl": metrics.get("ppl")})
            if not args.keep_files:
                _cleanup_model(ctx, source_group, src_local)
        else:
            log(f"✓ Reference logits already exist: {logits_path.name}")
            ctx.source = {"status": "done", "ppl": src_rec.get("ppl")}
            _refresh_live()

        # Upload (or retry) the source report
        if state.get("source", {}).get("status") == "evaluated":
            try:
                upload_report(ctx, source_group, src_report)
                state.set("source", {**state.get("source", {}),
                                     "status": "done"})
            except Exception as exc:
                log_err(f"Source report upload failed: {str(exc)[:300]}")

        # ── 4. Evaluate quants sequentially, prefetch next in background ──
        ctx.executor = ThreadPoolExecutor(max_workers=1)
        results = {}
        failed = []

        def _next_pending(start: int) -> Optional[dict]:
            for g in quant_groups[start:]:
                if state.status(g["name"]) != "done":
                    return g
            return None

        for i, group in enumerate(quant_groups):
            if _bq._interruption_requested:
                log("⚠ Pipeline interrupted by user.")
                break

            key = group["name"]
            if state.status(key) == "done":
                log(f"✓ {key}: already evaluated, skipping")
                continue

            # Download (or pick up prefetch result)
            local = ctx.downloaded.pop(key, None)
            if local is None or not Path(local).exists():
                try:
                    local = _download_entry(ctx, group)
                except Exception as exc:
                    log_err(f"{key}: download failed: {str(exc)[:300]}")
                    state.set_file(key, "error", error=str(exc)[:300])
                    failed.append(key)
                    continue
            held = Path(local).stat().st_size

            # Start prefetch of the next pending quant (single lookahead)
            fut = _maybe_prefetch(ctx, _next_pending(i + 1), held)

            # Evaluate
            report_path = ctx.reports_dir / report_name_for(key)
            state.set_file(key, "evaluating")
            ctx.active_eval = {"name": key, "start": time.time()}
            _refresh_live()
            try:
                run_llama_ppl(ppl_bin, local, data_file, report_path,
                              kl_base=logits_path, ngl=args.ngl,
                              chunks=args.chunks)
                metrics = parse_metrics(report_path.read_text())
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                err = str(exc)[:300]
                log_err(f"{key}: evaluation failed: {err}")
                state.set_file(key, "error", error=err)
                failed.append(key)
                if fut is not None:
                    fut.result()
                continue
            finally:
                ctx.active_eval = None
                _refresh_live()

            log(f"  {key}: PPL={metrics.get('ppl', '?')} "
                f"KL mean={metrics.get('kl_mean', '?')} "
                f"max={metrics.get('kl_max', '?')}")
            state.set_file(key, "evaluated", **metrics)

            # Upload report
            try:
                upload_report(ctx, group, report_path)
            except Exception as exc:
                log_err(f"{key}: report upload failed: {str(exc)[:300]}")

            # Free disk: the local model file is no longer needed
            if not args.keep_files:
                _cleanup_model(ctx, group, local)

            state.set_file(key, "done", **metrics)
            results[key] = metrics

            # Wait for the prefetch to complete before the next iteration
            if fut is not None:
                fut.result()

        ctx.executor.shutdown(wait=True)

    # ── 5. Final summary ──
    _print_summary(state, source_name, quant_groups)


def _cleanup_model(ctx: Context, group: dict, local_path: Path):
    """Delete local GGUF (and leftover shards) after the report is uploaded."""
    removed = 0
    for f in group["files"]:
        p = ctx.models_dir / f
        if p.exists():
            removed += p.stat().st_size
            p.unlink()
    if local_path.exists():
        removed += local_path.stat().st_size
        local_path.unlink()
    if removed:
        log(f"🗑  Removed {group['name']} from disk "
            f"(freed {removed / (1024**3):.2f} GB)")


def _print_summary(state: PplState, source_name: str, quant_groups: list):
    log("\n" + "=" * 60)
    log("  Final Report")
    log("=" * 60)
    src = state.get("source", {})
    log(f"\n  Source: {source_name}  PPL={src.get('ppl', '?')}")
    log(f"\n  {'Model':<40} {'PPL':>9} {'KL mean':>9} {'KL max':>9}  Status")
    log(f"  {'─' * 40} {'─' * 9} {'─' * 9} {'─' * 9}  ──────")
    for g in quant_groups:
        key = g["name"]
        e = state.entry(key)
        st = e.get("status", "pending")
        ppl = e.get("ppl", "—")
        km = e.get("kl_mean", "—")
        kx = e.get("kl_max", "—")
        log(f"  {Path(key).name:<40} {ppl:>9} {km:>9} {kx:>9}  {st}")
    errors = [g["name"] for g in quant_groups
              if state.status(g["name"]) == "error"]
    if errors:
        log(f"\n  ⚠  Failed: {errors}")
        log("     Re-run the script with the same arguments to retry.")
    log("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="APEX Batch Perplexity / KL-Divergence Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/batch_perplexity.py \\\n"
            "    --repo user/Qwen3.5-35B-A3B-APEX\n"
            "  python3 scripts/batch_perplexity.py --backend s3 \\\n"
            "    --prefix models/Qwen3.5-35B-A3B/ --model-pattern \"*APEX*\"\n"
        ),
    )
    parser.add_argument("--repo", "-r",
                        help="HF repo with source + quant GGUFs "
                             "(e.g. user/model-APEX). Required for the HF "
                             "backend.")
    parser.add_argument("--backend", choices=["auto", "s3", "hf"],
                        default="auto",
                        help="Where models come from: s3, hf or auto "
                             "(S3 if S3_ENDPOINT is configured, else HF; "
                             "default: auto)")
    parser.add_argument("--prefix",
                        help="S3 key prefix to narrow the listing "
                             "(S3 backend only)")
    parser.add_argument("--model-pattern",
                        help="fnmatch wildcard for the model file basename, "
                             "e.g. \"*APEX*\" or \"Qwen35-Tier*\" — use when "
                             "the bucket/repo holds files of several models")
    parser.add_argument("--source-file",
                        help="Explicit source GGUF filename "
                             "(skip BF16/F16/F32 auto-detection)")
    parser.add_argument("--data",
                        help="Path to the perplexity corpus "
                             "(default: dataset/combined_all_micro.txt)")
    parser.add_argument("--workspace", "-w",
                        default=str(Path.home() / "apex_perplexity"),
                        help="Workspace directory for state, models & reports "
                             "(default: ~/apex_perplexity)")
    parser.add_argument("--token", "-t",
                        help="HF token (default: $HF_TOKEN from env / .env)")
    parser.add_argument("--ppl-bin",
                        help="Path to llama-perplexity "
                             "(default: $LLAMA_PERPLEXITY, $LLAMA_CPP_DIR, "
                             "./llama.cpp/build/bin or PATH)")
    parser.add_argument("--ngl", type=int,
                        default=int(os.environ.get("NGL", "99")),
                        help="GPU layers for llama-perplexity "
                             "(default: $NGL or 99)")
    parser.add_argument("--chunks", type=int,
                        help="Number of chunks for perplexity "
                             "(default: llama.cpp default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List discovered files and the plan, then exit")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep downloaded GGUF files on disk after "
                             "evaluation (default: deleted once the report "
                             "is uploaded)")
    parser.add_argument("--s3-endpoint",
                        help="S3 endpoint URL with bucket embedded "
                             "(default: $S3_ENDPOINT from env / .env)")
    parser.add_argument("--s3-token",
                        help="S3 IAM token (default: $S3_TOKEN from env / .env)")
    parser.add_argument("--s3-key-id",
                        help="S3 static access key id "
                             "(default: $S3_KEY_ID from env / .env)")
    parser.add_argument("--s3-secret",
                        help="S3 static access secret key "
                             "(default: $S3_SECRET from env / .env)")

    args = parser.parse_args()

    try:
        run_pipeline(args)
    finally:
        sys.stdout.flush()


if __name__ == "__main__":
    main()
