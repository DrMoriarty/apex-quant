#!/usr/bin/env python3
"""mAPEX Batch Quantization Pipeline.

Downloads a source GGUF model (bf16/f16/f32) and an importance matrix from
HuggingFace, quantizes through mAPEX tiers using quantize.py, and
uploads every resulting GGUF to HuggingFace.

Each tier in the speed range (9-15) is produced in two variants:
  * normal   — default quantize.py parameters          (Tier9)
  * speed    — with the ``--speed`` flag passed on     (Tier9-s)

Manages a README.md in the output repo: downloads an existing README or
creates one with source info, mAPEX attribution, and a quantization table.
After all tiers are uploaded the README is pushed to the same repo.

Supports **resumable** execution: state is persisted to a JSON file so that
re-running with identical arguments only performs remaining work.

Concurrency:
  * model + imatrix downloads run sequentially in the main thread
    (so Ctrl+C can interrupt them)
  * tiers are quantized sequentially, but tier N upload overlaps with
    tier N+1 quantization (upload runs in a background thread)
  * when S3 upload is enabled (--s3-endpoint/--s3-token), each tier is
    uploaded to HF and to S3 in parallel: at most one quant is being
    uploaded to HF and at most one to S3 at any given moment
  * with --s3-upload-source the source model (the merged GGUF when the
    source was split into shards) is uploaded to S3 as a separate "SRC"
    task shown in the live table.  While it runs, tier uploads are
    blocked; quantization itself is not blocked.

Usage:
  python3 scripts/batch_quantize.py \\
      --model user/source-model-GGUF \\
      --imatrix user/imatrix-repo \\
      --output user/model-mAPEX

Output repo: {output}  (all tier GGUFs in one repo)

S3 upload (optional, S3-compatible API e.g. Yandex Object Storage):
  The bucket must be embedded in the endpoint URL:
      https://storage.yandexcloud.net/<bucket>/
      https://<bucket>.storage.yandexcloud.net/
  Authentication: static access keys (S3_KEY_ID + S3_SECRET, AWS SigV4
  signing — recommended for long runs) or an IAM token (S3_TOKEN,
  Authorization: Bearer, expires in ~12 h).

Environment / .env:
  HF_TOKEN     HuggingFace access token
  S3_ENDPOINT  S3 endpoint URL with bucket (enables S3 upload)
  S3_KEY_ID    S3 static access key id (with S3_SECRET)
  S3_SECRET    S3 static access secret key
  S3_TOKEN     S3 IAM token (alternative to static keys)
  S3_REGION    S3 region for SigV4 signing (default: ru-central1)
"""

import argparse
import io
import json
import os
import pty
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tty
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Split GGUF shard filename: <prefix>-00001-of-00002.gguf
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")

_interruption_requested = False


def _handle_sigint(sig, frame):
    global _interruption_requested
    _interruption_requested = True
    print("\n⚠  Interrupt requested — finishing current step …")


signal.signal(signal.SIGINT, _handle_sigint)


import contextlib


@contextlib.contextmanager
def _allow_hard_interrupt():
    """On Ctrl+C, kill the process immediately (no cleanup).

    huggingface_hub catches KeyboardInterrupt in its retry/tqdm loops
    and swallows it, so we must os._exit to actually stop.

    No-op outside the main thread (signal.signal only works there).
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.getsignal(signal.SIGINT)

    def _hard_exit(sig, frame):
        os.write(2, b"\n\033[?25h")
        os._exit(130)

    signal.signal(signal.SIGINT, _hard_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def _load_dotenv():
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("\"'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


# ---------------------------------------------------------------------------
# Network retry
# ---------------------------------------------------------------------------

try:
    import httpx as _httpx
    _HTTPX_ERRORS = (_httpx.HTTPError,)
except ImportError:
    _httpx = None
    _HTTPX_ERRORS = ()

_NETWORK_ERRORS = (
    OSError,
    ConnectionError,
    TimeoutError,
) + _HTTPX_ERRORS


def _retry_on_network_error(fn=None, *, max_retries=5, delay=10):
    """Decorator: retry a function on transient network errors.

    Prints a user-visible warning before each retry so the operator knows
    the script is waiting for connectivity to recover.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except _NETWORK_ERRORS as exc:
                    if attempt == max_retries:
                        raise
                    log_err(f"Network error in {func.__name__}: {exc}")
                    log(f"  → Проблемы с интернетом, повтор через {delay} сек. "
                        f"(попытка {attempt}/{max_retries})")
                    time.sleep(delay)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = list(range(1, 16))

# Tiers that get an extra "--speed" variant (Tier9-s alongside Tier9).
SPEED_TIERS = set(range(9, 16))

TIER_BASE_TYPE = {
    1: "Q8_0", 2: "Q8_0", 3: "Q8_0", 4: "Q8_0", 5: "Q8_0", 6: "Q8_0",
    7: "Q6_K", 8: "Q6_K", 9: "Q6_K",
    10: "Q5_K_M", 11: "Q5_K_M", 12: "Q5_K_M",
    13: "Q4_K_M",
    14: "Q4_K_M", 15: "Q4_K_M",
}


def variant_key(tier: int, speed: bool) -> str:
    """State key for a tier variant: '7' (normal) or '7-s' (speed)."""
    return f"{tier}-s" if speed else str(tier)


def variant_label(tier: int, speed: bool) -> str:
    """Display/README name for a tier variant: 'tier7' or 'tier7-s'."""
    return f"tier{tier}-s" if speed else f"tier{tier}"


def expand_variants(tiers: list) -> list:
    """Expand a sorted tier list into (tier, speed) variants.

    Tiers in SPEED_TIERS get both (tier, False) and (tier, True);
    other tiers only (tier, False).  Ordering: tier7, tier7-s, tier8, …
    """
    out = []
    for t in tiers:
        out.append((t, False))
        if t in SPEED_TIERS:
            out.append((t, True))
    return out


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str):
    line = f"[{_ts()}] {msg}"
    if _HAS_RICH and _lc is not None and _lc.live is not None:
        _lc.live.append_log(line)
    else:
        print(line, flush=True)


def log_err(msg: str):
    line = f"[{_ts()}] \u274c {msg}"
    # Always print errors to stderr so they are visible even when
    # Rich Live is overwriting the terminal.
    print(line, file=sys.stderr, flush=True)
    if _HAS_RICH and _lc is not None and _lc.live is not None:
        _lc.live.append_log(line)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class BatchState:
    """JSON-backed state for resumable batch processing."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

    # -- persistence --
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    # -- top-level keys --
    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self.save()

    # -- per-variant helpers (key: '7' or '7-s') --
    def _t(self, key: str) -> dict:
        return self.data.setdefault("tiers", {}).setdefault(str(key), {})

    def tier_status(self, key: str) -> str:
        return self._t(key).get("status", "pending")

    def set_tier(self, key: str, status: str, **extra):
        t = self._t(key)
        t["status"] = status
        t["updated"] = datetime.now().isoformat()
        t.update(extra)
        self.save()

    # -- source / imatrix helpers --
    def mark_source(self, path: str, fmt: str):
        self.data["source"] = {"path": path, "format": fmt, "status": "downloaded"}
        self.save()

    def mark_imatrix(self, path: str):
        self.data["imatrix"] = {"path": path, "status": "downloaded"}
        self.save()

    def source_info(self) -> Optional[dict]:
        return self.data.get("source") if self.data.get("source", {}).get("status") == "downloaded" else None

    def imatrix_info(self) -> Optional[dict]:
        return self.data.get("imatrix") if self.data.get("imatrix", {}).get("status") == "downloaded" else None

    # -- README helpers --
    def readme_initialized(self) -> bool:
        return self.data.get("readme_initialized", False)

    def mark_readme_initialized(self):
        self.data["readme_initialized"] = True
        self.save()

    def readme_tiers(self) -> set:
        return set(self.data.get("readme_tiers", []))

    def mark_readme_tier(self, tier: int):
        tiers = set(self.data.get("readme_tiers", []))
        tiers.add(tier)
        self.data["readme_tiers"] = sorted(tiers)
        self.save()

    # -- per-variant upload helpers (hf / s3) --
    def upload_done(self, key: str, target: str) -> bool:
        return bool(self._t(key).get(f"{target}_done"))

    def mark_upload_done(self, key: str, target: str):
        self._t(key)[f"{target}_done"] = True
        self.save()

    # -- source model S3 upload --
    def source_upload_done(self) -> bool:
        return bool(self.data.get("source_upload", {}).get("done", False))

    def mark_source_upload(self, status: str, **extra):
        rec = {"status": status, "updated": datetime.now().isoformat()}
        if status == "done":
            rec["done"] = True
        rec.update(extra)
        self.data["source_upload"] = rec
        self.save()


# ---------------------------------------------------------------------------
# Rich display  (optional — falls back to plain text)
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

# One-shot console for non-live display_status() calls.
if _HAS_RICH:
    console = Console(force_terminal=True)
else:
    console = None


# ---------------------------------------------------------------------------
# Live display context
# ---------------------------------------------------------------------------

class _LiveContext:
    """Mutable namespace shared between pipeline and display code."""

    def __init__(self):
        self.live: Optional["_RichLive"] = None
        self.active_tiers: dict[str, dict] = {}
        self.upload_progress: dict[str, dict] = {}
        self.s3_upload_progress: dict[str, dict] = {}
        self.source_upload: Optional[dict] = None
        self.downloads: dict[str, dict] = {}
        self.merge: Optional[dict] = None
        self.state: Optional["BatchState"] = None
        self.source_ok: bool = False
        self.imatrix_ok: bool = False
        self.tiers: list = []


_lc = _LiveContext() if _HAS_RICH else None


# ---------------------------------------------------------------------------
# Live display — thin wrapper around rich.live.Live
# ---------------------------------------------------------------------------
#
# Previous attempts to manually manage ANSI cursor-up via sys.stdout failed
# on macOS Terminal.app because Python's TTY buffering and Rich's internal
# cursor bookkeeping fight over the terminal.  We now use Rich's native
# ``Live`` widget, which handles all cursor positioning internally.
#
# The original duplicate-table bug was caused by calling ``console.print()``
# while a Live context was active.  In the current code, *all* output during
# live mode goes through ``live.update()`` and ``append_log()``, never
# through ``console.print()``.
# ---------------------------------------------------------------------------

_LOG_RING_SIZE = 8


class _RichLive:
    """Thin wrapper around ``rich.live.Live`` — handles spinner animation,
    log ring buffer, and Rich Live cursor management.

    Rich's ``get_renderable`` callback is used: Rich's own daemon thread calls
    ``_composite()`` on every refresh. No manual terminal I/O from our threads.
    Only callers (pipeline, capture) mutate state dicts; Rich renders.
    """

    def __init__(self):
        self._log_lines: list[str] = []
        self._lock = threading.Lock()
        self._last_refresh: float = 0.0
        self._last_good: Optional[object] = None   # fallback renderable

        # Pass file=sys.stdout explicitly so that Console stores a direct
        # reference.  Without it, self._file is None and Console's ``file``
        # property re-reads ``sys.stdout`` on every render — which breaks
        # during upload when sys.stdout is temporarily redirected.
        self._live = Live(
            console=Console(file=sys.stdout, force_terminal=True),
            auto_refresh=True,
            refresh_per_second=5,
            transient=True,
            get_renderable=self._composite,
        )

    # -- lifecycle (thin passthrough) --

    def start(self):
        self._live.start()

    def stop(self):
        self._live.stop()

    def update(self, _renderable=None):
        """Explicit refresh — throttled to avoid starving auto-refresh."""
        now = time.time()
        if now - self._last_refresh >= 0.2:       # ≤ 5/sec
            self._last_refresh = now
            self._live.refresh()

    def append_log(self, msg: str):
        with self._lock:
            self._log_lines.append(msg)
            if len(self._log_lines) > _LOG_RING_SIZE * 3:
                self._log_lines = self._log_lines[-_LOG_RING_SIZE:]

    # -- composite builder (called by Rich's daemon thread via
    #    ``_get_renderable`` on every auto-refresh tick) --

    def _composite(self):
        try:
            with self._lock:
                logs = list(self._log_lines[-_LOG_RING_SIZE:])
            if _lc is not None:
                renderable = display_status(
                    _lc.state, _lc.source_ok, _lc.imatrix_ok,
                    _lc.tiers, _live_ctx=_lc)
            else:
                renderable = Text("")
            parts = [renderable]
            padded = logs + [""] * (_LOG_RING_SIZE - len(logs))
            parts.append(Text("\n".join(padded)))
            result = Group(*parts)
            self._last_good = result
            return result
        except Exception:
            # If the refresh thread's _composite raises, Rich's
            # _RefreshThread.run() dies with no recovery.  Return
            # the last known-good renderable to keep the display
            # alive.
            if self._last_good is not None:
                return self._last_good
            return Text("… refreshing …")


def _init_live():
    if _lc is None or not _HAS_RICH:
        return
    if _lc.live is None:
        _lc.live = _RichLive()
        _lc.live.start()


def _stop_live():
    if not _HAS_RICH or _lc is None or _lc.live is None:
        return
    _lc.live.update(_build_live_renderable())
    _lc.live.stop()
    _lc.live = None
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _build_live_renderable():
    """Build the Rich renderable for the live display."""
    return display_status(_lc.state, _lc.source_ok, _lc.imatrix_ok,
                          _lc.tiers, _live_ctx=_lc)



_STATUS_STYLE = {
    "pending":    "dim",
    "quantizing": "yellow",
    "quantized":  "cyan",
    "uploading":  "magenta",
    "queued":     "dim cyan",
    "uploaded":   "green",
    "error":      "bold red",
    "done":       "bold green",
}

_STATUS_ICON = {
    "pending":    "○",
    "quantizing": "◉",
    "quantized":  "◇",
    "uploading":  "↑",
    "queued":     "↑",
    "uploaded":   "✓",
    "error":      "✗",
    "done":       "✓",
}


def _format_elapsed(seconds: float) -> str:
    """Format a duration as MM:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{m:02d}:{s:02d}" if not h else f"{h}:{m:02d}:{s:02d}"


def display_status(state: BatchState, source_ok: bool, imatrix_ok: bool,
                   tiers: list = None, *, _live_ctx=None):
    """Render the current status to the terminal.

    *tiers* is a list of (tier, speed) variants.  When *_live_ctx* is
    provided (internal), returns a Rich renderable for use inside a
    ``Live`` display.  Otherwise prints a one-shot table.
    """
    if tiers is None:
        tiers = expand_variants(TIERS)

    # ── Live mode (called by _build_live_renderable) ──
    if _HAS_RICH and _live_ctx is not None:
        from rich.text import Text

        table = Table(show_lines=False, padding=(0, 1), expand=True)
        table.add_column("Tier", justify="right", style="bold", width=8)
        table.add_column("Base", width=7)
        table.add_column("Status", width=56)
        table.add_column("Info", ratio=1, style="dim", overflow="fold")

        # Merge row (shard merge task)
        if _live_ctx.merge:
            mg = _live_ctx.merge
            mst = mg["status"]
            if mst == "merging":
                start = mg.get("start", time.time())
                elapsed_str = _format_elapsed(time.time() - start)
                frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
                total_gb = mg.get("total_gb")
                info = f"{total_gb:.1f} GB" if total_gb else ""
                table.add_row("MERGE", "—",
                              Text(f" {frame} merging… {elapsed_str}", style="yellow"),
                              info)
            elif mst == "error":
                elapsed_str = _format_elapsed(mg.get("elapsed", 0))
                table.add_row("MERGE", "—",
                              Text(f" ✗ merge failed {elapsed_str}", style="bold red"), "")
            else:
                elapsed_str = _format_elapsed(mg.get("elapsed", 0))
                note = " (cached)" if mst == "cached" else ""
                table.add_row("MERGE", "—",
                              Text(f" ✓ merged in {elapsed_str}{note}", style="green"), "")

        # Source-model S3 upload row
        if _live_ctx.source_upload:
            row = _render_source_upload(_live_ctx.source_upload, _live_ctx)
            if row:
                table.add_row("SRC", "—", row[0], row[1])

        # Download rows (above tier table)
        for key, dl in _live_ctx.downloads.items():
            label = dl.get("label", key)
            st = dl["status"]
            if st == "downloading":
                total = dl.get("total", 0)
                dl_bytes = dl.get("bytes", 0)
                start = dl.get("start", time.time())
                elapsed = time.time() - start
                m, s = divmod(int(elapsed), 60)
                elapsed_str = f"{m:02d}:{s:02d}"
                frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
                if total > 0 and dl_bytes > 0:
                    pct = dl_bytes / total * 100
                    bar_len = 20
                    filled = int(pct / 100 * bar_len)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    # ETA must be based on bytes transferred in this session:
                    # resumed bytes from a previous attempt arrived in 0s
                    # and would otherwise skew the rate estimate.
                    session_bytes = dl_bytes - dl.get("resumed_from", 0)
                    if session_bytes > 0:
                        eta_sec = elapsed / session_bytes * (total - dl_bytes)
                        em, es = divmod(int(eta_sec), 60)
                        eh, em = divmod(em, 60)
                        eta_str = f"{em:02d}:{es:02d}" if not eh else f"{eh}:{em:02d}:{es:02d}"
                    else:
                        # No bytes transferred yet this session (fresh resume)
                        eta_str = "--:--"
                    # Instantaneous speed, EMA-smoothed across renders.
                    # Session-based (resumed bytes excluded) so a resume
                    # spike doesn't show a bogus multi-GB/s value.
                    now = time.time()
                    prev_b = dl.get("_spd_bytes")
                    prev_t = dl.get("_spd_ts")
                    if prev_t is not None and now > prev_t:
                        inst = max(0.0, (dl_bytes - prev_b) / (now - prev_t) / (1024**2))
                        speed = 0.7 * dl.get("_spd", inst) + 0.3 * inst
                    else:
                        speed = dl.get("_spd", 0.0)
                    dl["_spd_bytes"] = dl_bytes
                    dl["_spd_ts"] = now
                    dl["_spd"] = speed
                    status_col = Text(f" {frame} {bar} {pct:.1f}% {elapsed_str} ETA {eta_str}",
                                      style="cyan")
                    info = (f"{dl_bytes / (1024**2):.0f}/{total / (1024**2):.0f} MB"
                            f" · {speed:.1f} MB/s")
                else:
                    status_col = Text(f" {frame} downloading… {elapsed_str}", style="cyan")
                    info = ""
                table.add_row("DL", label[:7], status_col, info)
            elif st == "done":
                table.add_row("DL", label[:7],
                              Text(" ✓ downloaded", style="green"), "")

        # Tier rows
        for tier, speed in tiers:
            key = variant_key(tier, speed)
            base = TIER_BASE_TYPE[tier]
            active = _live_ctx.active_tiers.get(key)
            if active:
                status_col = _render_active_tier(key, active, _live_ctx)
                info = _format_active_info(key, active, state)
            else:
                st = state.tier_status(key)
                # State says "uploading" but the tier is NOT in active_tiers
                # → the upload is sitting in the executor queue.
                show_st = "queued" if st == "uploading" else st
                icon = _STATUS_ICON.get(show_st, "?")
                style = _STATUS_STYLE.get(show_st, "")
                status_col = Text(f" {icon} {show_st}", style=style)
                info = ""
                if show_st == "error":
                    info = Text(state._t(key).get("error_short", ""), style="red")
                elif show_st == "uploaded":
                    sz = state._t(key).get("size_mb")
                    if sz:
                        info = f"{sz} MB"
            table.add_row(variant_label(tier, speed), base, status_col, info)

        src_label = "✓ downloaded" if source_ok else "… pending"
        imx_label = "✓ downloaded" if imatrix_ok else "… pending"
        header = Text(f"Source:  {src_label}\nImatrix: {imx_label}", style="bold")
        return Panel(table, title="[bold]mAPEX Batch Quantization[/bold]",
                     subtitle=header, border_style="blue")

    # ── One-shot mode (original behavior) ──
    if _HAS_RICH:
        from rich.text import Text

        table = Table(show_lines=False, padding=(0, 1), expand=True)
        table.add_column("Tier", justify="right", style="bold", width=8)
        table.add_column("Base", width=7)
        table.add_column("Status", width=56)
        table.add_column("Info", ratio=1, style="dim", overflow="fold")
        merge_rec = state.data.get("merge")
        if merge_rec:
            mel = _format_elapsed(merge_rec.get("elapsed", 0))
            mst = merge_rec.get("status", "done")
            note = " (cached)" if mst == "cached" else ""
            style = "green" if mst in ("done", "cached") else "bold red"
            table.add_row("MERGE", "—",
                          Text(f" ✓ merged in {mel}{note}", style=style), "")
        su_rec = state.data.get("source_upload")
        if su_rec:
            if su_rec.get("done"):
                table.add_row("SRC", "—",
                              Text(" ✓ source uploaded to S3", style="green"), "")
            elif su_rec.get("status") == "error":
                table.add_row("SRC", "—",
                              Text(" ✗ source S3 upload failed", style="bold red"), "")
        for tier, speed in tiers:
            key = variant_key(tier, speed)
            base = TIER_BASE_TYPE[tier]
            st = state.tier_status(key)
            icon = _STATUS_ICON.get(st, "?")
            style = _STATUS_STYLE.get(st, "")
            info = ""
            if st == "error":
                info = Text(state._t(key).get("error_short", ""), style="red")
            elif st == "uploaded":
                sz = state._t(key).get("size_mb")
                if sz:
                    info = f"{sz} MB"
            table.add_row(
                variant_label(tier, speed),
                base,
                Text(f" {icon} {st}", style=style),
                info,
            )
        src_label = "✓ downloaded" if source_ok else "… pending"
        imx_label = "✓ downloaded" if imatrix_ok else "… pending"
        header = Text(f"Source:  {src_label}\nImatrix: {imx_label}", style="bold")
        panel = Panel(table, title="[bold]mAPEX Batch Quantization[/bold]",
                       subtitle=header, border_style="blue")
        console.print(panel)
    else:
        print(f"\n  {'Tier':<9} {'Base':<8} {'Status'}")
        print(f"  {'─'*9} {'─'*8} {'─'*20}")
        for tier, speed in tiers:
            key = variant_key(tier, speed)
            base = TIER_BASE_TYPE[tier]
            st = state.tier_status(key)
            icon = _STATUS_ICON.get(st, "?")
            print(f"  {variant_label(tier, speed):<9} {base:<8} {icon} {st}")
        print()


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _render_source_upload(su: dict, lc):
    """Build (status_col, info) for the SRC row, or None if nothing to show."""
    from rich.text import Text
    st = su.get("status")
    if st == "uploading":
        start = su.get("start", time.time())
        elapsed_str = _format_elapsed(time.time() - start)
        frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]
        prog = lc.s3_upload_progress.get("src") if lc else None
        if prog and prog.get("total", 0) > 0:
            cur, total = prog["current"], prog["total"]
            pct = cur / total * 100
            bar_len = 16
            filled = min(int(pct / 100 * bar_len), bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            cur_mb, total_mb = cur / (1024**2), total / (1024**2)
            speed_str = ""
            start_ts = prog.get("start_ts")
            if start_ts and cur > 0 and time.time() > start_ts:
                speed_mb = cur / (1024**2) / (time.time() - start_ts)
                speed_str = f" {speed_mb:.1f} MB/s"
            status_col = Text(
                f" ↑ s3 {bar} {cur_mb:.0f}/{total_mb:.0f} MB "
                f"{pct:.0f}%{speed_str} {elapsed_str}",
                style="magenta")
        else:
            status_col = Text(
                f" {frame} uploading source to S3… {elapsed_str}",
                style="magenta")
        return status_col, ""
    if st == "done":
        return Text(" ✓ source uploaded to S3", style="green"), ""
    if st == "error":
        return Text(" ✗ source S3 upload failed", style="bold red"), ""
    return None


def _render_active_tier(key: str, active: dict, lc) -> "Text":
    """Build a Rich Text object for a currently active tier row."""
    from rich.text import Text
    st = active["status"]
    start = active.get("start", time.time())
    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    elapsed_str = f"{m:02d}:{s:02d}" if not h else f"{h}:{m:02d}:{s:02d}"
    frame = _SPINNER_FRAMES[int(time.time() * 4) % len(_SPINNER_FRAMES)]

    if st == "quantizing":
        cur = active.get("tensor_current")
        total = active.get("tensor_total")
        if cur is not None and total and total > 0 and cur > 0:
            pct = cur / total * 100
            bar_len = 16
            filled = int(pct / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            eta_sec = elapsed / cur * (total - cur)
            em, es = divmod(int(eta_sec), 60)
            eh, em = divmod(em, 60)
            eta_str = f"{em:02d}:{es:02d}" if not eh else f"{eh}:{em:02d}:{es:02d}"
            return Text(f" ◉ {bar} {cur}/{total} {elapsed_str} ETA {eta_str}", style="yellow")
        return Text(f" {frame} quantizing… {elapsed_str}", style="yellow")
    elif st == "uploading":
        prog = lc.upload_progress.get(key)
        target_tag = ""
        if not prog or prog.get("total", 0) <= 0:
            prog = lc.s3_upload_progress.get(key)
            target_tag = "s3 "
        if prog and prog.get("total", 0) > 0:
            cur = prog["current"]
            total = prog["total"]
            pct = cur / total * 100
            stage_start = prog.get("stage_start", active.get("start", time.time()))
            stage_elapsed = time.time() - stage_start
            bar_len = 16
            filled = min(int(pct / 100 * bar_len), bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            cur_mb = cur / (1024**2)
            total_mb = total / (1024**2)
            speed_str = ""
            start_ts = prog.get("start_ts")
            if start_ts and cur > 0 and time.time() > start_ts:
                speed_mb = cur / (1024**2) / (time.time() - start_ts)
                speed_str = f" {speed_mb:.1f} MB/s"
            if cur > 0 and stage_elapsed > 1:
                eta_sec = stage_elapsed / cur * (total - cur)
                em, es = divmod(int(eta_sec), 60)
                eh, em = divmod(em, 60)
                eta_str = f"{em:02d}:{es:02d}" if not eh else f"{eh}:{em:02d}:{es:02d}"
                return Text(
                    f" ↑ {target_tag}{bar} {cur_mb:.0f}/{total_mb:.0f} MB {pct:.0f}%{speed_str} {elapsed_str} ETA {eta_str}",
                    style="magenta",
                )
            return Text(
                f" ↑ {target_tag}{bar} {cur_mb:.0f}/{total_mb:.0f} MB {pct:.0f}%{speed_str} {elapsed_str}",
                style="magenta",
            )
        return Text(f" {frame} uploading{(' ' + target_tag.strip()) if target_tag else ''}… {elapsed_str}", style="magenta")
    else:
        return Text(f" ? {st}", style="dim")


def _format_active_info(key: str, active: dict, state: BatchState) -> str:
    """Return hint text for active tier's info column."""
    last = active.get("last_line", "")
    parts = []
    if last:
        parts.append(last)
    s3p = _lc.s3_upload_progress.get(key) if _lc else None
    if s3p and s3p.get("total", 0) > 0:
        pct = s3p["current"] / s3p["total"] * 100
        info = f"S3 {pct:.0f}% ({s3p['current'] / (1024**2):.0f} MB)"
        start_ts = s3p.get("start_ts")
        if start_ts and s3p["current"] > 0 and time.time() > start_ts:
            speed_mb = s3p["current"] / (1024**2) / (time.time() - start_ts)
            info += f" · {speed_mb:.1f} MB/s"
        parts.append(info)
    return " · ".join(parts)


class _Capture:
    """Captures subprocess output, exposing the last non-empty line.

    Used by ``run_quantize`` to feed tier status into the live display
    while preventing raw output from corrupting Rich's rendering.
    """

    _PROGRESS_RE = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def __init__(self, key: str, *, stream: Optional[io.TextIOBase] = None):
        self.key = key                 # variant key ('7' or '7-s')
        self._stream = stream          # original stderr (or None)
        self._buf = ""
        self._lines: list[str] = []
        self.last_line: str = ""
        self._lock = threading.Lock()
        self._last_live_update: float = 0.0

    def write(self, data: str) -> int:
        self._buf += data
        while True:
            idx_n = self._buf.find("\n")
            idx_r = self._buf.find("\r")
            if idx_n >= 0 and (idx_r < 0 or idx_n <= idx_r):
                idx = idx_n
            elif idx_r >= 0:
                idx = idx_r
            else:
                break
            line = self._ANSI_RE.sub("", self._buf[:idx]).rstrip("\r\n\t ")
            self._buf = self._buf[idx + 1:]
            if line:
                with self._lock:
                    self._lines.append(line)
                    self.last_line = line
                if _lc and _lc.active_tiers.get(self.key):
                    _lc.active_tiers[self.key]["last_line"] = self.last_line[:120]
                    m = self._PROGRESS_RE.search(line)
                    if m:
                        _lc.active_tiers[self.key]["tensor_current"] = int(m.group(1))
                        _lc.active_tiers[self.key]["tensor_total"] = int(m.group(2))
                    if _lc.live is not None:
                        _lc.live.update()
        if self._stream is not None and not (_lc and _lc.live):
            try:
                return self._stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def fileno(self):
        if self._stream is not None:
            return self._stream.fileno()
        raise io.UnsupportedOperation("fileno")

    @property
    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _check_hf_import():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log_err("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)


def _is_network_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient network failure."""
    return isinstance(exc, _NETWORK_ERRORS)


@_retry_on_network_error
def _list_repo_gguf_files(repo_id: str, token: Optional[str]) -> list:
    """Return list of .gguf filenames in a HF repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    return [f for f in files if f.endswith(".gguf")]


def _pick_source_gguf(gguf_files: list) -> str:
    """Pick the bf16/f16/f32 source file from a list of .gguf filenames."""
    import re
    # priority: bf16 > f16 > f32 (case-insensitive)
    for pattern in (r"bf16", r"Bf16", r"BF16"):
        for f in gguf_files:
            if re.search(pattern, f):
                return f
    for pattern in (r"\bf16\b", r"\bF16\b"):
        for f in gguf_files:
            if re.search(pattern, f, re.IGNORECASE):
                return f
    for pattern in (r"\bf32\b", r"\bF32\b"):
        for f in gguf_files:
            if re.search(pattern, f, re.IGNORECASE):
                return f
    if gguf_files:
        return gguf_files[0]
    return ""


_SRC_FMT_RE = re.compile(r"(?i)(?:bf16|f16|f32)")


def _expand_shard_files(filename: str, repo_files: list) -> list:
    """Expand a split-GGUF filename into all its shard filenames.

    Split GGUFs are distributed as Model-00001-of-00002.gguf,
    Model-00002-of-00002.gguf, … For a shard name returns the full
    ordered shard list (filtered against *repo_files*); for a regular
    single-file GGUF returns [filename].
    """
    m = SHARD_RE.match(filename)
    if not m:
        return [filename]
    prefix, count = m.group(1), int(m.group(3))
    shards = []
    for f in repo_files:
        mm = SHARD_RE.match(f)
        if mm and mm.group(1) == prefix and int(mm.group(3)) == count:
            shards.append(f)
    shards.sort()
    return shards if shards else [filename]


def _tier_filename(source_file: str, tier: int, speed: bool = False) -> str:
    """Derive tier output GGUF name from the source model filename.

    Replaces the source format token (BF16/F16/F32, any case) with Tier<N>
    (or Tier<N>-s for the speed variant):
      Model-BF16.gguf → Model-Tier3.gguf
      model_f16.gguf  → model_Tier7-s.gguf  (speed variant)
    Falls back to appending a -Tier<N> suffix if no format token is found.
    """
    name = Path(source_file).name
    tier_token = f"Tier{tier}-s" if speed else f"Tier{tier}"
    result, n = _SRC_FMT_RE.subn(tier_token, name)
    if n == 0:
        result = re.sub(r"(?i)\.gguf$", f"-{tier_token}.gguf", name)
        if result == name:
            result = f"{name}-{tier_token}.gguf"
    return result


def _pick_imatrix_file(files: list) -> str:
    """Pick an imatrix file from a HF repo file list."""
    import re
    cands = [f for f in files if re.search(r"imatrix", f, re.IGNORECASE)]
    if cands:
        cands.sort(reverse=True)
        return cands[0]
    # Also accept .dat files
    dat_files = [f for f in files if f.endswith(".dat")]
    if dat_files:
        dat_files.sort(reverse=True)
        return dat_files[0]
    return ""


# ---------------------------------------------------------------------------
# README management
# ---------------------------------------------------------------------------

README_PREAMBLE = """\
# mAPEX Quantized Models

## Source Data

The source data for quantization was taken from [{source_model}](https://huggingface.co/{source_model}).

Original source file: `{source_file}`

## Quantization Method

All quants were produced using the **modified APEX** quantization scheme.

mAPEX (modified Automated Precision EXpert allocation) assigns per-layer, per-tensor precision
for MoE models using `llama.cpp`'s `--tensor-type-file`.

For more information, see: <https://github.com/DrMoriarty/apex-quant/>

"""

README_TABLE_HEADER = """\
## Quantized Models

> **Note:** TierN-s quants are typically slightly larger than their regular
> counterparts, but provide roughly 10–20% faster inference (highly dependent
> on the model and GPU).

| Name | Size (GB) | Comments |
|------|-----------|----------|
"""


@_retry_on_network_error
def _readme_in_repo(repo_id: str, token: Optional[str]) -> bool:
    """Check if README.md exists in a HF repo."""
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        return "README.md" in files
    except Exception as exc:
        if _is_network_error(exc):
            raise
        return False


@_retry_on_network_error
def _download_readme(repo_id: str, token: Optional[str], dest: Path) -> bool:
    """Download README.md from a HF repo. Returns True on success."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            local_dir=str(dest.parent),
            token=token,
        )
        src = Path(local)
        if src.exists() and src != dest:
            dest.write_text(src.read_text())
        return dest.exists()
    except Exception as exc:
        if _is_network_error(exc):
            raise
        return False


def _create_readme(readme_path: Path, source_model: str, source_file: str):
    """Create a fresh README.md with source info and empty table at EOF."""
    digest = README_PREAMBLE.format(source_model=source_model, source_file=source_file)
    readme_path.write_text(digest + README_TABLE_HEADER)


def _extract_frontmatter_and_body(content: str) -> tuple[str, str]:
    """Split HF YAML frontmatter from the rest of the file.

    Returns ``(frontmatter, body)`` where *frontmatter* includes the
    closing ``---`` line (with a trailing newline) or is empty if the
    content has no frontmatter block.
    """
    if not content.startswith("---"):
        return "", content
    second = content.find("---", 3)
    if second < 0:
        return "", content
    end = second + 3
    # consume trailing newline after the closing ---
    if end < len(content) and content[end] == "\n":
        end += 1
    return content[:end], content[end:]


def _ensure_apex_header(readme_path: Path, source_model: str, source_file: str):
    """Inject mAPEX header into an existing README downloaded from HF.

    If the README already contains the mAPEX table marker (``## Quantized Models``),
    it is left untouched.  Otherwise the file is restructured as:

        <HF YAML frontmatter>        ← preserved verbatim at top
        <mAPEX source/method info>    ← new
        <original prose body>        ← preserved
        <quantization table>         ← new; must be last so ``open("a")`` row appends land inside
    """
    if not readme_path.exists():
        return
    content = readme_path.read_text()
    if "## Quantized Models" in content:
        return
    frontmatter, body = _extract_frontmatter_and_body(content)
    preamble = README_PREAMBLE.format(source_model=source_model, source_file=source_file)
    readme_path.write_text(frontmatter + preamble + body.strip("\n") + "\n\n" + README_TABLE_HEADER)
    log("✓ Injected mAPEX header into existing README.md")


def _readme_has_tier(readme_path: Path, tier_name: str) -> bool:
    """Check whether a tier row already exists in the README."""
    if not readme_path.exists():
        return False
    return f"| {tier_name} " in readme_path.read_text()


def _append_readme_row(readme_path: Path, tier_name: str, size_gb: float):
    """Append a row for a completed tier to the README table."""
    row = f"| {tier_name} | {size_gb:.2f} | |\n"
    with readme_path.open("a") as f:
        f.write(row)


def _rebuild_readme_rows(readme_path: Path, state: "BatchState"):
    """Populate README with rows for all variants already marked uploaded in state."""
    for tier, speed in expand_variants(TIERS):
        key = variant_key(tier, speed)
        if state.tier_status(key) != "uploaded":
            continue
        tier_name = variant_label(tier, speed)
        if _readme_has_tier(readme_path, tier_name):
            continue
        info = state._t(key)
        sz_mb = info.get("size_mb")
        if sz_mb:
            _append_readme_row(readme_path, tier_name, sz_mb / 1024)


@_retry_on_network_error
def _upload_readme(
    readme_path: Path,
    repo_id: str,
    token: Optional[str],
):
    """Upload README.md to the output repo."""
    from huggingface_hub import create_repo, upload_folder
    if not readme_path.exists():
        log_err(f"README not found at {readme_path}, skipping upload.")
        return

    try:
        create_repo(repo_id=repo_id, repo_type="model",
                    exist_ok=True, token=token)
        with tempfile.TemporaryDirectory(prefix="apex_readme_") as tmp:
            (Path(tmp) / "README.md").write_text(readme_path.read_text())
            upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message="Update README — mAPEX quantization info",
            )
        log(f"  ✓ README → {repo_id}")
    except Exception as exc:
        if _is_network_error(exc):
            raise
        log_err(f"  ✗ README upload failed ({repo_id}): {exc}")


# ---------------------------------------------------------------------------
# Download — resume support
# ---------------------------------------------------------------------------
# Download — resumable
# ---------------------------------------------------------------------------
#
# huggingface_hub does NOT support resuming downloads across process
# restarts: it uses per-process UUID temp file names (PR #4228) and
# deletes the blob before re-downloading.  Xet storage adds its own
# non-resumable layer on top.
#
# We implement our own resumable downloader: get the URL + file size
# from HuggingFace API, then download with httpx + Range headers,
# writing to a deterministic `.part` file.
#
# On restart the script checks the existing `.part` file size and
# resumes from that byte offset.  On success the file is renamed
# to the final `.gguf` name.
# ---------------------------------------------------------------------------

import httpx


def _resumable_download(url: str, dest: Path, *, token: Optional[str] = None,
                        label: str = "file") -> Path:
    """Download *url* to *dest* with resume support.

    Incomplete data is stored alongside *dest* as ``{dest}.part``.
    On resume the existing ``.part`` file is inspected and a ``Range``
    header is sent to skip already-downloaded bytes.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["Accept-Encoding"] = "identity"

    # Get total file size via HEAD
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        head = client.head(url, headers=headers)
        head.raise_for_status()
        total = int(head.headers.get("content-length", 0))

    if existing > 0 and existing < total:
        log(f"Resuming {label} from {existing / (1024**2):.1f} MB  "
            f"({existing / total * 100:.1f}%)")
        headers["Range"] = f"bytes={existing}-"
    elif existing >= total and total > 0:
        # Already complete — rename and return
        if dest.exists():
            dest.unlink()
        part.rename(dest)
        return dest
    else:
        existing = 0

    mode = "ab" if existing > 0 else "wb"
    bytes_downloaded = existing
    if _lc and _lc.downloads.get(label):
        _lc.downloads[label]["resumed_from"] = existing

    with open(part, mode) as f, \
         httpx.Client(follow_redirects=True, timeout=httpx.Timeout(300, connect=30)) as client:
        with client.stream("GET", url, headers=headers) as response:
            if existing > 0 and response.status_code == 200:
                # Server ignored Range — restart from scratch
                f.seek(0)
                f.truncate()
                bytes_downloaded = 0
                if _lc and _lc.downloads.get(label):
                    _lc.downloads[label]["resumed_from"] = 0

            response.raise_for_status()

            total_mb = total / (1024**2)
            last_log = time.time()
            last_live_update = time.time()
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                now = time.time()
                # Update live context download progress
                if _lc and _lc.downloads.get(label):
                    _lc.downloads[label]["bytes"] = bytes_downloaded
                    _lc.downloads[label]["total"] = total
                    # Force Rich Live to re-render (throttled to 1/sec)
                    if _lc.live and now - last_live_update >= 1.0:
                        _lc.live.update(_build_live_renderable())
                        last_live_update = now
                elif now - last_log >= 5:
                    pct = bytes_downloaded / total * 100 if total else 0
                    log(f"  {label}: {bytes_downloaded / (1024**2):.0f}/{total_mb:.0f} MB  ({pct:.1f}%)")
                    last_log = now

    # Verify size
    actual = part.stat().st_size
    if actual != total:
        raise ValueError(
            f"Download size mismatch: expected {total}, got {actual} ({label})")

    if dest.exists():
        dest.unlink()
    part.rename(dest)
    log(f"{label} complete: {dest.name}  ({actual / (1024**3):.2f} GB)")
    return dest


def _get_hf_download_url(repo_id: str, filename: str, token: Optional[str]) -> str:
    """Resolve the download URL for a file in a HuggingFace repo."""
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo_id, filename=filename, repo_type="model")


def _download_single_file(
    repo_id: str,
    filename: str,
    workspace: Path,
    token: Optional[str],
    *,
    label: str = "file",
) -> Path:
    """Download a single file from HF with resume support."""
    dest = workspace / filename
    if dest.exists() and dest.stat().st_size > 0:
        if _lc:
            _lc.downloads[label] = {"label": label, "status": "done",
                                     "bytes": dest.stat().st_size,
                                     "total": dest.stat().st_size, "start": time.time()}
        log(f"{label} already exists: {dest.name}  "
            f"({dest.stat().st_size / (1024**3):.2f} GB)")
        if _lc and _lc.live:
            _lc.live.update(_build_live_renderable())
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if _lc:
        _lc.downloads[label] = {"label": label, "status": "downloading",
                                 "bytes": 0, "total": 0, "start": time.time()}
        if _lc.live:
            _lc.live.update(_build_live_renderable())
    url = _get_hf_download_url(repo_id, filename, token)
    result = _resumable_download(url, dest, token=token, label=label)
    if _lc and label in _lc.downloads:
        _lc.downloads[label]["status"] = "done"
        if _lc.live:
            _lc.live.update(_build_live_renderable())
    return result


@_retry_on_network_error
def download_source_model(
    repo_id: str,
    workspace: Path,
    token: Optional[str],
    source_files: list,
) -> list:
    """Download the source GGUF shard(s) from HF, return list of local paths."""
    target_dir = workspace / "source_model"
    paths = []
    n = len(source_files)
    for i, filename in enumerate(source_files):
        label = "source model" if n == 1 else f"source model {i + 1}/{n}"
        result = _download_single_file(
            repo_id, filename, target_dir, token, label=label,
        )
        if not result.exists():
            raise FileNotFoundError(f"Source file not found after download: {result}")
        log(f"Source shard ready: {result.name}  ({result.stat().st_size / (1024**3):.2f} GB)")
        paths.append(result)
    return paths


@_retry_on_network_error
def download_imatrix(
    repo_id: str,
    workspace: Path,
    token: Optional[str],
    imatrix_file: str,
) -> Path:
    """Download the imatrix file from HF, return local path."""
    target_dir = workspace / "imatrix"
    result = _download_single_file(
        repo_id, imatrix_file, target_dir, token, label="imatrix",
    )
    if not result.exists():
        raise FileNotFoundError(f"Imatrix file not found after download: {result}")
    log(f"Imatrix ready: {result.name}  ({result.stat().st_size / (1024**2):.1f} MB)")
    return result


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------

def find_gguf_split():
    """Find llama-gguf-split binary."""
    q = os.environ.get("LLAMA_GGUF_SPLIT", "")
    if q and os.path.isfile(q):
        return q

    d = os.environ.get("LLAMA_CPP_DIR", "")
    if d:
        p = os.path.join(d, "llama-gguf-split")
        if os.path.isfile(p):
            return p

    candidates = [
        "./llama.cpp/build/bin",
        str(SCRIPT_DIR.parent / "llama.cpp" / "build" / "bin"),
    ]
    for d in candidates:
        p = os.path.join(d, "llama-gguf-split")
        if os.path.isfile(p):
            return p

    try:
        subprocess.check_output(["command", "-v", "llama-gguf-split"], shell=True)
        return "llama-gguf-split"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def merge_gguf_shards(shard_paths: list, *, keep_shards: bool = True,
                      state: Optional["BatchState"] = None) -> Path:
    """Merge split GGUF shards into a single GGUF.

    llama-quantize cannot read split GGUFs, so sharded source models must
    be merged before quantization. The merged file is written next to the
    first shard and reused on subsequent runs.

    When *keep_shards* is False the source shards are deleted after a
    successful merge to save disk space.

    The merge is shown as its own task row (with spinner and elapsed time)
    in the live table; the total time spent is recorded into *state*.
    """
    def _mark(status: str, elapsed: float):
        if _lc is not None:
            if status == "merging":
                _lc.merge = {"status": "merging", "start": time.time()}
            else:
                if _lc.merge is None:
                    _lc.merge = {}
                _lc.merge["status"] = status
                _lc.merge["elapsed"] = elapsed
            if _lc.live:
                _lc.live.update(_build_live_renderable())
        if state is not None:
            state.set("merge", {"status": status, "elapsed": round(elapsed, 1)})

    first = Path(shard_paths[0]).resolve()
    m = SHARD_RE.match(first.name)
    if not m:
        return first
    merged = first.with_name(f"{m.group(1)}.gguf")
    shards_exist = all(Path(p).exists() for p in shard_paths)
    if merged.exists() and merged.stat().st_size > 0:
        log(f"✓ Merged GGUF already exists: {merged.name}  "
            f"({merged.stat().st_size / (1024**3):.2f} GB)")
        _mark("cached", 0.0)
        if not keep_shards and shards_exist:
            _delete_shards(shard_paths)
        return merged

    split_bin = find_gguf_split()
    if not split_bin:
        log_err("llama-gguf-split not found. Set LLAMA_GGUF_SPLIT or LLAMA_CPP_DIR, "
                "or merge the shards manually:\n"
                f"  llama-gguf-split --merge {first} {merged}")
        sys.exit(1)

    total_gb = sum(p.stat().st_size for p in shard_paths) / (1024**3)
    log(f"Merging {len(shard_paths)} shards ({total_gb:.1f} GB) → {merged.name} …")
    _mark("merging", 0.0)
    if _lc is not None and _lc.merge is not None:
        _lc.merge["total_gb"] = total_gb
    t0 = time.time()
    try:
        subprocess.run([split_bin, "--merge", str(first), str(merged)], check=True)
        elapsed = time.time() - t0
    except BaseException:
        _mark("error", time.time() - t0)
        raise
    if merged.stat().st_size == 0:
        raise RuntimeError(f"Merged GGUF is empty: {merged}")
    _mark("done", elapsed)
    log(f"✓ Merged: {merged.name}  ({merged.stat().st_size / (1024**3):.2f} GB)  "
        f"in {_format_elapsed(elapsed)}")
    if not keep_shards:
        _delete_shards(shard_paths)
    return merged


def _delete_shards(shard_paths: list):
    """Delete source GGUF shards after a successful merge."""
    freed = 0
    for p in shard_paths:
        p = Path(p)
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    if freed:
        log(f"🗑  Removed {len(shard_paths)} source shard(s), "
            f"freed {freed / (1024**3):.2f} GB")


def run_quantize(
    tier: int,
    speed: bool,
    source_gguf: Path,
    imatrix_path: Path,
    output_gguf: Path,
) -> None:
    """Run quantize.py for a single tier variant. Raises on failure."""
    key = variant_key(tier, speed)
    label = variant_label(tier, speed)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "quantize.py"),
        "--profile", f"tier{tier}",
    ]
    if speed:
        cmd.append("--speed")
    cmd += ["--imatrix", str(imatrix_path), str(source_gguf), str(output_gguf)]

    if _HAS_RICH and _lc is not None and _lc.live is not None:
        cap = _Capture(key)

        # Use a PTY for stderr so llama-quantize sees a real terminal and
        # emits per-tensor progress lines (it suppresses them on pipes).
        master_fd, slave_fd = pty.openpty()
        tty.setraw(slave_fd)

        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=slave_fd, close_fds=True)
        os.close(slave_fd)

        err_buf = ""
        def _read_stderr():
            nonlocal err_buf
            try:
                while True:
                    try:
                        chunk = os.read(master_fd, 8192)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    cap.write(text)
                    err_buf += text
            finally:
                try:
                    os.close(master_fd)
                except OSError:
                    pass

        def _read_stdout():
            try:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    cap.write(chunk)
            except Exception:
                pass

        reader_err = threading.Thread(target=_read_stderr, daemon=True)
        reader_err.start()
        reader_out = threading.Thread(target=_read_stdout, daemon=True)
        reader_out.start()
        # Poll instead of blocking wait so Ctrl-C / _interruption_requested
        # is checked regularly (proc.wait() blocks the main thread entirely).
        while proc.poll() is None:
            if _interruption_requested:
                proc.kill()
                proc.wait()
                reader_err.join(timeout=5)
                reader_out.join(timeout=5)
                raise KeyboardInterrupt("Quantization interrupted by user")
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        reader_err.join(timeout=5)
        reader_out.join(timeout=5)
        if proc.returncode != 0:
            clean_err = _Capture._ANSI_RE.sub("", err_buf)
            tail_lines = [l for l in clean_err.splitlines() if l.strip()]
            tail = "\n".join(tail_lines[-15:]) or f"(exit code {proc.returncode})"
            raise RuntimeError(
                f"quantize.py ({label}) failed:\n{tail}"
            )
    else:
        log(f"Quantizing {label} → {output_gguf.name}")
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"quantize.py exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class _UploadProgressWrapper(io.BufferedIOBase):
    """Wraps a binary file object to track upload progress via _lc.upload_progress.

    huggingface_hub reads the file twice:
      1.  Preparing  — reads to compute SHA/hash
      2.  seek(0)    — rewinds
      3.  Uploading  — reads again to send bytes over the wire
      4.  Committing — API commit after upload

    Stage transitions fire when a full pass completes (bytes_read reaches total).
    """

    def __init__(self, path: Path, key: str):
        super().__init__()
        self._file = open(path, "rb")
        self._key = key
        self._total = path.stat().st_size
        self._bytes_read = 0
        self._pass = 0
        self.name = str(path)
        self.mode = "rb"
        self._set_stage("preparing")

    # -- stage helpers --

    def _on_pass_complete(self):
        self._pass += 1
        if self._pass == 1:
            self._set_stage("uploading")
        elif self._pass >= 2:
            self._set_stage("committing")

    def _set_stage(self, stage: str):
        if _lc and self._key in _lc.upload_progress:
            _lc.upload_progress[self._key]["stage"] = stage
            _lc.upload_progress[self._key]["stage_start"] = time.time()
        if _lc and self._key in _lc.active_tiers:
            _lc.active_tiers[self._key]["last_line"] = stage
            if _lc.live is not None:
                _lc.live.update()

    # -- io interface --

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        n = len(chunk)
        if n:
            self._bytes_read += n
            if self._bytes_read >= self._total:
                self._on_pass_complete()
            if _lc and self._key in _lc.upload_progress:
                _lc.upload_progress[self._key]["current"] = self._bytes_read
        return chunk

    def __len__(self) -> int:
        return self._total

    def seek(self, offset: int, whence: int = 0) -> int:
        pos = self._file.seek(offset, whence)
        self._bytes_read = pos
        return pos

    def tell(self) -> int:
        return self._file.tell()

    def close(self):
        self._file.close()
        super().close()


@_retry_on_network_error
def upload_tier(
    key: str,
    gguf_path: Path,
    repo_id: str,
    token: Optional[str],
) -> None:
    """Upload a quantized GGUF to HuggingFace. *key* is a variant key."""
    from huggingface_hub import create_repo, HfApi
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found for upload: {gguf_path}")
    if gguf_path.stat().st_size == 0:
        raise ValueError(f"GGUF file is empty: {gguf_path}")

    label = f"tier{key}"
    _live_active = _HAS_RICH and _lc is not None and _lc.live is not None
    if not _live_active:
        log(f"Uploading {label} ({gguf_path.name}, "
            f"{gguf_path.stat().st_size / (1024**3):.2f} GB) → {repo_id}")

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    if _live_active:
        import io as _io

        wrapper = _UploadProgressWrapper(gguf_path, key)
        _old_stdout = sys.stdout
        _old_stderr = sys.stderr
        sys.stdout = _io.StringIO()
        sys.stderr = _io.StringIO()
        try:
            try:
                HfApi(token=token).upload_file(
                    path_or_fileobj=wrapper,
                    path_in_repo=gguf_path.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"mAPEX {label} quantization",
                )
            except TypeError:
                wrapper.close()
                HfApi(token=token).upload_file(
                    path_or_fileobj=str(gguf_path),
                    path_in_repo=gguf_path.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"mAPEX {label} quantization",
                )
        finally:
            wrapper.close()
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
    else:
        HfApi(token=token).upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"mAPEX {label} quantization",
        )

    if not _live_active:
        log(f"✓ Uploaded {label} → {repo_id}")


# ---------------------------------------------------------------------------
# S3 upload (optional, S3-compatible API e.g. Yandex Object Storage)
# ---------------------------------------------------------------------------

def _parse_s3_endpoint(endpoint: str) -> tuple[str, str]:
    """Extract (base_url, bucket) from an S3 endpoint URL.

    The bucket must be embedded in the URL, either as a path segment or
    as a subdomain:
      https://storage.yandexcloud.net/<bucket>/   → path form
      https://<bucket>.storage.yandexcloud.net/   → subdomain form
    """
    ep = endpoint.strip()
    if "://" not in ep:
        ep = "https://" + ep
    u = urlsplit(ep)
    scheme = u.scheme or "https"
    netloc = u.netloc
    path = u.path.strip("/")
    if path:
        return f"{scheme}://{netloc}", path.split("/", 1)[0]
    host = (u.hostname or "").lower()
    labels = host.split(".")
    if host == "storage.yandexcloud.net" or len(labels) < 3:
        raise ValueError(
            f"Cannot determine S3 bucket from endpoint '{endpoint}'. "
            f"Use https://storage.yandexcloud.net/<bucket>/ or "
            f"https://<bucket>.storage.yandexcloud.net/")
    bucket_host = ".".join(labels[1:])
    if u.port:
        bucket_host = f"{bucket_host}:{u.port}"
    return f"{scheme}://{bucket_host}", labels[0]


def _get_s3_settings(args) -> Optional[dict]:
    """Resolve S3 settings from CLI args / env (.env), or None if disabled.

    Authentication modes (in priority order):
      * static keys — S3_KEY_ID + S3_SECRET (AWS SigV4 signing, does not
        expire; recommended for long batch runs)
      * IAM token — S3_TOKEN (Authorization: Bearer, expires in ~12 h)
    """
    endpoint = getattr(args, "s3_endpoint", None) or os.environ.get("S3_ENDPOINT")
    key_id = getattr(args, "s3_key_id", None) or os.environ.get("S3_KEY_ID")
    secret = getattr(args, "s3_secret", None) or os.environ.get("S3_SECRET")
    token = getattr(args, "s3_token", None) or os.environ.get("S3_TOKEN")

    if not endpoint and not (key_id or secret or token):
        return None
    if not endpoint:
        log_err("S3 auth found but --s3-endpoint (or S3_ENDPOINT in .env) "
                "is missing.")
        sys.exit(1)

    if key_id and secret:
        auth = {"mode": "sigv4", "key_id": key_id, "secret": secret}
    elif key_id or secret:
        log_err("Incomplete S3 static keys: both --s3-key-id and --s3-secret "
                "(or S3_KEY_ID / S3_SECRET in .env) are required.")
        sys.exit(1)
    elif token:
        auth = {"mode": "bearer", "token": token}
    else:
        log_err("No S3 credentials: set S3_KEY_ID + S3_SECRET (static keys) "
                "or S3_TOKEN (IAM token) in .env.")
        sys.exit(1)

    try:
        base_url, bucket = _parse_s3_endpoint(endpoint)
    except ValueError as exc:
        log_err(str(exc))
        sys.exit(1)
    return {"base_url": base_url, "bucket": bucket, "auth": auth}


# ---------------------------------------------------------------------------
# AWS SigV4 request signing (for static-key authentication)
# ---------------------------------------------------------------------------

def _aws_sigv4_headers(method: str, url: str, key_id: str, secret: str,
                       body: Optional[bytes] = None) -> dict:
    """Build AWS Signature Version 4 headers for an S3 request.

    Signs host, x-amz-content-sha256 and x-amz-date.  When *body* is None
    the payload is streamed and declared as UNSIGNED-PAYLOAD; when bytes
    are given their real hash is used.
    """
    import hashlib
    import hmac

    u = urlsplit(url)
    host = u.netloc
    region = os.environ.get("S3_REGION", "ru-central1")
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = amz_date[:8]
    payload_hash = (hashlib.sha256(body).hexdigest()
                    if body is not None else "UNSIGNED-PAYLOAD")

    canonical_uri = u.path or "/"
    if u.query:
        pairs = []
        for part in u.query.split("&"):
            k, _, v = part.partition("=")
            pairs.append((_uri_encode(k), _uri_encode(v)))
        pairs.sort()
        canonical_query = "&".join(f"{k}={v}" for k, v in pairs)
    else:
        canonical_query = ""

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n")
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    canonical_request = "\n".join([
        method, canonical_uri, canonical_query,
        canonical_headers, signed_headers, payload_hash,
    ])
    scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    def _h(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k = _h(("AWS4" + secret).encode(), datestamp)
    k = _h(k, region)
    k = _h(k, "s3")
    k = _h(k, "aws4_request")
    signature = hmac.new(k, string_to_sign.encode(),
                         hashlib.sha256).hexdigest()

    return {
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"),
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }


def _uri_encode(s: str) -> str:
    """RFC 3986 URI-encode (AWS SigV4 flavour: keep unreserved chars)."""
    import urllib.parse
    return urllib.parse.quote(s, safe="-_.~")


def _s3_auth_headers(s3: dict, method: str, url: str,
                     body: Optional[bytes] = None) -> dict:
    """Return authentication headers for an S3 request."""
    auth = s3["auth"]
    if auth["mode"] == "sigv4":
        return _aws_sigv4_headers(method, url, auth["key_id"],
                                  auth["secret"], body)
    return {"Authorization": f"Bearer {auth['token']}"}


_S3_PART_SIZE = 64 * 1024 * 1024   # 64 MB per multipart part


def _s3_progress(key: str, sent: int):
    """Update S3 upload progress in the live display."""
    if _lc is not None and key in _lc.s3_upload_progress:
        _lc.s3_upload_progress[key]["current"] = sent
        if _lc.live is not None:
            _lc.live.update()


def _s3_check(resp: httpx.Response, what: str):
    """Raise with server details on a non-2xx S3 response."""
    if resp.status_code >= 300:
        raise RuntimeError(
            f"S3 {what} failed: HTTP {resp.status_code}: {resp.text[:300]}")


@_retry_on_network_error
def _s3_put_part(url: str, s3: dict, data: bytes) -> str:
    """PUT a single multipart part, return its ETag. Retries on network
    errors and transient server errors (5xx/429)."""
    headers = _s3_auth_headers(s3, "PUT", url, data)
    headers["Content-Length"] = str(len(data))
    with httpx.Client(timeout=httpx.Timeout(600, connect=30)) as client:
        resp = client.put(url, content=data, headers=headers)
        if resp.status_code >= 500 or resp.status_code == 429:
            # Re-raised as httpx error so _retry_on_network_error retries it.
            raise httpx.HTTPStatusError(
                f"S3 part upload HTTP {resp.status_code}",
                request=resp.request, response=resp)
        _s3_check(resp, "part upload")
        return resp.headers.get("ETag", "")


@_retry_on_network_error
def upload_tier_s3(
    key: str,
    gguf_path: Path,
    s3: dict,
    *,
    label: Optional[str] = None,
) -> None:
    """Upload a quantized GGUF to an S3-compatible storage.

    Files larger than _S3_PART_SIZE are uploaded via multipart upload
    (simple PUT is rejected by the storage frontend for large bodies
    with HTTP 413).  Authentication: static keys (SigV4) or IAM token.
    """
    label = label or f"tier{key}"
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found for upload: {gguf_path}")
    if gguf_path.stat().st_size == 0:
        raise ValueError(f"GGUF file is empty: {gguf_path}")

    size = gguf_path.stat().st_size
    url = f"{s3['base_url']}/{s3['bucket']}/{gguf_path.name}"

    _live_active = _HAS_RICH and _lc is not None and _lc.live is not None
    if not _live_active:
        log(f"Uploading {label} ({gguf_path.name}, {size / (1024**3):.2f} GB) "
            f"→ S3 {s3['bucket']}")

    if size <= _S3_PART_SIZE:
        # Simple PUT — small files fit in a single request (unsigned payload).
        def _gen():
            sent = 0
            with open(gguf_path, "rb") as f:
                while True:
                    chunk = f.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    sent += len(chunk)
                    _s3_progress(key, sent)
                    yield chunk

        headers = _s3_auth_headers(s3, "PUT", url)
        headers["Content-Length"] = str(size)
        with httpx.Client(timeout=httpx.Timeout(600, connect=30)) as client:
            resp = client.put(url, content=_gen(), headers=headers)
            _s3_check(resp, "upload")
    else:
        _s3_multipart_upload(key, gguf_path, url, s3, size)

    if not _live_active:
        log(f"✓ Uploaded {label} → S3 {s3['bucket']}")


def _s3_multipart_upload(
    key: str,
    gguf_path: Path,
    url: str,
    s3: dict,
    size: int,
):
    """Multipart upload: initiate → sequential part PUTs → complete.

    Each part PUT is retried independently on network errors; on any
    unrecoverable failure the multipart upload is aborted server-side.
    """
    import re as _re

    part_size = _S3_PART_SIZE
    n_parts = (size + part_size - 1) // part_size
    timeout = httpx.Timeout(600, connect=30)

    with httpx.Client(timeout=timeout) as client:
        # 1. initiate
        resp = client.post(f"{url}?uploads=",
                           headers=_s3_auth_headers(s3, "POST",
                                                    f"{url}?uploads="))
        _s3_check(resp, "multipart initiate")
        m = _re.search(r"<UploadId>([^<]+)</UploadId>", resp.text)
        if not m:
            raise RuntimeError(
                f"S3 multipart initiate: no UploadId in response: "
                f"{resp.text[:300]}")
        upload_id = m.group(1)

        # 2. parts
        etags: list[tuple[int, str]] = []
        try:
            with open(gguf_path, "rb") as f:
                for i in range(1, n_parts + 1):
                    chunk = f.read(part_size)
                    if not chunk:
                        break
                    part_url = (f"{url}?partNumber={i}&uploadId={upload_id}")
                    etag = _s3_put_part(part_url, s3, chunk)
                    if not etag:
                        raise RuntimeError(
                            f"S3 part {i}/{n_parts}: missing ETag in response")
                    etags.append((i, etag))
                    _s3_progress(key, f.tell())

            # 3. complete
            parts_xml = "".join(
                f"<Part><PartNumber>{n}</PartNumber>"
                f"<ETag>{_escape_xml(etag)}</ETag></Part>"
                for n, etag in etags)
            complete_xml = (
                "<CompleteMultipartUpload>"
                f"{parts_xml}</CompleteMultipartUpload>")
            body = complete_xml.encode()
            resp = client.post(
                f"{url}?uploadId={upload_id}",
                content=body,
                headers={**_s3_auth_headers(s3, "POST",
                                            f"{url}?uploadId={upload_id}",
                                            body),
                         "Content-Type": "application/xml",
                         "Content-Length": str(len(body))},
            )
            _s3_check(resp, "multipart complete")
        except BaseException:
            # Abort server-side so no orphaned parts linger in the bucket.
            try:
                with httpx.Client(timeout=timeout) as abort_client:
                    abort_client.delete(
                        f"{url}?uploadId={upload_id}",
                        headers=_s3_auth_headers(s3, "DELETE",
                                                 f"{url}?uploadId={upload_id}"))
            except Exception:
                pass
            raise


def _escape_xml(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))


def _estimate_tier_size(tier: int, speed: bool, source_gguf: Path) -> int:
    """Estimate quantized output size in bytes via estimate_size.py --json."""
    cmd = [sys.executable, str(SCRIPT_DIR / "estimate_size.py"),
           "--profile", f"tier{tier}", "--json", str(source_gguf)]
    if speed:
        cmd.append("--speed")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return int(data["estimated_size_gb"] * 1e9)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    _check_hf_import()

    token = args.token or _get_hf_token()
    if not token:
        log_err("No HF token found. Set HF_TOKEN in .env or pass --token.")
        sys.exit(1)

    s3 = _get_s3_settings(args)

    # Optional S3 upload of the source (merged) model.  While the upload
    # task runs, tier uploads are blocked (they wait on *source_upload_done*);
    # quantization itself is not blocked.
    upload_source = bool(getattr(args, "s3_upload_source", False))
    if upload_source and s3 is None:
        log_err("--s3-upload-source ignored: S3 endpoint/credentials are "
                "not configured (set S3_ENDPOINT + keys in .env).")
        upload_source = False
    source_upload_done = threading.Event()
    _upload_queue_lock = threading.Lock()
    if not upload_source:
        source_upload_done.set()
    source_executor: Optional[ThreadPoolExecutor] = None
    source_upload_future: Optional[Future] = None

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / ".batch_quant_state.json"
    state = BatchState(state_path)

    # Parse output base repo id (org/name)
    output_base = args.output.rstrip("/")
    if not output_base or "/" not in output_base:
        log_err("--output must be in the form org/name  (e.g. MyOrg/Model-mAPEX)")
        sys.exit(1)

    tiers = args.tiers
    variants = expand_variants(tiers)
    output_dir = workspace / "quantized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── header ──
    log("=" * 60)
    if args.dry_run:
        log("  *** DRY RUN MODE ***")
    log("  mAPEX Batch Quantization Pipeline")
    log("=" * 60)
    log(f"  Model:    {args.model}")
    log(f"  Imatrix:  {args.imatrix}")
    log(f"  Output:   {output_base}")
    log(f"  Tiers:    {', '.join(variant_label(t, sp) for t, sp in variants)}")
    log(f"  Workspace: {workspace}")
    if s3:
        auth_desc = ("static key" if s3["auth"]["mode"] == "sigv4" else "IAM token")
        log(f"  S3:       {s3['base_url']}  (bucket: {s3['bucket']}, auth: {auth_desc})")
    if upload_source:
        log("  S3 source upload: enabled (merged source model → S3)")
    log("=" * 60)

    # Migrate state: variants uploaded before S3 support only went to HF.
    if s3:
        migrated = False
        for tier, speed in variants:
            key = variant_key(tier, speed)
            if state.tier_status(key) == "uploaded" and not state.upload_done(key, "hf"):
                state.mark_upload_done(key, "hf")
                migrated = True
        if migrated:
            log("✓ Marked previously uploaded variants as HF-complete (S3 enabled)")

    if _lc:
        _lc.state = state
        _lc.source_ok = state.source_info() is not None
        _lc.imatrix_ok = state.imatrix_info() is not None
        _lc.tiers = variants
        merge_rec = state.get("merge")
        if merge_rec:
            _lc.merge = {"status": merge_rec.get("status", "done"),
                         "elapsed": merge_rec.get("elapsed", 0.0)}

    _init_live()

    # ── 1. Discover & download source model + imatrix ──
    source_info = state.source_info()
    if source_info:
        source_gguf = Path(source_info["path"])
        if source_gguf.exists():
            if _lc:
                _lc.downloads["source model"] = {
                    "label": "source model", "status": "done",
                    "bytes": source_gguf.stat().st_size,
                    "total": source_gguf.stat().st_size, "start": time.time()}
                _lc.source_ok = True
                if _lc.live:
                    _lc.live.update(_build_live_renderable())
            log(f"✓ Source model already downloaded: {source_gguf.name}")
        else:
            source_info = None
            state.data.pop("source", None)
            state.save()

    imatrix_info = state.imatrix_info()
    if imatrix_info:
        imatrix_path = Path(imatrix_info["path"])
        if imatrix_path.exists():
            if _lc:
                _lc.downloads["imatrix"] = {
                    "label": "imatrix", "status": "done",
                    "bytes": imatrix_path.stat().st_size,
                    "total": imatrix_path.stat().st_size, "start": time.time()}
                _lc.imatrix_ok = True
                if _lc.live:
                    _lc.live.update(_build_live_renderable())
            log(f"✓ Imatrix already downloaded: {imatrix_path.name}")
        else:
            imatrix_info = None
            state.data.pop("imatrix", None)
            state.save()

    # ── discover files + download both in parallel ──
    if not source_info or not imatrix_info:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        @_retry_on_network_error
        def _list_all(repo_id):
            return api.list_repo_files(repo_id=repo_id, repo_type="model")

        if not source_info:
            src_files = _list_all(args.model)
        else:
            src_files = []
        if not imatrix_info:
            imx_files = _list_all(args.imatrix)
        else:
            imx_files = []

        # resolve source file
        if not source_info:
            gguf_files = [f for f in src_files if f.endswith(".gguf")]
            if not gguf_files:
                log_err(f"No .gguf files found in {args.model}")
                sys.exit(1)
            source_file = (getattr(args, "source_file", None)
                           or _pick_source_gguf(gguf_files))
            if not source_file:
                log_err(f"Could not determine source file in {args.model}. "
                        f"Found: {gguf_files}")
                sys.exit(1)
            if len(gguf_files) > 1:
                log(f"Found {len(gguf_files)} .gguf files, selected: {source_file}")
            source_files = _expand_shard_files(source_file, gguf_files)
            if len(source_files) > 1:
                log(f"Source model is split into {len(source_files)} shards: "
                    f"{source_files[0]} … {source_files[-1]}")

        # resolve imatrix file
        if not imatrix_info:
            imatrix_file = (getattr(args, "imatrix_file", None)
                            or _pick_imatrix_file(imx_files))
            if not imatrix_file:
                log_err(f"Could not find imatrix file in {args.imatrix}. "
                        f"Files: {imx_files}")
                sys.exit(1)
            log(f"Selected imatrix file: {imatrix_file}")

        # download both in the main thread so Ctrl+C actually stops them
        if args.dry_run:
            if not source_info:
                log(f"DRY RUN: would download source model {args.model} "
                    f"({', '.join(source_files)})")
                if len(source_files) > 1:
                    log("DRY RUN: shards would be merged into a single GGUF "
                        "before quantization")
                source_gguf = Path(f"/dry-run/{source_files[0]}")
                state.mark_source(str(source_gguf), source_files[0])
            if not imatrix_info:
                log(f"DRY RUN: would download imatrix {args.imatrix} ({imatrix_file})")
                imatrix_path = Path(f"/dry-run/{imatrix_file}")
                state.mark_imatrix(str(imatrix_path))
        else:
            with _allow_hard_interrupt():
                downloads = []
                if not source_info:
                    downloads.append(("source", download_source_model, (args.model, workspace, token, source_files)))
                if not imatrix_info:
                    downloads.append(("imatrix", download_imatrix, (args.imatrix, workspace, token, imatrix_file)))
                for key, fn, a in downloads:
                    try:
                        result = fn(*a)
                    except KeyboardInterrupt:
                        log("\nDownload interrupted by user.")
                        sys.exit(130)
                    except Exception as exc:
                        log_err(f"Download failed ({key}): {exc}")
                        sys.exit(1)
                    if key == "source":
                        if len(result) == 1:
                            source_gguf = result[0]
                        else:
                            source_gguf = merge_gguf_shards(
                                result, keep_shards=args.keep_files, state=state)
                        state.mark_source(str(source_gguf), source_files[0])
                    else:
                        imatrix_path = result
                        state.mark_imatrix(str(result))

        if _lc:
            _lc.source_ok = True
            _lc.imatrix_ok = True
            if _lc.live:
                _lc.live.update(_build_live_renderable())

        log("✓ Both downloads complete.")

    # ── 1b. Optional S3 upload of the source (merged) model ──
    # Runs in its own single-worker executor as a separate "SRC" task.
    # Tier upload workers wait on *source_upload_done*, so the source
    # upload blocks all tier uploads (HF and S3) but not quantization.
    if upload_source and not args.dry_run:
        if state.source_upload_done():
            log("✓ Source model already uploaded to S3")
            source_upload_done.set()
        else:
            def _do_upload_source_s3():
                fsize = (source_gguf.stat().st_size
                         if source_gguf.exists() else 0)
                if _lc:
                    with _upload_queue_lock:
                        _lc.source_upload = {"status": "uploading",
                                             "start": time.time()}
                        _lc.s3_upload_progress["src"] = {
                            "current": 0, "total": fsize,
                            "start_ts": time.time()}
                    if _lc.live:
                        _lc.live.update(_build_live_renderable())
                log(f"Uploading source model ({source_gguf.name}, "
                    f"{fsize / (1024**3):.2f} GB) → S3 {s3['bucket']}")
                try:
                    upload_tier_s3("src", source_gguf, s3, label="source")
                    elapsed = (time.time()
                               - _lc.source_upload.get("start", time.time())
                               ) if _lc and _lc.source_upload else 0.0
                    state.mark_source_upload("done", elapsed=round(elapsed, 1))
                    if _lc:
                        with _upload_queue_lock:
                            _lc.s3_upload_progress.pop("src", None)
                            _lc.source_upload = {"status": "done",
                                                 "elapsed": elapsed}
                    log(f"✓ Source model uploaded to S3 {s3['bucket']} "
                        f"in {_format_elapsed(elapsed)}")
                except Exception as exc:
                    err = str(exc)[:300]
                    log_err(f"Source model S3 upload failed: {err}")
                    state.mark_source_upload("error", error=err)
                    if _lc:
                        with _upload_queue_lock:
                            _lc.s3_upload_progress.pop("src", None)
                            _lc.source_upload = {"status": "error"}
                finally:
                    source_upload_done.set()
                    if _lc and _lc.live:
                        _lc.live.update(_build_live_renderable())

            source_executor = ThreadPoolExecutor(max_workers=1)
            source_upload_future = source_executor.submit(_do_upload_source_s3)
    elif upload_source and args.dry_run:
        log(f"DRY RUN: would upload source model {source_gguf.name} "
            f"→ S3 {s3['bucket']}")
        source_upload_done.set()

    # ── 2. Initialize README.md ──
    readme_path = workspace / "README.md"
    if not state.readme_initialized() or not readme_path.exists():
        if _readme_in_repo(output_base, token):
            log(f"Found README.md in {output_base}, downloading …")
            downloaded = _download_readme(output_base, token, readme_path)
            if downloaded:
                log(f"✓ README.md downloaded from {output_base}")
                _ensure_apex_header(readme_path, args.model,
                                   state.source_info()["format"])
            else:
                _create_readme(readme_path, args.model,
                               state.source_info()["format"])
                log("✓ Created new README.md (download failed, created fresh)")
        else:
            _create_readme(readme_path, args.model,
                           state.source_info()["format"])
            log("✓ Created new README.md")
        _rebuild_readme_rows(readme_path, state)
        state.mark_readme_initialized()
    else:
        log("✓ README.md already initialized")

    # ── Determine which tiers need work ──
    if _lc and _lc.live:
        _lc.live.update(_build_live_renderable())
    else:
        display_status(state, source_ok=True, imatrix_ok=True, tiers=variants)

    needs_work = []
    for tier, speed in variants:
        key = variant_key(tier, speed)
        label = variant_label(tier, speed)
        st = state.tier_status(key)
        if st == "uploaded":
            if s3 and not state.upload_done(key, "s3"):
                log(f"{label}: already on HF, will upload to S3")
            else:
                log(f"✓ {label}: already uploaded, skipping")
                continue
        needs_work.append((tier, speed))

    if not needs_work:
        log("\n✅ All variants completed. Nothing to do.")
        if source_upload_future is not None:
            log("Waiting for source model S3 upload to finish …")
            source_upload_future.result()
            source_executor.shutdown(wait=True)
        _print_summary(state, variants, output_base)
        if not args.dry_run:
            _upload_readme(readme_path, output_base, token)
        else:
            log(f"DRY RUN: would upload README.md → {output_base}")
        return

    log(f"\nVariants to process: "
        f"{[variant_label(t, sp) for t, sp in needs_work]}")

    # ── 3. Quantize + Upload pipeline ──
    # At any moment: 1 HF upload running, 1 S3 upload running (if enabled),
    # 1 quantize running.  HF and S3 uploads run in parallel, each via its
    # own single-threaded executor queue.  The live display shows only the
    # tier whose upload is *currently executing* (set via callback inside
    # the upload workers), not every queued tier.
    _init_live()
    upload_executor = ThreadPoolExecutor(max_workers=1)
    s3_executor = ThreadPoolExecutor(max_workers=1) if s3 else None
    # pending_uploads: ((tier, speed), target, Future)
    pending_uploads: list = []

    def _active_start(key: str):
        """Mark variant active in the display when its first upload starts."""
        if _lc and _lc.live is not None:
            with _upload_queue_lock:
                if key not in _lc.active_tiers:
                    _lc.active_tiers[key] = {
                        "status": "uploading",
                        "start": time.time(),
                        "last_line": "",
                    }

    def _maybe_finish_upload(key: str, p: Path):
        """Mark variant uploaded when every enabled target has finished."""
        if not state.upload_done(key, "hf"):
            return
        if s3 is not None and not state.upload_done(key, "s3"):
            return
        sz_gb = p.stat().st_size / (1024**3) if p.exists() else 0
        state.set_tier(key, "uploaded", size_mb=round(sz_gb * 1024, 1))
        # Free disk space: the file is fully uploaded to all targets.
        if not args.keep_files and p.exists():
            sz = p.stat().st_size / (1024**3)
            p.unlink()
            log(f"🗑  Removed uploaded file: {p.name}  ({sz:.2f} GB)")
        if _lc:
            with _upload_queue_lock:
                _lc.active_tiers.pop(key, None)
                if _lc.live:
                    _lc.live.update(_build_live_renderable())

    def _do_upload_hf(vk: tuple, p: Path):
        """Run in the HF executor thread: upload GGUF to HuggingFace."""
        tier, speed = vk
        key = variant_key(tier, speed)
        # While the source-model S3 upload (SRC task) is running, tier
        # uploads are blocked; quantization is not affected.
        source_upload_done.wait()
        _active_start(key)
        fsize = p.stat().st_size if p.exists() else 0
        if _lc and _lc.live is not None:
            with _upload_queue_lock:
                _lc.upload_progress[key] = {"current": 0, "total": fsize,
                                            "stage": "preparing"}
        upload_tier(key, p, output_base, token)
        state.mark_upload_done(key, "hf")
        if _lc:
            with _upload_queue_lock:
                _lc.upload_progress.pop(key, None)
        # README row tracks the HF copy
        sz_gb = p.stat().st_size / (1024**3) if p.exists() else 0
        tier_name = variant_label(tier, speed)
        if readme_path.exists() and not _readme_has_tier(readme_path, tier_name):
            _append_readme_row(readme_path, tier_name, sz_gb)
        _maybe_finish_upload(key, p)

    def _do_upload_s3(vk: tuple, p: Path):
        """Run in the S3 executor thread: upload GGUF to S3 storage."""
        tier, speed = vk
        key = variant_key(tier, speed)
        source_upload_done.wait()
        _active_start(key)
        fsize = p.stat().st_size if p.exists() else 0
        if _lc and _lc.live is not None:
            with _upload_queue_lock:
                _lc.s3_upload_progress[key] = {"current": 0, "total": fsize, "start_ts": time.time()}
        upload_tier_s3(key, p, s3)
        state.mark_upload_done(key, "s3")
        if _lc:
            with _upload_queue_lock:
                _lc.s3_upload_progress.pop(key, None)
        _maybe_finish_upload(key, p)

    def _check_completed_uploads():
        """Process all completed upload futures (non-blocking)."""
        nonlocal pending_uploads
        still_pending = []
        for vk, target, fut in pending_uploads:
            if not fut.done():
                still_pending.append((vk, target, fut))
                continue
            key = variant_key(*vk)
            label = variant_label(*vk)
            try:
                fut.result()
                if _lc:
                    with _upload_queue_lock:
                        if target == "hf":
                            _lc.upload_progress.pop(key, None)
                        else:
                            _lc.s3_upload_progress.pop(key, None)
                    if _lc.live:
                        _lc.live.update(_build_live_renderable())
            except Exception as exc:
                err = str(exc)[:300]
                log_err(f"{label} {target} upload failed: {err}")
                state.set_tier(key, "error", error=f"{target}: {err}",
                               error_short=f"{target}: {err[:180]}")
                if _lc:
                    with _upload_queue_lock:
                        if target == "hf":
                            _lc.upload_progress.pop(key, None)
                        else:
                            _lc.s3_upload_progress.pop(key, None)
                    if _lc.live:
                        _lc.live.update(_build_live_renderable())
        pending_uploads = still_pending

    def _wait_all_uploads():
        """Wait for every remaining upload future to finish."""
        while pending_uploads:
            _check_completed_uploads()
            if pending_uploads:
                time.sleep(0.5)

    # Disk-space gating: estimated output size + safety margin must fit
    # in the free space of the workspace volume before a tier is quantized.
    DISK_MARGIN = 1.25   # 25% headroom over the estimate

    def _wait_for_disk_space(needed_bytes: int, label: str):
        """Block until *needed_bytes* are free on the workspace volume.

        While waiting, completed uploads are collected — their files are
        deleted after a successful upload, freeing disk space.  Aborts the
        pipeline if nothing is queued and space still cannot be freed.
        """
        while True:
            free = shutil.disk_usage(workspace).free
            if free >= needed_bytes:
                return
            if not pending_uploads or _interruption_requested:
                log_err(
                    f"Not enough disk space for {label}: need "
                    f"{needed_bytes / (1024**3):.1f} GB, free "
                    f"{free / (1024**3):.1f} GB, and no pending uploads "
                    f"to wait for. Free up space and re-run.")
                sys.exit(1)
            log(f"⏳ {label}: need {needed_bytes / (1024**3):.1f} GB, "
                f"free {free / (1024**3):.1f} GB — waiting for pending "
                f"uploads to finish and free space …")
            # Sleep until at least one pending upload future completes.
            while not any(fut.done() for _, _, fut in pending_uploads):
                time.sleep(0.5)
                if _interruption_requested:
                    return
            _check_completed_uploads()

    processed = []
    failed = []

    # huggingface_hub catches KeyboardInterrupt in its retry/tqdm loops
    # and swallows it, so _wait_all_uploads / fut.result() will block
    # forever on a normal SIGINT.  Use _allow_hard_interrupt so Ctrl+C
    # during quantize or upload kills the process immediately.
    with _allow_hard_interrupt():
        try:
            for tier, speed in needs_work:
                if _interruption_requested:
                    log("⚠ Pipeline interrupted by user.")
                    break

                key = variant_key(tier, speed)
                label = variant_label(tier, speed)

                if args.dry_run:
                    log(f"DRY RUN: would quantize and upload {label}")
                    continue

                output_gguf = output_dir / _tier_filename(source_gguf.name, tier, speed)
                st = state.tier_status(key)

                # ── skip already uploaded ──
                if st == "uploaded":
                    if output_gguf.exists() and output_gguf.stat().st_size > 0:
                        if s3 and not state.upload_done(key, "s3"):
                            log(f"{label}: on HF, uploading to S3")
                        else:
                            log(f"✓ {label}: already uploaded, skipping")
                            continue
                    elif s3 and not state.upload_done(key, "s3"):
                        log(f"{label}: on HF, file was cleaned up → "
                            f"will re-quantize for S3 upload")
                        state.set_tier(key, "pending")
                        st = "pending"
                    else:
                        log(f"✓ {label}: already uploaded, skipping")
                        continue
                elif st == "quantizing":
                    log(f"{label}: was quantizing (interrupted) → will re-quantize")
                    if output_gguf.exists():
                        output_gguf.unlink(missing_ok=True)
                    state.set_tier(key, "pending")
                    st = "pending"
                elif st == "uploading":
                    if output_gguf.exists() and output_gguf.stat().st_size > 0:
                        log(f"{label}: was uploading, output exists → will re-upload")
                        state.set_tier(key, "quantized")
                        st = "quantized"
                    else:
                        log(f"{label}: was uploading, output missing → will re-quantize")
                        state.set_tier(key, "pending")
                        st = "pending"

                # ── collect completed uploads (non-blocking) ──
                _check_completed_uploads()

                # ── quantize (runs while executor processes upload queue) ──
                if st in ("pending", "error", "quantized"):
                    if st in ("quantized", "error") and output_gguf.exists() and output_gguf.stat().st_size > 0:
                        # Verify file size matches what was recorded at
                        # quantize time; truncated files (Ctrl-C) must be
                        # re-quantized.
                        expected_mb = state._t(key).get("file_size_mb")
                        actual_mb = round(output_gguf.stat().st_size / (1024**2), 1)
                        if expected_mb is None or abs(expected_mb - actual_mb) < 0.5:
                            log(f"✓ {label}: quantized file exists, skipping quantize")
                            state.set_tier(key, "quantized", size_mb=actual_mb,
                                           file_size_mb=actual_mb)
                        else:
                            log(f"{label}: size mismatch ({actual_mb} MB vs expected "
                                f"{expected_mb} MB) → will re-quantize")
                            state.set_tier(key, "pending")
                            st = "pending"
                    else:
                        est_size = _estimate_tier_size(tier, speed, source_gguf)
                        _wait_for_disk_space(int(est_size * DISK_MARGIN), label)
                        state.set_tier(key, "quantizing")
                        if _lc:
                            _lc.active_tiers[key] = {"status": "quantizing",
                                                     "start": time.time(),
                                                     "last_line": ""}
                        t0 = time.time()
                        try:
                            run_quantize(tier, speed, source_gguf, imatrix_path, output_gguf)
                        except Exception as exc:
                            err = str(exc)[:300]
                            log_err(f"{label} quantize failed: {err}")
                            state.set_tier(key, "error", error=str(exc)[:300],
                                           error_short=err[:200])
                            failed.append((tier, speed))
                            if _lc:
                                _lc.active_tiers.pop(key, None)
                            if output_gguf.exists() and output_gguf.stat().st_size == 0:
                                output_gguf.unlink(missing_ok=True)
                            continue
                        elapsed = time.time() - t0
                        m, s = divmod(int(elapsed), 60)
                        h, m = divmod(m, 60)
                        sz_mb = output_gguf.stat().st_size / (1024**2)
                        log(f"✓ {label}: quantized in {h:02d}:{m:02d}:{s:02d}  "
                            f"({sz_mb:.1f} MB)")
                        state.set_tier(key, "quantized", size_mb=round(sz_mb, 1),
                                       file_size_mb=round(sz_mb, 1))
                        if _lc:
                            _lc.active_tiers.pop(key, None)

                if not (_lc and _lc.live):
                    display_status(state, True, True, tiers=variants)

                # ── skip upload if interrupted ──
                if _interruption_requested:
                    break

                # ── submit uploads to executor queues ──
                # Doesn't block.  Each task sits in its queue until the
                # executor's single worker finishes the previous upload.
                # HF and S3 uploads run in parallel (separate executors),
                # so at most one quant uploads to HF and one to S3.
                state.set_tier(key, "uploading")
                info = state._t(key)
                if not info.get("hf_done"):
                    fut = upload_executor.submit(_do_upload_hf, (tier, speed), output_gguf)
                    pending_uploads.append(((tier, speed), "hf", fut))
                if s3 is not None and not info.get("s3_done"):
                    fut = s3_executor.submit(_do_upload_s3, (tier, speed), output_gguf)
                    pending_uploads.append(((tier, speed), "s3", fut))
                processed.append((tier, speed))

            # ── collect any remaining completions ──
            _wait_all_uploads()
            upload_executor.shutdown(wait=True)
            if s3_executor is not None:
                s3_executor.shutdown(wait=True)
            if source_upload_future is not None:
                source_upload_future.result()
                source_executor.shutdown(wait=True)

        except KeyboardInterrupt:
            log_err("Forced interrupt — saving state and cleaning up.")
            _check_completed_uploads()
            upload_executor.shutdown(wait=False, cancel_futures=True)
            if s3_executor is not None:
                s3_executor.shutdown(wait=False, cancel_futures=True)
            if source_executor is not None:
                source_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            log_err(f"Unexpected error: {exc}")
            _check_completed_uploads()
            upload_executor.shutdown(wait=False, cancel_futures=True)
            if s3_executor is not None:
                s3_executor.shutdown(wait=False, cancel_futures=True)
            if source_executor is not None:
                source_executor.shutdown(wait=False, cancel_futures=True)
        finally:
            _stop_live()

    # ── 4. Cleanup incomplete outputs ──
    if not args.dry_run and not args.keep_files:
        _cleanup_incomplete(state, output_dir,
                            source_name=source_gguf.name, variants=variants)
    elif args.keep_files:
        log("✓ --keep-files: quantized files are kept in " + str(output_dir))

    # ── 5. Final report ──
    _print_summary(state, variants, output_base)

    # ── 6. Upload README to output repo ──
    if readme_path.exists():
        if args.dry_run:
            log(f"\nDRY RUN: would upload README.md → {output_base}")
        else:
            log(f"\nUploading README.md → {output_base} …")
            _upload_readme(readme_path, output_base, token)


def _cleanup_incomplete(state: BatchState, output_dir: Path,
                        source_name: str = "", variants: list = None):
    """Delete quantized GGUFs that are no longer needed on disk.

    Removes files interrupted mid-quantize (status 'quantizing') and
    files already fully uploaded (status 'uploaded') to free disk space.
    Preserves files whose upload did not finish ('quantized', 'uploading',
    'error') so that a re-run only retries the upload.
    """
    if variants is None:
        variants = expand_variants(TIERS)
    count = 0
    for tier, speed in variants:
        key = variant_key(tier, speed)
        st = state.tier_status(key)
        if st not in ("quantizing", "uploaded"):
            continue
        gguf = output_dir / _tier_filename(source_name, tier, speed)
        if gguf.exists():
            sz = gguf.stat().st_size / (1024**3)
            gguf.unlink()
            if st == "quantizing":
                log(f"🗑  Removed incomplete file: {gguf.name}  ({sz:.2f} GB)")
            else:
                log(f"🗑  Removed uploaded file: {gguf.name}  ({sz:.2f} GB)")
            count += 1
    if count:
        log(f"Cleaned up {count} file(s).")


def _print_summary(state: BatchState, variants: list, output_base: str):
    """Print a final summary table."""
    log("\n" + "=" * 60)
    log("  Final Report")
    log("=" * 60)

    uploaded = []
    quantized_pending = []
    errors = []

    for tier, speed in variants:
        key = variant_key(tier, speed)
        label = variant_label(tier, speed)
        st = state.tier_status(key)
        info = state._t(key)
        sz = info.get("size_mb", "?")
        if st == "uploaded":
            uploaded.append((label, sz))
        elif st in ("quantized", "quantizing", "uploading"):
            quantized_pending.append((label, st))
        elif st == "error":
            errors.append((label, info.get("error_short", "unknown")))
        # pending variants just sit in neither list

    pending_count = len(variants) - len(uploaded) - len(quantized_pending) - len(errors)

    log(f"\n  Repo: https://huggingface.co/{output_base}")

    if uploaded:
        log(f"\n  ✅ Uploaded ({len(uploaded)}):")
        for label, sz in uploaded:
            log(f"     {label:<9}  {sz} MB")

    if quantized_pending:
        log(f"\n  📦 Quantized but not uploaded ({len(quantized_pending)}):")
        for label, st in quantized_pending:
            log(f"     {label:<9}  (status: {st})")

    if errors:
        log(f"\n  ❌ Errors ({len(errors)}):")
        for label, err in errors:
            log(f"     {label:<9}  {err}")

    if pending_count > 0:
        log(f"\n  ○  Not started: {pending_count}")

    failed_quant = [lbl for lbl, _ in errors]
    not_uploaded = [lbl for lbl, _ in quantized_pending]
    if failed_quant or not_uploaded:
        log(f"\n  ⚠  Incomplete variants: {sorted(failed_quant + not_uploaded)}")
        log("     Re-run the script with the same arguments to retry.")

    log("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tiers(s: str) -> list:
    """Parse tier spec into a sorted list of ints.

    Supports comma-separated ranges and individual values:
      '1-10,13'   → [1,2,3,4,5,6,7,8,9,10,13]
      '1-13'      → [1,2,3,4,5,6,7,8,9,10,11,12,13]
      '3-8'       → [3,4,5,6,7,8]
      '1,5,7'     → [1,5,7]
      '2'         → [2]
    """
    tiers = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            tiers.update(range(a, b + 1))
        else:
            tiers.add(int(part))
    return sorted(tiers)


def main():
    parser = argparse.ArgumentParser(
        description="mAPEX Batch Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/batch_quantize.py \\\n"
            "    --model bullerwins/Qwen3.5-35B-A3B-GGUF \\\n"
            "    --imatrix bullerwins/Qwen3.5-35B-A3B-imatrix-GGUF \\\n"
            "    --output user/Qwen3.5-35B-A3B-mAPEX\n"
        ),
    )
    parser.add_argument("--model", "-m", required=True,
                        help="HF repo with source GGUF  (e.g. user/model-GGUF)")
    parser.add_argument("--imatrix", "-i", required=True,
                        help="HF repo with imatrix file (e.g. user/imatrix-GGUF)")
    parser.add_argument("--output", "-o", required=True,
                        help="HF repo id for all output tiers (org/name), "
                             "e.g. user/model-mAPEX")
    parser.add_argument("--tiers", default="1-15",
                        help="Tier spec: '1-13', '1-10,13', '3-8', '1,5,7' "
                             "(default: 1-15; tiers 9-15 are each produced in "
                             "two variants: normal and -s with --speed)")
    parser.add_argument("--workspace", "-w",
                        default=str(Path.home() / "apex_batch"),
                        help="Workspace directory for state & intermediate files "
                             "(default: ~/apex_batch)")
    parser.add_argument("--token", "-t",
                        help="HF token (default: $HF_TOKEN from env / .env)")
    parser.add_argument("--s3-endpoint",
                        help="S3 endpoint URL with bucket embedded "
                             "(e.g. https://storage.yandexcloud.net/my-bucket/ or "
                             "https://my-bucket.storage.yandexcloud.net/). "
                             "Enables parallel S3 upload. "
                             "(default: $S3_ENDPOINT from env / .env)")
    parser.add_argument("--s3-token",
                        help="S3 IAM token, sent as Authorization: Bearer "
                             "(default: $S3_TOKEN from env / .env)")
    parser.add_argument("--s3-key-id",
                        help="S3 static access key id, used with --s3-secret "
                             "(AWS SigV4 signing; does not expire, recommended "
                             "for long runs). (default: $S3_KEY_ID from env / .env)")
    parser.add_argument("--s3-secret",
                        help="S3 static access secret key, used with "
                             "--s3-key-id (default: $S3_SECRET from env / .env)")
    parser.add_argument("--s3-upload-source", action="store_true",
                        help="Upload the source model to S3 (the merged GGUF "
                             "when the source is split into shards, otherwise "
                             "the single source file). Requires S3 credentials. "
                             "Runs as a separate SRC task in the live table and "
                             "blocks tier uploads until it finishes "
                             "(quantization is not blocked).")
    parser.add_argument("--source-file",
                        help="Explicit source GGUF filename (skip auto-detection)")
    parser.add_argument("--imatrix-file",
                        help="Explicit imatrix filename (skip auto-detection)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate the pipeline without downloading, quantizing, "
                             "or uploading. Still creates/updates README.md locally.")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep quantized GGUF files on disk even after "
                             "successful upload. Without this flag, files that "
                             "are fully uploaded (HF and S3 if enabled) are "
                             "deleted from disk to save space.")

    args = parser.parse_args()
    args.tiers = _parse_tiers(args.tiers)

    if not args.tiers or not all(1 <= t <= 15 for t in args.tiers):
        log_err("--tiers must be in range 1-15")
        sys.exit(1)

    try:
        run_pipeline(args)
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
