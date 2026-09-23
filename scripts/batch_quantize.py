#!/usr/bin/env python3
"""APEX Batch Quantization Pipeline.

Downloads a source GGUF model (bf16/f16/f32) and an importance matrix from
HuggingFace, quantizes through APEX tiers 1-13 using quantize.py, and
uploads every resulting GGUF to HuggingFace.

Manages a README.md in the output repo: downloads an existing README or
creates one with source info, APEX attribution, and a quantization table.
After all tiers are uploaded the README is pushed to the same repo.

Supports **resumable** execution: state is persisted to a JSON file so that
re-running with identical arguments only performs remaining work.

Concurrency:
  * model + imatrix downloads run sequentially in the main thread
    (so Ctrl+C can interrupt them)
  * tiers are quantized sequentially, but tier N upload overlaps with
    tier N+1 quantization (upload runs in a background thread)

Usage:
  python3 scripts/batch_quantize.py \\
      --model user/source-model-GGUF \\
      --imatrix user/imatrix-repo \\
      --output user/model-APEX

Output repo: {output}  (all tier GGUFs in one repo)

Environment / .env:
  HF_TOKEN   HuggingFace access token
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

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
        sys.stderr.write("\n")
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

TIERS = list(range(1, 14))

TIER_BASE_TYPE = {
    1: "Q8_0", 2: "Q8_0", 3: "Q8_0", 4: "Q8_0", 5: "Q8_0", 6: "Q8_0",
    7: "Q6_K", 8: "Q6_K", 9: "Q6_K",
    10: "Q5_K_M", 11: "Q5_K_M", 12: "Q5_K_M",
    13: "Q4_K_M",
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def log_err(msg: str):
    print(f"[{_ts()}] ❌ {msg}", file=sys.stderr, flush=True)


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

    # -- per-tier helpers --
    def _t(self, tier: int) -> dict:
        return self.data.setdefault("tiers", {}).setdefault(str(tier), {})

    def tier_status(self, tier: int) -> str:
        return self._t(tier).get("status", "pending")

    def set_tier(self, tier: int, status: str, **extra):
        t = self._t(tier)
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


# ---------------------------------------------------------------------------
# Rich display  (optional — falls back to plain text)
# ---------------------------------------------------------------------------

_HAS_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    pass

console = Console() if _HAS_RICH else None

_STATUS_STYLE = {
    "pending":    "dim",
    "quantizing": "yellow",
    "quantized":  "cyan",
    "uploading":  "magenta",
    "uploaded":   "green",
    "error":      "bold red",
    "done":       "bold green",
}

_STATUS_ICON = {
    "pending":    "○",
    "quantizing": "◉",
    "quantized":  "◇",
    "uploading":  "↑",
    "uploaded":   "✓",
    "error":      "✗",
    "done":       "✓",
}


def display_status(state: BatchState, source_ok: bool, imatrix_ok: bool,
                   tiers: list = None):
    """Render the current status to the terminal."""
    if tiers is None:
        tiers = TIERS
    if _HAS_RICH:
        from rich.text import Text
        table = Table(show_lines=False, expand=False, padding=(0, 1))
        table.add_column("Tier", justify="right", style="bold", width=6)
        table.add_column("Base", width=7)
        table.add_column("Status", min_width=26)
        table.add_column("Info", min_width=16, style="dim")
        for tier in tiers:
            base = TIER_BASE_TYPE[tier]
            st = state.tier_status(tier)
            icon = _STATUS_ICON.get(st, "?")
            style = _STATUS_STYLE.get(st, "")
            info = ""
            if st == "error":
                info = state._t(tier).get("error_short", "")
            elif st == "uploaded":
                sz = state._t(tier).get("size_mb")
                if sz:
                    info = f"{sz} MB"
            table.add_row(
                f"tier{tier}",
                base,
                Text(f" {icon} {st}", style=style),
                info,
            )
        src_label = "✓ downloaded" if source_ok else "… pending"
        imx_label = "✓ downloaded" if imatrix_ok else "… pending"
        header = Text(f"Source:  {src_label}\nImatrix: {imx_label}", style="bold")
        panel = Panel(table, title="[bold]APEX Batch Quantization[/bold]",
                       subtitle=header, border_style="blue")
        console.print(panel)
    else:
        print(f"\n  {'Tier':<6} {'Base':<8} {'Status'}")
        print(f"  {'─'*6} {'─'*8} {'─'*20}")
        for tier in tiers:
            base = TIER_BASE_TYPE[tier]
            st = state.tier_status(tier)
            icon = _STATUS_ICON.get(st, "?")
            print(f"  {tier:<6} {base:<8} {icon} {st}")
        print()


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

README_HEADER = """\
# APEX Quantized Models

## Source Data

The source data for quantization was taken from [{source_model}](https://huggingface.co/{source_model}).

Original source file: `{source_file}`

## Quantization Method

All quants were produced using the **modified APEX** quantization scheme.

APEX (Automated Precision EXpert allocation) assigns per-layer, per-tensor precision
for MoE models using `llama.cpp`'s `--tensor-type-file`.

For more information, see: <https://github.com/DrMoriarty/apex-quant/>

## Quantized Models

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
    """Create a fresh README.md with header and empty table."""
    readme_path.write_text(README_HEADER.format(
        source_model=source_model,
        source_file=source_file,
    ))


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
    """Populate README with rows for all tiers already marked uploaded in state."""
    for tier in TIERS:
        if state.tier_status(tier) != "uploaded":
            continue
        tier_name = f"tier{tier}"
        if _readme_has_tier(readme_path, tier_name):
            continue
        info = state._t(tier)
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
                commit_message="Update README — APEX quantization info",
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

    with open(part, mode) as f, \
         httpx.Client(follow_redirects=True, timeout=httpx.Timeout(300, connect=30)) as client:
        with client.stream("GET", url, headers=headers) as response:
            if existing > 0 and response.status_code == 200:
                # Server ignored Range — restart from scratch
                f.seek(0)
                f.truncate()
                bytes_downloaded = 0

            response.raise_for_status()

            total_mb = total / (1024**2)
            last_log = time.time()
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                now = time.time()
                if now - last_log >= 5:
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
        log(f"{label} already exists: {dest.name}  "
            f"({dest.stat().st_size / (1024**3):.2f} GB)")
        return dest

    workspace.mkdir(parents=True, exist_ok=True)
    url = _get_hf_download_url(repo_id, filename, token)
    return _resumable_download(url, dest, token=token, label=label)


@_retry_on_network_error
def download_source_model(
    repo_id: str,
    workspace: Path,
    token: Optional[str],
    source_file: str,
) -> Path:
    """Download the source GGUF from HF, return local path."""
    target_dir = workspace / "source_model"
    result = _download_single_file(
        repo_id, source_file, target_dir, token, label="source model",
    )
    if not result.exists():
        raise FileNotFoundError(f"Source file not found after download: {result}")
    log(f"Source model ready: {result.name}  ({result.stat().st_size / (1024**3):.2f} GB)")
    return result


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

def run_quantize(
    tier: int,
    source_gguf: Path,
    imatrix_path: Path,
    output_gguf: Path,
) -> None:
    """Run quantize.py for a single tier. Raises on failure."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "quantize.py"),
        "--profile", f"tier{tier}",
        "--imatrix", str(imatrix_path),
        str(source_gguf), str(output_gguf),
    ]
    log(f"Quantizing tier{tier} → {output_gguf.name}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"quantize.py exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@_retry_on_network_error
def upload_tier(
    tier: int,
    gguf_path: Path,
    repo_id: str,
    token: Optional[str],
) -> None:
    """Upload a quantized GGUF to HuggingFace."""
    from huggingface_hub import create_repo, upload_folder
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found for upload: {gguf_path}")
    if gguf_path.stat().st_size == 0:
        raise ValueError(f"GGUF file is empty: {gguf_path}")

    log(f"Uploading tier{tier} ({gguf_path.name}, "
        f"{gguf_path.stat().st_size / (1024**3):.2f} GB) → {repo_id}")

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    with tempfile.TemporaryDirectory(prefix=f"apex_upload_t{tier}_") as tmpdir:
        link = Path(tmpdir) / gguf_path.name
        link.symlink_to(gguf_path.resolve())
        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"APEX tier{tier} quantization",
        )
    log(f"✓ Uploaded tier{tier} → {repo_id}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    _check_hf_import()

    token = args.token or _get_hf_token()
    if not token:
        log_err("No HF token found. Set HF_TOKEN in .env or pass --token.")
        sys.exit(1)

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / ".batch_quant_state.json"
    state = BatchState(state_path)

    # Parse output base repo id (org/name)
    output_base = args.output.rstrip("/")
    if not output_base or "/" not in output_base:
        log_err("--output must be in the form org/name  (e.g. MyOrg/Model-APEX)")
        sys.exit(1)

    tiers = args.tiers
    output_dir = workspace / "quantized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── header ──
    log("=" * 60)
    if args.dry_run:
        log("  *** DRY RUN MODE ***")
    log("  APEX Batch Quantization Pipeline")
    log("=" * 60)
    log(f"  Model:    {args.model}")
    log(f"  Imatrix:  {args.imatrix}")
    log(f"  Output:   {output_base}")
    log(f"  Tiers:    {tiers}")
    log(f"  Workspace: {workspace}")
    log("=" * 60)

    display_status(state, source_ok=state.source_info() is not None, imatrix_ok=state.imatrix_info() is not None, tiers=tiers)

    # ── 1. Discover & download source model + imatrix ──
    source_info = state.source_info()
    if source_info:
        source_gguf = Path(source_info["path"])
        if source_gguf.exists():
            log(f"✓ Source model already downloaded: {source_gguf.name}")
        else:
            source_info = None
            state.data.pop("source", None)
            state.save()

    imatrix_info = state.imatrix_info()
    if imatrix_info:
        imatrix_path = Path(imatrix_info["path"])
        if imatrix_path.exists():
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
                log(f"DRY RUN: would download source model {args.model} ({source_file})")
                source_gguf = Path(f"/dry-run/{source_file}")
                state.mark_source(str(source_gguf), source_file)
            if not imatrix_info:
                log(f"DRY RUN: would download imatrix {args.imatrix} ({imatrix_file})")
                imatrix_path = Path(f"/dry-run/{imatrix_file}")
                state.mark_imatrix(str(imatrix_path))
        else:
            with _allow_hard_interrupt():
                downloads = []
                if not source_info:
                    downloads.append(("source", download_source_model, (args.model, workspace, token, source_file)))
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
                        source_gguf = result
                        state.mark_source(str(result), source_file)
                    else:
                        imatrix_path = result
                        state.mark_imatrix(str(result))

        log("✓ Both downloads complete.")

    # ── 2. Initialize README.md ──
    readme_path = workspace / "README.md"
    if not state.readme_initialized() or not readme_path.exists():
        if _readme_in_repo(output_base, token):
            log(f"Found README.md in {output_base}, downloading …")
            downloaded = _download_readme(output_base, token, readme_path)
            if downloaded:
                log(f"✓ README.md downloaded from {output_base}")
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
    display_status(state, source_ok=True, imatrix_ok=True, tiers=tiers)

    needs_work = []
    for tier in tiers:
        st = state.tier_status(tier)
        if st == "uploaded":
            log(f"✓ tier{tier}: already uploaded, skipping")
            continue
        needs_work.append(tier)

    if not needs_work:
        log("\n✅ All tiers completed. Nothing to do.")
        _print_summary(state, tiers, output_base)
        if not args.dry_run:
            _upload_readme(readme_path, output_base, token)
        else:
            log(f"DRY RUN: would upload README.md → {output_base}")
        return

    log(f"\nTiers to process: {needs_work}")

    # ── 3. Quantize + Upload pipeline ──
    # upload N-1 overlaps with quantize N
    upload_executor = ThreadPoolExecutor(max_workers=1)
    pending_upload: Optional[tuple] = None  # (tier, Future)

    def _wait_pending_upload():
        nonlocal pending_upload
        if pending_upload is None:
            return
        tier_p, fut = pending_upload
        try:
            fut.result()
            gguf_p = output_dir / f"tier{tier_p}.gguf"
            sz_mb_val = round(gguf_p.stat().st_size / (1024**2), 1) if gguf_p.exists() else None
            state.set_tier(tier_p, "uploaded", size_mb=sz_mb_val)
            # append row to README
            if gguf_p.exists() and readme_path.exists():
                tier_name = f"tier{tier_p}"
                if not _readme_has_tier(readme_path, tier_name):
                    _append_readme_row(readme_path, tier_name,
                                       gguf_p.stat().st_size / (1024**3))
                state.mark_readme_tier(tier_p)
        except Exception as exc:
            err = str(exc)[:120]
            log_err(f"tier{tier_p} upload failed: {err}")
            state.set_tier(tier_p, "error", error=str(exc)[:300],
                           error_short=err[:40])
        pending_upload = None

    processed = []
    failed = []

    try:
        for tier in needs_work:
            if _interruption_requested:
                log("⚠ Pipeline interrupted by user.")
                break

            if args.dry_run:
                log(f"DRY RUN: would quantize and upload tier{tier}")
                continue

            output_gguf = output_dir / f"tier{tier}.gguf"
            st = state.tier_status(tier)

            # ── quantize ──
            if st in ("pending", "error", "quantized"):
                # If quantized but file missing, re-quantize
                if st == "quantized" and output_gguf.exists() and output_gguf.stat().st_size > 0:
                    log(f"✓ tier{tier}: quantized file exists, skipping quantize")
                else:
                    state.set_tier(tier, "quantizing")
                    display_status(state, True, True, tiers=tiers)
                    t0 = time.time()
                    try:
                        run_quantize(tier, source_gguf, imatrix_path, output_gguf)
                    except Exception as exc:
                        err = str(exc)[:120]
                        log_err(f"tier{tier} quantize failed: {err}")
                        state.set_tier(tier, "error", error=str(exc)[:300],
                                       error_short=err[:40])
                        failed.append(tier)
                        if output_gguf.exists() and output_gguf.stat().st_size == 0:
                            output_gguf.unlink(missing_ok=True)
                        continue
                    elapsed = time.time() - t0
                    m, s = divmod(int(elapsed), 60)
                    h, m = divmod(m, 60)
                    sz_mb = output_gguf.stat().st_size / (1024**2)
                    log(f"✓ tier{tier}: quantized in {h:02d}:{m:02d}:{s:02d}  "
                        f"({sz_mb:.1f} MB)")
                    state.set_tier(tier, "quantized", size_mb=round(sz_mb, 1))
            elif st == "uploading":
                # interrupted during upload — re-upload below
                log(f"tier{tier}: was uploading, will retry")
            else:
                log(f"tier{tier}: status={st}, will proceed to upload")

            display_status(state, True, True, tiers=tiers)

            # ── wait for previous upload to finish ──
            _wait_pending_upload()

            # ── skip upload if interrupted ──
            if _interruption_requested:
                break

            # ── start upload in background ──
            repo_id = output_base
            state.set_tier(tier, "uploading")
            display_status(state, True, True, tiers=tiers)

            def _do_upload(t=tier, rid=repo_id, p=output_gguf):
                upload_tier(t, p, rid, token)

            pending_upload = (tier, upload_executor.submit(_do_upload))
            processed.append(tier)

        # ── wait for the very last upload ──
        _wait_pending_upload()
        upload_executor.shutdown(wait=True)

    except KeyboardInterrupt:
        log_err("Forced interrupt — saving state and cleaning up.")
        _wait_pending_upload()
        upload_executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        log_err(f"Unexpected error: {exc}")
        _wait_pending_upload()
        upload_executor.shutdown(wait=False, cancel_futures=True)

    # ── 4. Cleanup incomplete outputs ──
    if not args.dry_run:
        _cleanup_incomplete(state, output_dir, tiers=tiers)

    # ── 5. Final report ──
    _print_summary(state, tiers, output_base)

    # ── 6. Upload README to output repo ──
    if readme_path.exists():
        if args.dry_run:
            log(f"\nDRY RUN: would upload README.md → {output_base}")
        else:
            log(f"\nUploading README.md → {output_base} …")
            _upload_readme(readme_path, output_base, token)


def _cleanup_incomplete(state: BatchState, output_dir: Path, tiers: list = None):
    """Delete quantized GGUFs for tiers that are not fully uploaded."""
    if tiers is None:
        tiers = TIERS
    count = 0
    for tier in tiers:
        st = state.tier_status(tier)
        if st in ("uploaded", "done"):
            continue
        gguf = output_dir / f"tier{tier}.gguf"
        if gguf.exists():
            sz = gguf.stat().st_size / (1024**3)
            gguf.unlink()
            log(f"🗑  Removed incomplete file: {gguf.name}  ({sz:.2f} GB)")
            count += 1
    if count:
        log(f"Cleaned up {count} incomplete file(s).")


def _print_summary(state: BatchState, tiers: list, output_base: str):
    """Print a final summary table."""
    log("\n" + "=" * 60)
    log("  Final Report")
    log("=" * 60)

    uploaded = []
    quantized_pending = []
    errors = []

    for tier in tiers:
        st = state.tier_status(tier)
        info = state._t(tier)
        sz = info.get("size_mb", "?")
        if st == "uploaded":
            uploaded.append((tier, sz))
        elif st in ("quantized", "quantizing", "uploading"):
            quantized_pending.append((tier, st))
        elif st == "error":
            errors.append((tier, info.get("error_short", "unknown")))
        # pending tiers just sit in neither list

    pending_count = len(tiers) - len(uploaded) - len(quantized_pending) - len(errors)

    log(f"\n  Repo: https://huggingface.co/{output_base}")

    if uploaded:
        log(f"\n  ✅ Uploaded ({len(uploaded)}):")
        for tier, sz in uploaded:
            log(f"     tier{tier:<3}  {sz} MB")

    if quantized_pending:
        log(f"\n  📦 Quantized but not uploaded ({len(quantized_pending)}):")
        for tier, st in quantized_pending:
            log(f"     tier{tier:<3}  (status: {st})")

    if errors:
        log(f"\n  ❌ Errors ({len(errors)}):")
        for tier, err in errors:
            log(f"     tier{tier:<3}  {err}")

    if pending_count > 0:
        log(f"\n  ○  Not started: {pending_count}")

    failed_quant = [t for t, _ in errors]
    not_uploaded = [t for t, _ in quantized_pending]
    if failed_quant or not_uploaded:
        log(f"\n  ⚠  Incomplete tiers: {sorted(failed_quant + not_uploaded)}")
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
        description="APEX Batch Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/batch_quantize.py \\\n"
            "    --model bullerwins/Qwen3.5-35B-A3B-GGUF \\\n"
            "    --imatrix bullerwins/Qwen3.5-35B-A3B-imatrix-GGUF \\\n"
            "    --output user/Qwen3.5-35B-A3B-APEX\n"
        ),
    )
    parser.add_argument("--model", "-m", required=True,
                        help="HF repo with source GGUF  (e.g. user/model-GGUF)")
    parser.add_argument("--imatrix", "-i", required=True,
                        help="HF repo with imatrix file (e.g. user/imatrix-GGUF)")
    parser.add_argument("--output", "-o", required=True,
                        help="HF repo id for all output tiers (org/name), "
                             "e.g. user/model-APEX")
    parser.add_argument("--tiers", default="1-13",
                        help="Tier spec: '1-13', '1-10,13', '3-8', '1,5,7' (default: 1-13)")
    parser.add_argument("--workspace", "-w",
                        default=str(Path.home() / "apex_batch"),
                        help="Workspace directory for state & intermediate files "
                             "(default: ~/apex_batch)")
    parser.add_argument("--token", "-t",
                        help="HF token (default: $HF_TOKEN from env / .env)")
    parser.add_argument("--source-file",
                        help="Explicit source GGUF filename (skip auto-detection)")
    parser.add_argument("--imatrix-file",
                        help="Explicit imatrix filename (skip auto-detection)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate the pipeline without downloading, quantizing, "
                             "or uploading. Still creates/updates README.md locally.")

    args = parser.parse_args()
    args.tiers = _parse_tiers(args.tiers)

    if not args.tiers or not all(1 <= t <= 13 for t in args.tiers):
        log_err("--tiers must be in range 1-13")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
