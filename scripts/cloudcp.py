#!/usr/bin/env python3
"""cloudcp — s3cmd-like file tool for local FS, S3 and HuggingFace Hub.

Usage:
  cloudcp.py cp [options] SRC DST
  cloudcp.py ls PATH
  cloudcp.py rm PATH

Paths are distinguished by scheme:
  s3://bucket/folder/file.gguf                  S3-compatible storage
  hf://account/repository/folder/file.gguf      HuggingFace model repo
  hf://datasets/account/repo/folder/file.jsonl  HuggingFace dataset repo
  hf://spaces/account/repo/...                  HuggingFace Space
  /local/path/file.gguf                         local filesystem

Optional revision for HF: hf://[datasets/]account/repo@revision/path
(default: main).
`ls` supports a glob mask (* and ?) in the last path component.
`cp` accepts a single file, a directory (copied recursively) or a glob
mask (* and ?) in the last path component as SRC.

`cp` streams data in small chunks in both directions: the full file is never
held in memory and no temporary content files are created — the destination
is written directly.  Interrupted transfers are resumed:
  * destination exists and is smaller than the source
    -> copy continues from the byte offset where it stopped
       (HTTP Range for HF/S3 downloads, ListParts for S3 multipart uploads,
       append for local files);
  * destination equal or larger -> skipped (override with --force).

Note: uploads to HuggingFace go through the LFS protocol; a partially
uploaded HF object cannot be probed, so resume to hf:// works only when the
identical file was already fully committed (the copy is skipped then).
Resume to s3:// and to local files works across process restarts.

Environment (loaded from .env, same variable names as batch_quantize.py):
  HF_TOKEN                  HuggingFace token (write access for cp/rm)
  S3_ENDPOINT               e.g. https://storage.yandexcloud.net (no bucket)
  S3_KEY_ID / S3_SECRET     static keys (AWS SigV4) or S3_TOKEN (IAM)
  S3_REGION                 SigV4 region (default: ru-central1)

Dependencies: httpx, huggingface_hub (both already used by this repo).
"""

import io
import os
import re
import shutil
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace as dc_replace

import httpx

# ---------------------------------------------------------------------------
# Constants & .env loading
# ---------------------------------------------------------------------------

CHUNK_SIZE = 8 * 1024 * 1024          # 8 MB streaming buffer (constant memory)
READ_CHUNK = 256 * 1024               # network read granularity (smooth progress)
S3_PART_SIZE = 64 * 1024 * 1024       # 64 MB per multipart part
READER_WINDOW = 32 * 1024 * 1024      # read-ahead window for seekable streams
HTTP_TIMEOUT = httpx.Timeout(600, connect=30)
RETRY_ATTEMPTS = 5

_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing.

    Looks for *path* in the current directory, then next to this script
    (repo root), so the tool works from any cwd.
    """
    candidates = [path,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, path)]
    lines = None
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as f:
                lines = f.readlines()
                break
        except OSError:
            continue
    if lines is None:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def fmt_duration(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# Progress bar (wget-like)
# ---------------------------------------------------------------------------

class Bar:
    """Single-line wget-style progress bar on stderr."""

    def __init__(self, label: str, total: int, start: int = 0):
        self.label = label[-40:]
        self.total = total
        self.current = start
        self.t0 = time.monotonic()
        self._last_draw = 0.0
        self._done = False

    def update(self, current: int):
        self.current = current
        now = time.monotonic()
        if not self._done and now - self._last_draw >= 0.2:
            self._draw()
            self._last_draw = now

    def finish(self):
        if self._done:
            return
        self._done = True
        self._draw()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _draw(self):
        width = shutil.get_terminal_size((100, 20)).columns
        if self.total > 0:
            frac = min(1.0, self.current / self.total)
            speed = self.current / max(1e-6, time.monotonic() - self.t0)
            eta = (self.total - self.current) / speed if speed > 0 else 0
            body = (f"{self.label}  {frac * 100:5.1f}% "
                    f"{human(self.current)}/{human(self.total)}  "
                    f"{human(speed)}/s  eta {fmt_duration(eta)}")
        else:
            body = f"{self.label}  {human(self.current)}"
        sys.stderr.write("\r" + body[:width].ljust(width))
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# URI parsing
# ---------------------------------------------------------------------------

@dataclass
class URI:
    kind: str            # "s3" | "hf" | "local"
    bucket: str = ""     # s3
    repo_id: str = ""    # hf ("owner/name")
    revision: str = "main"
    repo_type: str = "model"  # hf: "model" | "dataset" | "space"
    path: str = ""       # object key / path-in-repo / local path
    raw: str = ""

    def label(self) -> str:
        if self.kind == "s3":
            return f"s3://{self.bucket}/{self.path}"
        if self.kind == "hf":
            prefix = "" if self.repo_type == "model" else self.repo_type + "s/"
            return f"hf://{prefix}{self.repo_id}/{self.path}"
        return self.path


def parse_uri(raw: str) -> URI:
    if raw.startswith("s3://"):
        bucket, _, path = raw[5:].partition("/")
        # bucket may be empty when S3_ENDPOINT embeds a default bucket
        return URI("s3", bucket=bucket, path=path, raw=raw)
    if raw.startswith("hf://"):
        parts = raw[5:].split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            sys.exit(f"error: HF path must be "
                     f"hf://[datasets/]account/repository[/path] "
                     f"— got '{raw}'")
        repo_type = "model"
        if parts[0] in ("datasets", "spaces"):
            repo_type = parts[0].rstrip("s")
            parts = parts[1:]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                sys.exit(f"error: HF path must be "
                         f"hf://{repo_type}s/account/repository[/path] "
                         f"— got '{raw}'")
        repo, revision = parts[1], "main"
        if "@" in repo:
            repo, _, revision = repo.partition("@")
        return URI("hf", repo_id=f"{parts[0]}/{repo}", revision=revision,
                   repo_type=repo_type, path="/".join(parts[2:]), raw=raw)
    return URI("local", path=raw, raw=raw)


def split_glob(path: str) -> tuple[str, str]:
    """Split into (literal_dir, mask); mask is '' when no wildcards."""
    if not any(c in path for c in "*?["):
        return path, ""
    dir_part, _, mask = path.rpartition("/")
    if any(c in dir_part for c in "*?["):
        sys.exit("error: glob mask is only supported in the last path "
                 f"component — got '{path}'")
    return dir_part, mask


def glob_to_regex(mask: str) -> re.Pattern:
    """fnmatch-style mask where * and ? do NOT cross '/' separators."""
    out = []
    for ch in mask:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$")


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

class _RetryableStatus(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def check_status(resp: httpx.Response, what: str):
    if resp.status_code >= 500 or resp.status_code == 429:
        raise _RetryableStatus(resp.status_code)
    if resp.status_code >= 300:
        try:
            body = resp.text[:300]
        except Exception:
            body = ""
        raise RuntimeError(f"{what}: HTTP {resp.status_code}: {body}")


def with_retries(what: str, fn, attempts: int = RETRY_ATTEMPTS):
    """Run fn() with exponential backoff on transport errors / 5xx / 429."""
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (httpx.TransportError, _RetryableStatus) as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"{what} failed after {attempts} attempts: {exc}") from exc
            sys.stderr.write(f"\n{what}: {exc} — retrying in {delay:.0f}s "
                             f"({attempt}/{attempts - 1})\n")
            time.sleep(delay)
            delay = min(30, delay * 2)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# AWS SigV4 (static keys) — same scheme as batch_quantize.py / s3_proxy.py
# ---------------------------------------------------------------------------

def _uri_encode(s: str) -> str:
    return urllib.parse.quote(s, safe="-_.~")


def _aws_sigv4_headers(method: str, url: str, key_id: str, secret: str,
                       region: str, body: bytes | None = None,
                       extra_headers: dict | None = None) -> dict:
    """AWS Signature Version 4 headers (UNSIGNED-PAYLOAD when streamed)."""
    import hashlib
    import hmac
    from datetime import datetime, timezone

    u = urllib.parse.urlsplit(url)
    host = u.netloc
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = amz_date[:8]
    payload_hash = (hashlib.sha256(body).hexdigest()
                    if body is not None else "UNSIGNED-PAYLOAD")

    canonical_uri = urllib.parse.quote(u.path or "/", safe="/-_.~")
    if u.query:
        pairs = []
        for part in u.query.split("&"):
            k, _, v = part.partition("=")
            # query values arrive URL-encoded — decode first so the
            # canonical form is encoded exactly once
            pairs.append((_uri_encode(urllib.parse.unquote(k)),
                          _uri_encode(urllib.parse.unquote(v))))
        pairs.sort()
        canonical_query = "&".join(f"{k}={v}" for k, v in pairs)
    else:
        canonical_query = ""

    extra = {k.lower(): v for k, v in (extra_headers or {}).items()}
    all_headers = {"host": host, "x-amz-content-sha256": payload_hash,
                   "x-amz-date": amz_date, **extra}
    signed = sorted(all_headers)
    canonical_headers = "".join(f"{k}:{all_headers[k]}\n" for k in signed)
    signed_headers = ";".join(signed)

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

    out = {"Authorization": (
               f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
               f"SignedHeaders={signed_headers}, Signature={signature}"),
           "x-amz-date": amz_date,
           "x-amz-content-sha256": payload_hash}
    out.update(extra_headers or {})
    return out


def _xml_findall(text: str, tag: str):
    """Elements by local tag name regardless of namespace."""
    root = ET.fromstring(text)
    return [el for el in root.iter()
            if el.tag == tag or el.tag.endswith("}" + tag)]


def _xml_text(el, tag: str) -> str:
    for child in el:
        if child.tag == tag or child.tag.endswith("}" + tag):
            return child.text or ""
    return ""


# ---------------------------------------------------------------------------
# S3 backend (plain REST + SigV4, no boto3)
# ---------------------------------------------------------------------------

class S3Client:
    def __init__(self):
        load_dotenv()
        endpoint = os.environ.get("S3_ENDPOINT")
        key_id = (os.environ.get("S3_KEY_ID")
                  or os.environ.get("AWS_ACCESS_KEY_ID"))
        secret = (os.environ.get("S3_SECRET")
                  or os.environ.get("AWS_SECRET_ACCESS_KEY"))
        token = os.environ.get("S3_TOKEN")
        if not endpoint:
            sys.exit("error: S3_ENDPOINT is not configured (set it in .env)")
        ep = endpoint.strip()
        if "://" not in ep:
            ep = "https://" + ep
        u = urllib.parse.urlsplit(ep)
        host = (u.hostname or "").lower()
        labels = host.split(".")
        self.default_bucket = ""
        if u.path.strip("/"):
            # path form: https://host/bucket/  (as in batch_quantize.py)
            self.base = f"{u.scheme}://{u.netloc}"
            self.default_bucket = u.path.strip("/").split("/", 1)[0]
        elif host == "storage.yandexcloud.net" or len(labels) < 3:
            # plain endpoint: bucket must come from the s3:// URL
            self.base = f"{u.scheme}://{u.netloc}"
        else:
            # virtual-host form: https://bucket.storage.yandexcloud.net/
            bucket_host = ".".join(labels[1:])
            if u.port:
                bucket_host = f"{bucket_host}:{u.port}"
            self.base = f"{u.scheme}://{bucket_host}"
            self.default_bucket = labels[0]
        self.region = os.environ.get("S3_REGION", "ru-central1")
        if key_id and secret:
            self.auth = ("sigv4", key_id, secret)
        elif token:
            self.auth = ("bearer", token)
        else:
            sys.exit("error: no S3 credentials — set S3_KEY_ID + S3_SECRET "
                     "or S3_TOKEN (in .env)")
        self.client = httpx.Client(timeout=HTTP_TIMEOUT)

    def _url(self, bucket: str, key: str = "", query: str = "") -> str:
        url = f"{self.base}/{bucket}"
        if key:
            # keep '/' separators in the key; encode everything else
            url += "/" + urllib.parse.quote(key, safe="/-_.~")
        if query:
            url += "?" + query
        return url

    def _headers(self, method: str, url: str, body: bytes | None = None,
                 extra: dict | None = None) -> dict:
        if self.auth[0] == "sigv4":
            return _aws_sigv4_headers(method, url, self.auth[1], self.auth[2],
                                      self.region, body, extra)
        out = {"Authorization": f"Bearer {self.auth[1]}"}
        out.update(extra or {})
        return out

    def _get(self, url: str) -> httpx.Response:
        def go():
            resp = self.client.get(url, headers=self._headers("GET", url))
            check_status(resp, f"GET {url}")
            return resp
        return with_retries("GET", go)

    # -- object operations ---------------------------------------------------

    def stat(self, bucket: str, key: str) -> int | None:
        """Object size or None if missing."""
        url = self._url(bucket, key)

        def go():
            resp = self.client.head(url, headers=self._headers("HEAD", url))
            if resp.status_code == 404:
                return None
            check_status(resp, f"HEAD {url}")
            return resp

        resp = with_retries("HEAD", go)
        return None if resp is None else int(
            resp.headers.get("content-length", 0))

    def delete(self, bucket: str, key: str) -> None:
        url = self._url(bucket, key)

        def go():
            resp = self.client.delete(url, headers=self._headers("DELETE", url))
            if resp.status_code == 404:
                return None
            check_status(resp, f"S3 delete {key}")
            return resp

        if with_retries("DELETE", go) is None:
            raise RuntimeError(f"not found: s3://{bucket}/{key}")

    def get_range_iter(self, bucket: str, key: str, offset: int, length: int):
        """Yield chunks of bytes [offset, offset+length) of an object."""
        url = self._url(bucket, key)
        headers = self._headers("GET", url, extra={
            "Range": f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity"})
        with self.client.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 500 or resp.status_code == 429:
                raise _RetryableStatus(resp.status_code)
            if resp.status_code >= 300:
                raise RuntimeError(f"S3 range GET: HTTP {resp.status_code}")
            if offset > 0 and resp.status_code != 206:
                raise RuntimeError(
                    "S3 range GET: server ignored Range (HTTP 200) — "
                    "resume is not possible")
            remaining = length
            for chunk in resp.iter_bytes(READ_CHUNK):
                remaining -= len(chunk)
                yield chunk
            if remaining > 0:
                raise RuntimeError(
                    f"S3 range GET: got {length - remaining}/{length} bytes")

    def put_stream(self, bucket: str, key: str, gen, size: int) -> None:
        url = self._url(bucket, key)
        headers = self._headers("PUT", url)
        headers["Content-Length"] = str(size)

        def go():
            return self.client.put(url, content=gen(), headers=headers)

        check_status(with_retries("PUT", go), f"S3 PUT {key}")

    # -- listing ---------------------------------------------------------------

    def list(self, bucket: str, prefix: str):
        """Yield (kind, name, size); kind is 'file' or 'dir'."""
        seen = set()
        token = ""
        while True:
            q = ["list-type=2", f"prefix={_uri_encode(prefix)}",
                 "delimiter=%2F"]
            if token:
                q.append(f"continuation-token={_uri_encode(token)}")
            resp = self._get(self._url(bucket, query="&".join(q)))
            for el in _xml_findall(resp.text, "Contents"):
                key = _xml_text(el, "Key")
                if key != prefix and key not in seen:
                    seen.add(key)
                    yield "file", key, int(_xml_text(el, "Size") or 0)
            for el in _xml_findall(resp.text, "CommonPrefixes"):
                pfx = _xml_text(el, "Prefix")
                if pfx not in seen:
                    seen.add(pfx)
                    yield "dir", pfx, 0
            tokens = _xml_findall(resp.text, "NextContinuationToken")
            token = tokens[0].text if tokens and tokens[0].text else ""
            if not token:
                return

    # -- multipart uploads -------------------------------------------------------

    def find_inprogress(self, bucket: str,
                        key: str) -> tuple[str, list] | None:
        """(upload_id, parts) for an in-progress upload of *key*.

        parts: list of (part_number, etag, size) sorted by number.
        """
        resp = self._get(self._url(bucket, query="uploads"))
        for el in _xml_findall(resp.text, "Upload"):
            if _xml_text(el, "Key") == key:
                upload_id = _xml_text(el, "UploadId")
                return upload_id, self.list_parts(bucket, key, upload_id)
        return None

    def list_parts(self, bucket: str, key: str,
                   upload_id: str) -> list[tuple[int, str, int]]:
        resp = self._get(self._url(
            bucket, key, f"uploadId={_uri_encode(upload_id)}"))
        parts = [(int(_xml_text(el, "PartNumber")), _xml_text(el, "ETag"),
                  int(_xml_text(el, "Size") or 0))
                 for el in _xml_findall(resp.text, "Part")]
        parts.sort()
        return parts

    def multipart_initiate(self, bucket: str, key: str) -> str:
        url = self._url(bucket, key, "uploads=")

        def go():
            return self.client.post(url, headers=self._headers("POST", url))

        resp = with_retries("multipart initiate", go)
        m = re.search(r"<UploadId>([^<]+)</UploadId>", resp.text)
        if not m:
            raise RuntimeError(
                f"S3 multipart initiate: no UploadId: {resp.text[:300]}")
        return m.group(1)

    def multipart_abort(self, bucket: str, key: str, upload_id: str) -> None:
        with_retries("abort", lambda: self.client.delete(
            self._url(bucket, key, f"uploadId={_uri_encode(upload_id)}"),
            headers=self._headers("DELETE", self._url(
                bucket, key, f"uploadId={_uri_encode(upload_id)}"))))

    def multipart_complete(self, bucket: str, key: str, upload_id: str,
                           parts: list[tuple[int, str]]) -> None:
        parts_xml = "".join(
            f"<Part><PartNumber>{n}</PartNumber>"
            f"<ETag>{etag}</ETag></Part>" for n, etag in parts)
        body = (f"<CompleteMultipartUpload>{parts_xml}"
                f"</CompleteMultipartUpload>").encode()
        url = self._url(bucket, key, f"uploadId={_uri_encode(upload_id)}")
        headers = self._headers("POST", url, body=body)
        headers["Content-Type"] = "application/xml"

        def go():
            return self.client.post(url, content=body, headers=headers)

        resp = with_retries("complete", go)
        if resp.status_code >= 300 or "<Error>" in resp.text:
            raise RuntimeError(
                f"S3 multipart complete failed: {resp.text[:300]}")

    def upload_part(self, bucket: str, key: str, upload_id: str, number: int,
                    body_factory, length: int) -> str:
        url = self._url(bucket, key,
                        f"partNumber={number}&uploadId={_uri_encode(upload_id)}")
        headers = self._headers("PUT", url)
        headers["Content-Length"] = str(length)

        def go():
            # fresh body stream per attempt (retries restart the part)
            return self.client.put(url, content=body_factory(),
                                   headers=headers)

        resp = with_retries("part upload", go)
        check_status(resp, f"S3 part {number}")
        etag = resp.headers.get("ETag", "")
        if not etag:
            raise RuntimeError(f"S3 part {number}: missing ETag")
        return etag

    def copy_part(self, bucket: str, key: str, upload_id: str, number: int,
                  src_bucket: str, src_key: str, offset: int, length: int) -> str:
        """Server-side UploadPartCopy (s3:// -> s3:// transfers)."""
        url = self._url(bucket, key,
                        f"partNumber={number}&uploadId={_uri_encode(upload_id)}")
        extra = {"x-amz-copy-source": f"/{src_bucket}/{_uri_encode(src_key)}",
                 "x-amz-copy-source-range":
                     f"bytes={offset}-{offset + length - 1}"}
        headers = self._headers("PUT", url, extra=extra)

        def go():
            return self.client.put(url, headers=headers)

        resp = with_retries("copy part", go)
        check_status(resp, f"S3 copy part {number}")
        etags = _xml_findall(resp.text, "ETag")
        etag = etags[0].text if etags and etags[0].text else \
            resp.headers.get("ETag", "")
        if not etag:
            raise RuntimeError(f"S3 copy part {number}: missing ETag")
        return etag


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class HFClient:
    def __init__(self):
        load_dotenv()
        self.endpoint = _HF_ENDPOINT
        self.token = os.environ.get("HF_TOKEN")
        if self.token is None:
            try:
                from huggingface_hub import get_token
                self.token = get_token()
            except Exception:
                self.token = None
        self.client = httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True)
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token, endpoint=self.endpoint)
        return self._api

    def _headers(self, extra: dict | None = None) -> dict:
        out = {"Accept-Encoding": "identity"}
        if self.token:
            out["Authorization"] = f"Bearer {self.token}"
        out.update(extra or {})
        return out

    def resolve_url(self, uri: URI) -> str:
        rev = urllib.parse.quote(uri.revision, safe="")
        prefix = "" if uri.repo_type == "model" else uri.repo_type + "s/"
        if not uri.path:
            return f"{self.endpoint}/{prefix}{uri.repo_id}/resolve/{rev}"
        return (f"{self.endpoint}/{prefix}{uri.repo_id}/resolve/{rev}/"
                f"{urllib.parse.quote(uri.path, safe='/')}")

    def stat(self, uri: URI) -> int | None:
        """File size in the repo, or None if missing."""
        url = self.resolve_url(uri)

        def go():
            resp = self.client.head(url, headers=self._headers())
            if resp.status_code == 404:
                return None
            check_status(resp, f"HEAD {url}")
            return resp

        resp = with_retries("HEAD", go)
        if resp is None:
            return None
        size = resp.headers.get("content-length")
        if not size:
            def go2():
                r = self.client.get(url, headers=self._headers(
                    {"Range": "bytes=0-0"}))
                if r.status_code == 404:
                    return None
                check_status(r, f"GET {url}")
                return r
            r = with_retries("GET", go2)
            if r is None:
                return None
            cr = r.headers.get("content-range", "")
            size = cr.rsplit("/", 1)[-1] if "/" in cr else "0"
        return int(size)

    def get_range_iter(self, uri: URI, offset: int, length: int):
        url = self.resolve_url(uri)
        headers = self._headers({
            "Range": f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity"})
        with self.client.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 500 or resp.status_code == 429:
                raise _RetryableStatus(resp.status_code)
            if resp.status_code >= 300:
                raise RuntimeError(f"HF range GET: HTTP {resp.status_code}")
            if offset > 0 and resp.status_code != 206:
                raise RuntimeError(
                    "HF range GET: server ignored Range (HTTP 200) — "
                    "resume is not possible")
            remaining = length
            for chunk in resp.iter_bytes(READ_CHUNK):
                remaining -= len(chunk)
                yield chunk
            if remaining > 0:
                raise RuntimeError(
                    f"HF range GET: got {length - remaining}/{length} bytes")

    def list(self, uri: URI):
        """Yield (kind, full_path, size) for one repo directory."""
        rev = urllib.parse.quote(uri.revision, safe="")
        api_url = (f"{self.endpoint}/api/{uri.repo_type}s/"
                   f"{uri.repo_id}/tree/{rev}")
        if uri.path:
            api_url += "/" + urllib.parse.quote(uri.path, safe="/")
        token = ""
        while True:
            url = (api_url + ("?cursor=" + urllib.parse.quote(token)
                              if token else ""))
            resp = with_retries("tree", lambda: self.client.get(
                url, headers=self._headers()))
            if resp.status_code == 404:
                raise RuntimeError(
                    f"hf://{uri.repo_id}/{uri.path}: repository, revision "
                    f"'{uri.revision}' or directory does not exist — or the "
                    f"current HF token has no access to it")
            if resp.status_code == 401:
                raise RuntimeError(
                    "HuggingFace: unauthorized — set a valid HF_TOKEN in "
                    ".env (private repos require a token with read access)")
            check_status(resp, "HF tree listing")
            for item in resp.json():
                size = (item.get("lfs") or {}).get("size") or \
                    item.get("size", 0)
                kind = "dir" if item["type"] == "directory" else "file"
                yield kind, item["path"], int(size)
            link = resp.headers.get("link", "")
            m = re.search(r'[?&]cursor=([^&>]+)', link)
            token = urllib.parse.unquote(m.group(1)) if m else ""
            if not token:
                return

    def delete(self, uri: URI) -> None:
        """Delete a file (creates a commit; git history keeps the blob)."""
        try:
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            from huggingface_hub.utils import RepositoryNotFoundError
        try:
            self.api.delete_file(path_in_repo=uri.path, repo_id=uri.repo_id,
                                 repo_type=uri.repo_type,
                                 revision=uri.revision)
        except RepositoryNotFoundError:
            raise RuntimeError(
                f"hf://{uri.repo_id}: repository does not exist — or the "
                f"current HF token has no access to it") from None

    def upload(self, src: URI, dest: URI, force: bool) -> None:
        """Upload to HF via huggingface_hub (LFS protocol for big files).

        The source is exposed to huggingface_hub as a seekable stream
        (RangeReader) — no temporary file is written, memory stays bounded.
        """
        from huggingface_hub import CommitOperationAdd
        try:
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:  # older hub
            from huggingface_hub.utils import RepositoryNotFoundError

        src_size = src_size_of(src)
        dest_path = dest.path
        if not dest_path or dest.raw.endswith("/"):
            dest_path = (dest_path.rstrip("/") + "/" if dest_path else "") + \
                os.path.basename(src.path.rstrip("/"))
        if not force:
            existing = self.stat(URI("hf", repo_id=dest.repo_id,
                                     revision=dest.revision,
                                     repo_type=dest.repo_type,
                                     path=dest_path))
            if existing is not None:
                if existing == src_size:
                    print(f"skipped (identical size, already on HF): "
                          f"{dest.repo_id}/{dest_path}")
                    return
                print(f"note: HF destination exists with different size "
                      f"({human(existing)} != {human(src_size)}) — "
                      f"overwriting")

        bar = None
        if src.kind == "local":
            # hub reads the local file directly (no temp copy); it shows
            # its own tqdm during the upload
            op = CommitOperationAdd(path_in_repo=dest_path,
                                    path_or_fileobj=src.path)
        else:
            reader = make_seekable_reader(src, src_size)
            op = CommitOperationAdd(path_in_repo=dest_path,
                                    path_or_fileobj=reader)
            # CommitOperationAdd.__post_init__ hashed the whole source
            reader.phase = "upload"
            reader.reset_progress()
            bar = Bar(f"hf:{dest_path}", src_size)
            reader.bar = bar
        try:
            self.api.create_commit(
                repo_id=dest.repo_id, repo_type=dest.repo_type,
                operations=[op],
                commit_message=f"Upload {dest_path} via cloudcp")
        except RepositoryNotFoundError:
            print(f"repo {dest.repo_id} does not exist — creating it")
            self.api.create_repo(repo_id=dest.repo_id, repo_type=dest.repo_type,
                                 exist_ok=True)
            self.api.create_commit(
                repo_id=dest.repo_id, repo_type=dest.repo_type,
                operations=[op],
                commit_message=f"Upload {dest_path} via cloudcp")
        finally:
            if bar is not None:
                bar.finish()


# ---------------------------------------------------------------------------
# Seekable remote reader (for uploads to HF through huggingface_hub)
# ---------------------------------------------------------------------------

class RangeReader(io.BufferedIOBase):
    """Seekable read-only stream over an HTTP-range source.

    huggingface_hub reads it twice (sha256 pass, then LFS part uploads);
    every read is served from a bounded read-ahead window fetched with
    ranged GETs.  Memory: READER_WINDOW + CHUNK_SIZE.
    """

    def __init__(self, src: URI, size: int):
        self.src = src
        self.size = size
        self.pos = 0
        self.phase = "hash"
        self.bar: Bar | None = None
        self._buf = b""
        self._buf_start = 0
        self._high = 0

    def reset_progress(self):
        self._high = 0
        self._buf = b""

    def _progress(self, served: int):
        self._high = max(self._high, self.pos + served)
        if self.bar is not None:
            self.bar.update(self._high)

    def _fetch(self, offset: int, length: int) -> bytes:
        chunks = []
        got = 0
        if self.src.kind == "s3":
            s3 = get_s3()
            it = s3.get_range_iter(self.src.bucket, self.src.path,
                                   offset, length)
        else:
            hf = get_hf()
            it = hf.get_range_iter(self.src, offset, length)
        for chunk in it:
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            self.pos = offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            buf_end = self._buf_start + len(self._buf)
            if self._buf_start <= self.pos < buf_end:
                take = min(n, buf_end - self.pos)
                out += self._buf[self.pos - self._buf_start:
                                 self.pos - self._buf_start + take]
                self.pos += take
                n -= take
                continue
            # refill window starting at current position
            window = min(max(READER_WINDOW, n), self.size - self.pos)
            self._buf = self._fetch(self.pos, window)
            self._buf_start = self.pos
            if not self._buf:
                break
        served = len(out)
        self._progress(served)
        return bytes(out)

    def read1(self, n: int = -1):
        return self.read(n)

    def close(self):
        pass


def make_seekable_reader(src: URI, size: int) -> RangeReader:
    return RangeReader(src, size)


def resolve_s3(uri: URI) -> URI:
    """Fill in the default bucket from S3_ENDPOINT when the URL omits it."""
    if uri.kind == "s3" and not uri.bucket:
        s3 = get_s3()
        if not s3.default_bucket:
            sys.exit("error: s3:// path must include a bucket "
                     "(or set S3_ENDPOINT with an embedded bucket)")
        uri.bucket = s3.default_bucket
    return uri


def src_size_of(src: URI) -> int:
    if src.kind == "local":
        if not os.path.isfile(src.path):
            sys.exit(f"error: no such file: {src.path}")
        return os.path.getsize(src.path)
    if src.kind == "s3":
        size = get_s3().stat(src.bucket, src.path)
    else:
        size = get_hf().stat(src)
    if size is None:
        sys.exit(f"error: no such file: {src.label()}")
    return size


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_S3: S3Client | None = None
_HF: HFClient | None = None


def get_s3() -> S3Client:
    global _S3
    if _S3 is None:
        _S3 = S3Client()
    return _S3


def get_hf() -> HFClient:
    global _HF
    if _HF is None:
        _HF = HFClient()
    return _HF


# ---------------------------------------------------------------------------
# Copy engine
# ---------------------------------------------------------------------------

def _range_body(src: URI, offset: int, length: int, bar: Bar | None):
    """Factory returning a generator over src[offset:offset+length].

    A factory (not a generator) is required so each HTTP retry gets a
    fresh body stream.
    """
    if src.kind == "local":
        def gen():
            sent = 0
            with open(src.path, "rb") as f:
                f.seek(offset)
                while sent < length:
                    chunk = f.read(min(CHUNK_SIZE, length - sent))
                    if not chunk:
                        raise RuntimeError(
                            f"local source shrank: {src.path}")
                    sent += len(chunk)
                    if bar:
                        bar.update(offset + sent)
                    yield chunk
        return gen
    if src.kind == "s3":
        s3 = get_s3()
        return lambda: s3.get_range_iter(src.bucket, src.path, offset, length)
    hf = get_hf()
    return lambda: hf.get_range_iter(src, offset, length)


def copy_to_s3(src: URI, dest: URI, force: bool) -> None:
    """Copy any source into s3:// via resumable multipart upload."""
    s3 = get_s3()
    if not dest.path:
        sys.exit("error: S3 destination must include a key "
                 "(s3://bucket/path/file.gguf)")
    src_size = src_size_of(src)
    dest_size = s3.stat(dest.bucket, dest.path)
    if dest_size == src_size and not force:
        print(f"skipped (identical size): {dest.label()}")
        return

    part_size = S3_PART_SIZE
    upload_id: str | None = None
    parts_kept: list[tuple[int, str]] = []
    offset = 0
    resumed = False

    inprog = s3.find_inprogress(dest.bucket, dest.path)
    if inprog:
        upload_id, parts = inprog
        if parts:
            part_size = max(p[2] for p in parts)
        for num, etag, size in parts:
            expected = len(parts_kept) + 1
            if num != expected:
                break
            if size == part_size or (size < part_size
                                     and offset + size == src_size):
                parts_kept.append((num, etag))
                offset += size
                if size < part_size:
                    break
            else:
                break  # partial trailing part -> re-upload this number
        if offset > src_size:
            # source shrank since the last attempt — restart from scratch
            s3.multipart_abort(dest.bucket, dest.path, upload_id)
            upload_id, parts_kept, offset, part_size = None, [], 0, S3_PART_SIZE
        elif parts_kept:
            resumed = True

    if src_size == 0:
        if upload_id is not None:
            s3.multipart_abort(dest.bucket, dest.path, upload_id)
        s3.put_stream(dest.bucket, dest.path, lambda: iter(()), 0)
        print(f"uploaded (empty file): {dest.label()}")
        return

    if offset == src_size and upload_id is not None:
        pass  # everything already uploaded — go straight to complete
    elif upload_id is None:
        if src_size <= part_size:
            bar = Bar(dest.label(), src_size)
            body = _range_body(src, 0, src_size, bar)
            s3.put_stream(dest.bucket, dest.path, body, src_size)
            bar.update(src_size)
            bar.finish()
            print(f"copied: {src.label()} -> {dest.label()} "
                  f"({human(src_size)})")
            return
        upload_id = s3.multipart_initiate(dest.bucket, dest.path)

    if resumed:
        print(f"resuming multipart upload at offset {offset} "
              f"({human(offset)}/{human(src_size)}, part size "
              f"{human(part_size)})")

    n_parts = (src_size + part_size - 1) // part_size
    next_num = len(parts_kept) + 1
    bar = Bar(dest.label(), src_size, start=offset)
    while offset < src_size:
        length = min(part_size, src_size - offset)
        if src.kind == "s3":
            etag = s3.copy_part(dest.bucket, dest.path, upload_id, next_num,
                                src.bucket, src.path, offset, length)
            bar.update(offset + length)
        else:
            body = _range_body(src, offset, length, bar)
            etag = s3.upload_part(dest.bucket, dest.path, upload_id,
                                  next_num, body, length)
        parts_kept.append((next_num, etag))
        offset += length
        next_num += 1
    bar.finish()
    s3.multipart_complete(dest.bucket, dest.path, upload_id, parts_kept)

    final = s3.stat(dest.bucket, dest.path)
    if final != src_size:
        raise RuntimeError(
            f"size mismatch after upload: expected {src_size}, got {final}")
    print(f"copied: {src.label()} -> {dest.label()} ({human(src_size)})")


def copy_to_local(src: URI, dest: URI, force: bool) -> None:
    """Stream a remote file to a local path with append-based resume."""
    src_size = src_size_of(src)
    dest_path = dest.path
    if dest_path.endswith("/") or os.path.isdir(dest_path):
        dest_path = os.path.join(
            dest_path, os.path.basename(src.path.rstrip("/")))
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    offset = 0
    if os.path.isfile(dest_path):
        local_size = os.path.getsize(dest_path)
        if local_size == src_size and not force:
            print(f"skipped (identical size): {dest_path}")
            return
        if 0 < local_size < src_size:
            offset = local_size
            print(f"resuming at offset {offset} "
                  f"({human(offset)}/{human(src_size)})")

    bar = Bar(dest_path, src_size, start=offset)
    mode = "r+b" if offset else "wb"
    with open(dest_path, mode) as f:
        if offset:
            f.seek(offset)
        it = (_range_body(src, offset, src_size - offset, None)()
              if src.kind != "local" else None)
        if src.kind == "local":
            with open(src.path, "rb") as g:
                g.seek(offset)
                while offset < src_size:
                    chunk = g.read(min(CHUNK_SIZE, src_size - offset))
                    if not chunk:
                        break
                    f.write(chunk)
                    offset += len(chunk)
                    bar.update(offset)
        else:
            for chunk in it:
                f.write(chunk)
                offset += len(chunk)
                bar.update(offset)
    bar.finish()
    final = os.path.getsize(dest_path)
    if final != src_size:
        raise RuntimeError(
            f"size mismatch: expected {src_size}, got {final} ({dest_path})")
    print(f"copied: {src.label()} -> {dest_path} ({human(src_size)})")


def copy_hf_to_hf(src: URI, dest: URI, force: bool) -> None:
    """Server-side copy between HF repos (CommitOperationCopy)."""
    try:
        from huggingface_hub import CommitOperationCopy
        try:
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        sys.exit("error: huggingface_hub is required for hf:// operations")

    dest_path = dest.path
    if not dest_path or dest.raw.endswith("/"):
        base = os.path.basename(src.path.rstrip("/"))
        dest_path = (dest_path.rstrip("/") + "/" if dest_path else "") + base
    hf = get_hf()
    op = CommitOperationCopy(path_in_repo=dest_path,
                             src_path_in_repo=src.path,
                             src_repo_id=src.repo_id,
                             source_repo_type=src.repo_type)
    try:
        hf.api.create_commit(repo_id=dest.repo_id, repo_type=dest.repo_type,
                             operations=[op],
                             commit_message=f"Copy {src.path} via cloudcp")
    except RepositoryNotFoundError:
        sys.exit(f"error: destination repo {dest.repo_id} does not exist")
    print(f"copied (server-side): hf://{src.repo_id}/{src.path} -> "
          f"hf://{dest.repo_id}/{dest_path}")


def copy_local_to_local(src: URI, dest: URI, force: bool) -> None:
    if not os.path.isfile(src.path):
        sys.exit(f"error: no such file: {src.path}")
    src_size = os.path.getsize(src.path)
    dest_path = dest.path
    if dest_path.endswith("/") or os.path.isdir(dest_path):
        dest_path = os.path.join(dest_path, os.path.basename(src.path))
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    offset = 0
    if os.path.isfile(dest_path):
        dst_size = os.path.getsize(dest_path)
        if dst_size == src_size and not force:
            print(f"skipped (identical size): {dest_path}")
            return
        if 0 < dst_size < src_size:
            offset = dst_size
            print(f"resuming at offset {offset} "
                  f"({human(offset)}/{human(src_size)})")
    bar = Bar(dest_path, src_size, start=offset)
    with open(src.path, "rb") as g, \
            open(dest_path, "r+b" if offset else "wb") as f:
        g.seek(offset)
        if offset:
            f.seek(offset)
        sent = offset
        while sent < src_size:
            chunk = g.read(min(CHUNK_SIZE, src_size - sent))
            if not chunk:
                break
            f.write(chunk)
            sent += len(chunk)
            bar.update(sent)
    bar.finish()
    print(f"copied: {src.path} -> {dest_path} ({human(src_size)})")


# ---------------------------------------------------------------------------
# ls / rm
# ---------------------------------------------------------------------------

def _print_entry(kind: str, size: int, display: str):
    size_str = "<DIR>" if kind == "dir" else human(size)
    print(f"{size_str:>10}  {display}")


def cmd_ls(args) -> None:
    uri = parse_uri(args.path)
    resolve_s3(uri)
    if uri.kind != "local":
        uri.path = uri.path.rstrip("/")
    if uri.kind == "local":
        import glob as _glob
        matches = sorted(_glob.glob(uri.path)) if any(
            c in uri.path for c in "*?[") else [uri.path]
        if not matches:
            sys.exit(f"error: no such path: {uri.path}")
        for m in matches:
            if os.path.isdir(m):
                for entry in sorted(os.scandir(m),
                                    key=lambda e: e.name):
                    if entry.is_dir():
                        _print_entry("dir", 0,
                                     os.path.join(m, entry.name) + "/")
                    else:
                        _print_entry("file", entry.stat().st_size,
                                     os.path.join(m, entry.name))
            else:
                _print_entry("file", os.path.getsize(m), m)
        return

    dir_part, mask = split_glob(uri.path)
    regex = glob_to_regex(mask) if mask else None

    if uri.kind == "s3":
        s3 = get_s3()
        prefix = dir_part + "/" if dir_part else ""
        found = False
        for kind, name, size in s3.list(uri.bucket, prefix):
            base = name.rstrip("/").rsplit("/", 1)[-1]
            if kind == "dir":
                base += "/"
            if regex and not regex.match(base):
                continue
            found = True
            _print_entry(kind, size, f"s3://{uri.bucket}/{name}")
        if not found:
            sys.exit(f"error: nothing found under: {uri.path}")
        return

    hf = get_hf()
    dir_uri = URI("hf", repo_id=uri.repo_id, revision=uri.revision,
                  repo_type=uri.repo_type, path=dir_part)
    dir_prefix = (dir_part + "/") if dir_part else ""
    found = False
    for kind, path, size in hf.list(dir_uri):
        base = path[len(dir_prefix):]
        if "/" in base:
            continue  # deeper entries should not appear, but just in case
        if kind == "dir":
            base += "/"
        if regex and not regex.match(base):
            continue
        found = True
        display = (f"hf://{uri.repo_id}/{path}" + ("/" if kind == "dir" else ""))
        _print_entry(kind, size, display)
    if not found:
        sys.exit(f"error: nothing found under: {uri.path}")


def cmd_rm(args) -> None:
    uri = parse_uri(args.path)
    resolve_s3(uri)
    if uri.kind == "local":
        if not os.path.isfile(uri.path):
            sys.exit(f"error: no such file: {uri.path}")
        os.remove(uri.path)
        print(f"deleted: {uri.path}")
    elif uri.kind == "s3":
        if not uri.path:
            sys.exit("error: s3:// rm requires a full object key")
        get_s3().delete(uri.bucket, uri.path)
        print(f"deleted: {uri.label()}")
    else:
        if not uri.path:
            sys.exit("error: hf:// rm requires a file path inside the repo")
        get_hf().delete(uri)
        print(f"deleted (commit created): {uri.label()}")


# ---------------------------------------------------------------------------
# cp dispatcher & CLI
# ---------------------------------------------------------------------------

def expand_cp_sources(src: URI) -> tuple[list[tuple[URI, str]], bool]:
    """Expand a cp source (file, directory or glob mask) into files.

    Returns (pairs, plain) where pairs is a list of (file_uri, rel_name)
    and plain is True when the source was a single literal file (rel_name
    is then just its basename).
    """
    if src.kind == "local":
        import glob as _glob
        if any(c in src.path for c in "*?["):
            matches = sorted(p for p in _glob.glob(src.path)
                             if os.path.isfile(p))
            if not matches:
                sys.exit(f"error: no such file: {src.path}")
            return ([(URI("local", path=m, raw=m), os.path.basename(m))
                     for m in matches], False)
        if os.path.isfile(src.path):
            return [(src, os.path.basename(src.path))], True
        if os.path.isdir(src.path):
            out = []
            for root, _dirs, files in os.walk(src.path):
                for name in sorted(files):
                    p = os.path.join(root, name)
                    out.append((URI("local", path=p, raw=p),
                                os.path.relpath(p, src.path)))
            if not out:
                sys.exit(f"error: directory is empty: {src.path}")
            return out, False
        sys.exit(f"error: no such file: {src.path}")

    dir_part, mask = split_glob(src.path)
    dir_part = dir_part.rstrip("/")
    prefix = (dir_part + "/") if dir_part else ""

    def make(path: str) -> URI:
        return dc_replace(src, path=path)

    def finish(out: list[tuple[URI, str]], what: str):
        if not out:
            sys.exit(f"error: {what}: {src.label()}")
        return out, False

    if src.kind == "hf":
        hf = get_hf()
        if mask:
            regex = glob_to_regex(mask)
            out = []
            for kind, path, _size in hf.list(dc_replace(src, path=dir_part)):
                base = path[len(prefix):]
                if kind == "file" and "/" not in base and regex.match(base):
                    out.append((make(path), base))
            return finish(out, "nothing found under")
        if src.path and hf.stat(src) is not None:
            return [(src, os.path.basename(src.path))], True
        out = []

        def walk_hf(u: URI) -> None:
            for kind, path, _size in hf.list(u):
                if kind == "dir":
                    walk_hf(dc_replace(src, path=path))
                else:
                    out.append((make(path), path[len(prefix):]))

        walk_hf(dc_replace(src, path=dir_part))
        return finish(out, "no such file or directory")

    s3 = get_s3()
    if mask:
        regex = glob_to_regex(mask)
        out = []
        for kind, key, _size in s3.list(src.bucket, prefix):
            base = key[len(prefix):]
            if kind == "file" and "/" not in base and regex.match(base):
                out.append((make(key), base))
        return finish(out, "nothing found under")
    if src.path and not src.path.endswith("/") \
            and s3.stat(src.bucket, src.path) is not None:
        return [(src, os.path.basename(src.path))], True
    out = []

    def walk_s3(pfx: str) -> None:
        for kind, key, _size in s3.list(src.bucket, pfx):
            if kind == "dir":
                walk_s3(key)
            else:
                out.append((make(key), key[len(prefix):]))

    walk_s3(prefix)
    return finish(out, "no such file or directory")


def cmd_cp(args) -> None:
    global S3_PART_SIZE
    if args.part_size_mb:
        S3_PART_SIZE = args.part_size_mb * 1024 * 1024
    src = parse_uri(args.src)
    dest = parse_uri(args.dst)
    resolve_s3(src)
    resolve_s3(dest)
    force = args.force

    pairs, plain = expand_cp_sources(src)
    multi = len(pairs) > 1
    if multi and dest.kind == "local" and not dest.path.endswith("/") \
            and not os.path.isdir(dest.path):
        sys.exit("error: multiple sources need a directory destination — "
                 f"add a trailing '/': {args.dst}")
    if multi and dest.kind != "local" and not dest.raw.endswith("/"):
        sys.exit("error: multiple sources need a directory destination — "
                 f"add a trailing '/': {args.dst}")
    dest_is_dir = (dest.kind == "local"
                   and (dest.path.endswith("/") or os.path.isdir(dest.path))) \
        or (dest.kind != "local" and dest.raw.endswith("/"))
    if multi and dest.kind == "local":
        os.makedirs(dest.path, exist_ok=True)
        dest_is_dir = True
    if multi:
        print(f"{len(pairs)} file(s) to copy")

    for i, (s, rel) in enumerate(pairs, 1):
        if plain or not dest_is_dir:
            d = dest
        else:
            rel_posix = rel.replace(os.sep, "/")
            if dest.kind == "local":
                dpath = os.path.join(dest.path, rel_posix)
            else:
                dpath = (dest.path.rstrip("/") + "/" + rel_posix) \
                    if dest.path else rel_posix
            d = dc_replace(dest, path=dpath)
        if multi:
            print(f"[{i}/{len(pairs)}] {s.label()}")

        if s.kind == "hf" and d.kind == "hf" and s.repo_id == d.repo_id \
                and s.path == d.path:
            sys.exit("error: source and destination are the same file")
        if s.kind == "local" and d.kind == "local":
            copy_local_to_local(s, d, force)
        elif d.kind == "s3":
            copy_to_s3(s, d, force)
        elif d.kind == "local":
            copy_to_local(s, d, force)
        elif d.kind == "hf":
            if s.kind == "hf":
                copy_hf_to_hf(s, d, force)
            else:
                get_hf().upload(s, d, force)
        else:
            sys.exit(f"error: unsupported combination: "
                     f"{s.kind} -> {d.kind}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cloudcp.py",
        description="cp/ls/rm across local FS, S3 and HuggingFace Hub "
                    "(streaming, resumable)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cp = sub.add_parser("cp", help="copy a file/directory/glob mask "
                                     "(any pair of local/s3/hf)")
    p_cp.add_argument("src")
    p_cp.add_argument("dst")
    p_cp.add_argument("--force", action="store_true",
                      help="re-copy even if destination size matches")
    p_cp.add_argument("--part-size-mb", type=int, default=None,
                      help="S3 multipart part size in MB (default: 64)")

    p_ls = sub.add_parser("ls", help="list a path (mask with * and ? allowed)")
    p_ls.add_argument("path")

    p_rm = sub.add_parser("rm", help="delete a file")
    p_rm.add_argument("path")

    args = parser.parse_args()
    try:
        {"cp": cmd_cp, "ls": cmd_ls, "rm": cmd_rm}[args.cmd](args)
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
    except Exception as exc:  # noqa: BLE001 — CLI prints errors, no tracebacks
        msg = str(exc).strip().replace("\n", " — ")
        if exc.__class__.__name__ not in ("RuntimeError", "OSError"):
            name = exc.__class__.__name__
            msg = f"{name}: {msg}" if msg else name
        if msg.startswith("HTTP ") and ":" in msg:
            code, _, body = msg.partition(": ")
            try:
                import json as _json
                body = _json.loads(body).get("error", body)
            except Exception:
                body = body.split("\n")[0][:120]
            msg = f"HTTP {code}: {body}"
        msg = msg[:300]
        sys.exit(f"error: {msg}")


if __name__ == "__main__":
    main()
