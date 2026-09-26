#!/usr/bin/env python3
"""Simple read-only S3 HTTP proxy (single file, stdlib only).

Serves GET/HEAD requests, proxying objects from an S3-compatible
storage (e.g. Yandex Object Storage). Endpoint, credentials and region
are taken from CLI args or from the environment / .env file, using the
same variable names as batch_quantize.py:

  S3_ENDPOINT   S3 endpoint URL with bucket embedded (path or subdomain form)
  S3_KEY_ID     static access key id (with S3_SECRET, AWS SigV4 signing)
  S3_SECRET     static access secret key
  S3_TOKEN      IAM token (alternative to static keys)
  S3_REGION     region for SigV4 signing (default: ru-central1)

Examples:
  python3 scripts/s3_proxy.py --port 8080
  python3 scripts/s3_proxy.py --endpoint https://storage.yandexcloud.net/mybucket/

Usage over HTTP:
  curl -O http://localhost:8080/models/Qwen35/model-tier5.gguf
  curl -C - -O http://localhost:8080/models/Qwen35/model-tier5.gguf   # resume

Download resume works via the standard Range header: the proxy forwards
it to S3 and relays the 206 response, so any client with resume support
(curl -C -, wget -c, aria2c, HTTP file managers) works out of the box.

Memory usage is constant: the body is streamed in small chunks, the
full file (10-30+ GB) is never held in memory. Only GET/HEAD are
supported — all other methods are rejected, nothing is ever written
to S3.
"""

import argparse
import hashlib
import hmac
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

CHUNK_SIZE = 256 * 1024          # 256 KB relay buffer (constant memory)
UPSTREAM_TIMEOUT = 600           # per-read socket timeout for huge files

FORWARDED_HEADERS = (
    "Content-Length", "Content-Range", "Content-Type", "ETag",
    "Last-Modified", "Accept-Ranges", "Content-Encoding",
)


# ---------------------------------------------------------------------------
# .env loading (same convention as the rest of the repo)
# ---------------------------------------------------------------------------

def load_dotenv(path: str) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# S3 settings & AWS SigV4 (adapted from batch_quantize.py)
# ---------------------------------------------------------------------------

def parse_s3_endpoint(endpoint: str) -> tuple:
    """Extract (base_url, bucket) from an endpoint URL.

    Bucket must be embedded in the URL:
      https://storage.yandexcloud.net/<bucket>/  -> path form
      https://<bucket>.storage.yandexcloud.net/  -> subdomain form
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


def _uri_encode(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s, safe="-_.~")


def aws_sigv4_headers(method: str, url: str, key_id: str, secret: str,
                      region: str) -> dict:
    """Build AWS Signature Version 4 headers for a bodyless S3 request."""
    u = urlsplit(url)
    host = u.netloc
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = amz_date[:8]
    payload_hash = hashlib.sha256(b"").hexdigest()

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
    signature = hmac.new(k, string_to_sign.encode(), hashlib.sha256) \
        .hexdigest()

    return {
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"),
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }


def get_s3_settings(args) -> dict:
    """Resolve S3 settings from CLI args / env / .env (same priority as
    batch_quantize.py: static keys first, then IAM token)."""
    endpoint = args.endpoint or os.environ.get("S3_ENDPOINT")
    key_id = args.key_id or os.environ.get("S3_KEY_ID")
    secret = args.secret or os.environ.get("S3_SECRET")
    token = args.token or os.environ.get("S3_TOKEN")
    region = args.region or os.environ.get("S3_REGION", "ru-central1")

    if not endpoint:
        sys.exit("error: S3 endpoint is not configured "
                 "(--endpoint or S3_ENDPOINT in .env)")
    if key_id and secret:
        auth = ("sigv4", key_id, secret)
    elif key_id or secret:
        sys.exit("error: incomplete static keys — both --key-id and "
                 "--secret (or S3_KEY_ID / S3_SECRET in .env) are required")
    elif token:
        auth = ("bearer", token)
    else:
        sys.exit("error: no S3 credentials — set S3_KEY_ID + S3_SECRET "
                 "or S3_TOKEN (in .env or via args)")

    try:
        base_url, bucket = parse_s3_endpoint(endpoint)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    return {"base_url": base_url, "bucket": bucket, "auth": auth,
            "region": region}


def s3_auth_headers(s3: dict, method: str, url: str) -> dict:
    auth = s3["auth"]
    if auth[0] == "sigv4":
        return aws_sigv4_headers(method, url, auth[1], auth[2], s3["region"])
    return {"Authorization": f"Bearer {auth[1]}"}


# ---------------------------------------------------------------------------
# Proxy request handler
# ---------------------------------------------------------------------------

class S3ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "s3-proxy/1.0"
    upstream_timeout = UPSTREAM_TIMEOUT

    def do_GET(self):
        self._proxy("GET")

    def do_HEAD(self):
        self._proxy("HEAD")

    def _proxy(self, method: str):
        s3 = self.server.s3

        # Only the object path is used; query strings are dropped so that
        # S3 API operations (list/delete/...) cannot be invoked through
        # the proxy — plain object GET/HEAD only.
        path = self.path.split("?", 1)[0]
        if not path.startswith("/") or path == "/":
            self._send_simple_error(400, "object path required")
            return

        url = f"{s3['base_url']}/{s3['bucket']}{path}"
        headers = s3_auth_headers(s3, method, url)

        # Forward Range (download resume) and a couple of harmless
        # cache validators from the client.
        for name in ("Range", "If-Match", "If-None-Match",
                     "If-Modified-Since", "If-Unmodified-Since"):
            value = self.headers.get(name)
            if value:
                headers[name] = value

        req = Request(url, method=method, headers=headers)
        try:
            upstream = urlopen(req, timeout=self.upstream_timeout)
        except HTTPError as exc:
            self._relay_error(exc)
            return
        except (URLError, OSError, TimeoutError) as exc:
            self._send_simple_error(502, f"upstream error: {exc}")
            return

        try:
            self.send_response(upstream.status)
            for name in FORWARDED_HEADERS:
                value = upstream.headers.get(name)
                if value:
                    self.send_header(name, value)
            if "Content-Length" not in upstream.headers:
                # Unknown length: we cannot delimit a keep-alive body.
                self.close_connection = True
            self.end_headers()

            if method == "GET":
                while True:
                    chunk = upstream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                self.wfile.flush()
        finally:
            upstream.close()

    def _relay_error(self, exc: HTTPError):
        """Forward the upstream error status and a short error body."""
        try:
            body = exc.read(4096)
        except OSError:
            body = b""
        try:
            self.send_response(exc.code)
            ctype = exc.headers.get("Content-Type") if exc.headers else None
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_simple_error(self, code: int, message: str):
        body = message.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} "
              f"{fmt % args}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Read-only S3 -> HTTP(S) proxy (GET/HEAD only, "
                    "streaming, Range/resume support)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8080")),
                        help="listen port (default: 8080)")
    parser.add_argument("--bind", default="",
                        help="bind address (default: all interfaces)")
    parser.add_argument("--env", default=".env",
                        help="path to .env file (default: ./.env)")
    parser.add_argument("--endpoint",
                        help="S3 endpoint URL with bucket (default: "
                             "$S3_ENDPOINT)")
    parser.add_argument("--key-id",
                        help="S3 static access key id (default: $S3_KEY_ID)")
    parser.add_argument("--secret",
                        help="S3 static access secret (default: $S3_SECRET)")
    parser.add_argument("--token",
                        help="S3 IAM token (default: $S3_TOKEN)")
    parser.add_argument("--region",
                        help="S3 region for SigV4 (default: $S3_REGION "
                             "or ru-central1)")
    args = parser.parse_args()

    load_dotenv(args.env)
    s3 = get_s3_settings(args)

    server = ThreadingHTTPServer((args.bind, args.port), S3ProxyHandler)
    server.s3 = s3
    server.daemon_threads = True
    print(f"s3-proxy: serving s3://{s3['bucket']} (read-only) "
          f"on port {args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
