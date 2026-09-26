#!/usr/bin/env python3
"""Run the rus_finance_benchmark.jsonl eval against a local GGUF model.

Loads the model once via llama-server and streams all questions through it,
then reports accuracy split by level (Basic / Intermediate / Advanced).

Usage:
    python3 scripts/rus_finance_benchmark.py --model <model.gguf> [options]

Options:
    --model PATH        GGUF model file (required)
    --lora PATH         Optional LoRA adapter (gguf)
    --level LEVEL       Run only one level: Basic | Intermediate | Advanced
    --temperature T     Sampling temperature (default: 0.0 = greedy)
    --parallel N        Concurrent requests to the server (default: 1)
    --limit N           Limit number of questions (for smoke runs)
    --ngl N             GPU layers (default: 99, env NGL)
    --context N         Context size (default: 4096)
    --dataset PATH      Path to benchmark jsonl (default: dataset/rus_finance_benchmark.jsonl)
    --keep-server       Keep the spawned llama-server process running after the run
    --port PORT         Port for llama-server (default: 8471)

Environment:
    LLAMA_CPP_DIR       Path to llama.cpp build/bin directory (required to
                        spawn the server; or point LLAMA_SERVER directly at
                        a llama-server binary)
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "dataset" / "rus_finance_benchmark.jsonl"

LEVELS = ["Basic", "Intermediate", "Advanced"]

SYSTEM_PROMPT = (
    "Ты — точный вычислительный ассистент. Реши задачу и дай ответ в виде "
    "одного числа. В самом конце ответа обязательно напиши строку вида "
    "'Ответ: <число>'. Никаких пояснений после этой строки не давай."
)

USER_TEMPLATE = "{question}\n\nДай финальный ответ в формате 'Ответ: <число>'."


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val:
            env[key] = val
    return env


def find_server_binary(repo_env: dict) -> str:
    candidate = repo_env.get("LLAMA_SERVER") or os.environ.get("LLAMA_SERVER")
    if candidate and Path(candidate).exists():
        return candidate
    cpp_dir = repo_env.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP_DIR")
    if not cpp_dir:
        sys.exit(
            "Error: set LLAMA_CPP_DIR (or LLAMA_SERVER) in .env or environment. "
            "See .env comments."
        )
    binary = Path(cpp_dir) / "llama-server"
    if not binary.exists():
        sys.exit(f"Error: llama-server not found in {cpp_dir}")
    return str(binary)


def wait_port(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def wait_healthy(port: int, timeout: float = 600.0) -> bool:
    """Wait until /health reports the model is fully loaded."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except (HTTPError, URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    return False


def chat_completion(port: int, temperature: float, system: str, user: str,
                    max_tokens: int = 512, retries: int = 3) -> str:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    last_err = None
    for _ in range(retries):
        try:
            req = Request(url, data=payload,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except (HTTPError, URLError, OSError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"chat completion failed after {retries} retries: {last_err}")


NUM_RE = re.compile(r"-?\d[\d\s\u00a0]*(?:[.,]\d+)?")


def extract_answer(text: str):
    """Extract a float from the model output. Prefers the 'Ответ:' marker,
    falls back to the last number in the text."""
    if not text:
        return None
    marker = text.rfind("Ответ")
    if marker != -1:
        tail = text[marker:]
        matches = NUM_RE.findall(tail.replace("Ответ", "", 1))
        if matches:
            return parse_num(matches[-1])
    matches = NUM_RE.findall(text)
    if matches:
        return parse_num(matches[-1])
    return None


def parse_num(s: str):
    s = s.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(predicted, expected, rel_tol=0.005, abs_tol=1e-9):
    if predicted is None:
        return False
    return abs(predicted - expected) <= max(abs_tol, rel_tol * abs(expected))


def load_dataset(path: Path, level: str | None, limit: int | None):
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if level and d["level"] != level:
                continue
            items.append(d)
    if limit:
        items = items[:limit]
    return items


def run_items(items, port, temperature, parallel):
    def work(idx_item):
        idx, item = idx_item
        user = USER_TEMPLATE.format(question=item["question"])
        out = chat_completion(port, temperature, SYSTEM_PROMPT, user)
        pred = extract_answer(out)
        ok = is_correct(pred, float(item["final_answer"]))
        return idx, item, out, pred, ok

    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for idx, item, out, pred, ok in pool.map(work, enumerate(items)):
            results[idx] = (item, out, pred, ok)
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"  progress: {done}/{len(items)}", file=sys.stderr)
    return results


def report(results):
    """Print accuracy report split by level + overall."""
    by_level = defaultdict(lambda: [0, 0])
    for item, _out, _pred, ok in results:
        by_level[item["level"]][0] += int(ok)
        by_level[item["level"]][1] += 1

    print("\n" + "=" * 60)
    print("rus_finance_benchmark results")
    print("=" * 60)
    total_ok = total_n = 0
    for level in LEVELS:
        if level not in by_level:
            continue
        ok, n = by_level[level]
        total_ok += ok
        total_n += n
        print(f"{level:<14} {ok:>5}/{n:<5}  {100.0 * ok / n:6.2f}%")
    if total_n:
        print("-" * 60)
        print(f"{'Overall':<14} {total_ok:>5}/{total_n:<5}  "
              f"{100.0 * total_ok / total_n:6.2f}%")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--lora", default=None, help="Optional LoRA adapter (gguf)")
    ap.add_argument("--level", choices=LEVELS, default=None,
                    help="Run only one level")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Concurrent requests (default: 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of questions")
    ap.add_argument("--ngl", type=int,
                    default=int(os.environ.get("NGL", 99)),
                    help="GPU layers (default: env NGL or 99)")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--keep-server", action="store_true",
                    help="Do not kill llama-server after the run")
    args = ap.parse_args()

    repo_env = load_env(REPO_ROOT / ".env")
    server_bin = find_server_binary(repo_env)
    items = load_dataset(args.dataset, args.level, args.limit)
    if not items:
        sys.exit("No questions selected (check --level / --limit).")

    print(f"Model:        {args.model}")
    if args.lora:
        print(f"LoRA:         {args.lora}")
    print(f"Server:       {server_bin} (port {args.port})")
    print(f"Questions:    {len(items)}"
          + (f" (level={args.level})" if args.level else ""))
    print(f"Temperature:  {args.temperature}")

    cmd = [
        server_bin,
        "-m", str(Path(args.model).resolve()),
        "-c", str(args.context),
        "-ngl", str(args.ngl),
        "--port", str(args.port),
        "--temp", str(args.temperature),
    ]
    if args.lora:
        cmd += ["--lora", str(Path(args.lora).resolve())]

    print("Starting llama-server...")
    server_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    try:
        if not wait_port(args.port, timeout=60):
            sys.exit("Error: llama-server did not open port in time")
        print("Waiting for model to load...")
        if not wait_healthy(args.port):
            sys.exit("Error: llama-server /health never reported ok")
        print("Model loaded. Running benchmark...")
        t0 = time.time()
        results = run_items(items, args.port, args.temperature, args.parallel)
        elapsed = time.time() - t0
        report(results)
        print(f"\nElapsed: {elapsed:.1f}s "
              f"({elapsed / len(results):.2f}s per question)")
    finally:
        if args.keep_server:
            print(f"llama-server left running on port {args.port} (pid "
                  f"{server_proc.pid})")
        else:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                else:
                    server_proc.terminate()
                server_proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                server_proc.kill()

    # Exit code: non-zero if accuracy below 100% is not the point; keep 0.
    # Save raw outputs for debugging failed cases.
    out_path = Path("rus_finance_benchmark_raw.json")
    with out_path.open("w") as f:
        json.dump([
            {"id": it["id"], "level": it["level"], "topic": it["topic"],
             "expected": it["final_answer"], "predicted": pred,
             "correct": ok, "raw": out}
            for it, out, pred, ok in results
        ], f, ensure_ascii=False, indent=1)
    print(f"Raw outputs saved to {out_path}")


if __name__ == "__main__":
    main()
