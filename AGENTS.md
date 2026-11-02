# AGENTS.md

## What this repo produces

APEX quantizes MoE (and dense/hybrid) GGUF models via per-layer, per-tensor-type precision assignments using llama.cpp's `--tensor-type-file`. No C patches — everything is config-driven.

## Directory layout: `scripts/` vs `old_scripts/`

- `scripts/` — the active Python interface (tier profiles, dense allocator, batch runner, cloud copy). Nothing here is a shell script.
- `old_scripts/` — the legacy shell interface (`generate_config.sh`, `quantize.sh`, `apex_pipeline.sh`, `eval.sh`, `benchmark.sh`, `estimate_config_size.py`, sensitivity/MTP/plot scripts, `push_to_hf.sh`). Reference only: these scripts produced the published APEX quants, but nothing runs them (the test suite covers `scripts/` exclusively). Don't delete or "modernize" them.

## Active scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `generate_config.py` | Emits tensor-type file; `--profile` (tier1–tier15, i-tierN, dense-*), `--layers`, `--dense-layers`, `--arch moe\|dense`. Quant modes: `--speed`/`--size`/`--mixed` (default `mixed`: experts IQ-band table, rest K-quant speed table). Extra role-shift modifiers (`--edge-exp`, `--mid-attn`, `--embd-type`, …) and dense overrides (`--linattn`, `--fullattn`, `--embd`, `--output-type`, `--f16-tensors`) |
| `quantize.py` | Quantize GGUF; `--profile` (default tier1), `--config`, `--imatrix`, `--dry-run`, `--generate-config`, `--base-type` (default Q8_0). Auto-loads `.env`, auto-detects layers/arch/experts via `detect_gguf_params.py` |
| `estimate_size.py` | Predict GGUF size from a profile or config; `--compare` shows tensor types × groups; also accepts `--imatrix`, `--dense-layers`, quant mode flags |
| `detect_gguf_params.py` | Reads GGUF header: layers, arch, expert count (used by quantize.py) |
| `grab_config.py` | Reverse direction: extracts a tensor-type config from an existing GGUF header; auto-detects MoE vs dense format |
| `generate_dense_config.py` | Primary dense path. Imatrix-importance-driven allocator: reads imatrix GGUF + source GGUF header (or `--inventory` tensor JSON), ranks tensor groups by importance, maps rank → quality continuum → quant ladder. `--tier` (tier1–15) sets the continuum's lower bound, NOT a size budget. Emits anchored configs for `quantize.py --config` |
| `batch_quantize.py` | End-to-end runner: HF download → imatrix → quantize tiers (default `--tiers 1-15`) → HF/S3 upload. Resumable state in `--workspace` (default `~/apex_batch`) |
| `cloudcp.py` | s3cmd-like `cp`/`ls`/`rm` across `s3://`, `hf://` and local paths; streaming, resumable (Range / S3 multipart ListParts) |
| `batch_perplexity.py` | Batch perplexity/KLD eval of quants from S3 or HF; saves `<model>-perplexity-report.txt` and uploads it back |
| `generate_imatrix_calibration.py` | Builds imatrix calibration file from a text block + rendered tool conversations (needs `transformers` tokenizer, HF model id — not a .gguf) |
| `reduce_calibration.py` | Shrinks a calibration text file, preserving text diversity (`-light`/`-ultralight` variants) |
| `s3_proxy.py` | Read-only S3 HTTP proxy; reads the same `S3_*` env vars as `batch_quantize.py` |

## Canonical workflow

Operational order — always size-check before the multi-hour quantize:

1. `scripts/estimate_size.py --profile <p> --layers N <model>.gguf` → predict target size from a profile.
2. `scripts/generate_config.py --profile <p> --layers N [--dense-layers N] [--arch moe|dense]` → emit the tensor-type file.
3. Re-verify the config lands in the target band (`estimate_size.py --config <cfg>`). This is the gate before quantizing.
4. `scripts/quantize.py --profile <p> --config <cfg> <input>.gguf <out>.gguf` (`--dry-run` skips the write).

For dense models skip step 2's profile and use `generate_dense_config.py` instead (see below).

## Running tests

```bash
python3 tests/run_all.py
```

Single stdlib-unittest entry point (no package manager, no CI, no test dependencies). It discovers the unittest modules in `tests/` and additionally runs the two script-style tests as subprocesses; it returns a non-zero exit code if anything fails.

| Module | Covers |
|--------|--------|
| `tests/test_generate_config.py` | `scripts/generate_config.py` behaviour: dense experiment arms (flat control / FFN gradient / full-vs-linear attention split), anchored-pattern regression (`attn_output` ≠ output head), `--arch dense` validation, tier1–15 profiles against `PROFILES` × quant modes (`--speed`/`--size`/`--mixed`), role-shift modifiers, edge/near/mid zoning, `--f16-tensors` |
| `tests/test_generate_dense_config.py` | `scripts/generate_dense_config.py` imatrix allocator via `--inventory` + a synthetic imatrix GGUF (no model files): anchored patterns, full inventory coverage, `token_embd` tier-pinned quant on the imatrix-free ladder (`TIER_TOKEN_EMBD`), Q6_K ladder cap (except tier1–2, token_embd exempt), `output.weight` Q3_K floor, importance-monotone allocation |
| `tests/test_estimate_size.py` | `scripts/estimate_size.py` internals (`load_config`, `match_type`, `estimate`, `estimate_breakdown`) on synthetic tensors, plus the CLI `--json` schema and per-profile base types on a synthetic GGUF header |
| `tests/test_reduce_calibration.py` | `scripts/reduce_calibration.py`: `-light`/`-ultralight` targets, even-stride diversity, sentence-boundary chunk splitting |
| `tests/test_config_collisions.py` | Replays llama-quantize's regex matching against real tensor-name fixtures (`tests/fixtures/`, header dumps of published GGUFs) to catch silent first-match-wins collisions. Also runnable standalone: `python3 tests/test_config_collisions.py` |
| `tests/test_cloudcp_stream_retry.py` | `scripts/cloudcp.py` stream-resume paths (Range GET, full GET, concurrent parts, HF). Also runnable standalone: `python3 tests/test_cloudcp_stream_retry.py` |

The tests exercise `scripts/` only. Committed configs in `configs/` are frozen artifacts of the legacy generator (byte-for-byte reproduction is no longer asserted); regex-collision safety for them is guarded by `test_config_collisions.py`.

## Critical implementation detail: regex anchoring

llama-quantize matches `--tensor-type-file` entries with `std::regex_search` — **unanchored, first match wins** (`src/llama-quant.cpp`). An unanchored `output.weight` pattern captures `blk.N.attn_output.weight` and silently mis-quantizes all attention output projections. Dense configs use `^...$` anchored patterns for this reason (`generate_config.py` dense path and `old_scripts/generate_config.sh`); the MoE emitter uses unanchored patterns but relies on tensor-name structure (`ffn_gate_exps` ≠ `ffn_gate`). Known-benign ambiguity (guarded by `test_config_collisions.py`): unanchored `blk.N.attn_q` also matches `attn_qkv.weight`, but every MoE profile assigns both the same type — one edit away from a real bug.

## MoE vs dense/hybrid architecture

- `--arch moe` (default): emits `ffn_gate_exps`, `ffn_gate_shexp`, etc. Standard MoE path.
- `--arch dense`: for models with no experts. Emits `ffn_gate.weight` (with `.weight` suffix), `token_embd.weight`, `output.weight`, and splits attention into full-attn vs linear-attn tensors (`--linattn`/`--fullattn`).
- `--dense-layers N`: leading dense (non-MoE) FFN layers inside an MoE model (e.g., LFM2). Incompatible with `--arch dense`.
- Dense allocation (`generate_dense_config.py`) enforces measured constraints, don't "fix" them: `token_embd` has zero imatrix coverage (get_rows, not matmul; types needing imatrix, e.g. IQ2_XXS, abort at quantize time with rc=133) → imatrix-free ladder only, with a tier-pinned quant (`TIER_TOKEN_EMBD`: tier1–3 → Q8_0, tier4–6 → Q6_K, tier7–9 → Q5_K, tier10–12 → Q4_K, tier13–15 → Q3_K) — exempt from the ladder cap; ladder capped at Q6_K except tiers 1–2 (Q8_0 away from near-full precision measured worse); `output.weight` keeps a Q3_K floor (dearest group on the measured curve). Static `dense-*` profiles (`dense-flat`, `dense-grad`, `dense-hybrid`, `dense-hybrid-quality`) are size-matched A/B controls, not the primary path.

## Config naming convention

`configs/{prefix}_{profile}.txt` where prefix identifies the model (e.g., `lfm25_8b`, `laguna_xs21`) and profile is a legacy name (`quality`, `balanced`, `compact`, `mini`, plus `nano`/`micro`), an I-variant, a dense profile (`dense_flat`, `dense_grad`, …, `opt_*` for the Qwen3.8 experiment). 200+ committed configs are the ground truth — `test_generate_config.sh` asserts the generator reproduces them byte-for-byte. Tier-profile configs are generated on the fly, not committed.

## Model definitions

`models/*.yaml` drive the pipeline. Single model currently: `qwen38_27b.yaml` — a dense-FFN/hybrid-attention pilot (64 layers, 3× linear-attn + 1× full-attn interleave, vision tower, real MTP/NextN head). Fields: `model_id`, `layers`, `arch` (moe|dense), `config_prefix`, `hf_repo`, `source_gguf`, `calibration`, `eval_suite`, `baselines`. `models/qwen38_27b_tensors.json` is the tensor inventory for `generate_dense_config.py --inventory`.

## Environment variables

`quantize.py` and `batch_quantize.py` auto-load `.env` (see `.env.example`; git-ignored) without overwriting existing vars:

- `LLAMA_QUANTIZE` — path to the `llama-quantize` binary; or `LLAMA_CPP_DIR` — llama.cpp build/bin dir
- `LLAMA_GGUF_SPLIT`, `LLAMA_CONVERT_HF_TO_GGUF` — specific binaries for `batch_quantize.py` (converter is the vendored `llama-scripts/convert_hf_to_gguf.py`)
- `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) — HF download/upload
- `S3_ENDPOINT`, `S3_TOKEN` or `S3_KEY_ID`/`S3_SECRET`, `S3_REGION` — S3 upload for batch runs
- `NGL` — GPU layers for `batch_perplexity.py` (default 99)

`NUM_LAYERS`/`WORK_DIR` only exist in `old_scripts/apex_pipeline.sh` and friends.

## No package manager / no CI

No package.json, Makefile, pyproject.toml, CI workflows, or linter config. Scripts are standalone Python/bash. Python deps (`requirements.txt`): `huggingface_hub`, `httpx`, `rich`, `numpy`, `transformers` (calibration), `torch`/`sentencepiece`/`safetensors` (conversion); `matplotlib` commented out (only for plotting). The `gguf` package is NOT pip-installed: it is vendored in `llama-scripts/gguf-py` and used via `llama-scripts/convert_hf_to_gguf.py`. Testing is manual: run the three test scripts above.

## Secrets / git hygiene

`.env` holds `HF_TOKEN` and S3 keys — **never commit it**. `.gitignore` excludes `.env`, `*.gguf`, `*.bin`, `*.log`, `.venv/`; keep raw models and GGUFs out of git.
