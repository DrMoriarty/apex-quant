# AGENTS.md

## What this repo produces

APEX quantizes MoE (and dense/hybrid) GGUF models via per-layer, per-tensor-type precision assignments using llama.cpp's `--tensor-type-file`. No C patches — everything is config-driven.

## Key scripts and their roles

Two parallel interfaces exist — shell (legacy profiles) and Python (tier-based profiles). The Python scripts are the current primary interface shown in README.

### Python scripts (primary — tier profiles: tier1–tier15, dense-*)

| Script | Purpose |
|--------|---------|
| `scripts/generate_config.py` | Emits tensor-type file; supports `--profile`, `--edge-exp`, `--layers`, `--dense-layers`, `--arch` |
| `scripts/quantize.py` | Quantize GGUF; `--profile`, `--config`, `--imatrix`, `--dry-run`, `--generate-config` |
| `scripts/estimate_size.py` | Predict GGUF size from a profile or config; `--compare` inspects tensor types × groups |
| `scripts/detect_gguf_params.py` | Reads GGUF header to detect layers, arch, expert count (used by quantize.py) |

### Shell scripts (legacy — named profiles: quality, balanced, compact, mini, nano, micro)

| Script | Purpose |
|--------|---------|
| `scripts/generate_config.sh` | Emits a tensor-type file for `llama-quantize --tensor-type-file` |
| `scripts/quantize.sh` | Wraps generate + llama-quantize; selects profile, finds binary |

### Shared scripts

| Script | Purpose |
|--------|---------|
| `scripts/estimate_config_size.py` | Predicts GGUF size from a config + tensor inventory |
| `scripts/eval.sh` | Runs PPL, KL, HellaSwag, Winogrande, MMLU, ARC, TruthfulQA, speed; outputs JSON |
| `scripts/benchmark.sh` | Batch benchmarking with TSV output and optional plots |
| `scripts/batch_perplexity.py` | Batch perplexity/KLD eval of quants from S3 or HF; saves `<model>-perplexity-report.txt` and uploads it back |
| `scripts/apex_pipeline.sh` | Full pipeline: download → convert → quantize → imatrix → eval → publish (YAML-driven) |
| `scripts/generate_sensitivity_configs.py` | Builds perturbation sweep configs for measuring tensor-group sensitivity |
| `scripts/generate_opt_config.py` | Turns a measured sensitivity curve into an allocation config |
| `scripts/push_to_hf.sh` | Uploads GGUFs to HuggingFace |

## Running tests

```bash
./tests/test_generate_config.sh   # bash; tests generate_config.sh (regression guards + dense mode)
./tests/test_config_collisions.py  # python3; checks for regex pattern conflicts across all configs
```

Both must pass. The bash test asserts committed configs reproduce byte-for-byte from the generator. The Python test replays regex matching against real tensor name fixtures (`tests/fixtures/`) to catch silent first-match-wins collisions.

## Critical implementation detail: regex anchoring

llama-quantize matches `--tensor-type-file` entries with `std::regex_search` — **unanchored, first match wins**. An unanchored `output.weight` pattern captures `blk.N.attn_output.weight` and silently mis-quantizes all attention output projections. The dense emitter (`generate_config.sh`) uses `^...$` anchored patterns for this reason. The MoE emitter uses unanchored patterns but relies on tensor name structure (e.g., `ffn_gate_exps` differs from `ffn_gate`). Test `test_config_collisions.py` guards against these collisions.

## MoE vs dense/hybrid architecture

- `--arch moe` (default): emits `ffn_gate_exps`, `ffn_gate_shexp`, etc. Standard MoE model path.
- `--arch dense`: emits `ffn_gate.weight` (with `.weight` suffix), `token_embd.weight`, `output.weight`, full-attn vs linear-attn tensors. For models with no experts.
- `--dense-layers N`: leading dense (non-MoE) FFN layers inside an MoE model (e.g., LFM2, Step-3.x). Incompatible with `--arch dense` — the test suite explicitly rejects this combination.

## Config naming convention

`configs/{prefix}_{profile}.txt` where prefix identifies the model (e.g., `qwen35a3b`, `laguna_xs21`) and profile is one of:
- **Legacy**: `quality`, `balanced`, `compact`, `mini`, plus I-variants and experimental tiers (`nano`, `micro`)
- **Tier-based** (Python scripts): `tier1`–`tier15`
- **Dense/hybrid**: `dense-flat`, `dense-grad`, `dense-hybrid`, `dense-hybrid-quality`

Committed configs are the ground truth — `test_generate_config.sh` asserts the generator reproduces them byte-for-byte.

## Model definitions

`models/*.yaml` drive `apex_pipeline.sh`. Fields: `model_id`, `layers`, `arch` (moe|dense), `config_prefix`, `hf_repo`, `source_gguf`, `calibration`, `eval_suite`, `baselines`.

## Size estimation before quantizing

Always use `scripts/estimate_size.py` (or `scripts/estimate_config_size.py` for raw config files) to verify a config hits the target size band before running a 6-hour quantize. The dense experiment arms must be size-matched (≤0.20 GB spread) or the A/B is uninterpretable — the bash test enforces this.

## Environment variables

- `LLAMA_CPP_DIR` or `LLAMA_QUANTIZE` — path to llama.cpp build/bin or specific binary
- `NUM_LAYERS` — override transformer layer count (default: 40)
- `WORK_DIR` — pipeline working directory (default: `/workspace/data/apex`)
- `NGL` — GPU layers for eval (default: 99)

## No package manager / no CI

This repo has no package.json, Makefile, pyproject.toml, CI workflows, or linter config. Scripts are standalone bash/python with `set -euo pipefail`. Python dependencies: `matplotlib`, `numpy` (for plot scripts). Testing is manual: run the two test scripts above.
