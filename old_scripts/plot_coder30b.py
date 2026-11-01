#!/usr/bin/env python3
"""
plot_coder30b.py -- Generate benchmark plots for Qwen3-Coder-30B-A3B APEX.

Usage:
  python scripts/plot_coder30b.py
  python scripts/plot_coder30b.py --input-dir benchmark_results/coder30b/ --output-dir plots/coder30b/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "Qwen3-Coder-30B-Q8_0":              "Q8_0 (30.3 GB)",
    "Qwen3-Coder-30B-APEX-Quality":      "APEX Quality (18.1 GB)",
    "Qwen3-Coder-30B-APEX-I-Quality":    "APEX I-Quality (18.1 GB)",
    "Qwen3-Coder-30B-APEX-Balanced":     "APEX Balanced (20.5 GB)",
    "Qwen3-Coder-30B-APEX-I-Balanced":   "APEX I-Balanced (20.8 GB)",
    "Qwen3-Coder-30B-APEX-Compact":      "APEX Compact (13.8 GB)",
    "Qwen3-Coder-30B-APEX-I-Compact":    "APEX I-Compact (13.8 GB)",
    "Qwen3-Coder-30B-APEX-Mini":         "APEX Mini (11.3 GB)",
    "Qwen3-Coder-30B-UD-Q4_K_XL":       "Unsloth UD-Q4_K_XL (16.5 GB)",
    "Qwen3-Coder-30B-Q5_K_S":           "Unsloth Q5_K_S (19.6 GB)",
    # Handle alternate model key from JSON
    "q8_0":                               "Q8_0 (30.3 GB)",
}

ALLOWED_MODELS = set(DISPLAY_NAMES.keys())

CATEGORY_MAP = {
    "Qwen3-Coder-30B-Q8_0":              "baseline",
    "q8_0":                               "baseline",
    "Qwen3-Coder-30B-UD-Q4_K_XL":       "external",
    "Qwen3-Coder-30B-Q5_K_S":           "external",
    "Qwen3-Coder-30B-APEX-Quality":      "apex",
    "Qwen3-Coder-30B-APEX-I-Quality":    "apex",
    "Qwen3-Coder-30B-APEX-Balanced":     "apex",
    "Qwen3-Coder-30B-APEX-I-Balanced":   "apex",
    "Qwen3-Coder-30B-APEX-Compact":      "apex",
    "Qwen3-Coder-30B-APEX-I-Compact":    "apex",
    "Qwen3-Coder-30B-APEX-Mini":         "apex",
}

COLORS = {
    "baseline": "#D32F2F",
    "external": "#F57C00",
    "apex":     "#388E3C",
}

MARKERS = {
    "baseline": "s",
    "external": "D",
    "apex":     "o",
}

CATEGORY_LABELS = {
    "baseline": "Baselines (Q8_0)",
    "external": "Unsloth",
    "apex":     "APEX Configurations",
}

MODEL_COLORS = {
    "Q8_0 (30.3 GB)":                   "#D32F2F",
    "APEX Quality (18.1 GB)":            "#1B5E20",
    "APEX I-Quality (18.1 GB)":          "#2E7D32",
    "APEX Balanced (20.5 GB)":           "#33691E",
    "APEX I-Balanced (20.8 GB)":         "#388E3C",
    "APEX Compact (13.8 GB)":            "#558B2F",
    "APEX I-Compact (13.8 GB)":          "#66BB6A",
    "APEX Mini (11.3 GB)":               "#81C784",
    "Unsloth UD-Q4_K_XL (16.5 GB)":     "#F57C00",
    "Unsloth Q5_K_S (19.6 GB)":          "#FF9800",
}


def load_benchmarks(input_dir):
    models = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path) as fh:
            data = json.load(fh)
        model_key = data.get("model", fname.replace(".json", ""))
        if model_key not in ALLOWED_MODELS:
            print(f"  [skip] {model_key} (not in ALLOWED_MODELS)")
            continue
        data["display_name"] = DISPLAY_NAMES[model_key]
        data["category"] = CATEGORY_MAP.get(model_key, "apex")
        models.append(data)
    if not models:
        print(f"ERROR: No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(models)} benchmark files from {input_dir}")
    for m in models:
        print(f"  - {m['display_name']} (PPL={m['perplexity']})")
    return models


def _apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=10)


# --- Plot 1: PPL vs Size ---
def plot_pareto_ppl_size(models, output_dir):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []
    q8_ppl = None
    for m in models:
        if "Q8_0" in m["display_name"]:
            q8_ppl = m["perplexity"]

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        marker = MARKERS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)
        ax.scatter(m["size_gb"], m["perplexity"], color=color, marker=marker,
                   s=100, zorder=5, edgecolors="white", linewidths=0.6, label=label)
        texts.append(ax.text(m["size_gb"], m["perplexity"],
                             m["display_name"], fontsize=8))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    if q8_ppl is not None:
        ax.axhline(y=q8_ppl, color="#757575", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(f"Q8_0 baseline ({q8_ppl:.4f})", xy=(ax.get_xlim()[1], q8_ppl),
                    xytext=(-8, 6), textcoords="offset points",
                    fontsize=8, color="#757575", ha="right", va="bottom")

    _apply_style(ax, "Qwen3-Coder-30B: Perplexity vs Model Size",
                 "Model Size (GB)", "Perplexity (lower is better)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(output_dir, "pareto_ppl_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 2: PPL vs Speed ---
def plot_pareto_ppl_speed(models, output_dir):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []
    q8_ppl = None
    for m in models:
        if "Q8_0" in m["display_name"]:
            q8_ppl = m["perplexity"]

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        marker = MARKERS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)
        ax.scatter(m["tg128_ts"], m["perplexity"], color=color, marker=marker,
                   s=100, zorder=5, edgecolors="white", linewidths=0.6, label=label)
        texts.append(ax.text(m["tg128_ts"], m["perplexity"],
                             m["display_name"], fontsize=8))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    if q8_ppl is not None:
        ax.axhline(y=q8_ppl, color="#757575", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(f"Q8_0 baseline ({q8_ppl:.4f})", xy=(ax.get_xlim()[1], q8_ppl),
                    xytext=(-8, 6), textcoords="offset points",
                    fontsize=8, color="#757575", ha="right", va="bottom")

    _apply_style(ax, "Qwen3-Coder-30B: Perplexity vs Inference Speed",
                 "Speed tg128 (tokens/sec, higher is better)",
                 "Perplexity (lower is better)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(output_dir, "pareto_ppl_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 3: Radar ---
def plot_radar_chart(models, output_dir):
    radar_names = [
        "APEX I-Quality (18.1 GB)", "APEX I-Balanced (20.8 GB)",
        "APEX I-Compact (13.8 GB)", "APEX Mini (11.3 GB)",
        "Unsloth UD-Q4_K_XL (16.5 GB)", "Unsloth Q5_K_S (19.6 GB)",
        "Q8_0 (30.3 GB)",
    ]
    radar_models = [m for m in models if m["display_name"] in radar_names]
    radar_models.sort(key=lambda m: radar_names.index(m["display_name"]))

    if len(radar_models) < 2:
        print("  [!] Not enough models for radar chart, skipping.")
        return None

    axis_labels = ["PPL\n(inverted)", "HellaSwag", "MMLU", "ARC\nChallenge",
                   "Speed\n(tg128)", "Size\nEfficiency"]

    raw_data = {}
    for m in radar_models:
        name = m["display_name"]
        raw_data[name] = {
            "ppl_inv": 1.0 / m["perplexity"],
            "hellaswag": m["hellaswag"],
            "mmlu": m["mmlu"],
            "arc": m["arc_challenge"],
            "speed": m["tg128_ts"],
            "size_eff": 1.0 / m["size_gb"],
        }

    keys = ["ppl_inv", "hellaswag", "mmlu", "arc", "speed", "size_eff"]
    mins = {k: min(raw_data[n][k] for n in raw_data) for k in keys}
    maxs = {k: max(raw_data[n][k] for n in raw_data) for k in keys}

    def normalize(val, key):
        r = maxs[key] - mins[key]
        return (val - mins[key]) / r if r != 0 else 0.5

    norm_data = {name: [normalize(raw_data[name][k], k) for k in keys]
                 for name in raw_data}

    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    radar_colors = {
        "APEX I-Quality (18.1 GB)":       "#2E7D32",
        "APEX I-Balanced (20.8 GB)":      "#388E3C",
        "APEX I-Compact (13.8 GB)":       "#66BB6A",
        "APEX Mini (11.3 GB)":            "#81C784",
        "Unsloth UD-Q4_K_XL (16.5 GB)":  "#E65100",
        "Unsloth Q5_K_S (19.6 GB)":      "#F57C00",
        "Q8_0 (30.3 GB)":                "#D32F2F",
    }
    radar_ls = {
        "APEX I-Quality (18.1 GB)":       "-",
        "APEX I-Balanced (20.8 GB)":      "--",
        "APEX I-Compact (13.8 GB)":       "-.",
        "APEX Mini (11.3 GB)":            ":",
        "Unsloth UD-Q4_K_XL (16.5 GB)":  "-.",
        "Unsloth Q5_K_S (19.6 GB)":      "-",
        "Q8_0 (30.3 GB)":                "--",
    }

    for name in radar_names:
        if name not in norm_data:
            continue
        values = norm_data[name] + norm_data[name][:1]
        ax.plot(angles, values, color=radar_colors.get(name, "#999"),
                linewidth=2.0, linestyle=radar_ls.get(name, "-"), label=name)
        ax.fill(angles, values, color=radar_colors.get(name, "#999"), alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#757575")
    ax.set_title("Qwen3-Coder-30B: Multi-Metric Comparison",
                 fontsize=14, fontweight="bold", pad=24)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(output_dir, "radar_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 4: Accuracy comparison ---
def plot_accuracy_comparison(models, output_dir):
    model_order = [
        "Q8_0 (30.3 GB)",
        "APEX I-Quality (18.1 GB)", "APEX I-Balanced (20.8 GB)",
        "APEX I-Compact (13.8 GB)", "APEX Mini (11.3 GB)",
        "APEX Quality (18.1 GB)", "APEX Balanced (20.5 GB)",
        "APEX Compact (13.8 GB)",
        "Unsloth UD-Q4_K_XL (16.5 GB)", "Unsloth Q5_K_S (19.6 GB)",
    ]
    ordered = []
    for name in model_order:
        for m in models:
            if m["display_name"] == name:
                ordered.append(m)
                break

    if not ordered:
        return None

    benchmarks = [
        ("HellaSwag", "hellaswag"), ("Winogrande", "winogrande"),
        ("MMLU", "mmlu"), ("ARC-Challenge", "arc_challenge"),
        ("TruthfulQA", "truthfulqa"),
    ]

    n_models = len(ordered)
    n_benchmarks = len(benchmarks)
    bar_width = 0.07
    group_gap = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    group_centers = []
    for gi, (bench_label, bench_key) in enumerate(benchmarks):
        group_start = gi * (n_models * bar_width + group_gap)
        group_center = group_start + (n_models - 1) * bar_width / 2
        group_centers.append(group_center)
        for mi, m in enumerate(ordered):
            val = m.get(bench_key, 0) or 0
            x = group_start + mi * bar_width
            color = MODEL_COLORS.get(m["display_name"], "#999999")
            ax.bar(x, val, width=bar_width * 0.88, color=color,
                   edgecolor="white", linewidth=0.5)
            ax.text(x, val + 0.3, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6, rotation=45)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([b[0] for b in benchmarks], fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS.get(m["display_name"], "#999"),
               edgecolor="white", linewidth=0.5, label=m["display_name"])
               for m in ordered]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2, framealpha=0.9)

    all_vals = [m.get(k, 0) or 0 for m in ordered for _, k in benchmarks if (m.get(k, 0) or 0) > 0]
    if all_vals:
        ax.set_ylim(max(0, min(all_vals) - 3), max(all_vals) + 4)

    _apply_style(ax, "Qwen3-Coder-30B: Accuracy Benchmark Comparison", "", "Accuracy (%)")
    fig.tight_layout()
    out = os.path.join(output_dir, "accuracy_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 5: KL comparison ---
def plot_kl_comparison(models, output_dir):
    kl_models = [m for m in models if m.get("kl_mean") is not None]
    if not kl_models:
        return None

    kl_order = [
        "Q8_0 (30.3 GB)",
        "APEX I-Balanced (20.8 GB)", "APEX Balanced (20.5 GB)",
        "APEX I-Quality (18.1 GB)", "APEX Quality (18.1 GB)",
        "Unsloth Q5_K_S (19.6 GB)", "Unsloth UD-Q4_K_XL (16.5 GB)",
        "APEX I-Compact (13.8 GB)", "APEX Compact (13.8 GB)",
        "APEX Mini (11.3 GB)",
    ]
    ordered = []
    for name in kl_order:
        for m in kl_models:
            if m["display_name"] == name:
                ordered.append(m)
                break
    for m in kl_models:
        if m not in ordered:
            ordered.append(m)

    names = [m["display_name"] for m in ordered]
    kl_means = [m["kl_mean"] for m in ordered]
    kl_999s = [m.get("kl_99_9", 0) or 0 for m in ordered]

    n = len(names)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(x - bar_width / 2, kl_means, bar_width, color="#1565C0",
           edgecolor="white", linewidth=0.5, label="KL Mean")
    ax.bar(x + bar_width / 2, kl_999s, bar_width, color="#E65100",
           edgecolor="white", linewidth=0.5, label="KL 99.9th Percentile")

    for i, (mean_v, p99_v) in enumerate(zip(kl_means, kl_999s)):
        if mean_v > 0:
            ax.text(x[i] - bar_width / 2, mean_v * 1.15,
                    f"{mean_v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
        if p99_v > 0:
            ax.text(x[i] + bar_width / 2, p99_v * 1.15,
                    f"{p99_v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    for i, m in enumerate(ordered):
        color = MODEL_COLORS.get(m["display_name"], "#333")
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight("bold")

    _apply_style(ax, "Qwen3-Coder-30B: KL Divergence from F16 Reference",
                 "", "KL Divergence (log scale, lower is better)")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(output_dir, "kl_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 6: Efficiency bubble ---
def plot_efficiency(models, output_dir):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []

    inv_ppls = [1.0 / m["perplexity"] for m in models]
    min_inv, max_inv = min(inv_ppls), max(inv_ppls)
    inv_range = max_inv - min_inv if max_inv != min_inv else 1.0

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)
        inv_ppl = 1.0 / m["perplexity"]
        bubble_size = 120 + 480 * (inv_ppl - min_inv) / inv_range
        ax.scatter(m["size_gb"], m["tg128_ts"], s=bubble_size, color=color,
                   alpha=0.7, zorder=5, edgecolors="white", linewidths=0.8, label=label)
        texts.append(ax.text(m["size_gb"], m["tg128_ts"],
                             m["display_name"], fontsize=8))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax.annotate("Bubble size = 1/PPL\n(larger = better quality)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#757575",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#BDBDBD", alpha=0.9))

    _apply_style(ax, "Qwen3-Coder-30B: Size vs Speed (Quality as Bubble Size)",
                 "Model Size (GB)", "Speed tg128 (tokens/sec)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(output_dir, "efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Plot 7: APEX vs Unsloth KL ---
def plot_kl_apex_vs_unsloth(models, output_dir):
    from matplotlib.patches import Patch

    model_order = [
        "APEX I-Balanced (20.8 GB)", "APEX I-Quality (18.1 GB)",
        "APEX I-Compact (13.8 GB)", "APEX Mini (11.3 GB)",
        "APEX Quality (18.1 GB)", "APEX Balanced (20.5 GB)", "APEX Compact (13.8 GB)",
        "Unsloth UD-Q4_K_XL (16.5 GB)", "Unsloth Q5_K_S (19.6 GB)",
        "Q8_0 (30.3 GB)",
    ]

    bar_colors = {}
    for n in model_order:
        if "APEX" in n:
            bar_colors[n] = "#388E3C"
        elif "Unsloth" in n:
            bar_colors[n] = "#F57C00"
        else:
            bar_colors[n] = "#D32F2F"

    ordered = []
    for name in model_order:
        for m in models:
            if m["display_name"] == name and m.get("kl_mean") is not None:
                ordered.append(m)
                break

    if not ordered:
        return None

    names = [m["display_name"] for m in ordered]
    kl_means = [m["kl_mean"] for m in ordered]
    kl_999s = [m.get("kl_99_9", 0) or 0 for m in ordered]
    colors = [bar_colors.get(n, "#999") for n in names]

    n = len(names)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(x, kl_means, 0.6, color=colors, edgecolor="white",
                  linewidth=0.5, label="KL Mean", zorder=3)
    ax.scatter(x, kl_999s, marker='v', s=80, color='#333', zorder=5,
               label="KL 99.9th Percentile")
    for i in range(n):
        ax.plot([x[i], x[i]], [kl_means[i], kl_999s[i]],
                color='#666', linewidth=1.0, linestyle='--', zorder=4)

    for i, (bar, mv, pv) in enumerate(zip(bars, kl_means, kl_999s)):
        ax.text(bar.get_x() + bar.get_width() / 2, mv + 0.001,
                f"{mv:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(x[i] + 0.15, pv + 0.02, f"{pv:.3f}", ha="left", va="bottom",
                fontsize=6, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    for i, name in enumerate(names):
        ax.get_xticklabels()[i].set_color(colors[i])
        ax.get_xticklabels()[i].set_fontweight("bold")

    legend_elements = [
        Patch(facecolor="#388E3C", edgecolor="white", label="APEX"),
        Patch(facecolor="#F57C00", edgecolor="white", label="Unsloth"),
        Patch(facecolor="#D32F2F", edgecolor="white", label="Q8_0 Baseline"),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#333',
                   markersize=8, label="KL 99.9th Pctl"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left", framealpha=0.9)

    _apply_style(ax, "Qwen3-Coder-30B: APEX vs Unsloth KL Divergence",
                 "", "KL Divergence (lower is better)")
    fig.tight_layout()
    out = os.path.join(output_dir, "kl_apex_vs_unsloth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="benchmark_results/coder30b/")
    parser.add_argument("--output-dir", default="plots/coder30b/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = load_benchmarks(args.input_dir)

    plots = [
        ("1/7", "Pareto: PPL vs Size", plot_pareto_ppl_size),
        ("2/7", "Pareto: PPL vs Speed", plot_pareto_ppl_speed),
        ("3/7", "Radar chart", plot_radar_chart),
        ("4/7", "Accuracy comparison", plot_accuracy_comparison),
        ("5/7", "KL comparison", plot_kl_comparison),
        ("6/7", "Efficiency bubble", plot_efficiency),
        ("7/7", "APEX vs Unsloth KL", plot_kl_apex_vs_unsloth),
    ]

    generated = []
    for step, desc, fn in plots:
        print(f"\n[{step}] Generating {desc} ...")
        result = fn(models, args.output_dir)
        if result:
            generated.append(result)
            print(f"       Saved: {result}")

    print(f"\nDone. {len(generated)} plot(s) generated in {args.output_dir}")


if __name__ == "__main__":
    main()
