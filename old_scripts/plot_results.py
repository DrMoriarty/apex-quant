#!/usr/bin/env python3
"""
plot_results.py -- Generate Pareto frontier and comparison plots for APEX quantizations.

Reads benchmark data from a TSV file (or uses hardcoded defaults) and produces:
  1. pareto_ppl_vs_size.png   -- Perplexity vs Model Size with Pareto frontier
  2. pareto_ppl_vs_speed.png  -- Perplexity vs Inference Speed with Pareto frontier
  3. comparison_bars.png      -- Grouped bar chart comparing key configurations

Usage:
  python scripts/plot_results.py
  python scripts/plot_results.py --tsv results.tsv --output-dir ./plots
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Default data (so the script works standalone without a TSV file)
# ---------------------------------------------------------------------------
# (name, size_gb, perplexity, speed_tps, category,
#  hellaswag, winogrande, mmlu, arc_challenge, truthfulqa)
DEFAULT_CONFIGS = [
    ("F16",              64.6,  6.537, 30.4, "reference",
     82.5, 74.5, 41.5, 56.9, 37.2),
    ("Unsloth Q8_K_XL",  46.4,  6.536, 36.4, "external",
     82.5, 74.8, 41.3, 57.9, 38.1),
    ("Q8_0",             34.4,  6.533, 52.5, "baseline",
     83.0, 75.3, 41.2, 57.9, 37.7),
    ("APEX Quality",     21.3,  6.527, 62.3, "apex",
     83.0, 74.5, 41.2, 56.2, 37.7),
    ("APEX Balanced",    23.6,  6.533, 60.8, "apex",
     83.0, 74.5, 41.3, 56.9, 36.8),
    ("APEX Compact",     16.1,  6.783, 69.8, "apex",
     82.5, 73.3, 40.9, 55.2, 36.5),
]

# Mapping from TSV quant_type to friendly display names (used by the TSV reader)
_FRIENDLY_NAMES = {
    # Experiment results.tsv names
    "MOE_3TIER_IQ":  "APEX Quality",
    "MOE_LAYER_v1":  "APEX Balanced",
    "MOE_3TIER_v1":  "APEX Compact",
    "UNSLOTH_Q8KXL": "Unsloth Q8_K_XL",
    # Benchmark results.tsv filenames (without .gguf)
    "Qwen3.5-35B-A3B-APEX-Quality":  "APEX Quality",
    "Qwen3.5-35B-A3B-APEX-Balanced": "APEX Balanced",
    "Qwen3.5-35B-A3B-APEX-Compact":  "APEX Compact",
    "Qwen3.5-35B-A3B-UD-Q8_K_XL":   "Unsloth Q8_K_XL",
}

# Category classification for TSV rows
_CATEGORY_MAP = {
    "Q4_K_M":        "baseline",
    "Q8_0":          "baseline",
    "UNSLOTH_Q8KXL": "external",
    # Benchmark filenames
    "Qwen3.5-35B-A3B-UD-Q8_K_XL": "external",
}

# Configs to include in the grouped bar chart
_BAR_CONFIGS = [
    "F16",
    "Unsloth Q8_K_XL",
    "Q8_0",
    "APEX Quality",
    "APEX Balanced",
    "APEX Compact",
]

# Reference perplexity line (Q8_0)
Q8_0_PPL = 6.533


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
COLORS = {
    "baseline":  "#D32F2F",   # red
    "external":  "#F57C00",   # orange
    "apex":      "#388E3C",   # green
    "reference": "#9E9E9E",   # gray
    "discard":   "#BDBDBD",   # light gray
}

MARKERS = {
    "baseline":  "s",
    "external":  "D",
    "apex":      "o",
    "reference": "^",
    "discard":   "x",
}

CATEGORY_LABELS = {
    "baseline":  "Standard Baselines",
    "external":  "External (Unsloth)",
    "apex":      "APEX Configurations",
    "reference": "Reference (F16)",
    "discard":   "Discarded Experiments",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv(path):
    """Load results from a TSV file and return a list of config tuples.

    Supports two formats:
    1. Experiment results.tsv: columns include quant_type, size_mb, perplexity, tokens_per_sec, status
    2. Benchmark results.tsv: columns include model, size_gb, perplexity, tg128_ts, kl_mean, kl_max
    """
    configs = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []

        # Detect format
        is_benchmark_format = "model" in fields and "size_gb" in fields

        for row in reader:
            if is_benchmark_format:
                # Benchmark format: model, size_gb, perplexity, ppl_error, kl_mean, kl_max, pp512_ts, tg128_ts
                name = row.get("model", "").strip()
                name = _FRIENDLY_NAMES.get(name, name)
                category = _CATEGORY_MAP.get(row.get("model", "").strip(), "apex")

                try:
                    size_gb = float(row["size_gb"])
                except (ValueError, KeyError):
                    size_gb = None
                try:
                    ppl = float(row["perplexity"])
                except (ValueError, KeyError):
                    ppl = None
                try:
                    speed = float(row["tg128_ts"])
                    if speed <= 0:
                        speed = None
                except (ValueError, KeyError):
                    speed = None

                # Store KL divergence data as extra fields (accessed via index 5, 6)
                try:
                    kl_mean = float(row.get("kl_mean", "N/A"))
                except (ValueError, TypeError):
                    kl_mean = None
                try:
                    kl_max = float(row.get("kl_max", "N/A"))
                except (ValueError, TypeError):
                    kl_max = None

                configs.append((name, size_gb, ppl, speed, category, kl_mean, kl_max))
            else:
                # Experiment results.tsv format
                quant = row["quant_type"].strip()
                status = row.get("status", "").strip()

                name = _FRIENDLY_NAMES.get(quant, quant)
                category = _CATEGORY_MAP.get(quant, "apex")

                if status == "discard":
                    category = "discard"

                try:
                    size_gb = float(row["size_mb"]) / 1024.0
                except (ValueError, KeyError):
                    size_gb = None
                try:
                    ppl = float(row["perplexity"])
                except (ValueError, KeyError):
                    ppl = None
                try:
                    speed = float(row["tokens_per_sec"])
                    if speed <= 0:
                        speed = None
                except (ValueError, KeyError):
                    speed = None

                configs.append((name, size_gb, ppl, speed, category, None, None))

    return configs


def get_configs(tsv_path):
    """Return config list -- from TSV if available, otherwise defaults."""
    if tsv_path and os.path.isfile(tsv_path):
        print(f"Loading data from: {tsv_path}")
        return load_tsv(tsv_path)
    print("Using hardcoded default data.")
    return list(DEFAULT_CONFIGS)


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------

def pareto_frontier_2d(points, minimize_x=True, minimize_y=True):
    """
    Given a list of (x, y, label) tuples, return the subset on the Pareto
    frontier sorted by x.  By default both objectives are minimised.
    """
    sorted_pts = sorted(points, key=lambda p: p[0],
                        reverse=(not minimize_x))
    frontier = []
    best_y = float("inf") if minimize_y else float("-inf")
    for pt in sorted_pts:
        y = pt[1]
        if (minimize_y and y <= best_y) or (not minimize_y and y >= best_y):
            frontier.append(pt)
            best_y = y
    return sorted(frontier, key=lambda p: p[0])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_style(ax, title, xlabel, ylabel):
    """Apply consistent professional styling to an axes object."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)


def _add_reference_line(ax, y=Q8_0_PPL):
    """Add a horizontal dashed reference line at Q8_0 perplexity."""
    ax.axhline(y=y, color="#757575", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.annotate(
        f"Q8_0 baseline ({y:.3f})",
        xy=(ax.get_xlim()[1], y),
        xytext=(-6, 6),
        textcoords="offset points",
        fontsize=8,
        color="#757575",
        ha="right",
        va="bottom",
    )


def _offset_label(x, y, existing, x_range, y_range):
    """Compute a small offset to reduce label overlap."""
    dx = x_range * 0.015
    dy = y_range * 0.025
    ox, oy = dx, dy
    # Simple collision avoidance: nudge if too close to an existing label
    for ex, ey in existing:
        if abs(x - ex) < x_range * 0.06 and abs(y - ey) < y_range * 0.06:
            oy += dy
    return ox, oy


# ---------------------------------------------------------------------------
# Plot 1: Perplexity vs Size
# ---------------------------------------------------------------------------

def plot_ppl_vs_size(configs, output_dir):
    """Generate pareto_ppl_vs_size.png."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Separate plottable vs skipped
    plotted_labels = []
    pareto_candidates = []

    for name, size, ppl, _speed, cat in configs:
        if size is None or ppl is None:
            continue
        if cat == "discard":
            continue

        color = COLORS.get(cat, "#999999")
        marker = MARKERS.get(cat, "o")
        label = CATEGORY_LABELS.get(cat)

        # Only add legend entry once per category
        if label in ax.get_legend_handles_labels()[1]:
            label = None

        ax.scatter(size, ppl, color=color, marker=marker, s=80, zorder=5,
                   edgecolors="white", linewidths=0.5, label=label)
        plotted_labels.append((size, ppl, name))
        pareto_candidates.append((size, ppl, name))

    # Pareto frontier (minimise size, minimise PPL)
    if pareto_candidates:
        frontier = pareto_frontier_2d(pareto_candidates,
                                      minimize_x=True, minimize_y=True)
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, color="#1565C0", linewidth=1.5, linestyle="-",
                alpha=0.6, zorder=3, label="Pareto Frontier")

    # Labels
    if plotted_labels:
        x_vals = [p[0] for p in plotted_labels]
        y_vals = [p[1] for p in plotted_labels]
        x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 1.0
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0.1
        placed = []
        for x, y, name in plotted_labels:
            ox, oy = _offset_label(x, y, placed, x_range, y_range)
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(ox * 40, oy * 20), fontsize=8, ha="left",
                        arrowprops=dict(arrowstyle="-", color="#BDBDBD",
                                        lw=0.5))
            placed.append((x, y))

    _apply_style(ax, "APEX Quantization: Perplexity vs Model Size",
                 "Model Size (GB)", "Perplexity (lower is better)")
    _add_reference_line(ax)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "pareto_ppl_vs_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2: Perplexity vs Speed
# ---------------------------------------------------------------------------

def plot_ppl_vs_speed(configs, output_dir):
    """Generate pareto_ppl_vs_speed.png."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted_labels = []
    pareto_candidates = []

    for name, _size, ppl, speed, cat in configs:
        if speed is None or ppl is None or speed <= 0:
            continue
        if cat == "discard":
            continue

        color = COLORS.get(cat, "#999999")
        marker = MARKERS.get(cat, "o")
        label = CATEGORY_LABELS.get(cat)

        if label in ax.get_legend_handles_labels()[1]:
            label = None

        ax.scatter(speed, ppl, color=color, marker=marker, s=80, zorder=5,
                   edgecolors="white", linewidths=0.5, label=label)
        plotted_labels.append((speed, ppl, name))
        pareto_candidates.append((speed, ppl, name))

    # Pareto frontier (maximise speed => minimise -speed, minimise PPL)
    if pareto_candidates:
        neg_candidates = [(-s, p, n) for s, p, n in pareto_candidates]
        frontier_neg = pareto_frontier_2d(neg_candidates,
                                          minimize_x=True, minimize_y=True)
        frontier = [(-p[0], p[1], p[2]) for p in frontier_neg]
        frontier.sort(key=lambda p: p[0])
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, color="#1565C0", linewidth=1.5, linestyle="-",
                alpha=0.6, zorder=3, label="Pareto Frontier")

    # Labels
    if plotted_labels:
        x_vals = [p[0] for p in plotted_labels]
        y_vals = [p[1] for p in plotted_labels]
        x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 1.0
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0.1
        placed = []
        for x, y, name in plotted_labels:
            ox, oy = _offset_label(x, y, placed, x_range, y_range)
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(ox * 40, oy * 20), fontsize=8, ha="left",
                        arrowprops=dict(arrowstyle="-", color="#BDBDBD",
                                        lw=0.5))
            placed.append((x, y))

    _apply_style(ax, "APEX Quantization: Perplexity vs Inference Speed",
                 "Speed (tokens/sec, higher is better)",
                 "Perplexity (lower is better)")
    _add_reference_line(ax)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "pareto_ppl_vs_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3: Grouped bar comparison
# ---------------------------------------------------------------------------

def plot_comparison_bars(configs, output_dir):
    """Generate comparison_bars.png with grouped bars for key configs.

    Includes Perplexity, Size, Speed, HellaSwag, and MMLU groups when
    accuracy benchmark data is available in the config tuples.
    """
    # Build lookup -- config tuples may have 5 or 10 elements
    lookup = {}
    for cfg in configs:
        name = cfg[0]
        if name in _BAR_CONFIGS:
            entry = {"size": cfg[1], "ppl": cfg[2], "speed": cfg[3]}
            # Accuracy benchmarks (elements 5-9 if present)
            if len(cfg) > 5:
                entry["hellaswag"] = cfg[5]
                entry["mmlu"] = cfg[7] if len(cfg) > 7 else None
            else:
                entry["hellaswag"] = None
                entry["mmlu"] = None
            lookup[name] = entry

    # Filter to configs that actually exist in the data
    names = [n for n in _BAR_CONFIGS if n in lookup]
    if not names:
        print("  [!] No matching configs for bar chart, skipping.")
        return None

    ppls   = [lookup[n]["ppl"]   for n in names]
    sizes  = [lookup[n]["size"]  for n in names]
    speeds = [lookup[n]["speed"] for n in names]
    hellaswags = [lookup[n].get("hellaswag") for n in names]
    mmlus      = [lookup[n].get("mmlu") for n in names]

    # Replace None speeds with 0 for plotting
    speeds = [s if s is not None else 0.0 for s in speeds]
    hellaswags = [h if h is not None else 0.0 for h in hellaswags]
    mmlus      = [m if m is not None else 0.0 for m in mmlus]

    # Normalise PPL to make it visually comparable with other metrics.
    # Scale so Q8_0 baseline = 1.0; display as "Perplexity (normalised)".
    ppl_norm = [p / Q8_0_PPL if p is not None else 0.0 for p in ppls]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    n_configs = len(names)
    group_labels = ["Perplexity (norm.)", "Size (GB)", "Speed (t/s)",
                    "HellaSwag (%)", "MMLU (%)"]
    n_groups = len(group_labels)

    # Bar positions
    bar_width = 0.10
    group_gap = 0.18
    group_width = n_configs * bar_width

    # Colours for each config -- follow the same category scheme
    bar_colors = []
    for n in names:
        for cfg in configs:
            if cfg[0] == n:
                bar_colors.append(COLORS.get(cfg[4], "#999999"))
                break

    # Five groups of metrics
    data_groups = [ppl_norm, sizes, speeds, hellaswags, mmlus]

    for gi, (metric_vals, glabel) in enumerate(zip(data_groups, group_labels)):
        group_center = gi * (group_width + group_gap)
        for ci, (val, color) in enumerate(zip(metric_vals, bar_colors)):
            x = group_center + ci * bar_width
            bar = ax.bar(x, val, width=bar_width * 0.85, color=color,
                         edgecolor="white", linewidth=0.5)
            # Value label on top
            if val > 0:
                if gi == 0:
                    fmt = f"{val:.2f}"
                elif gi == 1:
                    fmt = f"{val:.1f}"
                elif gi == 2:
                    fmt = f"{val:.0f}"
                else:
                    fmt = f"{val:.1f}"
                ax.text(x, val + max(metric_vals) * 0.02, fmt,
                        ha="center", va="bottom", fontsize=6, rotation=45)

    # X-axis: group labels centred under each group
    group_centers = []
    for gi in range(n_groups):
        gc = gi * (group_width + group_gap) + (group_width - bar_width) / 2
        group_centers.append(gc)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10)

    # Legend (one entry per config)
    handles = []
    for ci, (n, c) in enumerate(zip(names, bar_colors)):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=c, edgecolor="white",
                                     linewidth=0.5, label=n))
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              ncol=2, framealpha=0.9)

    _apply_style(ax, "APEX vs Standard Quantization", "", "")
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=8))

    fig.tight_layout()
    out = os.path.join(output_dir, "comparison_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto frontier plots for APEX quantization results."
    )
    parser.add_argument(
        "--tsv",
        default="results.tsv",
        help="Path to results TSV file (default: results.tsv in repo root).",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots",
        help="Directory to save generated PNG files (default: ./plots/).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    configs_raw = get_configs(args.tsv)
    # 5-tuples (name, size, ppl, speed, category) for Pareto plots
    configs_5 = [(c[0], c[1], c[2], c[3], c[4]) for c in configs_raw]
    print(f"Loaded {len(configs_5)} configurations.\n")

    generated = []

    print("[1/3] Generating Pareto: Perplexity vs Model Size ...")
    p = plot_ppl_vs_size(configs_5, output_dir)
    if p:
        generated.append(p)
        print(f"      Saved: {p}")

    print("[2/3] Generating Pareto: Perplexity vs Inference Speed ...")
    p = plot_ppl_vs_speed(configs_5, output_dir)
    if p:
        generated.append(p)
        print(f"      Saved: {p}")

    print("[3/3] Generating grouped bar comparison ...")
    # Pass full tuples so the bar chart can access accuracy benchmarks
    p = plot_comparison_bars(configs_raw, output_dir)
    if p:
        generated.append(p)
        print(f"      Saved: {p}")

    print(f"\nDone. {len(generated)} plot(s) generated in {output_dir}/")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
