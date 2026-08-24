from __future__ import annotations
import pathlib
from typing import Dict, Any, List

# matplotlib must be deterministic – use Agg, fixed style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#0d0f14",
    "figure.facecolor": "#0d0f14",
    "axes.edgecolor": "#2a2f3a",
    "axes.labelcolor": "#e6e8ee",
    "xtick.color": "#a1a1aa",
    "ytick.color": "#a1a1aa",
    "text.color": "#e6e8ee",
    "grid.color": "#1e232f",
    "grid.alpha": 0.6,
})

COLORS = {
    "accent": "#a78bfa",
    "cyan": "#22d3ee",
    "green": "#34d399",
    "amber": "#fbbf24",
    "pink": "#f472b6",
    "bg": "#0d0f14",
}

def _save(fig, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    return path

def chart_performance(perf: Dict[str, Any], out_path: pathlib.Path):
    if not perf or "error" in perf:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Baseline Performance (Measured vs Derived)", fontsize=12, color="#e6e8ee", y=1.02)

    # left: latency breakdown
    ax = axes[0]
    labels = ["Prefill\n(TTFT≈)", "Decode", "Total"]
    vals = [perf.get("prefill_time_s", 0), perf.get("decode_time_s", 0), perf.get("total_time_s", 0)]
    bars = ax.bar(labels, vals, color=[COLORS["amber"], COLORS["cyan"], COLORS["accent"]], edgecolor="#2a2f3a")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{v:.2f}s", ha="center", va="bottom", fontsize=9, color="#e6e8ee")
    ax.set_ylabel("Seconds")
    ax.set_title("Timing breakdown", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # right: tok/s
    ax = axes[1]
    labels2 = ["Prefill", "Decode", "Overall"]
    vals2 = [perf.get("prefill_tok_per_s", 0), perf.get("decode_tok_per_s", 0), perf.get("overall_tok_per_s", 0)]
    bars = ax.bar(labels2, vals2, color=[COLORS["pink"], COLORS["green"], COLORS["accent"]], edgecolor="#2a2f3a")
    for b, v in zip(bars, vals2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#e6e8ee")
    ax.set_ylabel("tok/s")
    ax.set_title("Throughput (tok/s) – decode is primary", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.text(0.5, -0.02, "Note: TTFT is streaming-approximation (first token via TextIteratorStreamer), not true prefill.  •  Fixed ordering is a stated limitation.", ha="center", fontsize=7, color="#9ca3af", style="italic")
    return _save(fig, out_path)

def chart_latency_percentiles(perf: Dict[str, Any], out_path: pathlib.Path):
    if not perf or "error" in perf:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["Mean", "Median", "P95", "P99", "Min", "Max"]
    vals = [
        perf.get("inter_token_latency_ms_mean", 0),
        perf.get("inter_token_latency_ms_median", 0),
        perf.get("inter_token_latency_ms_p95", 0),
        perf.get("inter_token_latency_ms_p99", 0) or 0,
        perf.get("inter_token_latency_ms_min", 0),
        perf.get("inter_token_latency_ms_max", 0),
    ]
    colors = [COLORS["accent"], COLORS["cyan"], COLORS["amber"], COLORS["pink"], COLORS["green"], "#f87171"]
    bars = ax.bar(metrics, vals, color=colors, edgecolor="#2a2f3a")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}", ha="center", fontsize=8, color="#e6e8ee")
    ax.set_ylabel("ms")
    ax.set_title("Inter-token latency distribution (ms)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _save(fig, out_path)

def chart_context_scaling(ctx: Dict[str, Any], out_path: pathlib.Path):
    if not ctx or "error" in ctx or not ctx.get("points"):
        return None
    pts = ctx["points"]
    xs = [p["num_input_tokens"] for p in pts]
    ys = [p["tok_per_sec"] for p in pts]
    labels = [p["label"] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("H1 – Context Scaling: tok/s vs input length (descriptive only)", fontsize=11, color="#e6e8ee", y=1.02)

    ax = axes[0]
    ax.plot(xs, ys, marker="o", color=COLORS["accent"], linewidth=2.5, markersize=8, markerfacecolor=COLORS["cyan"], markeredgecolor="white")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0,10), ha="center", fontsize=7, color="#cbd5e1")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("tok/s (decode)")
    ax.set_title(f"Raw sweep – k={ctx.get('fit_k'):.1f} α={ctx.get('fit_alpha'):.2f} r={ctx.get('pearson_r'):.2f}" if ctx.get("fit_k") else "Raw sweep")
    ax.grid(True, linestyle="--", alpha=0.4)

    # log-log fit overlay if available
    ax2 = axes[1]
    if ctx.get("fit_k") and ctx.get("fit_alpha"):
        import numpy as np
        k, a = ctx["fit_k"], ctx["fit_alpha"]
        xs_fit = np.linspace(min(xs)*0.9, max(xs)*1.1, 100)
        ys_fit = k / (xs_fit ** a)
        ax2.scatter(xs, ys, color=COLORS["accent"], s=60, edgecolor="white", zorder=3, label="measured")
        ax2.plot(xs_fit, ys_fit, color=COLORS["amber"], linestyle="--", linewidth=2, label=f"fit: r=k/n^α")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Input tokens (log)")
        ax2.set_ylabel("tok/s (log)")
        ax2.set_title(f"Power-law fit (Pearson r={ctx.get('pearson_r'):.2f})")
        ax2.legend(framealpha=0.3, facecolor="#111827")
        ax2.grid(True, linestyle="--", alpha=0.4, which="both")
    else:
        # fallback: total time vs tokens
        tot = [p["total_time_s"] for p in pts]
        ax2.bar(labels, tot, color=COLORS["cyan"], edgecolor="#2a2f3a")
        ax2.set_ylabel("Total time (s)")
        ax2.set_title("Total time per context length")
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.text(0.5, -0.02, ctx.get("conclusion","Descriptive only. No pass/fail about quality."), ha="center", fontsize=7, color="#9ca3af", style="italic", wrap=True)
    return _save(fig, out_path)

def chart_decode_position(dec: Dict[str, Any], out_path: pathlib.Path):
    if not dec or "error" in dec or not dec.get("latencies_ms"):
        return None
    lats = dec["latencies_ms"]
    binned = dec.get("binned_means_ms", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("H2 – Decode-position trace: per-token latency vs output position", fontsize=11, color="#e6e8ee", y=1.02)

    ax = axes[0]
    xs = list(range(1, len(lats)+1))
    ax.plot(xs, lats, color=COLORS["cyan"], alpha=0.35, linewidth=0.8)
    # moving average
    import numpy as np
    if len(lats) > 10:
        window = 20
        ma = np.convolve(lats, np.ones(window)/window, mode="valid")
        ax.plot(xs[window-1:], ma, color=COLORS["accent"], linewidth=2, label=f"{window}-token MA")
    ax.set_xlabel("Output token position")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(dec.get("conclusion","")[:60])
    ax.legend(framealpha=0.3, facecolor="#111827")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    if binned:
        bins_labels = [f"{i*50+1}-{(i+1)*50}" for i in range(len(binned))]
        bars = ax.bar(bins_labels, binned, color=COLORS["amber"], edgecolor="#2a2f3a")
        for b, v in zip(bars, binned):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}", ha="center", fontsize=8, color="#e6e8ee")
        ax.set_xlabel("Token window")
        ax.set_ylabel("Mean latency (ms)")
        ax.set_title(f"Slope {dec.get('slope_ms_per_token',0):.4f} ms/token – binned (50-tok)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        ax.text(0.5,0.5,"No binned data", ha="center", color="#9ca3af")

    return _save(fig, out_path)

def chart_accuracy(acc: Dict[str, Any], out_path: pathlib.Path):
    if not acc or "error" in acc:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Accuracy / Grounding Probe (synthetic, descriptive only)", fontsize=11, color="#e6e8ee", y=1.02)

    # left: score gauge
    ax = axes[0]
    score = acc.get("accuracy_score", 0)
    # donut
    wedges, _ = ax.pie([score, 1-score], colors=[COLORS["green"], "#1f2937"], startangle=90, counterclock=False, wedgeprops=dict(width=0.35, edgecolor="#0d0f14"))
    ax.text(0, 0, f"{score*100:.0f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#e6e8ee")
    ax.text(0, -0.22, f"{len(acc.get('matched_facts',[]))}/{len(acc.get('expected_facts',[]))} facts", ha="center", fontsize=9, color="#9ca3af")
    ax.set_title("Audit score")

    # right: per-fact
    ax = axes[1]
    details = acc.get("audit_details", [])
    if details:
        facts_short = [d["fact"][:28]+"…" if len(d["fact"])>30 else d["fact"] for d in details]
        hits = [1 if d["matched"] else 0 for d in details]
        colors = [COLORS["green"] if h else "#f87171" for h in hits]
        bars = ax.barh(facts_short, hits, color=colors, edgecolor="#2a2f3a")
        ax.set_xlim(0,1.2)
        for b, h in zip(bars, hits):
            lab = "✓" if h else "✗"
            ax.text(0.05, b.get_y()+b.get_height()/2, lab, va="center", ha="left", fontsize=14, color="white", fontweight="bold")
        ax.set_title("Per-fact audit")
        ax.set_xlabel("Matched (1) / Missed (0)")
    else:
        ax.text(0.5,0.5,"No audit details", ha="center", color="#9ca3af")
    return _save(fig, out_path)

def generate_all_charts(run_dir: pathlib.Path, perf, ctx, dec, acc):
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True, parents=True)
    made = []
    try:
        p = charts_dir / "01_performance.png"
        if chart_performance(perf, p):
            made.append(p)
    except Exception as e:
        print(f"perf chart failed: {e}")
    try:
        p = charts_dir / "02_latency.png"
        if chart_latency_percentiles(perf, p):
            made.append(p)
    except Exception as e:
        print(f"latency chart failed: {e}")
    try:
        p = charts_dir / "03_context_scaling.png"
        if chart_context_scaling(ctx, p):
            made.append(p)
    except Exception as e:
        print(f"context chart failed: {e}")
    try:
        p = charts_dir / "04_decode_position.png"
        if chart_decode_position(dec, p):
            made.append(p)
    except Exception as e:
        print(f"decode chart failed: {e}")
    try:
        p = charts_dir / "05_accuracy.png"
        if chart_accuracy(acc, p):
            made.append(p)
    except Exception as e:
        print(f"acc chart failed: {e}")
    return made
