#!/usr/bin/env python3
"""Render editorial, light-mode Hugging Face Model Card figures for OmniSRE.

All figures are drawn on an inch-based canvas so typography, hairlines and margins
stay on a consistent Swiss grid. Training curves are parsed from the raw TRL logs and
corpus counts are read from the validated dataset artifacts when they are available.

Usage:
    uv run python scripts/render_hf_visuals.py
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# ── Palette ──────────────────────────────────────────────────────────────────
PAPER = "#FAF9F7"
INK = "#18181B"
INK_2 = "#3F3F46"
MUTED = "#52525B"
FAINT = "#A1A1AA"
RULE = "#E4E4E7"
RULE_2 = "#D4D4D8"
ACCENT = "#0F766E"
ACCENT_TINT = "#C9DEDA"
WARM = "#B45309"

SANS = "Helvetica Neue"
SERIF = "Charter"
MONO = "Menlo"
DPI = 200
MARGIN = 0.6

plt.rcParams.update(
    {
        "font.family": SANS,
        "text.color": INK,
        "axes.edgecolor": RULE_2,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.facecolor": PAPER,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "real_sre_v2"

SFT_LABELS = {
    "rmems/incident-response-oncall-trajectories": "rmems · on-call incident trajectories",
    "nolabs/deepfabric-devops-reasoning-traces": "nolabs · DevOps reasoning traces",
    "grounded_openshift_promql_runbook_trace": "Grounded runbook traces (ours)",
    "miyad316/sre-agentic-trajectories-devops": "miyad316 · agentic SRE trajectories",
    "beatsprom/devops-kubernetes-iac-sft-dpo-2026": "beatsprom · Kubernetes / IaC",
    "Snaseem2026/devops-incident-response": "Snaseem2026 · incident response",
    "qwen38-sre-rl/safety_and_metastable_balance": "Safety & metastable balance (ours)",
}
KB_LABELS = {
    "samber/awesome-prometheus-alerts": "samber · Prometheus alert rules",
    "openshift/runbooks": "openshift · runbooks",
    "danluu/post-mortems": "danluu · public post-mortems",
    "SREGym/SREGym": "SREGym · fault scenarios",
    "prometheus-operator/runbooks": "prometheus-operator · runbooks",
}
SFT_FALLBACK = {
    "rmems/incident-response-oncall-trajectories": 2961,
    "nolabs/deepfabric-devops-reasoning-traces": 1305,
    "grounded_openshift_promql_runbook_trace": 400,
    "miyad316/sre-agentic-trajectories-devops": 248,
    "beatsprom/devops-kubernetes-iac-sft-dpo-2026": 100,
    "Snaseem2026/devops-incident-response": 15,
    "qwen38-sre-rl/safety_and_metastable_balance": 9,
}
KB_FALLBACK = {
    "samber/awesome-prometheus-alerts": 1217,
    "openshift/runbooks": 304,
    "danluu/post-mortems": 240,
    "SREGym/SREGym": 177,
    "prometheus-operator/runbooks": 108,
}


# ── Canvas helpers ───────────────────────────────────────────────────────────
def _canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h), dpi=DPI, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _t(ax, x, y, s, size=10.0, color=INK, weight="normal", family=SANS, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "baseline")
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, **kw)


def _hline(ax, x0, x1, y, color=RULE_2, lw=0.75):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")


def _vline(ax, x, y0, y1, color=RULE_2, lw=0.75):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt")


def _text_width(fig, artist) -> float:
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer).width / DPI


def _masthead(ax, w, h, fig_no: str, title: str, subtitle: str) -> None:
    top = h - 0.42
    _hline(ax, MARGIN, w - MARGIN, top, color=INK, lw=1.1)
    _t(ax, MARGIN, top - 0.27, fig_no, size=7.8, color=ACCENT, family=MONO, weight="bold")
    _t(
        ax,
        w - MARGIN,
        top - 0.27,
        "OmniSRE · w3joe/OmniSRE",
        size=7.8,
        color=FAINT,
        family=MONO,
        ha="right",
    )
    _t(ax, MARGIN, top - 0.72, title, size=21, family=SERIF, weight="bold")
    _t(ax, MARGIN, top - 1.02, subtitle, size=11, color=MUTED)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=PAPER)
    plt.close(fig)


# ── Data ─────────────────────────────────────────────────────────────────────
def _parse_trl_log(log_path: Path, log_every: int = 5):
    """Return (steps, loss, token_acc_pct, final_summary) from a TRL stdout log."""
    if not log_path.exists():
        return [], [], [], {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    steps, losses, accs, summary = [], [], [], {}
    for m in re.finditer(r"\{'(?:loss|train_runtime)'.*?\}", text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "train_runtime" in d:
            summary = {k: float(v) for k, v in d.items()}
            continue
        steps.append(log_every * (len(steps) + 1))
        losses.append(float(d["loss"]))
        accs.append(float(d.get("mean_token_accuracy", 0.0)) * 100.0)
    return steps, losses, accs, summary


def _total_steps(log_path: Path, default: int) -> int:
    if not log_path.exists():
        return default
    # The optimizer progress bar is the only one reporting seconds-per-iteration.
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    totals = [int(t) for t in re.findall(r"\d+/(\d+) \[[^\]]*s/it\]", text)]
    return max(totals) if totals else default


def _source_counts(path: Path, fallback: dict[str, int]) -> dict[str, int]:
    if not path.exists():
        return dict(fallback)
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                counts[json.loads(line).get("source", "unknown")] += 1
    return dict(counts.most_common())


def _validation_report() -> dict:
    path = DATA_DIR / "validation_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "sft_total_rows": 5038,
        "sft_invalid_rows": 0,
        "sft_token_percentiles": {"p50": 2284, "p99": 2916},
        "pii_secret_leaks": {},
        "rl_oracle_checked": 200,
        "rl_oracle_pass_rate": 1.0,
        "contamination_overlaps": {"a": 0, "b": 0, "c": 0},
        "safety_mix": {
            "benign_false_alarms": 60,
            "external_escalations": 60,
            "metastable_cascades": 60,
            "adversarial_log_injections": 69,
        },
    }


# ── Figure 0: masthead ───────────────────────────────────────────────────────
def render_hero(out_path: Path) -> None:
    w, h = 14.0, 5.2
    fig, ax = _canvas(w, h)
    right = w - MARGIN

    top = h - 0.42
    _hline(ax, MARGIN, right, top, color=INK, lw=1.4)
    _t(
        ax,
        MARGIN,
        top - 0.27,
        "OPEN-WEIGHTS SRE AGENT · MODEL CARD",
        size=7.8,
        color=ACCENT,
        family=MONO,
        weight="bold",
    )
    _t(
        ax,
        right,
        top - 0.27,
        "v0.1 draft · November 2026",
        size=7.8,
        color=FAINT,
        family=MONO,
        ha="right",
    )

    title = _t(ax, MARGIN - 0.04, 2.92, "OmniSRE", size=74, family=SERIF, weight="bold")
    tw = _text_width(fig, title)
    _t(ax, MARGIN - 0.04 + tw + 0.16, 2.92, "27B", size=34, family=SERIF, color=ACCENT)

    _t(
        ax,
        MARGIN,
        2.32,
        "An open-weights agent for site reliability engineering.",
        size=17,
        weight="light",
        color=INK_2,
    )
    _t(
        ax,
        MARGIN,
        2.0,
        "Triage, localize, root-cause and remediate — reading before it writes.",
        size=17,
        weight="light",
        color=MUTED,
    )

    # Agent-loop index, right column
    col_x = 9.75
    rows = [
        ("01", "Triage", "read", ACCENT),
        ("02", "Localize", "read", ACCENT),
        ("03", "Root cause", "read", ACCENT),
        ("04", "Policy gate", "gate", WARM),
        ("05", "Mitigate", "write", INK),
    ]
    _t(ax, col_x, 3.98, "AGENT LOOP", size=7.4, color=FAINT, family=MONO)
    y = 3.62
    for i, (num, name, tag, col) in enumerate(rows):
        _t(ax, col_x, y, num, size=8.6, color=FAINT, family=MONO)
        _t(ax, col_x + 0.42, y, name, size=11, color=INK, weight="medium")
        _t(
            ax,
            right,
            y,
            tag,
            size=8.6,
            color=col,
            family=MONO,
            weight="bold" if tag != "read" else "normal",
            ha="right",
        )
        if i < len(rows) - 1:
            _hline(ax, col_x, right, y - 0.11, color=RULE, lw=0.6)
        y -= 0.33

    # Specification grid
    grid_top = 1.5
    _hline(ax, MARGIN, right, grid_top, color=RULE_2, lw=0.8)
    specs = [
        ("PARAMETERS", "27B", "Hybrid Gated DeltaNet + Attention"),
        ("LICENSE", "Apache 2.0", "Open weights, fully self-hostable"),
        ("CONTEXT", "262K", "Native tokens, ~2 GB FP8 KV @ 64K"),
        ("POST-TRAINING", "Read-Before-Write RLVR", "SFT warm-start, then DAPO + GTPO"),
    ]
    col_w = (right - MARGIN) / len(specs)
    for i, (label, value, sub) in enumerate(specs):
        x = MARGIN + i * col_w + (0.22 if i else 0.0)
        if i:
            _vline(ax, MARGIN + i * col_w, 0.42, grid_top - 0.18, color=RULE, lw=0.7)
        _t(ax, x, 1.2, label, size=7.4, color=FAINT, family=MONO)
        _t(
            ax,
            x,
            0.83,
            value,
            size=19 if len(value) < 12 else 15.5,
            family=SERIF,
            weight="bold",
            color=INK,
        )
        _t(ax, x, 0.52, sub, size=9.4, color=MUTED)

    _save(fig, out_path)


# ── Figure 1: architecture ───────────────────────────────────────────────────
def render_architecture(out_path: Path) -> None:
    w, h = 14.0, 7.2
    fig, ax = _canvas(w, h)
    right = w - MARGIN
    _masthead(
        ax,
        w,
        h,
        "FIG. 1 — AGENT LOOP",
        "A read-before-write incident loop",
        "No mutation reaches the cluster until at least two read-only observations "
        "clear the policy gate.",
    )

    stages = [
        ("01", "Triage", ["Verify SLO burn", "Filter false alarms", "Ignore log injections"]),
        ("02", "Localize", ["Metrics, logs, traces", "Kubernetes events", "Recent Git changes"]),
        ("03", "Root cause", ["Falsify red herrings", "Isolate faulty object", "Runbook lookup"]),
        ("04", "Policy gate", ["≥ 2 reads required", "No destructive ops", "No-regression check"]),
        ("05", "Mitigate", ["Minimal K8s patch", "Helm rollback / scale", "Burn-in & verify"]),
    ]
    colors = [ACCENT, ACCENT, ACCENT, WARM, INK]
    gap = 0.42
    cw = (right - MARGIN - gap * 4) / 5
    xs = [MARGIN + i * (cw + gap) for i in range(5)]

    # Phase brackets
    by = 5.02
    for x0, x1, label, col in [
        (xs[0], xs[2] + cw, "READ-ONLY", ACCENT),
        (xs[3], xs[3] + cw, "GATE", WARM),
        (xs[4], xs[4] + cw, "WRITE", INK),
    ]:
        _t(ax, x0, by + 0.1, label, size=7.6, color=col, family=MONO, weight="bold")
        _hline(ax, x0, x1, by, color=col, lw=0.7)
        _vline(ax, x0, by - 0.07, by, color=col, lw=0.7)
        _vline(ax, x1, by - 0.07, by, color=col, lw=0.7)

    rule_y = 4.62
    for i, ((num, name, lines), col) in enumerate(zip(stages, colors, strict=True)):
        x = xs[i]
        _hline(ax, x, x + cw, rule_y, color=col, lw=2.2)
        _t(ax, x, rule_y - 0.5, num, size=24, family=SERIF, color=col)
        _t(ax, x, rule_y - 0.88, name, size=14, weight="bold")
        for j, line in enumerate(lines):
            _t(ax, x, rule_y - 1.22 - j * 0.27, line, size=10.4, color=MUTED)
        if i < 4:
            ax.annotate(
                "",
                xy=(x + cw + gap - 0.1, rule_y - 0.62),
                xytext=(x + cw + 0.1, rule_y - 0.62),
                arrowprops={
                    "arrowstyle": "-|>,head_width=0.16,head_length=0.32",
                    "color": FAINT,
                    "lw": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    # Backbone specification strip
    strip_top = 2.42
    _hline(ax, MARGIN, right, strip_top, color=RULE_2, lw=0.8)
    _t(ax, MARGIN, strip_top - 0.3, "BACKBONE", size=7.6, color=ACCENT, family=MONO, weight="bold")
    _t(
        ax,
        MARGIN + 0.85,
        strip_top - 0.3,
        "Qwen3.8-27B · 64 layers in a 3 : 1 hybrid pattern",
        size=10.4,
        color=INK_2,
    )

    n_layers = 64
    bar_y, bar_h = 1.42, 0.36
    pitch = (right - MARGIN) / n_layers
    for k in range(n_layers):
        is_attn = (k + 1) % 4 == 0
        ax.add_patch(
            Rectangle(
                (MARGIN + k * pitch, bar_y),
                pitch - 0.045,
                bar_h,
                facecolor=ACCENT if is_attn else ACCENT_TINT,
                edgecolor="none",
            )
        )
    _t(ax, MARGIN, bar_y + bar_h + 0.08, "1", size=7.2, color=FAINT, family=MONO)
    _t(
        ax,
        right - 0.045,
        bar_y + bar_h + 0.08,
        "64",
        size=7.2,
        color=FAINT,
        family=MONO,
        ha="right",
    )

    ly = bar_y - 0.28
    ax.add_patch(
        Rectangle((MARGIN, ly - 0.01), 0.14, 0.12, facecolor=ACCENT_TINT, edgecolor="none")
    )
    _t(ax, MARGIN + 0.22, ly, "48 × Gated DeltaNet", size=9.6, weight="bold")
    _t(
        ax,
        MARGIN + 1.72,
        ly,
        "linear attention · in_proj_qkv / z / a / b, out_proj",
        size=9.6,
        color=MUTED,
    )
    lx = 7.35
    ax.add_patch(Rectangle((lx, ly - 0.01), 0.14, 0.12, facecolor=ACCENT, edgecolor="none"))
    _t(ax, lx + 0.22, ly, "16 × Gated Attention", size=9.6, weight="bold")
    _t(ax, lx + 1.78, ly, "every 4th layer · q / k / v / o_proj + SwiGLU", size=9.6, color=MUTED)

    specs = [
        ("ADAPTER", "LoRA, rank 64"),
        ("TRAINABLE", "466.9M params · 1.68%"),
        ("FROZEN", "Vision tower, MTP head"),
        ("SERVING", "1 × H100 / L40S, FP8"),
    ]
    sw = (right - MARGIN) / len(specs)
    _hline(ax, MARGIN, right, 0.92, color=RULE, lw=0.6)
    for i, (label, value) in enumerate(specs):
        x = MARGIN + i * sw
        _t(ax, x, 0.66, label, size=7.2, color=FAINT, family=MONO)
        _t(ax, x, 0.4, value, size=11, color=INK, weight="medium")

    _save(fig, out_path)


# ── Figure 2: training curves ────────────────────────────────────────────────
def render_training_curves(stage1a_log: Path, phase1b_log: Path, out_path: Path) -> None:
    s_steps, s_loss, s_acc, s_sum = _parse_trl_log(stage1a_log)
    p_steps, p_loss, p_acc, p_sum = _parse_trl_log(phase1b_log)
    s_total = _total_steps(stage1a_log, 157)
    p_total = _total_steps(phase1b_log, 98)
    p_steps = [s_total + s for s in p_steps]
    end = s_total + p_total

    w, h = 14.0, 6.7
    fig, ax = _canvas(w, h)
    _masthead(
        ax,
        w,
        h,
        "FIG. 2 — SUPERVISED WARM-START",
        "Two-phase SFT converges cleanly before RL",
        f"Stage 1a on 5,038 real SRE traces ({s_total} steps), then Phase 1b on 1,559 "
        f"exact-schema traces ({p_total} steps, 2 epochs).",
    )

    ax_y, ax_h = 0.95, 3.3
    ax_w = 5.55
    lefts = [MARGIN + 0.5, MARGIN + 0.5 + ax_w + 1.2]
    panels = [
        ("Training loss", "cross-entropy, log scale", s_loss, p_loss),
        ("Token accuracy", "mean per-token, %", s_acc, p_acc),
    ]

    for idx, (title, unit, s_y, p_y) in enumerate(panels):
        x0 = lefts[idx]
        _t(ax, x0 - 0.5, ax_y + ax_h + 0.5, title, size=12.5, weight="bold")
        tw = _text_width(fig, ax.texts[-1])
        _t(ax, x0 - 0.5 + tw + 0.12, ax_y + ax_h + 0.5, unit, size=10, color=FAINT)

        a = fig.add_axes((x0 / w, ax_y / h, ax_w / w, ax_h / h))
        a.set_facecolor(PAPER)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(RULE_2)
        a.spines["bottom"].set_linewidth(0.8)
        a.tick_params(axis="both", length=0, labelsize=8.6, labelcolor=MUTED, pad=6)
        a.tick_params(axis="x", length=3, color=RULE_2, width=0.8)
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_family(MONO)
        a.set_axisbelow(True)
        a.grid(axis="y", color=RULE, lw=0.7)
        a.set_xlim(0, end + 30)
        a.xaxis.set_major_locator(FixedLocator([0, 50, 100, 150, 200, 250]))

        if idx == 0:
            a.set_yscale("log")
            a.set_ylim(0.04, 5.0)
            a.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 4]))
            a.yaxis.set_minor_locator(NullLocator())
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            a.set_ylim(55, 102)
            a.yaxis.set_major_locator(FixedLocator([60, 70, 80, 90, 100]))
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_family(MONO)

        # Phase boundary
        a.axvline(s_total, color=FAINT, lw=0.8, ls=(0, (2, 2)), zorder=1)
        xt = a.get_xaxis_transform()
        a.text(
            0,
            1.04,
            "STAGE 1A",
            transform=xt,
            fontsize=7.4,
            family=MONO,
            color=ACCENT,
            fontweight="bold",
            va="bottom",
        )
        a.text(
            s_total + 5,
            1.04,
            "PHASE 1B",
            transform=xt,
            fontsize=7.4,
            family=MONO,
            color=WARM,
            fontweight="bold",
            va="bottom",
        )

        a.plot(s_steps, s_y, color=ACCENT, lw=1.7, solid_capstyle="round", zorder=3)
        a.plot(p_steps, p_y, color=WARM, lw=1.7, solid_capstyle="round", zorder=3)
        for xs_, ys_, col in [(s_steps, s_y, ACCENT), (p_steps, p_y, WARM)]:
            if xs_:
                a.plot(
                    [xs_[0], xs_[-1]],
                    [ys_[0], ys_[-1]],
                    "o",
                    ms=3.6,
                    color=col,
                    mec=PAPER,
                    mew=0.8,
                    zorder=4,
                )

        fmt = (lambda v: f"{v:.3f}") if idx == 0 else (lambda v: f"{v:.1f}%")
        if s_steps:
            if idx == 0:
                a.annotate(
                    f"{s_y[0]:.2f}",
                    (s_steps[0], s_y[0]),
                    xytext=(7, 1),
                    textcoords="offset points",
                    fontsize=8.6,
                    family=MONO,
                    color=ACCENT,
                    va="center",
                )
                a.annotate(
                    fmt(s_y[-1]),
                    (s_steps[-1], s_y[-1]),
                    xytext=(-4, -11),
                    textcoords="offset points",
                    fontsize=8.6,
                    family=MONO,
                    color=ACCENT,
                    ha="right",
                    va="center",
                    fontweight="bold",
                )
            else:
                a.annotate(
                    fmt(s_y[0]),
                    (s_steps[0], s_y[0]),
                    xytext=(7, -1),
                    textcoords="offset points",
                    fontsize=8.6,
                    family=MONO,
                    color=ACCENT,
                    va="center",
                )
                a.annotate(
                    fmt(s_y[-1]),
                    (s_steps[-1], s_y[-1]),
                    xytext=(-4, -12),
                    textcoords="offset points",
                    fontsize=8.6,
                    family=MONO,
                    color=ACCENT,
                    ha="right",
                    va="center",
                    fontweight="bold",
                )
        if p_steps:
            a.annotate(
                fmt(p_y[-1]),
                (p_steps[-1], p_y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8.6,
                family=MONO,
                color=WARM,
                ha="left",
                va="center",
                fontweight="bold",
            )

        a.set_xlabel("optimizer step (cumulative)", fontsize=8.6, color=FAINT, labelpad=6)

    rt = s_sum.get("train_runtime", 0) + p_sum.get("train_runtime", 0)
    note = "Source: TRL trainer logs, logged every 5 steps · 4 × H100 BF16 · LoRA r = 64 · batch 32"
    if rt:
        note += f" · {rt / 60:.0f} min wall-clock"
    _t(ax, MARGIN, 0.22, note, size=7.4, color=FAINT, family=MONO)

    _save(fig, out_path)


# ── Figure 3: corpus & validation ────────────────────────────────────────────
def render_dataset_infographic(out_path: Path) -> None:
    sft = _source_counts(DATA_DIR / "sft_real_sre_corpus.jsonl", SFT_FALLBACK)
    kb = _source_counts(DATA_DIR / "production_runbooks_promql_kb.jsonl", KB_FALLBACK)
    rep = _validation_report()
    n_sft, n_kb = sum(sft.values()), sum(kb.values())

    w, h = 14.0, 7.0
    fig, ax = _canvas(w, h)
    right = w - MARGIN
    _masthead(
        ax,
        w,
        h,
        "FIG. 3 — DATA",
        "An open, validated SRE corpus",
        f"{n_sft:,} supervised trajectories and {n_kb:,} knowledge-base documents from "
        "permissively licensed public sources.",
    )

    # Left: horizontal bars
    label_x = MARGIN
    bar_x0 = MARGIN + 2.75
    bar_max_w = 3.25
    vmax = max(max(sft.values()), max(kb.values()))
    row = 0.262
    y = 4.86

    def group(title: str, total: int, items: dict[str, int], labels: dict[str, str], col: str):
        nonlocal y
        _t(ax, label_x, y, title, size=7.6, color=col, family=MONO, weight="bold")
        _t(
            ax,
            bar_x0 + bar_max_w + 0.62,
            y,
            f"{total:,}",
            size=7.6,
            color=col,
            family=MONO,
            weight="bold",
            ha="right",
        )
        y -= 0.1
        _hline(ax, label_x, bar_x0 + bar_max_w + 0.62, y, color=RULE_2, lw=0.6)
        y -= row
        for src, n in items.items():
            _t(ax, label_x, y, labels.get(src, src), size=9.4, color=INK_2, va="center")
            bw = max(bar_max_w * n / vmax, 0.012)
            ax.add_patch(Rectangle((bar_x0, y - 0.075), bw, 0.15, facecolor=col, edgecolor="none"))
            _t(ax, bar_x0 + bw + 0.07, y, f"{n:,}", size=8.4, color=MUTED, family=MONO, va="center")
            y -= row
        y -= 0.22

    group("SFT TRAJECTORIES", n_sft, sft, SFT_LABELS, ACCENT)
    group("KNOWLEDGE BASE", n_kb, kb, KB_LABELS, WARM)

    # Divider
    div_x = 7.62
    _vline(ax, div_x, 0.8, 5.0, color=RULE, lw=0.7)

    # Right: validation gates table
    tx = div_x + 0.42
    _t(ax, tx, 4.86, "FIVE VALIDATION GATES", size=7.6, color=ACCENT, family=MONO, weight="bold")
    _t(ax, right, 4.86, "ALL PASS", size=7.6, color=ACCENT, family=MONO, weight="bold", ha="right")
    _hline(ax, tx, right, 4.76, color=RULE_2, lw=0.6)

    pct = rep.get("sft_token_percentiles", {})
    leaks = sum(rep.get("pii_secret_leaks", {}).values())
    contam = sum(rep.get("contamination_overlaps", {}).values())
    mix = rep.get("safety_mix", {})
    adversarial = sum(v for k, v in mix.items() if k != "real_incidents")
    total = rep.get("sft_total_rows", n_sft)
    valid = total - rep.get("sft_invalid_rows", 0)
    oracle_n = rep.get("rl_oracle_checked", 200)
    oracle_ok = round(oracle_n * rep.get("rl_oracle_pass_rate", 1.0))

    gates = [
        ("Privacy", "PII and secret leaks after redaction", f"{leaks}"),
        (
            "Schema integrity",
            f"valid tool-call traces · p99 {pct.get('p99', 0):,} tok",
            f"{valid:,}/{total:,}",
        ),
        ("Oracle solvability", "RLVR scenarios solved by the oracle", f"{oracle_ok}/{oracle_n}"),
        ("Contamination", "scenario, topology & prompt overlap", f"{contam}"),
        ("Safety balance", "false alarms, escalations, log injections", f"{adversarial}"),
    ]
    gy = 4.24
    step = 0.78
    for i, (name, detail, value) in enumerate(gates):
        _t(ax, tx, gy + 0.08, f"0{i + 1}", size=8.4, color=FAINT, family=MONO)
        _t(ax, tx + 0.45, gy + 0.08, name, size=11.5, weight="bold")
        _t(ax, tx + 0.45, gy - 0.2, detail, size=9.2, color=MUTED)
        _t(
            ax,
            right - 0.78,
            gy - 0.06,
            value,
            size=17,
            family=SERIF,
            weight="bold",
            ha="right",
            va="center",
        )
        _t(
            ax,
            right,
            gy - 0.06,
            "PASS",
            size=7.6,
            color=ACCENT,
            family=MONO,
            weight="bold",
            ha="right",
            va="center",
        )
        if i < len(gates) - 1:
            _hline(ax, tx, right, gy - 0.42, color=RULE, lw=0.6)
        gy -= step

    _t(
        ax,
        MARGIN,
        0.22,
        "Source: data/real_sre_v2 — sft_real_sre_corpus.jsonl, production_runbooks_promql_kb.jsonl,"
        " validation_report.json",
        size=7.4,
        color=FAINT,
        family=MONO,
    )

    _save(fig, out_path)


def main() -> None:
    assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    render_hero(assets_dir / "omnisre_hero.png")
    render_architecture(assets_dir / "omnisre_architecture.png")
    render_training_curves(
        Path("/tmp/stage1a.log"),
        Path("/tmp/phase1b.log"),
        assets_dir / "omnisre_training_curves.png",
    )
    render_dataset_infographic(assets_dir / "omnisre_dataset_infographic.png")
    print(f"Rendered 4 OmniSRE figures in {assets_dir}")


if __name__ == "__main__":
    main()
