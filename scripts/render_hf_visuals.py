#!/usr/bin/env python3
"""Render visual-first Hugging Face Model Card graphics for OmniSRE from real logs & data."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

BG = "#0B1120"
PANEL = "#111A2E"
BORDER = "#1E293B"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
EMERALD = "#10B981"
CYAN = "#38BDF8"
VIOLET = "#A78BFA"
AMBER = "#F59E0B"
ROSE = "#F43F5E"


def _parse_trl_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    if not log_path.exists():
        return [], [], []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    steps, losses, accs = [], [], []
    step_num = 0
    for line in text.splitlines():
        m = re.search(r"(\{'loss':.*?\})", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(1))
            step_num += 5
            steps.append(step_num)
            losses.append(float(d["loss"]))
            accs.append(float(d.get("mean_token_accuracy", 0.0)) * 100.0)
        except (ValueError, SyntaxError, KeyError):
            continue
    return steps, losses, accs


def _rounded_box(ax, x, y, w, h, fc=PANEL, ec=BORDER, lw=1.5, radius=0.03, zorder=2):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def render_hero(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.4), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle background grid & telemetry waveform
    xs = np.linspace(0.02, 0.98, 300)
    for offset, amp, col, alpha in [
        (0.24, 0.055, CYAN, 0.14),
        (0.32, 0.045, EMERALD, 0.16),
        (0.18, 0.03, VIOLET, 0.10),
    ]:
        spike = np.exp(-((xs - 0.72) ** 2) / 0.0025) * 0.22
        ys = offset + amp * np.sin(xs * 24) + spike * (1.0 if col == EMERALD else 0.55)
        ax.plot(xs, ys, color=col, linewidth=2.2, alpha=alpha, zorder=1)
        ax.fill_between(xs, 0.02, ys, color=col, alpha=alpha * 0.22, zorder=1)

    # Outer sleek border
    _rounded_box(ax, 0.015, 0.04, 0.97, 0.92, fc="none", ec="#1E3A5F", lw=1.6, radius=0.035, zorder=3)

    # Top pill badge
    _rounded_box(ax, 0.055, 0.75, 0.27, 0.11, fc="#0F2928", ec=EMERALD, lw=1.2, radius=0.05, zorder=4)
    ax.text(
        0.19,
        0.805,
        "OPEN-WEIGHTS AUTONOMOUS SRE",
        color="#34D399",
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=5,
    )

    # Main Title
    ax.text(
        0.055,
        0.57,
        "OmniSRE",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        ha="left",
        va="center",
        family="sans-serif",
        zorder=5,
    )
    ax.text(
        0.292,
        0.558,
        "27B",
        color=CYAN,
        fontsize=28,
        fontweight="bold",
        ha="left",
        va="center",
        family="sans-serif",
        zorder=5,
    )

    # Punchy 1-line subtitle
    ax.text(
        0.055,
        0.38,
        "Open-source Site Reliability Engineering for every team — Triage, Localize, RCA & Safe Remediation.",
        color="#CBD5E1",
        fontsize=13.5,
        ha="left",
        va="center",
        family="sans-serif",
        zorder=5,
    )

    # 4 crisp metric cards across bottom
    specs = [
        ("ARCHITECTURE", "27B Dense Hybrid", "48 DeltaNet + 16 Attn", CYAN),
        ("CONTEXT", "262K Native", "~2 GB / 64K FP8 KV", EMERALD),
        ("POST-TRAINING", "DAPO + GTPO", "Read-Before-Write RLVR", VIOLET),
        ("LICENSE", "Apache 2.0", "100% Self-Hostable", AMBER),
    ]
    card_w = 0.205
    gap = 0.023
    start_x = 0.055
    for idx, (label, val, sub, accent) in enumerate(specs):
        cx = start_x + idx * (card_w + gap)
        _rounded_box(ax, cx, 0.10, card_w, 0.21, fc=PANEL, ec=BORDER, lw=1.3, radius=0.022, zorder=4)
        # Left accent bar
        _rounded_box(ax, cx + 0.008, 0.125, 0.006, 0.16, fc=accent, ec="none", lw=0, radius=0.003, zorder=5)
        ax.text(cx + 0.022, 0.265, label, color=MUTED, fontsize=8.2, fontweight="bold", ha="left", va="center", zorder=5)
        ax.text(cx + 0.022, 0.20, val, color=TEXT, fontsize=13.5, fontweight="bold", ha="left", va="center", zorder=5)
        ax.text(cx + 0.022, 0.135, sub, color=accent, fontsize=8.8, ha="left", va="center", zorder=5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)


def render_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.92,
        "Read-Write Separated Autonomous SRE Loop",
        color=TEXT,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.03,
        0.845,
        "Hard safety guardrails enforce >=2 read-only diagnostic observations before any cluster mutation",
        color=MUTED,
        fontsize=10.5,
        ha="left",
        va="center",
    )

    stages = [
        ("1. TRIAGE", "Read-Only", "Verify SLO burn\nFilter false alarms\nIgnore log injections", CYAN),
        ("2. LOCALIZE", "Read-Only", "Metrics + Logs\nOTel Traces\nK8s Events & Git", CYAN),
        ("3. RCA", "Read-Only", "Falsify red herrings\nIsolate root object\nRunbook lookup", VIOLET),
        ("4. GUARDRAIL", "Policy Gate", ">=2 Reads check\nNo destructive ops\nTNR no-regression", AMBER),
        ("5. MITIGATE", "Write-Gated", "Minimal K8s patch\nHelm rollback\nScale & Burn-In", EMERALD),
    ]

    bw = 0.162
    gap = 0.032
    sx = 0.03
    by = 0.36
    bh = 0.40

    for i, (title, tag, desc, col) in enumerate(stages):
        x = sx + i * (bw + gap)
        _rounded_box(ax, x, by, bw, bh, fc=PANEL, ec=col, lw=1.6, radius=0.025, zorder=3)
        _rounded_box(ax, x + 0.012, by + bh - 0.085, bw - 0.024, 0.062, fc=col, ec="none", lw=0, radius=0.015, zorder=4)
        ax.text(x + bw / 2, by + bh - 0.054, title, color="#06101C", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x + bw / 2, by + bh - 0.125, tag, color=col, fontsize=9, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x + bw / 2, by + 0.125, desc, color="#E2E8F0", fontsize=9.2, ha="center", va="center", linespacing=1.55, zorder=5)

        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + bw + gap - 0.005, by + bh / 2),
                xytext=(x + bw + 0.005, by + bh / 2),
                arrowprops={"arrowstyle": "->,head_width=0.35,head_length=0.45", "color": "#64748B", "lw": 2.2},
                zorder=5,
            )

    # Bottom hybrid architecture bar
    _rounded_box(ax, 0.03, 0.07, 0.94, 0.21, fc="#0E172A", ec="#253652", lw=1.4, radius=0.022, zorder=3)
    ax.text(0.05, 0.22, "HYBRID 3:1 MODEL BACKBONE (Rank-64 LoRA: 466.9M Trainable / 1.68%)", color=EMERALD, fontsize=10, fontweight="bold", va="center", zorder=4)
    pills = [
        ("48x Gated DeltaNet", "in_proj_qkv / z / a / b, out_proj", CYAN, 0.05, 0.28),
        ("16x Gated Attention + SwiGLU", "q / k / v / o_proj, gate / up / down_proj", VIOLET, 0.355, 0.31),
        ("Frozen Vision & MTP", "460.7M frozen -> fits 1x H100 / L40S FP8", AMBER, 0.69, 0.26),
    ]
    for label, sub, col, px, pw in pills:
        _rounded_box(ax, px, 0.095, pw, 0.092, fc=PANEL, ec=col, lw=1.1, radius=0.015, zorder=4)
        ax.text(px + 0.012, 0.157, label, color=TEXT, fontsize=9.2, fontweight="bold", va="center", zorder=5)
        ax.text(px + 0.012, 0.118, sub, color=MUTED, fontsize=8.0, va="center", zorder=5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)


def render_training_curves(stage1a_log: Path, phase1b_log: Path, out_path: Path) -> None:
    s1_steps, s1_loss, s1_acc = _parse_trl_log(stage1a_log)
    p1_steps, p1_loss, p1_acc = _parse_trl_log(phase1b_log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), dpi=220)
    fig.patch.set_facecolor(BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.8, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9.5)

    # Panel 1: Training Loss
    ax1.plot(s1_steps, s1_loss, color=CYAN, lw=2.6, marker="o", markersize=3.5, label="Stage 1a: Real SRE Corpus (5,038 traces)")
    if p1_steps:
        ax1.plot(p1_steps, p1_loss, color=EMERALD, lw=2.6, marker="s", markersize=4.0, label="Phase 1b: Exact-Schema Calibration (1,559 traces)")
    ax1.set_title("Training Loss (4x NVIDIA H100 BF16 LoRA-64)", color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Optimizer Step (Global Batch = 32 Trajectories)", color=MUTED, fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", color=MUTED, fontsize=10)
    ax1.set_ylim(bottom=0.0)
    ax1.legend(facecolor="#0B1120", edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="upper right")

    if s1_loss:
        ax1.annotate(
            f"Final: {s1_loss[-1]:.3f}",
            xy=(s1_steps[-1], s1_loss[-1]),
            xytext=(s1_steps[-1] - 42, s1_loss[-1] + 0.55),
            color=CYAN,
            fontsize=9.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": CYAN, "lw": 1.3},
        )
    if p1_loss:
        ax1.annotate(
            f"Phase 1b: {p1_loss[-1]:.3f}",
            xy=(p1_steps[-1], p1_loss[-1]),
            xytext=(p1_steps[-1] + 15, p1_loss[-1] + 0.45),
            color=EMERALD,
            fontsize=9.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": EMERALD, "lw": 1.3},
        )

    # Panel 2: Mean Token Accuracy
    ax2.plot(s1_steps, s1_acc, color=VIOLET, lw=2.6, marker="o", markersize=3.5, label="Stage 1a Token Accuracy")
    if p1_steps:
        ax2.plot(p1_steps, p1_acc, color=EMERALD, lw=2.6, marker="s", markersize=4.0, label="Phase 1b Schema Accuracy")
    ax2.set_title("Multi-Turn Tool-Use Token Accuracy (%)", color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Optimizer Step (Global Batch = 32 Trajectories)", color=MUTED, fontsize=10)
    ax2.set_ylabel("Mean Token Accuracy (%)", color=MUTED, fontsize=10)
    ax2.set_ylim(55, 100.5)
    ax2.legend(facecolor="#0B1120", edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="lower right")

    if s1_acc:
        ax2.annotate(
            f"{s1_acc[-1]:.1f}%",
            xy=(s1_steps[-1], s1_acc[-1]),
            xytext=(s1_steps[-1] - 32, s1_acc[-1] - 7.5),
            color=VIOLET,
            fontsize=10,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": VIOLET, "lw": 1.3},
        )
    if p1_acc:
        ax2.annotate(
            f"{p1_acc[-1]:.1f}%",
            xy=(p1_steps[-1], p1_acc[-1]),
            xytext=(p1_steps[-1] + 14, p1_acc[-1] - 6.0),
            color=EMERALD,
            fontsize=10,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": EMERALD, "lw": 1.3},
        )

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)


def render_dataset_infographic(out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.7), dpi=220, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor(BG)

    # Left: Open SRE Corpus Horizontal Bar Chart
    ax1.set_facecolor(PANEL)
    for spine in ax1.spines.values():
        spine.set_color(BORDER)

    sources = [
        "OpenSRE On-Call Traces\n(rmems, Apache-2.0)",
        "DevOps Reasoning Traces\n(deepfabric)",
        "PromQL Production Rules\n(awesome-prom-alerts)",
        "K8s/OpenShift Runbooks\n(openshift + kube-prom)",
        "Grounded Runbook Traces\n(exact-schema SFT)",
        "Agentic SRE + K8s RCA\n(miyad316 + beatsprom)",
        "Real Postmortems + SREGym\n(danluu + SREGym MIT)",
    ]
    counts = [2961, 1305, 1217, 412, 400, 373, 417]
    colors = [EMERALD, CYAN, AMBER, VIOLET, "#2DD4BF", "#60A5FA", ROSE]

    y_pos = np.arange(len(sources))
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color="#1E293B", linestyle="--", linewidth=0.8, zorder=0)
    bars = ax1.barh(y_pos, counts, color=colors, height=0.62, edgecolor="none", zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sources, color="#E2E8F0", fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 3550)
    ax1.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax1.set_xlabel("Verified Records / Rules Collected", color=MUTED, fontsize=9.5)
    ax1.set_title("Open-Source SRE Corpus (5,038 SFT + 2,046 KB)", color=TEXT, fontsize=12.5, fontweight="bold", pad=12)

    for bar, val in zip(bars, counts, strict=True):
        ax1.text(
            val + 55,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
            va="center",
        )

    # Right: 5-Gate Validation Scorecard
    ax2.set_facecolor(BG)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    ax2.text(0.02, 0.96, "5-Gate Dataset Validation", color=TEXT, fontsize=13.5, fontweight="bold", va="center")
    ax2.text(0.02, 0.895, "Every training split passes 5/5 automated gates before GPU launch", color=MUTED, fontsize=9.0, va="center")

    gates = [
        ("GATE 1", "Zero PII / Secret Leaks", "0 AWS/GCP/JWT/IPv4/Email leaks"),
        ("GATE 2", "Tool-Call Schema Integrity", "5,038/5,038 valid • p99 = 2,916 tok"),
        ("GATE 3", "RLVR Oracle Solvability", "200/200 (100%) deterministic solve"),
        ("GATE 4", "Zero Train/Eval Leakage", "0 topology • 0 scenario • 0 prompt"),
        ("GATE 5", "Safety & Adversarial Mix", "False alarms + Log-injection traps"),
    ]

    gy = 0.69
    step = 0.165
    for code, title, detail in gates:
        _rounded_box(ax2, 0.02, gy, 0.96, 0.135, fc=PANEL, ec="#1E3A5F", lw=1.2, radius=0.02, zorder=2)
        _rounded_box(ax2, 0.04, gy + 0.032, 0.16, 0.072, fc="#064E3B", ec=EMERALD, lw=1.0, radius=0.015, zorder=3)
        ax2.text(0.12, gy + 0.068, "PASS", color="#34D399", fontsize=8.8, fontweight="bold", ha="center", va="center", zorder=4)
        ax2.text(0.23, gy + 0.088, f"{code}: {title}", color=TEXT, fontsize=9.8, fontweight="bold", va="center", zorder=4)
        ax2.text(0.23, gy + 0.035, detail, color=MUTED, fontsize=8.5, va="center", zorder=4)
        gy -= step

    fig.tight_layout(pad=1.8)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)


def main() -> None:
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    render_hero(assets_dir / "omnisre_hero.png")
    render_architecture(assets_dir / "omnisre_architecture.png")
    render_training_curves(
        Path("/tmp/stage1a.log"),
        Path("/tmp/phase1b.log"),
        assets_dir / "omnisre_training_curves.png",
    )
    render_dataset_infographic(assets_dir / "omnisre_dataset_infographic.png")
    print("Rendered all 4 OmniSRE visuals in assets/")


if __name__ == "__main__":
    main()
