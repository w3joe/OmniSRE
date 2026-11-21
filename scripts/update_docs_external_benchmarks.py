#!/usr/bin/env python3
"""Render Swiss-editorial HTML + GitHub/HF Markdown tables from official external benchmarks.

Reads the output of `summarize_external_benchmarks.py` (`openrca_official` + `sregym_lite_live_k8s`)
and outputs ready-to-paste Markdown (`README.md`, `HF_MODEL_CARD_V1.md`) and HTML (`docs/index.html`)
tables adhering strictly to the OmniSRE light-mode editorial specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val) * 100.0:.1f}%"


def render_markdown_table(report: dict[str, Any], omni_key: str = "omnisre_v1") -> str:
    openrca = report.get("openrca_official") or {}
    omni_orca = openrca.get(omni_key) or {}
    base_orca = openrca.get("base_qwen38") or {}

    lines = [
        "### Official Third-Party External Benchmarks (`microsoft/OpenRCA` + Live `SREGym-Lite`)",
        "",
        "| External Suite | Metric | Base `Qwen3.8-27B` | **`OmniSRE v1.0`** | Delta |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    if "Telecom" in omni_orca or "Telecom" in base_orca:
        o_t = omni_orca.get("Telecom") or {}
        b_t = base_orca.get("Telecom") or {}
        o_acc = float(o_t.get("strict_accuracy", 0.0)) * 100.0
        b_acc = float(b_t.get("strict_accuracy", 0.0)) * 100.0
        lines.append(
            f"| **`microsoft/OpenRCA`** *(Telecom)* | Strict Root-Cause Accuracy | `{b_acc:.1f}%` | **`{o_acc:.1f}%`** | **`+{o_acc - b_acc:.1f} pp`** |"
        )
        o_cr = float(o_t.get("sandbox_crash_rate", 0.0)) * 100.0
        b_cr = float(b_t.get("sandbox_crash_rate", 0.0)) * 100.0
        lines.append(
            f"| **`microsoft/OpenRCA`** *(Telecom)* | Sandbox Crash Rate *(lower is better)* | `{b_cr:.1f}%` | **`{o_cr:.1f}%`** | **`{o_cr - b_cr:.1f} pp`** |"
        )

    sregym_runs = report.get("sregym_lite_live_k8s") or []
    if len(sregym_runs) >= 2:
        omni_sg, base_sg = sregym_runs[0], sregym_runs[1]
        o_diag = float(omni_sg.get("diagnosis_pass_rate") or 0.0) * 100.0
        b_diag = float(base_sg.get("diagnosis_pass_rate") or 0.0) * 100.0
        o_mit = float(omni_sg.get("mitigation_pass_rate") or 0.0) * 100.0
        b_mit = float(base_sg.get("mitigation_pass_rate") or 0.0) * 100.0
        n_prob = omni_sg.get("problems", 17)
        lines.append(
            f"| **Live `SREGym-Lite`** *({n_prob} Kind K8s faults)* | Diagnosis Pass Rate | `{b_diag:.1f}%` | **`{o_diag:.1f}%`** | **`{o_diag - b_diag:+.1f} pp`** |"
        )
        lines.append(
            f"| **Live `SREGym-Lite`** *({n_prob} Kind K8s faults)* | Live Cluster Mitigation Rate | `{b_mit:.1f}%` | **`{o_mit:.1f}%`** | **`{o_mit - b_mit:+.1f} pp`** |"
        )
    elif len(sregym_runs) == 1:
        omni_sg = sregym_runs[0]
        n_prob = omni_sg.get("problems", 0)
        lines.append(
            f"| **Live `SREGym-Lite`** *({n_prob} Kind K8s faults)* | Diagnosis / Mitigation | *(running)* | **`{_pct(omni_sg.get('diagnosis_pass_rate'))}` / `{_pct(omni_sg.get('mitigation_pass_rate'))}`** | — |"
        )

    return "\n".join(lines)


def render_html_table(report: dict[str, Any], omni_key: str = "omnisre_v1") -> str:
    openrca = report.get("openrca_official") or {}
    omni_orca = (openrca.get(omni_key) or {}).get("Telecom") or {}
    base_orca = (openrca.get("base_qwen38") or {}).get("Telecom") or {}
    sregym_runs = report.get("sregym_lite_live_k8s") or []

    omni_sg = sregym_runs[0] if len(sregym_runs) >= 1 else {}
    base_sg = sregym_runs[1] if len(sregym_runs) >= 2 else {}

    return f"""      <h3>Table 3 — Official External Benchmarks (<code>microsoft/OpenRCA</code> &amp; Live 4-Node Kind <code>SREGym-Lite</code>)</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>External Benchmark</th>
              <th>Model</th>
              <th class="num">Tasks</th>
              <th class="num">Primary Pass / Strict Acc.</th>
              <th class="num">Secondary / Mitigation</th>
              <th class="num">Crash / Timeout Rate</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>microsoft/OpenRCA</code> (Telecom)</td>
              <td><code>Qwen/Qwen3.8-27B</code> (Base)</td>
              <td class="num">{base_orca.get("tasks", "—")}</td>
              <td class="num">{_pct(base_orca.get("strict_accuracy"))}</td>
              <td class="num">{_pct(base_orca.get("partial_score_mean"))} partial</td>
              <td class="num">{_pct(base_orca.get("sandbox_crash_rate"))}</td>
            </tr>
            <tr class="highlight">
              <td><strong><code>microsoft/OpenRCA</code> (Telecom)</strong></td>
              <td><strong><code>OmniSRE-27B v1.0</code> (Ours)</strong></td>
              <td class="num">{omni_orca.get("tasks", "—")}</td>
              <td class="num"><span class="gain">{_pct(omni_orca.get("strict_accuracy"))}</span></td>
              <td class="num">{_pct(omni_orca.get("partial_score_mean"))} partial</td>
              <td class="num"><span class="gain">{_pct(omni_orca.get("sandbox_crash_rate"))}</span></td>
            </tr>
            <tr>
              <td>Live <code>SREGym-Lite</code> (4-Node Kind K8s)</td>
              <td><code>Qwen/Qwen3.8-27B</code> (Base)</td>
              <td class="num">{base_sg.get("problems", "—")}</td>
              <td class="num">{_pct(base_sg.get("diagnosis_pass_rate"))} diag</td>
              <td class="num">{_pct(base_sg.get("mitigation_pass_rate"))} mitigated</td>
              <td class="num">—</td>
            </tr>
            <tr class="highlight">
              <td><strong>Live <code>SREGym-Lite</code> (4-Node Kind K8s)</strong></td>
              <td><strong><code>OmniSRE-27B v1.0</code> (Ours)</strong></td>
              <td class="num">{omni_sg.get("problems", "—")}</td>
              <td class="num"><span class="gain">{_pct(omni_sg.get("diagnosis_pass_rate"))} diag</span></td>
              <td class="num"><span class="gain">{_pct(omni_sg.get("mitigation_pass_rate"))} mitigated</span></td>
              <td class="num">—</td>
            </tr>
          </tbody>
        </table>
      </div>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Format Swiss-editorial Markdown + HTML external benchmark tables")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--omni-key", type=str, default="omnisre_v1")
    args = parser.parse_args()

    report = json.loads(args.summary_json.read_text(encoding="utf-8"))
    print("=== MARKDOWN TABLE ===")
    print(render_markdown_table(report, args.omni_key))
    print("\n=== HTML TABLE ===")
    print(render_html_table(report, args.omni_key))


if __name__ == "__main__":
    main()
