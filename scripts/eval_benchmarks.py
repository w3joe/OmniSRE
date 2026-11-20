#!/usr/bin/env python3
"""Unified SRE Benchmark Runner for OmniSRE vs. Base Models (`RCAEval`, `OpenRCA`, `SREGym`, `internal_heldout`).

Supports both:
- `--mode subset`: Fast, deterministic stratified slices for rapid iteration (~15-30 min)
- `--mode full`:   Full benchmark suites for leaderboard reporting & Model Card tables

Connects to any OpenAI-compatible SGLang / vLLM server (`--endpoint-url http://127.0.0.1:30000/v1`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sre_rl.env.tier1_snapshot_env import Tier1SnapshotEnvironment
from sre_rl.eval.metrics import summarize_benchmark
from sre_rl.model.thinking import build_sre_system_prompt, format_multiturn_preserve_thinking
from sre_rl.rewards.verifiable_reward import score_episode
from sre_rl.schemas import IncidentScenario

# Curated high-signal SREGym subset covering K8s, DB, Network/Mesh, Metastable & Real Outages
SREGYM_SUBSET_PROBLEMS: tuple[str, ...] = (
    "ad_service_failure",
    "ad_service_high_cpu",
    "auth_miss_mongodb",
    "cart_service_failure",
    "coredns_failure",
    "k8s_target_port_misconfig",
    "kafka_queue_problems",
    "liveness_probe_misconfiguration",
    "loadgenerator_flood_homepage",
    "missing_service",
    "network_loss_hotel_res",
    "oom_kill_social_net",
    "pod_anti_affinity_deadlock",
    "readiness_probe_misconfiguration",
    "wrong_dns_policy",
)

MODE_DEFAULTS: dict[str, dict[str, int]] = {
    "subset": {
        "internal_heldout_scenarios": 30,
        "internal_heldout_k": 3,
        "rcaeval_tasks": 30,
        "openrca_tasks": 30,
        "sregym_tasks": 15,
    },
    "full": {
        "internal_heldout_scenarios": 366,
        "internal_heldout_k": 5,
        "rcaeval_tasks": 270,
        "openrca_tasks": 335,
        "sregym_tasks": 86,
    },
}


@dataclass
class BenchmarkRunResult:
    benchmark: str
    model: str
    mode: str
    num_tasks: int
    primary_metric_name: str
    primary_metric_value: float
    wall_time_s: float
    status: str = "completed"
    details: dict[str, Any] = field(default_factory=dict)


def _chat_completion(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 90,
) -> str:
    url = endpoint_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95 if temperature > 0.0 else 1.0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    reasoning = (msg.get("reasoning_content") or "").strip()
    content = (msg.get("content") or "").strip()
    if reasoning and "<think>" not in content:
        return f"<think>{reasoning}</think>\n{content}"
    return content


def run_internal_heldout_eval(
    endpoint_url: str,
    model: str,
    scenarios_json: Path,
    mode: str,
    limit: int | None = None,
    k: int | None = None,
    workers: int = 16,
    max_turns: int = 14,
) -> BenchmarkRunResult:
    """Evaluate Pass@1, Pass^k, and Safety Violation Rate on unseen real-runbook/PromQL topologies."""
    t0 = time.time()
    defaults = MODE_DEFAULTS[mode]
    n_sc = limit if limit is not None else defaults["internal_heldout_scenarios"]
    k_runs = k if k is not None else defaults["internal_heldout_k"]

    raw = json.loads(scenarios_json.read_text(encoding="utf-8"))
    scenarios = [IncidentScenario.model_validate(item) for item in raw[:n_sc]]

    def _run_one(sc: IncidentScenario, temp: float):
        env = Tier1SnapshotEnvironment(max_turns=max_turns, max_output_chars=2800)
        alert = env.reset(sc)
        msgs = [
            {"role": "system", "content": build_sre_system_prompt(sc.reasoning_effort)},
            {"role": "user", "content": alert},
        ]
        for _ in range(max_turns):
            fmt = format_multiturn_preserve_thinking(
                msgs, preserve_thinking=True, keep_last_k_thoughts=max_turns
            )
            assistant_text = _chat_completion(
                endpoint_url=endpoint_url,
                model=model,
                messages=fmt,
                temperature=temp,
                max_tokens=512,
            )
            out = env.step(assistant_text)
            msgs.append({"role": "assistant", "content": assistant_text})
            msgs.append(
                {"role": "user", "content": f"<tool_response>\n{out.observation}\n</tool_response>"}
            )
            if out.done:
                break
        ep = env.finalize_episode()
        score_episode(sc, ep)
        return ep

    grouped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs: list[tuple[IncidentScenario, list[concurrent.futures.Future]]] = []
        for sc in scenarios:
            sc_futs = [
                pool.submit(_run_one, sc, 0.0 if run_idx == 0 else 0.4)
                for run_idx in range(k_runs)
            ]
            futs.append((sc, sc_futs))
        for sc, sc_futs in futs:
            grouped.append((sc, [f.result() for f in sc_futs]))

    summary = summarize_benchmark(grouped, k=k_runs)
    return BenchmarkRunResult(
        benchmark="internal_heldout",
        model=model,
        mode=mode,
        num_tasks=summary.num_scenarios,
        primary_metric_name=f"pass_hat_{summary.k}",
        primary_metric_value=summary.pass_hat_k,
        wall_time_s=round(time.time() - t0, 1),
        details=asdict(summary),
    )


def run_rcaeval_benchmark(
    endpoint_url: str,
    model: str,
    work_dir: Path,
    mode: str,
    limit: int | None = None,
    workers: int = 12,
) -> BenchmarkRunResult:
    """Run RCAEval blind multi-modal RCA tasks (`re2-ob` / `re2-ss` / `re2-tt`) via OpenAI API."""
    t0 = time.time()
    n_limit = limit if limit is not None else MODE_DEFAULTS[mode]["rcaeval_tasks"]
    datasets = ["re2-ob"] if mode == "subset" else ["re2-ob", "re2-ss", "re2-tt"]
    tasks_root = work_dir / "rcaeval_tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    # Check if `rcaeval-agent` CLI is installed; if not, install RCAEval cleanly
    if subprocess.run(["which", "rcaeval-agent"], capture_output=True, check=False).returncode != 0:
        subprocess.run(
            ["pip", "install", "-q", "git+https://github.com/phamquiluan/RCAEval.git"],
            check=True,
        )

    all_task_dirs: list[tuple[str, Path]] = []
    for ds in datasets:
        ds_dir = tasks_root / ds
        if not ds_dir.exists():
            subprocess.run(
                ["rcaeval-agent", "export", "--dataset", ds, "--out", str(tasks_root)],
                check=True,
            )
        case_dirs = sorted(p for p in ds_dir.iterdir() if p.is_dir())
        for cdir in case_dirs:
            all_task_dirs.append((ds, cdir))

    selected = all_task_dirs[:n_limit]

    def _solve_case(ds: str, case_dir: Path) -> dict[str, Any]:
        # Summarize available telemetry CSV headers & top anomalies for LLM diagnosis
        snippets = []
        for fname in ("metrics.csv", "simple_metrics.csv", "logs.csv", "traces.csv"):
            fpath = case_dir / fname
            if fpath.exists():
                lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()[:45]
                snippets.append(f"### {fname}\n" + "\n".join(lines)[:2500])
        prompt = (
            f"RCAEval Incident Case: {case_dir.name} (dataset={ds})\n"
            "Analyze the telemetry excerpts below and return a JSON object with:\n"
            '{"root_cause_services": ["top1_service", "top2_service", "top3_service"], '
            '"fault_type": "cpu|mem|disk|socket|delay|loss", "reasoning": "..."}\n\n'
            + "\n\n".join(snippets)
        )
        reply = _chat_completion(
            endpoint_url=endpoint_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are OmniSRE. Identify the ranked root-cause microservice names strictly from telemetry evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=768,
        )
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        ranked: list[str] = []
        if m:
            try:
                parsed = json.loads(m.group(0))
                ranked = [str(x) for x in parsed.get("root_cause_services", [])[:5]]
            except json.JSONDecodeError:
                pass
        return {"dataset": ds, "case": case_dir.name, "predictions": ranked}

    preds_by_ds: dict[str, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_solve_case, ds, cdir) for ds, cdir in selected]
        for f in futs:
            res = f.result()
            preds_by_ds.setdefault(res["dataset"], []).append(
                {"case": res["case"], "ranks": res["predictions"]}
            )

    eval_scores: dict[str, Any] = {}
    top1_hits = 0
    for ds, rows in preds_by_ds.items():
        pred_file = work_dir / f"rcaeval_preds_{ds}_{model.replace('/', '_')}.jsonl"
        with pred_file.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        proc = subprocess.run(
            [
                "rcaeval-agent",
                "evaluate",
                "--tasks",
                str(tasks_root / ds),
                "--predictions",
                str(pred_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        eval_scores[ds] = proc.stdout.strip() or proc.stderr.strip()
        m_acc = re.search(r"(?:AC@1|Avg@1|top1)[^0-9]*([0-9.]+)", eval_scores[ds], re.IGNORECASE)
        if m_acc:
            top1_hits += float(m_acc.group(1))

    avg_top1 = round(top1_hits / max(1, len(preds_by_ds)), 4)
    return BenchmarkRunResult(
        benchmark="rcaeval",
        model=model,
        mode=mode,
        num_tasks=len(selected),
        primary_metric_name="ac_at_1",
        primary_metric_value=avg_top1,
        wall_time_s=round(time.time() - t0, 1),
        details={"datasets": datasets, "raw_evaluator_output": eval_scores},
    )


def run_openrca_benchmark(
    endpoint_url: str,
    model: str,
    work_dir: Path,
    mode: str,
    limit: int | None = None,
) -> BenchmarkRunResult:
    """Run microsoft/OpenRCA RCA-Agent against an OpenAI-compatible SGLang endpoint."""
    t0 = time.time()
    repo_dir = work_dir / "OpenRCA"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/microsoft/OpenRCA.git", str(repo_dir)],
            check=True,
        )

    n_limit = limit if limit is not None else MODE_DEFAULTS[mode]["openrca_tasks"]
    datasets = ["Market/cloudbed-1"] if mode == "subset" else ["Market/cloudbed-1", "Market/cloudbed-2", "Bank", "Telecom"]

    env = {
        **os.environ,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sglang-local"),
        "OPENAI_BASE_URL": endpoint_url,
        "OPENRCA_MODEL": model,
        "OPENRCA_MAX_CASES": str(n_limit),
    }

    outputs: dict[str, str] = {}
    for ds in datasets:
        proc = subprocess.run(
            ["python3", "-m", "rca.run_agent_standard", "--dataset", ds],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        outputs[ds] = (proc.stdout + "\n" + proc.stderr).strip()[-1500:]

    return BenchmarkRunResult(
        benchmark="openrca",
        model=model,
        mode=mode,
        num_tasks=n_limit,
        primary_metric_name="exact_rca_accuracy",
        primary_metric_value=0.0,
        wall_time_s=round(time.time() - t0, 1),
        details={"datasets": datasets, "logs": outputs},
    )


def run_sregym_benchmark(
    endpoint_url: str,
    model: str,
    work_dir: Path,
    mode: str,
    limit: int | None = None,
) -> BenchmarkRunResult:
    """Run SREGym live Kubernetes SRE scenarios against an OpenAI-compatible SGLang endpoint."""
    t0 = time.time()
    repo_dir = work_dir / "SREGym"
    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--recurse-submodules",
                "https://github.com/SREGym/SREGym.git",
                str(repo_dir),
            ],
            check=True,
        )

    n_limit = limit if limit is not None else MODE_DEFAULTS[mode]["sregym_tasks"]
    problems = list(SREGYM_SUBSET_PROBLEMS[:n_limit]) if mode == "subset" else ["ALL"]

    env = {
        **os.environ,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sglang-local"),
        "OPENAI_BASE_URL": endpoint_url,
        "SREGYM_MODEL": model,
    }

    cmd = ["uv", "run", "python", "-m", "sregym.conductor.main"]
    if problems != ["ALL"]:
        cmd.extend(["--problems", ",".join(problems)])

    proc = subprocess.run(
        cmd,
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    m_rate = re.search(r"(?:success[_ ]rate|pass@1)[^0-9]*([0-9.]+)", combined, re.IGNORECASE)
    score = float(m_rate.group(1)) if m_rate else 0.0
    if score > 1.0:
        score = round(score / 100.0, 4)

    return BenchmarkRunResult(
        benchmark="sregym",
        model=model,
        mode=mode,
        num_tasks=len(problems) if problems != ["ALL"] else MODE_DEFAULTS["full"]["sregym_tasks"],
        primary_metric_name="mitigation_success_rate",
        primary_metric_value=score,
        wall_time_s=round(time.time() - t0, 1),
        status="completed" if proc.returncode == 0 else "check_logs",
        details={"problems": problems, "log_tail": combined[-2000:]},
    )


def format_comparison_markdown(results: list[BenchmarkRunResult]) -> str:
    """Format side-by-side results into a clean Markdown table ready for HF/GitHub."""
    lines = [
        "| Benchmark | Mode | Model | Tasks | Metric | Score | Wall Time |",
        "| :--- | :---: | :--- | --: | :--- | --: | --: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.benchmark}` | {r.mode} | `{r.model}` | {r.num_tasks} | "
            f"`{r.primary_metric_name}` | **{r.primary_metric_value:.4f}** | {r.wall_time_s:.1f}s |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SRE Benchmark Harness (internal_heldout, RCAEval, OpenRCA, SREGym)"
    )
    parser.add_argument("--mode", choices=["subset", "full"], default="subset")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["internal_heldout", "rcaeval", "openrca", "sregym", "all"],
        default=["internal_heldout"],
    )
    parser.add_argument("--model", type=str, default="omnisre-27b")
    parser.add_argument("--endpoint-url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument(
        "--heldout-json",
        type=Path,
        default=Path("data/real_sre_v2/eval_real_heldout_scenarios.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/nvme/benchmarks"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/benchmark_results.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional per-benchmark task cap override")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    selected = (
        ["internal_heldout", "rcaeval", "openrca", "sregym"]
        if "all" in args.benchmarks
        else list(dict.fromkeys(args.benchmarks))
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkRunResult] = []
    for bench in selected:
        print(f"[eval] Running {bench} (mode={args.mode}, model={args.model})...", flush=True)
        if bench == "internal_heldout":
            results.append(
                run_internal_heldout_eval(
                    endpoint_url=args.endpoint_url,
                    model=args.model,
                    scenarios_json=args.heldout_json,
                    mode=args.mode,
                    limit=args.limit,
                    workers=args.workers,
                )
            )
        elif bench == "rcaeval":
            results.append(
                run_rcaeval_benchmark(
                    endpoint_url=args.endpoint_url,
                    model=args.model,
                    work_dir=args.work_dir,
                    mode=args.mode,
                    limit=args.limit,
                    workers=args.workers,
                )
            )
        elif bench == "openrca":
            results.append(
                run_openrca_benchmark(
                    endpoint_url=args.endpoint_url,
                    model=args.model,
                    work_dir=args.work_dir,
                    mode=args.mode,
                    limit=args.limit,
                )
            )
        elif bench == "sregym":
            results.append(
                run_sregym_benchmark(
                    endpoint_url=args.endpoint_url,
                    model=args.model,
                    work_dir=args.work_dir,
                    mode=args.mode,
                    limit=args.limit,
                )
            )

    md_table = format_comparison_markdown(results)
    payload = {
        "mode": args.mode,
        "model": args.model,
        "results": [asdict(r) for r in results],
        "markdown_table": md_table,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + md_table + "\n")
    print(f"[eval] Saved benchmark report to {args.output_json}")


if __name__ == "__main__":
    main()
