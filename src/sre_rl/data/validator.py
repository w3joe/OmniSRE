"""5-Gate Production SRE Dataset Quality, Hygiene & Anti-Contamination Validator.

Audits scraped SFT corpora, Knowledge Bases, and RLVR splits before GPU training:
- Gate 1 (PII & Secret Leak Audit): Scans every record for raw AWS/GCP/GitHub/Slack keys,
  PEM private keys, JWTs, DB passwords, raw emails, and non-preserved IPv4 addresses.
- Gate 2 (Structural & Tool-Call Integrity): Verifies ChatML role sequencing (`system ->
  user -> assistant`), balanced `<think>...</think>`, valid `<tool_call>` JSON schema,
  and token-length distribution (rejecting empty or runaway >32K traces).
- Gate 3 (Executable RLVR Oracle Solvability): Runs `Tier1SnapshotEnvironment` on sampled
  RL scenarios to prove 100% deterministic solvability (`mitigated=True, reward>=0.90`).
- Gate 4 (Zero Train/Held-Out Contamination): Verifies strictly disjoint `scenario_id`
  and `topology_name` between train and held-out splits + zero prompt hash overlap.
- Gate 5 (Safety & Distribution Balance): Checks coverage of multi-turn tool traces,
  benign false alarms, external escalations, metastable cascades, and log injection traps.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sre_rl.data.scrubber import EMAIL_RE, IPV4_RE, PRESERVED_IPS, SECRET_PATTERNS
from sre_rl.data.trajectory_builder import synthesize_expert_trajectory
from sre_rl.model.thinking import estimate_tokens, parse_qwen38_assistant_turn
from sre_rl.schemas import IncidentClassification, IncidentScenario


@dataclass
class ValidationReport:
    passed: bool
    gate_results: dict[str, bool] = field(default_factory=dict)
    sft_total_rows: int = 0
    sft_multiturn_tool_rows: int = 0
    sft_invalid_rows: int = 0
    sft_token_percentiles: dict[str, int] = field(default_factory=dict)
    pii_secret_leaks: dict[str, int] = field(default_factory=dict)
    rl_train_count: int = 0
    rl_val_count: int = 0
    rl_heldout_count: int = 0
    rl_oracle_checked: int = 0
    rl_oracle_pass_rate: float = 0.0
    contamination_overlaps: dict[str, int] = field(default_factory=dict)
    safety_mix: dict[str, int] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


def scan_text_for_leaks(text: str) -> dict[str, int]:
    """Return counts of any remaining unredacted secrets, emails, or non-preserved IPv4s."""
    hits: dict[str, int] = {}
    for label, pat in SECRET_PATTERNS:
        matches = pat.findall(text)
        if not matches:
            continue
        unredacted = 0
        for m in matches:
            blob = " ".join(m) if isinstance(m, tuple) else str(m)
            if "<REDACTED_" in blob:
                continue
            unredacted += 1
        if unredacted > 0:
            hits[label] = hits.get(label, 0) + unredacted

    emails = EMAIL_RE.findall(text)
    if emails:
        hits["EMAIL"] = hits.get("EMAIL", 0) + len(emails)

    for ip in IPV4_RE.findall(text):
        if ip in PRESERVED_IPS:
            continue
        octets = [int(x) for x in ip.split(".")]
        if all(0 <= o <= 255 for o in octets):
            hits["IPV4"] = hits.get("IPV4", 0) + 1
    return hits


def validate_sft_row(row: dict[str, Any], max_tokens: int = 32_768) -> tuple[bool, list[str], int, bool]:
    """Validate one SFT record. Returns `(ok, errors, est_tokens, has_tool_calls)`."""
    errors: list[str] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, ["messages_missing_or_too_short"], 0, False

    if msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        errors.append("invalid_opening_roles")

    total_tokens = 0
    has_tool_call = False
    assistant_count = 0

    for idx, msg in enumerate(msgs):
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role not in ("system", "user", "assistant"):
            errors.append(f"invalid_role_at_{idx}:{role}")
        if not content:
            errors.append(f"empty_content_at_{idx}")
            continue
        total_tokens += estimate_tokens(content)

        if role == "assistant":
            assistant_count += 1
            parsed = parse_qwen38_assistant_turn(content)
            if parsed.parse_error is not None:
                errors.append(f"assistant_parse_error_at_{idx}:{parsed.parse_error}")
            if parsed.tool_call is not None:
                has_tool_call = True

    if assistant_count == 0:
        errors.append("no_assistant_turns")
    if total_tokens > max_tokens:
        errors.append(f"exceeds_max_tokens:{total_tokens}")

    return len(errors) == 0, errors, total_tokens, has_tool_call


def validate_dataset_directory(
    dataset_dir: Path,
    oracle_sample_size: int = 120,
    seed: int = 42,
    auto_filter_invalid_sft: bool = False,
) -> ValidationReport:
    """Run all 5 validation gates on a scraped Dataset v2 directory."""
    sft_path = dataset_dir / "sft_real_sre_corpus.jsonl"
    kb_path = dataset_dir / "production_runbooks_promql_kb.jsonl"
    rl_train_path = dataset_dir / "rl_real_train.jsonl"
    rl_val_path = dataset_dir / "rl_real_val.jsonl"
    rl_heldout_path = dataset_dir / "rl_real_heldout.jsonl"

    failures: list[str] = []
    total_leaks: dict[str, int] = {}

    def _merge_leaks(chunk: dict[str, int]) -> None:
        for k, v in chunk.items():
            total_leaks[k] = total_leaks.get(k, 0) + v

    # --- Gate 1 & 2: SFT Structural Integrity + PII/Secret Leak Scan ---
    sft_rows = [json.loads(ln) for ln in sft_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    valid_sft_rows: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    multiturn_tool_count = 0
    invalid_sft = 0
    sft_prompt_hashes: set[str] = set()

    for row in sft_rows:
        ok, _errs, toks, has_tc = validate_sft_row(row)
        # Scan all messages for leaks
        joined = "\n".join(str(m.get("content", "")) for m in (row.get("messages") or []))
        leaks = scan_text_for_leaks(joined)
        _merge_leaks(leaks)

        if ok and not leaks:
            valid_sft_rows.append(row)
            token_lengths.append(toks)
            if has_tc and len(row["messages"]) >= 5:
                multiturn_tool_count += 1
            first_user = str(row["messages"][1].get("content", "")).strip()
            sft_prompt_hashes.add(hashlib.sha256(first_user.encode("utf-8")).hexdigest()[:16])
        else:
            invalid_sft += 1

    if auto_filter_invalid_sft and invalid_sft > 0:
        with sft_path.open("w", encoding="utf-8") as fh:
            for row in valid_sft_rows:
                fh.write(json.dumps(row) + "\n")

    # Scan KB for leaks too
    if kb_path.exists():
        for ln in kb_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                _merge_leaks(scan_text_for_leaks(ln))

    # --- Load RLVR Splits & Check Gate 3 (Oracle Solvability) + Gate 4 (Anti-Contamination) ---
    def _load_rl_scenarios(p: Path) -> list[IncidentScenario]:
        if not p.exists():
            return []
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            gt = row["reward_model"]["ground_truth"]
            sc = IncidentScenario.model_validate_json(gt) if isinstance(gt, str) else IncidentScenario.model_validate(gt)
            out.append(sc)
        return out

    rl_train = _load_rl_scenarios(rl_train_path)
    rl_val = _load_rl_scenarios(rl_val_path)
    rl_heldout = _load_rl_scenarios(rl_heldout_path)

    train_ids = {s.scenario_id for s in rl_train}
    heldout_ids = {s.scenario_id for s in rl_heldout}
    train_topos = {s.topology_name for s in rl_train}
    heldout_topos = {s.topology_name for s in rl_heldout}

    id_overlap = len(train_ids & heldout_ids)
    topo_overlap = len(train_topos & heldout_topos)
    prompt_overlap = 0
    for sc in rl_heldout:
        h = hashlib.sha256(sc.alert_pager_payload.strip().encode("utf-8")).hexdigest()[:16]
        if h in sft_prompt_hashes:
            prompt_overlap += 1

    # Gate 3: Run deterministic Tier-1 simulator on sampled RL scenarios across all splits
    rng = random.Random(seed)
    combined_rl = rl_train + rl_val + rl_heldout
    sample = rng.sample(combined_rl, min(oracle_sample_size, len(combined_rl))) if combined_rl else []
    oracle_passes = 0
    for sc in sample:
        traj = synthesize_expert_trajectory(sc)
        if traj["mitigated"] and traj["reward"] >= 0.90:
            oracle_passes += 1
    oracle_rate = round(oracle_passes / max(1, len(sample)), 4)

    # Gate 5: Safety & Scenario Mix Check
    safety_mix = {
        "real_incidents": sum(1 for s in combined_rl if s.fault.classification == IncidentClassification.REAL_INCIDENT),
        "benign_false_alarms": sum(1 for s in combined_rl if s.fault.classification == IncidentClassification.BENIGN_FALSE_ALARM),
        "external_escalations": sum(1 for s in combined_rl if s.fault.classification == IncidentClassification.EXTERNAL_ESCALATION),
        "metastable_cascades": sum(1 for s in combined_rl if s.fault.requires_secondary_drain_or_scale),
        "adversarial_log_injections": sum(1 for s in combined_rl if s.fault.contains_adversarial_log_injection),
    }

    # Evaluate the 5 Gates
    # Note: allow strictly 0 secret/credential leaks (AWS/GCP/GH/Slack/PEM/JWT/DB_PASS/EMAIL)
    hard_secret_leaks = sum(v for k, v in total_leaks.items() if k != "IPV4")
    # Note: IPv4 inside PromQL/runbooks like 192.168.x.x or 10.x.x.x should also be 0 after scrubbing!
    gate_pii_secrets = hard_secret_leaks == 0 and total_leaks.get("IPV4", 0) == 0
    gate_sft_structure = (invalid_sft == 0 or auto_filter_invalid_sft) and len(valid_sft_rows) >= 100
    gate_rl_oracle = oracle_rate == 1.0 and len(sample) > 0
    gate_decontamination = id_overlap == 0 and topo_overlap == 0 and prompt_overlap == 0
    gate_safety_balance = (
        safety_mix["benign_false_alarms"] >= 20
        and safety_mix["external_escalations"] >= 20
        and safety_mix["metastable_cascades"] >= 20
        and safety_mix["adversarial_log_injections"] >= 20
        and multiturn_tool_count >= 500
    )

    if not gate_pii_secrets:
        failures.append(f"Gate 1 FAILED: Unscrubbed PII/Secret leaks found: {total_leaks}")
    if not gate_sft_structure:
        failures.append(f"Gate 2 FAILED: {invalid_sft} invalid SFT rows (valid={len(valid_sft_rows)})")
    if not gate_rl_oracle:
        failures.append(f"Gate 3 FAILED: RLVR oracle pass rate = {oracle_rate} (expected 1.0)")
    if not gate_decontamination:
        failures.append(
            f"Gate 4 FAILED: Contamination detected (id={id_overlap}, topo={topo_overlap}, prompt={prompt_overlap})"
        )
    if not gate_safety_balance:
        failures.append(f"Gate 5 FAILED: Safety/multiturn mix below thresholds: {safety_mix}, multiturn={multiturn_tool_count}")

    pcts = {}
    if token_lengths:
        arr = np.array(token_lengths)
        pcts = {
            "p50": int(np.percentile(arr, 50)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
        }

    gate_results = {
        "gate1_zero_pii_secret_leaks": gate_pii_secrets,
        "gate2_sft_structural_integrity": gate_sft_structure,
        "gate3_rlvr_oracle_solvability": gate_rl_oracle,
        "gate4_zero_train_eval_contamination": gate_decontamination,
        "gate5_safety_and_multiturn_balance": gate_safety_balance,
    }
    report = ValidationReport(
        passed=all(gate_results.values()),
        gate_results=gate_results,
        sft_total_rows=len(valid_sft_rows) if auto_filter_invalid_sft else len(sft_rows),
        sft_multiturn_tool_rows=multiturn_tool_count,
        sft_invalid_rows=invalid_sft,
        sft_token_percentiles=pcts,
        pii_secret_leaks=total_leaks,
        rl_train_count=len(rl_train),
        rl_val_count=len(rl_val),
        rl_heldout_count=len(rl_heldout),
        rl_oracle_checked=len(sample),
        rl_oracle_pass_rate=oracle_rate,
        contamination_overlaps={
            "scenario_id_overlap": id_overlap,
            "topology_overlap": topo_overlap,
            "sft_to_heldout_prompt_overlap": prompt_overlap,
        },
        safety_mix=safety_mix,
        failure_reasons=failures,
    )
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    return report
