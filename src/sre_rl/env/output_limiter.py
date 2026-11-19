"""Bounded tool-output truncator for 32K-64K RL rollouts.

Raw `kubectl describe`, stack traces, and log dumps can easily exceed 50K tokens in a
single turn. This limiter caps each tool observation to `max_chars` (~4K tokens by
default) while preserving:
1. Header / top summary lines
2. High-signal error/exception/event lines from the middle
3. Most recent tail lines
"""

from __future__ import annotations

import re

HIGH_SIGNAL_LINE_RE = re.compile(
    r"(error|exception|fatal|panic|oomkilled|crashloopbackoff|timeout|refused|denied|unhealthy|failed|5\d\d)",
    re.IGNORECASE,
)


def bound_tool_output(
    raw_output: str,
    max_chars: int = 14_000,
    head_lines: int = 35,
    tail_lines: int = 35,
    max_error_highlights: int = 25,
) -> str:
    """Truncate oversized telemetry outputs while preserving high-signal diagnostic lines."""
    if len(raw_output) <= max_chars:
        return raw_output

    lines = raw_output.splitlines()
    if len(lines) <= head_lines + tail_lines:
        half = max_chars // 2 - 64
        return (
            f"{raw_output[:half]}\n"
            f"... [TRUNCATED {len(raw_output) - 2 * half} chars] ...\n"
            f"{raw_output[-half:]}"
        )

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    middle = lines[head_lines:-tail_lines]

    highlights: list[str] = []
    for idx, line in enumerate(middle, start=head_lines + 1):
        if HIGH_SIGNAL_LINE_RE.search(line):
            highlights.append(f"[line {idx}] {line[:300]}")
            if len(highlights) >= max_error_highlights:
                break

    omitted_count = len(middle) - len(highlights)
    sections = ["\n".join(head)]
    if highlights:
        sections.append(
            f"--- [EXTRACTED {len(highlights)} HIGH-SIGNAL LINES FROM {len(middle)} MIDDLE LINES] ---\n"
            + "\n".join(highlights)
        )
    sections.append(f"--- [OMITTED {omitted_count} ROUTINE LINES] ---")
    sections.append("\n".join(tail))

    assembled = "\n".join(sections)
    if len(assembled) > max_chars:
        assembled = assembled[: max_chars - 48] + "\n... [HARD CAP TRUNCATED] ..."
    return assembled
