"""Deterministic PII, Secret, & Internal-IP Scrubber for Production Incident Mining.

Before historical PagerDuty/Slack/Loki/Bastion traces enter Dataset A (SFT) or
Dataset B (Tier-1 Snapshot Simulators), all credentials and customer identifiers must
be scrubbed while keeping structural correlation intact (e.g. two occurrences of the
same internal pod IP map to the exact same `<IP_xxxx>` alias so network localization
remains solvable).
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PEM_KEY", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL)),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GCP_API_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("BEARER", re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*")),
    ("CONN_PASSWORD", re.compile(r"(?i)((?:postgres|mysql|redis|mongodb)://[^:\s]+:)([^@\s]+)(@)")),
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Keep standard loopback / link-local / K8s service CIDR hints recognizable if desired, or alias RFC1918
PRESERVED_IPS: frozenset[str] = frozenset({"127.0.0.1", "0.0.0.0", "169.254.169.254"})


@dataclass
class TelemetryScrubber:
    salt: str = "qwen38-sre-scrubber-v1"
    redaction_counts: dict[str, int] = field(default_factory=dict)

    def _alias(self, prefix: str, raw_value: str) -> str:
        digest = hmac.new(
            self.salt.encode("utf-8"),
            raw_value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:6]
        return f"<{prefix}_{digest}>"

    def _bump(self, category: str) -> None:
        self.redaction_counts[category] = self.redaction_counts.get(category, 0) + 1

    def scrub_text(self, text: str) -> str:
        out = text
        for label, pattern in SECRET_PATTERNS:
            if label == "BEARER":
                def _sub_bearer(m: re.Match[str]) -> str:
                    self._bump("BEARER")
                    return f"{m.group(1)}<REDACTED_BEARER_TOKEN>"
                out = pattern.sub(_sub_bearer, out)
            elif label == "CONN_PASSWORD":
                def _sub_conn(m: re.Match[str]) -> str:
                    self._bump("CONN_PASSWORD")
                    return f"{m.group(1)}<REDACTED_DB_PASS>{m.group(3)}"
                out = pattern.sub(_sub_conn, out)
            else:
                def _sub_secret(m: re.Match[str], lbl: str = label) -> str:
                    self._bump(lbl)
                    return f"<REDACTED_{lbl}>"
                out = pattern.sub(_sub_secret, out)

        def _sub_email(m: re.Match[str]) -> str:
            self._bump("EMAIL")
            return self._alias("EMAIL", m.group(0).lower())

        out = EMAIL_RE.sub(_sub_email, out)

        def _sub_ip(m: re.Match[str]) -> str:
            ip = m.group(0)
            if ip in PRESERVED_IPS:
                return ip
            octets = [int(x) for x in ip.split(".")]
            if any(o > 255 for o in octets):
                return ip
            self._bump("IPV4")
            return self._alias("IP", ip)

        out = IPV4_RE.sub(_sub_ip, out)
        return out
