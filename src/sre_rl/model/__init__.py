"""Qwen3.8-27B Dense hybrid architecture utilities (Gated DeltaNet + Thinking control)."""

from sre_rl.model.qwen38_lora import (
    DELTANET_PROJECTION_PATTERNS,
    GATED_ATTN_MLP_PATTERNS,
    VISION_AND_HEAD_SKIP_PATTERNS,
    Qwen38LoRASpec,
    build_qwen38_peft_lora_kwargs,
    discover_qwen38_lora_targets,
    recommended_qwen38_env_vars,
)
from sre_rl.model.thinking import (
    ParsedAssistantTurn,
    build_sre_system_prompt,
    estimate_tokens,
    format_multiturn_preserve_thinking,
    parse_qwen38_assistant_turn,
)

__all__ = [
    "DELTANET_PROJECTION_PATTERNS",
    "GATED_ATTN_MLP_PATTERNS",
    "VISION_AND_HEAD_SKIP_PATTERNS",
    "ParsedAssistantTurn",
    "Qwen38LoRASpec",
    "build_qwen38_peft_lora_kwargs",
    "build_sre_system_prompt",
    "discover_qwen38_lora_targets",
    "estimate_tokens",
    "format_multiturn_preserve_thinking",
    "parse_qwen38_assistant_turn",
    "recommended_qwen38_env_vars",
]
