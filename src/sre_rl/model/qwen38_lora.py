"""Qwen3.8-27B Dense (3:1 Gated DeltaNet + Gated Attention) LoRA & Vision-Freezing setup.

`Qwen/Qwen3.8-27B` uses 64 layers arranged as:
    16 x (3 x (Gated DeltaNet -> SwiGLU FFN) -> 1 x (Gated Attention -> SwiGLU FFN))
plus a native multimodal Vision Encoder and Multi-Token Prediction (MTP) heads.

Standard Transformer LoRA recipes only adapt `q_proj/k_proj/v_proj/o_proj` and miss
75% of Qwen3.8-27B's sequence-mixing layers (the 48 Gated DeltaNet blocks). This module
discovers and validates both Gated Attention, SwiGLU MLP, and Gated DeltaNet linear
projections while keeping the Vision Tower and MTP heads frozen to fit 4x H100 80GB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Standard Gated Attention (16 layers) + SwiGLU FFN (64 layers)
GATED_ATTN_MLP_PATTERNS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Gated DeltaNet linear-attention blocks (48 layers in Qwen3.5 / Qwen3.8 `qwen3_5` arch)
DELTANET_PROJECTION_PATTERNS: tuple[str, ...] = (
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj",
    "out_proj",
    "ba_proj",
)

# Modules to explicitly freeze / exclude during text-centric SRE RLVR on 4x H100
VISION_AND_HEAD_SKIP_PATTERNS: tuple[str, ...] = (
    "visual",
    "vision_tower",
    "vision_model",
    "merger",
    "mm_projector",
    "lm_head",
    "embed_tokens",
    "mtp",
)


@dataclass(frozen=True)
class Qwen38LoRASpec:
    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    freeze_vision_tower: bool = True
    freeze_mtp_heads: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: sorted(
            set(GATED_ATTN_MLP_PATTERNS + DELTANET_PROJECTION_PATTERNS)
        )
    )


def discover_qwen38_lora_targets(
    named_module_types: list[tuple[str, str]],
    require_deltanet: bool = True,
) -> list[str]:
    """Inspect `(module_full_name, class_name)` pairs and return clean leaf LoRA targets.

    Works on real `torch.nn.Module` trees (`[(n, type(m).__name__) for n, m in model.named_modules()]`)
    as well as dry-run unit tests without requiring GPU weights loaded.
    """
    linear_class_hints = ("Linear", "QuantLinear", "Linear4bit", "Linear8bitLt")
    leaf_names: set[str] = set()

    for full_name, cls_name in named_module_types:
        if not any(hint in cls_name for hint in linear_class_hints):
            continue
        if any(skip in full_name for skip in VISION_AND_HEAD_SKIP_PATTERNS):
            continue
        leaf = full_name.split(".")[-1]
        if leaf in GATED_ATTN_MLP_PATTERNS or leaf in DELTANET_PROJECTION_PATTERNS:
            leaf_names.add(leaf)

    if require_deltanet and not any(t in DELTANET_PROJECTION_PATTERNS for t in leaf_names):
        raise ValueError(
            "No Gated DeltaNet projection modules found in model tree. "
            f"Expected at least one of {DELTANET_PROJECTION_PATTERNS}; got {sorted(leaf_names)}."
        )
    return sorted(leaf_names)


def freeze_non_lora_subsystems(model: Any, spec: Qwen38LoRASpec | None = None) -> dict[str, int]:
    """Explicitly freeze vision tower & MTP parameters on a loaded HuggingFace model."""
    spec = spec or Qwen38LoRASpec()
    frozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        should_freeze_vision = spec.freeze_vision_tower and any(
            v in name for v in ("visual", "vision_tower", "vision_model", "merger", "mm_projector")
        )
        should_freeze_mtp = spec.freeze_mtp_heads and "mtp" in name
        if should_freeze_vision or should_freeze_mtp:
            param.requires_grad = False
            frozen_params += count

    return {"frozen_params": frozen_params, "total_params": total_params}


def build_qwen38_peft_lora_kwargs(spec: Qwen38LoRASpec | None = None) -> dict[str, Any]:
    """Return `peft.LoraConfig` keyword arguments tuned for Qwen/Qwen3.8-27B Dense."""
    spec = spec or Qwen38LoRASpec()
    return {
        "r": spec.rank,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(spec.target_modules),
        "exclude_modules": [
            ".*visual.*",
            ".*vision_tower.*",
            ".*merger.*",
            ".*mtp.*",
        ],
    }


def recommended_qwen38_env_vars(nvme_root: str = "/mnt/nvme") -> dict[str, str]:
    """Environment flags required for stable Qwen3.8-27B Gated DeltaNet + LoRA on 4x H100."""
    return {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        # Prevent torch.compile Inductor issues on mixed bf16/fp32 Gated DeltaNet backward passes
        "UNSLOTH_COMPILE_DISABLE": "1",
        # Keep fused Triton Gated DeltaNet (`flash-linear-attention`) enabled; do not force deterministic
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HOME": f"{nvme_root}/hf_cache",
        "WANDB_DIR": f"{nvme_root}/wandb",
    }
