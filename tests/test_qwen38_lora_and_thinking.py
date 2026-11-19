"""Tests for Qwen3.8-27B hybrid Gated DeltaNet LoRA discovery and thinking parser."""

import pytest

from sre_rl.model.qwen38_lora import (
    Qwen38LoRASpec,
    build_qwen38_peft_lora_kwargs,
    discover_qwen38_lora_targets,
)
from sre_rl.model.thinking import (
     format_multiturn_preserve_thinking,
    parse_qwen38_assistant_turn,
)


def test_discover_qwen38_lora_targets_includes_deltanet_and_skips_vision_mtp() -> None:
    mock_qwen38_tree = [
        ("model.visual.blocks.0.attn.q_proj", "Linear"),
        ("model.merger.mlp.0", "Linear"),
        ("model.layers.0.linear_attn.in_proj_qkvz", "Linear"),
        ("model.layers.0.linear_attn.in_proj_ba", "Linear"),
        ("model.layers.0.linear_attn.out_proj", "Linear"),
        ("model.layers.0.mlp.gate_proj", "Linear"),
        ("model.layers.0.mlp.up_proj", "Linear"),
        ("model.layers.0.mlp.down_proj", "Linear"),
        ("model.layers.3.self_attn.q_proj", "Linear"),
        ("model.layers.3.self_attn.k_proj", "Linear"),
        ("model.layers.3.self_attn.v_proj", "Linear"),
        ("model.layers.3.self_attn.o_proj", "Linear"),
        ("model.mtp.layers.0.q_proj", "Linear"),
        ("lm_head", "Linear"),
    ]

    targets = discover_qwen38_lora_targets(mock_qwen38_tree, require_deltanet=True)
    assert "in_proj_qkvz" in targets
    assert "in_proj_ba" in targets
    assert "out_proj" in targets
    assert "q_proj" in targets
    assert "gate_proj" in targets
    assert "lm_head" not in targets

    peft_kwargs = build_qwen38_peft_lora_kwargs(Qwen38LoRASpec(rank=64))
    assert peft_kwargs["r"] == 64
    assert ".*visual.*" in peft_kwargs["exclude_modules"]


def test_discover_qwen38_raises_if_deltanet_missing() -> None:
    attn_only_tree = [
        ("model.layers.0.self_attn.q_proj", "Linear"),
        ("model.layers.0.mlp.up_proj", "Linear"),
    ]
    with pytest.raises(ValueError, match="No Gated DeltaNet"):
        discover_qwen38_lora_targets(attn_only_tree, require_deltanet=True)


def test_parse_qwen38_assistant_turn_and_preserve_thinking() -> None:
    raw = (
        "<think>Checkout error ratio is 34%; check pod status first.</think>\n"
        '<tool_call>{"name": "k8s_inspect", "arguments": {"namespace": "astronomy-prod"}}</tool_call>'
    )
    parsed = parse_qwen38_assistant_turn(raw)
    assert parsed.parse_error is None
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "k8s_inspect"
    assert parsed.thinking_tokens > 0

    unclosed = "<think>Endless reasoning spiral without closing tag..."
    bad = parse_qwen38_assistant_turn(unclosed)
    assert bad.parse_error == "unclosed_think_block"
    assert bad.tool_call is None

    # Verify bounded preserve_thinking strips old <think> blocks while keeping recent k
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "alert"},
        {"role": "assistant", "content": "<think>old1</think>\n<tool_call>{\"name\":\"a\"}</tool_call>"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "<think>recent2</think>\n<tool_call>{\"name\":\"b\"}</tool_call>"},
    ]
    trimmed = format_multiturn_preserve_thinking(msgs, preserve_thinking=True, keep_last_k_thoughts=1)
    assert "<think>" not in trimmed[2]["content"]
    assert "<think>recent2</think>" in trimmed[4]["content"]
