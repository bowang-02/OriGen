"""LoRA for the Qwen3-VL text decoder's attention and FFN projections."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_TARGETS = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        # FP32 adapter weights and gradients; the frozen base remains BF16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(self.dropout(x).float(), self.lora_A), self.lora_B)
        return base_output + (update * self.scale).to(base_output.dtype)


def _target_modules(layer):
    for name in ATTENTION_TARGETS:
        yield layer.self_attn, name
    for name in FFN_TARGETS:
        yield layer.mlp, name


def install_text_lora(model, rank: int = 32, alpha: float = 64.0, dropout: float = 0.05) -> int:
    layers = model.model.language_model.layers
    expected = int(model.config.text_config.num_hidden_layers)
    if len(layers) != expected:
        raise RuntimeError(f"Expected {expected} text decoder layers, got {len(layers)}.")
    count = 0
    for layer in layers:
        for parent, name in _target_modules(layer):
            base = getattr(parent, name)
            if type(base) is not nn.Linear:
                raise TypeError(f"Expected nn.Linear at {parent.__class__.__name__}.{name}, got {type(base).__name__}.")
            setattr(parent, name, LoRALinear(base, rank, alpha, dropout))
            count += 1
    if count != expected * (len(ATTENTION_TARGETS) + len(FFN_TARGETS)):
        raise RuntimeError("The LoRA module count does not match the decoder architecture.")
    return count


def lora_parameters(model):
    for module in model.model.language_model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


@torch.no_grad()
def merge_text_lora(model) -> int:
    """Fold trained LoRA deltas into base weights for a standard full checkpoint."""
    merged = 0
    for layer in model.model.language_model.layers:
        for parent, name in _target_modules(layer):
            module = getattr(parent, name)
            if not isinstance(module, LoRALinear):
                raise TypeError(f"Expected a LoRA layer at {parent.__class__.__name__}.{name} before merging.")
            delta = (module.lora_B @ module.lora_A) * module.scale
            module.base.weight.add_(delta.to(module.base.weight.dtype))
            setattr(parent, name, module.base)
            merged += 1
    return merged
