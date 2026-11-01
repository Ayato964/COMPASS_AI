"""
Built-in LoRA Linear Layer & In-Place Weight Folding.
100% state_dict-compatible with `loralib.Linear` (`weight`, `bias`, `lora_A`, `lora_B`),
eliminating the external `loralib` dependency and enabling zero-overhead folded inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """
    Drop-in replacement for `loralib.Linear`.
    Supports in-place weight merging (`merge()`) so inference executes a single `F.linear`
    without computing the `(x @ lora_A.T @ lora_B.T) * scaling` branch at every step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.fan_in_fan_out = bool(fan_in_fan_out)
        self.merge_weights = bool(merge_weights)
        self.merged = False

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.r)))
            self.scaling = float(self.lora_alpha) / float(self.r)
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0

        self.reset_lora_parameters()
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_lora_parameters(self) -> None:
        if hasattr(self, "lora_A") and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _transpose(self, w: torch.Tensor) -> torch.Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    @torch.no_grad()
    def merge_lora_inplace(self) -> bool:
        """
        Folds LoRA weights into the base weight matrix:
            W <- W + (lora_B @ lora_A) * scaling
        """
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            delta = (self.lora_B.to(dtype=torch.float32) @ self.lora_A.to(dtype=torch.float32)) * self.scaling
            delta = self._transpose(delta).to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight.data.add_(delta)
            self.merged = True
            return True
        return False

    @torch.no_grad()
    def unmerge_lora_inplace(self) -> bool:
        if self.r > 0 and self.merged and self.lora_A is not None and self.lora_B is not None:
            delta = (self.lora_B.to(dtype=torch.float32) @ self.lora_A.to(dtype=torch.float32)) * self.scaling
            delta = self._transpose(delta).to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight.data.sub_(delta)
            self.merged = False
            return True
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            result = F.linear(x, self._transpose(self.weight), bias=self.bias)
            lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result + lora_out
        return F.linear(x, self._transpose(self.weight), bias=self.bias)


@torch.no_grad()
def merge_lora_weights_inplace(module: nn.Module) -> int:
    """
    Recursively folds all `LoRALinear` (and any `loralib.Linear`) modules in `module`
    so runtime inference has zero LoRA overhead.
    Returns the number of merged layers.
    """
    merged_count = 0
    for m in module.modules():
        if hasattr(m, "merge_lora_inplace") and callable(m.merge_lora_inplace):
            if m.merge_lora_inplace():
                merged_count += 1
        elif hasattr(m, "lora_A") and hasattr(m, "lora_B") and getattr(m, "r", 0) > 0 and not getattr(m, "merged", False):
            delta = (m.lora_B.to(torch.float32) @ m.lora_A.to(torch.float32)) * float(getattr(m, "scaling", 1.0))
            if getattr(m, "fan_in_fan_out", False):
                delta = delta.transpose(0, 1)
            m.weight.data.add_(delta.to(dtype=m.weight.dtype, device=m.weight.device))
            m.merged = True
            merged_count += 1
    return merged_count
