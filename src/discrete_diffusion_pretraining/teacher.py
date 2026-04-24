from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn


class EMATeacher(nn.Module):
    """EMA copy of a visual backbone used only to produce pooled frame targets.

    The teacher is kept in eval mode, receives no gradients, and is updated
    exclusively via `update(student)`. It is expected to expose a callable
    returning pooled per-frame features shaped `[B, T, D]`.
    """

    def __init__(
        self,
        student: nn.Module,
        *,
        decay: float = 0.999,
        warmup_steps: int = 0,
        final_decay: float | None = None,
    ) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(student)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.base_decay = float(decay)
        self.final_decay = float(final_decay) if final_decay is not None else self.base_decay
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True):  # always eval
        return super().train(False)

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.base_decay
        step = int(self._step.item())
        if step >= self.warmup_steps:
            return self.final_decay
        frac = step / max(1, self.warmup_steps)
        return self.base_decay + (self.final_decay - self.base_decay) * frac

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        decay = self.current_decay()
        for tp, sp in zip(self.backbone.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
        for tb, sb in zip(self.backbone.buffers(), student.buffers()):
            if tb.dtype.is_floating_point:
                tb.data.mul_(decay).add_(sb.data.to(tb.dtype), alpha=1.0 - decay)
            else:
                tb.data.copy_(sb.data)
        self._step += 1

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        was_training = self.backbone.training
        self.backbone.eval()
        out = self.backbone(*args, **kwargs)
        if was_training:
            self.backbone.train(False)
        return out
