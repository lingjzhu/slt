from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AccuracyCounts:
    top1_correct: int = 0
    top5_correct: int = 0
    total: int = 0

    def update(self, similarities: torch.Tensor, targets: torch.Tensor) -> None:
        topk = similarities.topk(k=min(5, similarities.size(1)), dim=1).indices
        self.total += targets.numel()
        self.top1_correct += int((topk[:, 0] == targets).sum().item())
        self.top5_correct += int((topk == targets.unsqueeze(1)).any(dim=1).sum().item())

    def as_dict(self) -> dict[str, float]:
        denom = max(self.total, 1)
        return {
            "top1": self.top1_correct / denom,
            "top5": self.top5_correct / denom,
            "count": float(self.total),
        }
