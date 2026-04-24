from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


@dataclass(frozen=True)
class WindowConfig:
    clip_length: int = 16
    stride: int = 16


class MViTIsolatedSignModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        clip_length: int = 16,
        clip_stride: int = 16,
        segment_batch_size: int = 16,
    ) -> None:
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        self.network = mvit_v2_s(weights=weights)
        self.window_config = WindowConfig(clip_length=clip_length, stride=clip_stride)
        self.segment_batch_size = segment_batch_size

        in_features = self.network.head[1].in_features
        dropout = self.network.head[0].p
        self.network.head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, videos: list[torch.Tensor]) -> torch.Tensor:
        window_batches: list[torch.Tensor] = []
        window_counts: list[int] = []

        for video in videos:
            clip_windows = self._window_video(video)
            window_batches.append(clip_windows)
            window_counts.append(clip_windows.shape[0])

        flat_windows = torch.cat(window_batches, dim=0)
        logits = self._run_window_batches(flat_windows)
        pooled_logits = logits.split(window_counts, dim=0)
        return torch.stack([clip_logits.mean(dim=0) for clip_logits in pooled_logits], dim=0)

    def _run_window_batches(self, flat_windows: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, flat_windows.shape[0], self.segment_batch_size):
            end = start + self.segment_batch_size
            outputs.append(self.network(flat_windows[start:end]))
        return torch.cat(outputs, dim=0)

    def _window_video(self, video: torch.Tensor) -> torch.Tensor:
        clip_length = self.window_config.clip_length
        stride = self.window_config.stride
        _, total_frames, _, _ = video.shape

        if total_frames < clip_length:
            pad_count = clip_length - total_frames
            pad_frame = video[:, -1:, :, :].expand(-1, pad_count, -1, -1)
            return torch.cat([video, pad_frame], dim=1).unsqueeze(0)

        if total_frames == clip_length:
            return video.unsqueeze(0)

        start_indices = list(range(0, total_frames - clip_length + 1, stride))
        final_start = total_frames - clip_length
        if start_indices[-1] != final_start:
            start_indices.append(final_start)

        clips = [video[:, start : start + clip_length, :, :] for start in start_indices]
        return torch.stack(clips, dim=0)
