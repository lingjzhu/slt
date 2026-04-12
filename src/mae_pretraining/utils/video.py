import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch


def get_num_padding_frames(
    idx: torch.Tensor,
    num_frames: int,
    sampling_rate: int,
    fps: float,
    target_fps: float,
) -> int:
    num_unique = len(torch.unique(idx))
    if target_fps > (fps * sampling_rate):
        num_non_padding = math.floor(num_unique * target_fps / (fps * sampling_rate))
    else:
        num_non_padding = num_unique
    return num_frames - num_non_padding


def get_start_end_idx(
    video_size: int,
    clip_size: int,
    clip_idx: int,
    num_clips: int,
    use_offset: bool = False,
) -> Tuple[int, int]:
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                start_idx = math.floor(delta / 2)
            else:
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def temporal_sampling(
    frames: torch.Tensor,
    start_idx: int,
    end_idx: int,
    num_samples: int,
    return_index: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    index = torch.linspace(start_idx + 0.5, end_idx + 0.5, num_samples, device=frames.device)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    if return_index:
        return new_frames, index
    return new_frames


def tensor_normalize(
    tensor: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float]],
    std: Union[torch.Tensor, Tuple[float, float, float]],
) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, tuple):
        mean = torch.tensor(mean, device=tensor.device)
    if isinstance(std, tuple):
        std = torch.tensor(std, device=tensor.device)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def horizontal_flip(prob: float, images: torch.Tensor) -> torch.Tensor:
    if np.random.uniform() < prob:
        images = images.flip((-1))
    return images


def random_crop(images: torch.Tensor, size: int) -> torch.Tensor:
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = int(np.random.randint(0, height - size)) if height > size else 0
    x_offset = int(np.random.randint(0, width - size)) if width > size else 0
    return images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]


def uniform_crop(
    images: torch.Tensor, size: int, spatial_idx: int, scale_size: Optional[int] = None
) -> torch.Tensor:
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size

    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped
