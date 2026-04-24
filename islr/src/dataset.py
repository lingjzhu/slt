from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import decord
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CSV = PROJECT_ROOT / "preprocessed_data.csv"
FULL_DATA_CSV = PROJECT_ROOT / "data" / "preprocessed_data_full.csv"
VIDEO_ROOT = PROJECT_ROOT / "data" / "preprocessed_videos_full"
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)


@dataclass(frozen=True)
class Sample:
    video_path: str
    label_index: int
    raw_label: int
    transcription: str
    view: str


def make_label_mapping(dataframes: Iterable[pd.DataFrame]) -> dict[int, int]:
    ids: set[int] = set()
    for dataframe in dataframes:
        ids.update(int(label_id) for label_id in dataframe["id"].unique())
    return {label_id: index for index, label_id in enumerate(sorted(ids))}


def split_dataset_frame(
    csv_path: str | Path,
    val_samples_per_class: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = pd.read_csv(csv_path)
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_indices: list[int] = []
    for _, group in shuffled.groupby("id", sort=True):
        take = min(val_samples_per_class, len(group) - 1) if len(group) > 1 else 0
        if take > 0:
            val_indices.extend(group.index[:take].tolist())

    val_frame = shuffled.loc[val_indices].reset_index(drop=True)
    train_frame = shuffled.drop(index=val_indices).reset_index(drop=True)
    return train_frame, val_frame


class ISLRDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        training: bool = True,
        label_to_index: dict[int, int] | None = None,
        view_mode: str = "both",
    ) -> None:
        if dataframe is None and csv_path is None:
            raise ValueError("Either csv_path or dataframe must be provided.")

        self.source_name = str(csv_path) if csv_path is not None else "in-memory dataframe"
        self.dataframe = dataframe.copy() if dataframe is not None else pd.read_csv(csv_path)
        self.training = training
        self.view_mode = view_mode
        self.label_to_index = label_to_index or make_label_mapping([self.dataframe])
        self.transform = self._build_transform(training=training)
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(f"No usable samples were found in {self.source_name}.")

        print(
            f"Loaded {len(self.samples)} samples from {self.source_name} "
            f"across {len(self.label_to_index)} classes."
        )

    def _build_transform(self, training: bool) -> T.Compose:
        ops: list[torch.nn.Module] = [
            T.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.RandomCrop(224) if training else T.CenterCrop(224),
        ]
        if training:
            ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.Normalize(mean=MEAN, std=STD))
        return T.Compose(ops)

    def _build_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for _, row in self.dataframe.iterrows():
            raw_label = int(row["id"])
            label_index = self.label_to_index[raw_label]
            transcription = str(row.get("transcription", ""))

            for view_name, column_name in (("front", "filepath_front"), ("left", "filepath_left")):
                if self.view_mode != "both" and self.view_mode != view_name:
                    continue

                raw_path = row.get(column_name)
                resolved_path = self._resolve_video_path(raw_path, view_name=view_name)
                if resolved_path is None:
                    continue

                samples.append(
                    Sample(
                        video_path=str(resolved_path),
                        label_index=label_index,
                        raw_label=raw_label,
                        transcription=transcription,
                        view=view_name,
                    )
                )
        return samples

    def _resolve_video_path(self, raw_path: object, view_name: str) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path:
            return None

        candidate = Path(raw_path)
        fallback_candidates = [
            candidate,
            PROJECT_ROOT / candidate.name,
            VIDEO_ROOT / view_name / candidate.name,
            Path(str(candidate).replace("/preprocessed_videos_full/", "/data/preprocessed_videos_full/")),
            Path(str(candidate).replace("/preprocessed_videos/", "/data/preprocessed_videos_full/")),
        ]

        for fallback in fallback_candidates:
            if fallback.exists():
                return fallback
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]

        try:
            video_reader = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
            frame_indices = list(range(len(video_reader)))
            video = video_reader.get_batch(frame_indices)
            video_tensor = torch.from_numpy(video.asnumpy()).permute(0, 3, 1, 2).float() / 255.0
        except Exception as exc:
            raise RuntimeError(f"Failed to decode {sample.video_path}: {exc}") from exc

        video_tensor = self.transform(video_tensor)
        video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()
        return video_tensor, sample.label_index


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[list[torch.Tensor], torch.Tensor]:
    videos, labels = zip(*batch)
    return list(videos), torch.tensor(labels, dtype=torch.long)
