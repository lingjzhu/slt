from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class FeatureExample:
    sample_id: str
    video_path: str
    feature_path: str
    dataset_name: str
    language: str
    caption: str
    length: int


def _resolve_feature_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    split_root = manifest_path.parent.parent
    return (split_root / candidate).resolve()


def load_manifest(
    manifest_path: str | Path,
    *,
    languages: set[str] | None = None,
    datasets: set[str] | None = None,
    max_source_length: int | None = None,
) -> list[FeatureExample]:
    manifest_path = Path(manifest_path)
    examples: list[FeatureExample] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            language = row["language"].strip().lower()
            dataset_name = row["dataset"].strip()
            if languages and language not in languages:
                continue
            if datasets and dataset_name not in datasets:
                continue

            length = int(row["num_feature_vectors"])
            if max_source_length is not None:
                length = min(length, max_source_length)

            feature_path = _resolve_feature_path(manifest_path, row["feature_path"])
            sample_id = Path(row["video_path"]).stem

            examples.append(
                FeatureExample(
                    sample_id=sample_id,
                    video_path=row["video_path"],
                    feature_path=str(feature_path),
                    dataset_name=dataset_name,
                    language=language,
                    caption=row["caption"].strip(),
                    length=length,
                )
            )

    if not examples:
        raise ValueError(f"No examples loaded from {manifest_path}")

    return examples


class SignT5Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        prompt_template: str = "translate to {language}",
        languages: set[str] | None = None,
        datasets: set[str] | None = None,
        max_source_length: int | None = None,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.prompt_template = prompt_template
        self.examples = load_manifest(
            manifest_path,
            languages=languages,
            datasets=datasets,
            max_source_length=max_source_length,
        )
        self.lengths = [example.length for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        features = torch.load(example.feature_path, map_location="cpu")
        if isinstance(features, dict):
            for key in ("features", "hidden_states", "x"):
                if key in features and torch.is_tensor(features[key]):
                    features = features[key]
                    break
            else:
                raise ValueError(
                    f"Unsupported feature dict in {example.feature_path}: {sorted(features.keys())}"
                )

        if not torch.is_tensor(features):
            raise TypeError(f"Expected a tensor in {example.feature_path}, got {type(features)}")

        features = features.float()
        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D [time, dim] features in {example.feature_path}, got {tuple(features.shape)}"
            )

        if features.size(0) != example.length:
            features = features[: example.length]

        return {
            "sample_id": example.sample_id,
            "video_path": example.video_path,
            "dataset_name": example.dataset_name,
            "language": example.language,
            "target_text": example.caption,
            "prompt_text": self.prompt_template.format(language=example.language),
            "input_features": features,
            "length": features.size(0),
        }


class SignT5Collator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_target_length: int,
        max_prompt_length: int = 16,
        include_metadata: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.max_prompt_length = max_prompt_length
        self.include_metadata = include_metadata

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(features)
        max_source_length = max(feature["input_features"].size(0) for feature in features)
        feature_dim = features[0]["input_features"].size(1)

        input_features = torch.zeros(batch_size, max_source_length, feature_dim, dtype=torch.float32)
        feature_attention_mask = torch.zeros(batch_size, max_source_length, dtype=torch.long)

        for row, feature in enumerate(features):
            length = feature["input_features"].size(0)
            input_features[row, :length] = feature["input_features"]
            feature_attention_mask[row, :length] = 1

        prompt_batch = self.tokenizer(
            [feature["prompt_text"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        label_batch = self.tokenizer(
            text_target=[feature["target_text"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = label_batch["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "labels": labels,
            "length": torch.tensor([feature["length"] for feature in features], dtype=torch.long),
        }

        if self.include_metadata:
            batch.update(
                {
                    "sample_ids": [feature["sample_id"] for feature in features],
                    "video_paths": [feature["video_path"] for feature in features],
                    "dataset_names": [feature["dataset_name"] for feature in features],
                    "languages": [feature["language"] for feature in features],
                    "target_texts": [feature["target_text"] for feature in features],
                    "prompt_texts": [feature["prompt_text"] for feature in features],
                }
            )

        return batch

