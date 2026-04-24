from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Any
import unicodedata

import numpy as np
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


@dataclass(frozen=True)
class TarFeatureExample:
    sample_id: str
    tar_path: str
    npz_member: str
    dataset_name: str
    split: str
    language: str
    sign_language: str
    output_language: str
    caption: str
    length: int


_OUTPUT_LANGUAGE_MAP = {
    "asl": "english",
    "bsl": "english",
    "csl": "chinese",
}

_SIGN_LANGUAGE_MAP = {
    "asl": "american sign language",
    "bsl": "british sign language",
    "csl": "chinese sign language",
}

_DATASET_LANGUAGE_MAP = {
    "asl_citizen": "asl",
    "wlasl": "asl",
    "csl_large": "csl",
}

_FEATURE_KEYS = (
    "body/global_pose",
    "body/body_pose",
    "body/left_hand_pose",
    "body/right_hand_pose",
    "body/hand_scale",
    "body/head_scale",
    "body/exp",
    "body/shape",
    "flame/eye_pose_params",
    "flame/pose_params",
    "flame/jaw_params",
    "flame/eyelid_params",
    "flame/expression_params",
    "flame/shape_params",
    "camera/pd_cam",
)


def _normalize_english_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_chinese_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str, language: str | None = None) -> str:
    language = (language or "").strip().lower()
    if language == "csl":
        return _normalize_chinese_text(text)
    if language in {"asl", "bsl"}:
        return _normalize_english_text(text)

    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_output_language_name(language: str) -> str:
    return _OUTPUT_LANGUAGE_MAP.get(language.strip().lower(), language.strip().lower())


def get_sign_language_name(language: str) -> str:
    return _SIGN_LANGUAGE_MAP.get(language.strip().lower(), language.strip().lower())


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
                    caption=normalize_text(row["caption"], language),
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
        prompt_template: str = "translate {sign_language} to {language}",
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
            "prompt_text": self.prompt_template.format(
                sign_language=get_sign_language_name(example.language),
                language=get_output_language_name(example.language),
            ),
            "input_features": features,
            "length": features.size(0),
        }


def _stable_val_membership(sample_id: str, ratio: float) -> bool:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return score < ratio


def _extract_target_text_from_metadata(dataset_name: str, metadata: dict[str, Any]) -> str:
    for key in ("transcription", "gloss", "caption", "text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if dataset_name == "asl_citizen":
        key = str(metadata.get("key", ""))
        if "-" in key:
            return key.split("-", 1)[1]
    if dataset_name == "csl_large":
        raw_id = str(metadata.get("raw_label", metadata.get("id", ""))).strip()
        if raw_id:
            return raw_id
    return str(metadata.get("key", "")).strip()


def build_webdataset_manifest(
    *,
    split: str,
    manifest_path: str | Path,
    data_root: str | Path,
    metadata_root: str | Path,
    datasets: set[str] | None = None,
    languages: set[str] | None = None,
    csl_val_ratio: float = 0.02,
) -> Path:
    data_root = Path(data_root)
    metadata_root = Path(metadata_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    target_datasets = sorted(datasets or _DATASET_LANGUAGE_MAP.keys())
    rows: list[dict[str, Any]] = []

    for dataset_name in target_datasets:
        language = _DATASET_LANGUAGE_MAP.get(dataset_name)
        if language is None:
            continue
        if languages and language not in languages:
            continue

        pear_split_dir = data_root / dataset_name / ("train" if dataset_name == "csl_large" and split == "val" else split)
        if not pear_split_dir.exists():
            continue

        metadata_split_dir = metadata_root / dataset_name / ("train" if dataset_name == "csl_large" and split == "val" else split)
        metadata_lookup: dict[str, dict[str, Any]] = {}
        if metadata_split_dir.exists():
            for tar_path in sorted(metadata_split_dir.glob("*.tar")):
                with tarfile.open(tar_path, "r") as tar_handle:
                    for member in tar_handle:
                        if not member.isfile() or not member.name.endswith(".json"):
                            continue
                        sample_id = Path(member.name).stem
                        with tar_handle.extractfile(member) as handle:
                            if handle is None:
                                continue
                            metadata_lookup[sample_id] = json.load(handle)

        for tar_path in sorted(pear_split_dir.glob("*.tar")):
            with tarfile.open(tar_path, "r") as tar_handle:
                for member in tar_handle:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue

                    sample_id = Path(member.name).stem
                    if dataset_name == "csl_large" and split in {"train", "val"}:
                        is_val = _stable_val_membership(sample_id, csl_val_ratio)
                        if split == "train" and is_val:
                            continue
                        if split == "val" and not is_val:
                            continue

                    with tar_handle.extractfile(member) as handle:
                        if handle is None:
                            continue
                        feature_meta = json.load(handle)

                    source_meta = metadata_lookup.get(sample_id, {})
                    raw_caption = _extract_target_text_from_metadata(dataset_name, source_meta)
                    caption = normalize_text(raw_caption, language)
                    if not caption:
                        continue

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "tar_path": str(tar_path.resolve()),
                            "npz_member": f"{sample_id}.npz",
                            "dataset": dataset_name,
                            "split": split,
                            "language": language,
                            "sign_language": get_sign_language_name(language),
                            "output_language": get_output_language_name(language),
                            "caption": caption,
                            "num_feature_vectors": int(feature_meta["num_frames"]),
                        }
                    )

    if not rows:
        raise ValueError(f"No WebDataset PEAR samples found for split={split} under {data_root}")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "tar_path",
                "npz_member",
                "dataset",
                "split",
                "language",
                "sign_language",
                "output_language",
                "caption",
                "num_feature_vectors",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def load_tar_manifest(
    manifest_path: str | Path,
    *,
    languages: set[str] | None = None,
    datasets: set[str] | None = None,
    max_source_length: int | None = None,
) -> list[TarFeatureExample]:
    manifest_path = Path(manifest_path)
    examples: list[TarFeatureExample] = []

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

            examples.append(
                TarFeatureExample(
                    sample_id=row["sample_id"],
                    tar_path=row["tar_path"],
                    npz_member=row["npz_member"],
                    dataset_name=dataset_name,
                    split=row["split"],
                    language=language,
                    sign_language=row["sign_language"],
                    output_language=row["output_language"],
                    caption=normalize_text(row["caption"], language),
                    length=length,
                )
            )

    if not examples:
        raise ValueError(f"No tar examples loaded from {manifest_path}")

    return examples


def _load_npz_features(npz_bytes: bytes) -> torch.Tensor:
    with np.load(io.BytesIO(npz_bytes)) as npz_file:
        frames: list[np.ndarray] = []
        expected_length: int | None = None
        for key in _FEATURE_KEYS:
            array = np.asarray(npz_file[key], dtype=np.float32)
            if expected_length is None:
                expected_length = int(array.shape[0])
            elif int(array.shape[0]) != expected_length:
                raise ValueError(f"Feature key {key} has inconsistent time dimension: {array.shape[0]} vs {expected_length}")
            frames.append(array.reshape(array.shape[0], -1))

    features = np.concatenate(frames, axis=1)
    return torch.from_numpy(features)


class SignTarFeatureDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        prompt_template: str = "translate {sign_language} to {language}",
        languages: set[str] | None = None,
        datasets: set[str] | None = None,
        max_source_length: int | None = None,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.prompt_template = prompt_template
        self.examples = load_tar_manifest(
            manifest_path,
            languages=languages,
            datasets=datasets,
            max_source_length=max_source_length,
        )
        self.lengths = [example.length for example in self.examples]
        self._tar_handles: dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _get_tar_handle(self, tar_path: str) -> tarfile.TarFile:
        handle = self._tar_handles.get(tar_path)
        if handle is None:
            handle = tarfile.open(tar_path, "r")
            self._tar_handles[tar_path] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        tar_handle = self._get_tar_handle(example.tar_path)
        member = tar_handle.getmember(example.npz_member)
        with tar_handle.extractfile(member) as handle:
            if handle is None:
                raise FileNotFoundError(f"Unable to read {example.npz_member} from {example.tar_path}")
            features = _load_npz_features(handle.read()).float()

        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D [time, dim] gesture features in {example.tar_path}:{example.npz_member}, got {tuple(features.shape)}"
            )

        if features.size(0) != example.length:
            features = features[: example.length]

        return {
            "sample_id": example.sample_id,
            "video_path": f"{example.tar_path}:{example.npz_member}",
            "dataset_name": example.dataset_name,
            "language": example.language,
            "target_text": example.caption,
            "prompt_text": self.prompt_template.format(
                sign_language=example.sign_language,
                language=example.output_language,
            ),
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
