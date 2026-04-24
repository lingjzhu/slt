from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, LengthGroupedSampler

from .data import (
    SignT5Collator,
    SignT5Dataset,
    SignTarFeatureDataset,
    build_webdataset_manifest,
    normalize_text,
)
from .metrics import compute_bleurt, compute_translation_metrics, save_json
from .model import SignLanguageT5


class FeatureSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, generation_dump_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_dump_dir = Path(generation_dump_dir) if generation_dump_dir else None

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if not self.args.group_by_length:
            return super()._get_train_sampler()
        lengths = getattr(dataset, "lengths", None)
        if lengths is None:
            return super()._get_train_sampler()
        if self.args.world_size <= 1:
            return LengthGroupedSampler(
                self.args.train_batch_size,
                lengths=lengths,
            )
        return DistributedLengthGroupedSampler(
            self.args.train_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            lengths=lengths,
        )

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **gen_kwargs: Any,
    ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        generation_model = model.module if hasattr(model, "module") else model

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")
        if "max_length" not in gen_kwargs and self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if "num_beams" not in gen_kwargs and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        generation_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {
                "input_features",
                "feature_attention_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
            }
        }
        generated_tokens = generation_model.generate(**generation_inputs, **gen_kwargs)
        gen_config = (
            generation_model.t5.generation_config
            if hasattr(generation_model, "t5")
            else generation_model.generation_config
        )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = generation_model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).detach().mean()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
            else:
                loss = None

        labels = inputs["labels"] if has_labels else None
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
        if labels is not None:
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)

        return loss, generated_tokens, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.generation_dump_dir and self.is_world_process_zero():
            target_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if target_dataset is not None:
                tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                if tokenizer is None:
                    return metrics
                generation_output_dir = self.generation_dump_dir / "eval_generations"
                model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
                device = next(model_for_eval.parameters()).device
                evaluate_split(
                    split_name=f"{metric_key_prefix}_step{self.state.global_step}",
                    model=model_for_eval,
                    tokenizer=tokenizer,
                    dataset=target_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    num_workers=self.args.dataloader_num_workers,
                    max_length=self.args.generation_max_length,
                    num_beams=self.args.generation_num_beams,
                    output_dir=generation_output_dir,
                    device=device,
                    include_metrics=False,
                )
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a T5 SLT model over extracted SignHiera features.")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="webdataset_tar",
        choices=["feature_manifest", "tar_manifest", "webdataset_tar"],
    )
    parser.add_argument("--train-manifest", type=str, default="")
    parser.add_argument("--val-manifest", type=str, default="")
    parser.add_argument("--test-manifest", type=str, default="")
    parser.add_argument("--data-root", type=str, default="/mnt/data2/sign_language_24fps")
    parser.add_argument("--metadata-root", type=str, default="/home/slimelab/Projects/slt/islr/webdataset_224")
    parser.add_argument("--csl-val-ratio", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--feature-dim", type=int, default=1104)
    parser.add_argument("--prompt-template", type=str, default="translate {sign_language} to {language}")
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--languages", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="sign-t5-slt")
    parser.add_argument("--wandb-run-name", type=str, default="t5_signhiera_run001")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--compute-bleurt", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def _parse_csv_set(raw: str) -> set[str] | None:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return set(values) if values else None


def build_compute_metrics(tokenizer):
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        # Guard against occasional invalid ids introduced by distributed gather/padding.
        predictions = predictions.astype(np.int64, copy=False)
        predictions = np.where((predictions >= 0) & (predictions < vocab_size), predictions, pad_token_id)

        labels = np.where(labels != -100, labels, pad_token_id)
        labels = labels.astype(np.int64, copy=False)
        labels = np.where((labels >= 0) & (labels < vocab_size), labels, pad_token_id)

        decoded_predictions = [
            normalize_text(text) for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ]
        decoded_labels = [normalize_text(text) for text in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return compute_translation_metrics(decoded_predictions, decoded_labels)

    return compute_metrics


def evaluate_split(
    *,
    split_name: str,
    model: SignLanguageT5,
    tokenizer,
    dataset: SignT5Dataset,
    batch_size: int,
    num_workers: int,
    max_length: int,
    num_beams: int,
    output_dir: Path,
    device: torch.device,
    include_metrics: bool = True,
    compute_bleurt_metrics: bool = False,
) -> dict[str, Any]:
    collator = SignT5Collator(
        tokenizer,
        max_target_length=max_length,
        include_metadata=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    predictions: list[dict[str, Any]] = []
    model.eval()

    for batch in dataloader:
        sample_ids = batch.pop("sample_ids")
        video_paths = batch.pop("video_paths")
        dataset_names = batch.pop("dataset_names")
        languages = batch.pop("languages")
        target_texts = batch.pop("target_texts")
        prompt_texts = batch.pop("prompt_texts")

        generation_inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key in {
                "input_features",
                "feature_attention_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
            }
        }

        generated = model.generate(
            **generation_inputs,
            max_length=max_length,
            num_beams=num_beams,
        )
        decoded_predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for sample_id, video_path, dataset_name, language, prompt_text, prediction, reference in zip(
            sample_ids,
            video_paths,
            dataset_names,
            languages,
            prompt_texts,
            decoded_predictions,
            target_texts,
        ):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "dataset": dataset_name,
                    "language": language,
                    "prompt": prompt_text,
                    "prediction": normalize_text(prediction),
                    "reference": normalize_text(reference),
                }
            )

    split_output_dir = output_dir / "eval"
    split_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = split_output_dir / f"{split_name}_predictions.tsv"
    with predictions_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id\tlanguage\tdataset\tprompt\tprediction\treference\tvideo_path\n")
        for row in predictions:
            handle.write(
                "\t".join(
                    [
                        row["sample_id"],
                        row["language"],
                        row["dataset"],
                        row["prompt"].replace("\t", " "),
                        row["prediction"].replace("\t", " "),
                        row["reference"].replace("\t", " "),
                        row["video_path"],
                    ]
                )
                + "\n"
            )

    result: dict[str, Any] = {
        "split": split_name,
        "num_samples": len(predictions),
    }
    if include_metrics:
        overall = compute_translation_metrics(
            [row["prediction"] for row in predictions],
            [row["reference"] for row in predictions],
        )
        if compute_bleurt_metrics:
            overall["bleurt"] = compute_bleurt(
                [row["prediction"] for row in predictions],
                [row["reference"] for row in predictions],
            )
        by_language: dict[str, dict[str, float]] = {}
        by_dataset: dict[str, dict[str, float]] = {}

        grouped_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped_language[row["language"]].append(row)
            grouped_dataset[row["dataset"]].append(row)

        for key, rows in grouped_language.items():
            by_language[key] = compute_translation_metrics(
                [row["prediction"] for row in rows],
                [row["reference"] for row in rows],
            )
            if compute_bleurt_metrics:
                by_language[key]["bleurt"] = compute_bleurt(
                    [row["prediction"] for row in rows],
                    [row["reference"] for row in rows],
                )
            by_language[key]["num_samples"] = len(rows)
        for key, rows in grouped_dataset.items():
            by_dataset[key] = compute_translation_metrics(
                [row["prediction"] for row in rows],
                [row["reference"] for row in rows],
            )
            if compute_bleurt_metrics:
                by_dataset[key]["bleurt"] = compute_bleurt(
                    [row["prediction"] for row in rows],
                    [row["reference"] for row in rows],
                )
            by_dataset[key]["num_samples"] = len(rows)

        result["overall"] = overall
        result["by_language"] = by_language
        result["by_dataset"] = by_dataset
        save_json(result, split_output_dir / f"{split_name}_metrics.json")
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    languages = _parse_csv_set(args.languages)
    datasets = _parse_csv_set(args.datasets)
    manifest_dir = output_dir / "manifests"

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    if args.dataset_format == "webdataset_tar":
        train_manifest = manifest_dir / "train_webdataset.tsv"
        val_manifest = manifest_dir / "val_webdataset.tsv"
        test_manifest = manifest_dir / "test_webdataset.tsv"
        is_rank_zero = int(os.environ.get("RANK", "0")) == 0
        if is_rank_zero:
            if not train_manifest.exists():
                build_webdataset_manifest(
                    split="train",
                    manifest_path=train_manifest,
                    data_root=args.data_root,
                    metadata_root=args.metadata_root,
                    datasets=datasets,
                    languages=languages,
                    csl_val_ratio=args.csl_val_ratio,
                )
            if not val_manifest.exists():
                build_webdataset_manifest(
                    split="val",
                    manifest_path=val_manifest,
                    data_root=args.data_root,
                    metadata_root=args.metadata_root,
                    datasets=datasets,
                    languages=languages,
                    csl_val_ratio=args.csl_val_ratio,
                )
            if not test_manifest.exists():
                build_webdataset_manifest(
                    split="test",
                    manifest_path=test_manifest,
                    data_root=args.data_root,
                    metadata_root=args.metadata_root,
                    datasets=datasets,
                    languages=languages,
                    csl_val_ratio=args.csl_val_ratio,
                )
        else:
            expected_manifests = (train_manifest, val_manifest, test_manifest)
            deadline = time.time() + 3600
            while time.time() < deadline:
                if all(path.exists() for path in expected_manifests):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"Timed out waiting for cached manifests in {manifest_dir}")

        dataset_cls = SignTarFeatureDataset
        train_manifest_path = train_manifest
        val_manifest_path = val_manifest
        test_manifest_path = test_manifest
    elif args.dataset_format == "tar_manifest":
        if not args.train_manifest or not args.val_manifest or not args.test_manifest:
            raise ValueError(
                "--train-manifest, --val-manifest, and --test-manifest are required when --dataset-format=tar_manifest"
            )
        dataset_cls = SignTarFeatureDataset
        train_manifest_path = Path(args.train_manifest)
        val_manifest_path = Path(args.val_manifest)
        test_manifest_path = Path(args.test_manifest)
    else:
        if not args.train_manifest or not args.val_manifest or not args.test_manifest:
            raise ValueError(
                "--train-manifest, --val-manifest, and --test-manifest are required when --dataset-format=feature_manifest"
            )
        dataset_cls = SignT5Dataset
        train_manifest_path = Path(args.train_manifest)
        val_manifest_path = Path(args.val_manifest)
        test_manifest_path = Path(args.test_manifest)

    train_dataset = dataset_cls(
        train_manifest_path,
        prompt_template=args.prompt_template,
        languages=languages,
        datasets=datasets,
        max_source_length=args.max_source_length,
    )
    val_dataset = dataset_cls(
        val_manifest_path,
        prompt_template=args.prompt_template,
        languages=languages,
        datasets=datasets,
        max_source_length=args.max_source_length,
    )
    test_dataset = dataset_cls(
        test_manifest_path,
        prompt_template=args.prompt_template,
        languages=languages,
        datasets=datasets,
        max_source_length=args.max_source_length,
    )

    model = SignLanguageT5(
        args.model_name_or_path,
        feature_dim=args.feature_dim,
        projection_dropout=args.projection_dropout,
        attn_implementation=args.attn_implementation,
        use_efficient=True,
        efficient_sdpa=True,
        efficient_flex=False,
        efficient_compile=False,
    )

    train_collator = SignT5Collator(tokenizer, max_target_length=args.max_target_length)
    use_bf16 = args.bf16 or (not args.fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.wandb_run_name,
        ddp_find_unused_parameters=False,
        fp16=args.fp16 and not use_bf16,
        bf16=use_bf16,
        save_safetensors=False,
    )

    trainer = FeatureSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        generation_dump_dir=str(output_dir),
    )

    save_json(vars(args), output_dir / "run_config.json")

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if training_args.process_index != 0:
        return

    model_for_eval = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    device = next(model_for_eval.parameters()).device
    eval_results = {
        "val": evaluate_split(
            split_name="val",
            model=model_for_eval,
            tokenizer=tokenizer,
            dataset=val_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
            max_length=args.generation_max_length,
            num_beams=args.num_beams,
            output_dir=output_dir,
            device=device,
            compute_bleurt_metrics=args.compute_bleurt,
        ),
        "test": evaluate_split(
            split_name="test",
            model=model_for_eval,
            tokenizer=tokenizer,
            dataset=test_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
            max_length=args.generation_max_length,
            num_beams=args.num_beams,
            output_dir=output_dir,
            device=device,
            compute_bleurt_metrics=args.compute_bleurt,
        ),
    }
    save_json(eval_results, output_dir / "eval" / "summary.json")


if __name__ == "__main__":
    main()
