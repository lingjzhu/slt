from __future__ import annotations

import argparse
import os
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

from .data import SignT5Collator, SignT5Dataset
from .metrics import compute_bleurt, compute_translation_metrics, save_json
from .model import SignLanguageT5


class FeatureSeq2SeqTrainer(Seq2SeqTrainer):
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
        generation_inputs = {
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
            "prompt_input_ids": inputs["prompt_input_ids"],
            "prompt_attention_mask": inputs["prompt_attention_mask"],
        }

        if not gen_kwargs:
            gen_kwargs = dict(self._gen_kwargs)
        gen_kwargs.setdefault("max_length", self.args.generation_max_length)
        gen_kwargs.setdefault("num_beams", self.args.generation_num_beams)

        generated_tokens = model.generate(**generation_inputs, **gen_kwargs)

        loss = None
        with torch.no_grad():
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss.detach().mean()

        labels = inputs["labels"] if has_labels else None
        if labels is not None and labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        return loss, generated_tokens, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a T5 SLT model over extracted SignHiera features.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--prompt-template", type=str, default="translate to {language}")
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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def _parse_csv_set(raw: str) -> set[str] | None:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return set(values) if values else None


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
                    "prediction": prediction.strip(),
                    "reference": reference.strip(),
                }
            )

    overall = compute_translation_metrics(
        [row["prediction"] for row in predictions],
        [row["reference"] for row in predictions],
    )
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
        by_dataset[key]["bleurt"] = compute_bleurt(
            [row["prediction"] for row in rows],
            [row["reference"] for row in rows],
        )
        by_dataset[key]["num_samples"] = len(rows)

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

    result = {
        "split": split_name,
        "num_samples": len(predictions),
        "overall": overall,
        "by_language": by_language,
        "by_dataset": by_dataset,
    }
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

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    train_dataset = SignT5Dataset(
        args.train_manifest,
        prompt_template=args.prompt_template,
        languages=languages,
        datasets=datasets,
        max_source_length=args.max_source_length,
    )
    val_dataset = SignT5Dataset(
        args.val_manifest,
        prompt_template=args.prompt_template,
        languages=languages,
        datasets=datasets,
        max_source_length=args.max_source_length,
    )
    test_dataset = SignT5Dataset(
        args.test_manifest,
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
        metric_for_best_model="bleu4",
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
        ),
    }
    save_json(eval_results, output_dir / "eval" / "summary.json")


if __name__ == "__main__":
    main()
