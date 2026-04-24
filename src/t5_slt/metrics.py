from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from .data import normalize_text


def _compute_bleu_from_stats(
    counts: list[int], totals: list[int], sys_len: int, ref_len: int, order: int
) -> float:
    try:
        from sacrebleu.metrics import BLEU

        compute_bleu = BLEU.compute_bleu
    except ImportError:
        compute_bleu = sacrebleu.compute_bleu

    kwargs = {}
    arg_names = inspect.getfullargspec(compute_bleu)[0]
    if "smooth_method" in arg_names:
        kwargs["smooth_method"] = "exp"
    else:
        kwargs["smooth"] = "exp"

    bleu = compute_bleu(
        correct=np.array(counts[:order]),
        total=np.array(totals[:order]),
        sys_len=int(sys_len),
        ref_len=int(ref_len),
        max_ngram_order=order,
        **kwargs,
    )
    return round(float(bleu.score), 2)


def _compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return float(np.mean(scores)) if scores else 0.0


def _compute_meteor(predictions: list[str], references: list[str]) -> float:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    for package in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)

    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(meteor_score([ref.split()], pred.split()))
    return float(np.mean(scores)) if scores else 0.0


def compute_translation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    predictions = [normalize_text(text) for text in predictions]
    references = [normalize_text(text) for text in references]

    exact_matches = [int(pred == ref) for pred, ref in zip(predictions, references)]
    accuracy = float(np.mean(exact_matches)) if exact_matches else 0.0

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score

    metrics = {
        "accuracy": accuracy,
        "exact_match": accuracy,
        "bleu1": _compute_bleu_from_stats(bleu.counts, bleu.totals, bleu.sys_len, bleu.ref_len, 1),
        "bleu2": _compute_bleu_from_stats(bleu.counts, bleu.totals, bleu.sys_len, bleu.ref_len, 2),
        "bleu3": _compute_bleu_from_stats(bleu.counts, bleu.totals, bleu.sys_len, bleu.ref_len, 3),
        "bleu4": _compute_bleu_from_stats(bleu.counts, bleu.totals, bleu.sys_len, bleu.ref_len, 4),
        "rougeL": _compute_rouge_l(predictions, references),
        "meteor": _compute_meteor(predictions, references),
        "chrf": float(chrf),
    }
    return metrics


def compute_bleurt(predictions: list[str], references: list[str]) -> float:
    import evaluate

    predictions = [normalize_text(text) for text in predictions]
    references = [normalize_text(text) for text in references]
    bleurt_metric = evaluate.load("bleurt", module_type="metric", config_name="BLEURT-20")
    scores = bleurt_metric.compute(predictions=predictions, references=references)["scores"]
    return float(np.mean(scores)) if scores else 0.0


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
