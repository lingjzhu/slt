from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LossMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass
class TrainingMetrics:
    loss: LossMeter = field(default_factory=LossMeter)
    text_loss: LossMeter = field(default_factory=LossMeter)
    visual_loss: LossMeter = field(default_factory=LossMeter)

    def update_from_step(self, step_loss: float, pass_a, pass_b, n: int = 1) -> None:
        self.loss.update(step_loss, n)
        self.text_loss.update(0.5 * (float(pass_a.text_loss) + float(pass_b.text_loss)), n)
        self.visual_loss.update(0.5 * (float(pass_a.visual_loss) + float(pass_b.visual_loss)), n)

    def as_dict(self) -> dict:
        return {
            "loss": self.loss.avg,
            "text_loss": self.text_loss.avg,
            "visual_loss": self.visual_loss.avg,
        }


def bleu_corpus(predictions: list[str], references: list[str]) -> float:
    """Lightweight corpus BLEU via sacrebleu if available, else 0."""
    try:
        import sacrebleu
    except Exception:
        return 0.0
    score = sacrebleu.corpus_bleu(predictions, [references])
    return float(score.score)
