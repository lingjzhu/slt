"""Smoke training test for the discrete-diffusion stage-2 package.

Runs a couple of optimizer steps with a tiny synthetic dataset, a tiny
BERT-based text backbone (avoids needing ModernBERT/flash_attn on this box),
and a tiny student visual backbone wrapped by an EMA teacher.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, BertModel

# ensure src is importable when run as `python smoke_train.py`
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from discrete_diffusion_pretraining.data import (  # noqa: E402
    DiffusionExample,
    DiscreteDiffusionCollator,
    DiscreteDiffusionDataset,
    sample_masks,
)
from discrete_diffusion_pretraining.decode import iterative_decode  # noqa: E402
from discrete_diffusion_pretraining.model import DiscreteDiffusionModel  # noqa: E402
from discrete_diffusion_pretraining.teacher import EMATeacher  # noqa: E402


class PooledVisualBackbone(nn.Module):
    """Tiny student backbone: linear over precomputed pooled frame features.

    Accepts `[B, T, D_in]` and returns `[B, T, D_out]` (identity by default).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        nn.init.eye_(self.proj.weight[: min(d_in, d_out), : min(d_in, d_out)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_fake_features(tmp: Path, n: int, T: int, D: int) -> list[DiffusionExample]:
    rows = []
    tmp.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        feat = torch.randn(T, D)
        p = tmp / f"feat_{i}.pt"
        torch.save(feat, p)
        rows.append(
            DiffusionExample(
                sample_id=f"s{i}",
                feature_path=str(p),
                teacher_feature_path=None,
                prompt_text="translate to english",
                target_text=f"sample number {i} is a test sentence.",
                language="asl",
            )
        )
    return rows


def main() -> None:
    torch.manual_seed(0)

    # Tiny BERT-based text encoder used in place of ModernBERT (flash_attn issues).
    config = BertConfig(
        vocab_size=30522,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=256,
    )
    text_backbone = BertModel(config)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    D_vis = 32
    student_backbone = PooledVisualBackbone(D_vis, D_vis)
    teacher = EMATeacher(student_backbone, decay=0.99)

    # EMA-init sanity
    for p_s, p_t in zip(student_backbone.parameters(), teacher.backbone.parameters()):
        assert torch.allclose(p_s, p_t), "teacher must init from student"

    model = DiscreteDiffusionModel(
        visual_feature_dim=D_vis,
        teacher_feature_dim=D_vis,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        text_backbone=text_backbone,
        student_backbone=student_backbone,
        vocab_size=config.vocab_size,
    )

    # Build dataset
    tmp = Path("/tmp/ddp_smoke")
    examples = _make_fake_features(tmp, n=8, T=6, D=D_vis)
    dataset = DiscreteDiffusionDataset(examples)
    collator = DiscreteDiffusionCollator(tokenizer, max_text_length=32)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

    # mask-inversion check on one batch
    batch = next(iter(loader))
    union = (batch["text_mask_a"] | batch["text_mask_b"])
    inter = (batch["text_mask_a"] & batch["text_mask_b"])
    assert torch.equal(union, batch["text_maskable"]), "base ∪ inverse must equal maskable"
    assert inter.sum().item() == 0, "base ∩ inverse must be empty"
    print(
        f"[mask] text_maskable={int(batch['text_maskable'].sum())} "
        f"a={int(batch['text_mask_a'].sum())} b={int(batch['text_mask_b'].sum())}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    teacher_param_before = next(teacher.backbone.parameters()).detach().clone()

    model.train()
    for step, batch in enumerate(loader):
        # student visual encoding
        visual_features_raw = batch["visual_features"]
        # teacher runs on the same (pre-pooled) input in this smoke test
        with torch.no_grad():
            visual_targets = teacher(visual_features_raw)
        visual_features = model.encode_visual(visual_features_raw)

        out = model(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            text_labels=batch["text_labels"],
            text_mask_a=batch["text_mask_a"],
            text_mask_b=batch["text_mask_b"],
            visual_features=visual_features,
            visual_attention_mask=batch["visual_attention_mask"],
            visual_targets=visual_targets,
            visual_mask_a=batch["visual_mask_a"],
            visual_mask_b=batch["visual_mask_b"],
        )
        loss = out["loss"]

        # step-loss check
        expected = 0.5 * (out["pass_a"].loss + out["pass_b"].loss)
        assert torch.allclose(loss, expected), "step loss must equal mean of pass losses"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # teacher gradient-freeness check
        for p in teacher.backbone.parameters():
            assert p.grad is None

        optimizer.step()
        teacher.update(student_backbone)

        print(
            f"[step {step}] loss={loss.item():.4f} "
            f"text={out['pass_a'].text_loss.item():.4f}/{out['pass_b'].text_loss.item():.4f} "
            f"vis={out['pass_a'].visual_loss.item():.4f}/{out['pass_b'].visual_loss.item():.4f}"
        )
        if step >= 1:
            break

    teacher_param_after = next(teacher.backbone.parameters()).detach().clone()
    assert not torch.allclose(teacher_param_before, teacher_param_after), (
        "teacher weights must change after EMA update"
    )

    # decode smoke
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        vis_feats = model.encode_visual(batch["visual_features"])
    editable = batch["text_maskable"]
    out_ids = iterative_decode(
        model,
        visual_features=vis_feats,
        visual_attention_mask=batch["visual_attention_mask"],
        text_input_ids=batch["text_input_ids"],
        text_attention_mask=batch["text_attention_mask"],
        editable_mask=editable,
        num_steps=4,
    )
    # all editable positions have been committed (no MASK remaining there)
    remaining_masks = (out_ids == model.mask_token_id) & editable.bool()
    assert remaining_masks.sum().item() == 0, "decoding must commit all editable positions"
    # prefix (non-editable) unchanged
    non_edit = ~editable.bool()
    assert torch.equal(out_ids[non_edit], batch["text_input_ids"][non_edit]), "prefix must stay fixed"

    print("smoke train + decode OK")


if __name__ == "__main__":
    main()
