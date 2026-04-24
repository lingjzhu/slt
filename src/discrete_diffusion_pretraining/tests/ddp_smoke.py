"""DDP smoke test. Launch with:
  torchrun --standalone --nproc_per_node=2 src/discrete_diffusion_pretraining/tests/ddp_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from discrete_diffusion_pretraining.data import (  # noqa: E402
    DiffusionExample,
    DiscreteDiffusionCollator,
    DiscreteDiffusionDataset,
)
from discrete_diffusion_pretraining.model import DiscreteDiffusionModel  # noqa: E402
from discrete_diffusion_pretraining.teacher import EMATeacher  # noqa: E402


class PooledStudent(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


def _build_fake_manifest(tmp: Path, n: int, T: int, D: int) -> list[DiffusionExample]:
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        feat_path = tmp / f"f_{i}.pt"
        if int(os.environ.get("RANK", 0)) == 0 and not feat_path.exists():
            torch.save(torch.randn(T, D), feat_path)
        rows.append(
            DiffusionExample(
                sample_id=f"s{i}",
                feature_path=str(feat_path),
                teacher_feature_path=None,
                prompt_text="translate into English:",
                target_text=f"sample number {i} is a test sentence.",
                language="asl",
            )
        )
    return rows


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=256,
    )

    D_vis = 32
    tmp = Path("/tmp/ddp_smoke_ddp")
    examples = _build_fake_manifest(tmp, n=16, T=6, D=D_vis)
    torch.distributed.barrier()  # ensure rank 0 finished writing files

    dataset = DiscreteDiffusionDataset(examples)
    collator = DiscreteDiffusionCollator(tokenizer, max_text_length=32)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collator)

    student = PooledStudent(D_vis).to(device)
    teacher = EMATeacher(student, decay=0.9).to(device)
    teacher_p_before = next(teacher.backbone.parameters()).detach().clone()

    text_backbone = BertModel(config).to(device)
    model = DiscreteDiffusionModel(
        visual_feature_dim=D_vis,
        teacher_feature_dim=D_vis,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        text_backbone=text_backbone,
        student_backbone=student,
        vocab_size=config.vocab_size,
    ).to(device)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optim = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=1e-3)

    sampler.set_epoch(0)
    ddp.train()
    for step, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            vis_targets = teacher(batch["visual_features"])
        out = ddp(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            text_labels=batch["text_labels"],
            text_mask_a=batch["text_mask_a"],
            text_mask_b=batch["text_mask_b"],
            visual_raw=batch["visual_features"],
            visual_attention_mask=batch["visual_attention_mask"],
            visual_targets=vis_targets,
            visual_mask_a=batch["visual_mask_a"],
            visual_mask_b=batch["visual_mask_b"],
        )
        loss = out["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        # inspect grad of mlm_head on each rank BEFORE step
        g = next(ddp.module.mlm_head.parameters()).grad.detach().clone()
        g_gathered = [torch.zeros_like(g) for _ in range(world)]
        torch.distributed.all_gather(g_gathered, g)
        grad_diff = max(float((g_gathered[0] - gg).abs().max()) for gg in g_gathered)
        if is_main:
            print(f"[step{step}] grad_max_diff_across_ranks={grad_diff:.2e}")
        optim.step()
        teacher.update(ddp.module.student_backbone)

        # sanity: all-reduce a cloned param to confirm DDP synced
        p = next(ddp.module.mlm_head.parameters()).detach().clone()
        gathered = [torch.zeros_like(p) for _ in range(world)]
        torch.distributed.all_gather(gathered, p)
        max_diff = max(float((gathered[0] - g).abs().max()) for g in gathered)
        if is_main:
            print(f"[rank{rank} step{step}] loss={loss.item():.4f} ddp_param_max_diff={max_diff:.2e}")
        assert max_diff < 1e-5, f"DDP params diverged across ranks: {max_diff}"

        # teacher should be identical across ranks too (student syncs via DDP, EMA deterministic)
        tp = next(teacher.backbone.parameters()).detach().clone()
        tp_gathered = [torch.zeros_like(tp) for _ in range(world)]
        torch.distributed.all_gather(tp_gathered, tp)
        max_teacher_diff = max(float((tp_gathered[0] - g).abs().max()) for g in tp_gathered)
        assert max_teacher_diff < 1e-5, f"teacher diverged across ranks: {max_teacher_diff}"

        if step >= 2:
            break

    teacher_p_after = next(teacher.backbone.parameters()).detach().clone()
    assert not torch.allclose(teacher_p_before, teacher_p_after), "teacher unchanged"

    if is_main:
        print("DDP smoke OK")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
