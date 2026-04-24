from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .model import DiscreteDiffusionModel


@torch.no_grad()
def iterative_decode(
    model: DiscreteDiffusionModel,
    *,
    visual_features: torch.Tensor,
    visual_attention_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    editable_mask: torch.Tensor,
    num_steps: int = 8,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Iterative confidence-based masked decoding on text conditioned on visual.

    `editable_mask` marks positions (`1`) that may be refined; all others stay fixed.
    Visual tokens are never corrupted (mask of zeros) so they remain observed.
    """
    model.eval()
    device = text_input_ids.device
    B, L = text_input_ids.shape

    editable_bool = editable_mask.bool()
    # initialize all editable positions to [MASK]
    ids = text_input_ids.clone()
    ids[editable_bool] = model.mask_token_id
    pending = editable_bool.clone()

    visual_zero_mask = torch.zeros_like(visual_attention_mask, dtype=torch.uint8)

    for step in range(num_steps):
        still_masked = pending
        n_pending = still_masked.sum(dim=1)  # [B]
        if int(n_pending.max()) == 0:
            break

        text_hidden, _ = model.forward_encoder(
            text_input_ids=ids,
            text_attention_mask=text_attention_mask,
            text_corruption_mask=still_masked.to(torch.uint8),
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            visual_corruption_mask=visual_zero_mask,
        )
        logits = model.mlm_head(text_hidden)

        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            preds = sampled
        else:
            probs = F.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

        # fraction to commit at this step (cosine schedule, final step commits all)
        frac = (step + 1) / num_steps
        commit_fraction = frac  # linearly schedule; simple & predictable

        # per-example: choose top-k by confidence among pending
        large_neg = torch.finfo(conf.dtype).min
        conf_masked = torch.where(still_masked, conf, torch.full_like(conf, large_neg))
        for b in range(B):
            pend = int(n_pending[b].item())
            if pend == 0:
                continue
            if step == num_steps - 1:
                k = pend
            else:
                k = max(1, int(round(commit_fraction * int(editable_bool[b].sum().item()))))
                already_committed = int(editable_bool[b].sum().item()) - pend
                k = max(0, min(pend, k - already_committed))
                if k == 0 and step == num_steps - 1:
                    k = pend
            if k <= 0:
                continue
            topk = torch.topk(conf_masked[b], k=k)
            positions = topk.indices
            ids[b, positions] = preds[b, positions]
            pending[b, positions] = False

    return ids
