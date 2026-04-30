from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class FrozenTextEncoder(nn.Module):
    """Frozen text encoder used by gesture pretraining contrastive loss."""

    def __init__(
        self,
        *,
        encoder_kind: str = "qwen3-embedding",
        model_name_or_path: str | None = None,
        max_text_length: int = 32,
        attn_implementation: str = "flash_attention_2",
        apply_liger: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if encoder_kind not in {"jina", "qwen3-embedding"}:
            raise ValueError(f"Unsupported text encoder kind: {encoder_kind}")

        from transformers import AutoModel, AutoTokenizer

        self.encoder_kind = encoder_kind
        self.max_text_length = int(max_text_length)

        if encoder_kind == "qwen3-embedding":
            if apply_liger:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

                    apply_liger_kernel_to_qwen3(
                        rope=True,
                        rms_norm=True,
                        swiglu=True,
                        cross_entropy=False,
                        fused_linear_cross_entropy=False,
                    )
                except Exception as exc:
                    logger.warning("liger qwen3 patch failed (%s); continuing without", exc)

            model_name_or_path = model_name_or_path or "Qwen/Qwen3-Embedding-0.6B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            model = AutoModel.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
            self._embed_dim = int(model.config.hidden_size)
        else:
            model_name_or_path = model_name_or_path or "jinaai/jina-embeddings-v5-text-nano"
            self.tokenizer = None
            model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                _attn_implementation="sdpa",
                dtype=dtype,
            )
            self._embed_dim = 768

        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.model.eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padded = bool(int(attention_mask[:, -1].sum().item()) == int(attention_mask.shape[0]))
        if left_padded:
            return hidden_states[:, -1]
        seq_lens = attention_mask.long().sum(dim=1) - 1
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, seq_lens]

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Returns (N, embed_dim) float32 normalised embeddings."""
        device = next(self.model.parameters()).device
        if self.encoder_kind == "qwen3-embedding":
            assert self.tokenizer is not None
            features = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            features = {key: value.to(device) for key, value in features.items()}
            outputs = self.model(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            )
            embeddings = self._last_token_pool(outputs.last_hidden_state, features["attention_mask"])
        else:
            embeddings = self.model.encode(texts=texts, task="text-matching")
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            embeddings = embeddings.to(device=device)
        return F.normalize(embeddings.float(), dim=-1)
