from __future__ import annotations

import contextlib
import logging
import math
import re
from pathlib import Path
from typing import Callable, ContextManager, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import SentenceTransformerTextEncoder


_GESTURE_STEP_RE = re.compile(r"gesture-(\d+)\.pt$")


class _RandContext:
    """Snapshot CPU + per-device CUDA RNG so dropout-like ops are reproducible across passes.

    Mirrors :class:`sentence_transformers.losses.cached_multiple_negatives_ranking.RandContext`,
    used by GradCache to make the second (with-grad) forward match the first (no-grad) one.
    """

    def __init__(self, *tensors: torch.Tensor) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices = sorted({t.device for t in tensors if t.is_cuda}, key=lambda d: d.index or 0)
        self.fwd_gpu_states = [torch.cuda.get_rng_state(d) for d in self.fwd_gpu_devices]

    def __enter__(self) -> None:
        self._saved_cpu = torch.get_rng_state()
        self._saved_gpu = [torch.cuda.get_rng_state(d) for d in self.fwd_gpu_devices]
        torch.set_rng_state(self.fwd_cpu_state)
        for d, st in zip(self.fwd_gpu_devices, self.fwd_gpu_states):
            torch.cuda.set_rng_state(st, device=d)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        torch.set_rng_state(self._saved_cpu)
        for d, st in zip(self.fwd_gpu_devices, self._saved_gpu):
            torch.cuda.set_rng_state(st, device=d)


logger = logging.getLogger(__name__)


class Qwen3EmbeddingTextEncoder(nn.Module):
    """Qwen3-Embedding loaded as a raw HF model with flash-attn 2 + liger kernels.

    Uses last-token pooling (decoder-only embedding recipe). Inputs are tokenized
    with right-padding; we recover the last non-pad position via attention_mask.
    Output is L2-normalized to match the upstream Qwen3-Embedding convention.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_text_length: int,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        apply_liger: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if apply_liger:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3

                # Qwen3Model has no LM head, so disable the fused LCE path.
                apply_liger_kernel_to_qwen3(
                    rope=True,
                    rms_norm=True,
                    swiglu=True,
                    cross_entropy=False,
                    fused_linear_cross_entropy=False,
                )
            except Exception as exc:
                logger.warning("liger qwen3 patch failed (%s); continuing without", exc)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self.max_text_length = int(max_text_length)

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self._embedding_dim = int(self.model.config.hidden_size)
        self._dtype = dtype

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def _last_token_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: [B, T, C], attention_mask: [B, T] of 0/1
        left_padded = bool(int(attention_mask[:, -1].sum().item()) == int(attention_mask.shape[0]))
        if left_padded:
            return hidden_states[:, -1]
        seq_lens = attention_mask.long().sum(dim=1) - 1
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, seq_lens]

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._last_token_pool(outputs.last_hidden_state, attention_mask)


class GestureTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        max_frames: int,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_frames = int(max_frames)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, self.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        gesture: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(gesture)
        x = x + self.pos_embed[:, : x.shape[1]]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if attention_mask is None:
            return x.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GestureSignCLIPModel(nn.Module):
    def __init__(
        self,
        *,
        text_model_name: str = "answerdotai/ModernBERT-base",
        max_text_length: int = 16,
        feature_dim: int = 1104,
        max_frames: int = 256,
        gesture_embed_dim: int = 768,
        gesture_depth: int = 12,
        gesture_num_heads: int = 12,
        gesture_mlp_ratio: float = 4.0,
        projection_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
        text_encoder_kind: str = "modernbert",
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        text_attn_implementation: str = "flash_attention_2",
        apply_liger: bool = True,
        freeze_text: bool = False,
    ) -> None:
        super().__init__()
        if loss_type not in {"infonce", "sigmoid"}:
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        if text_encoder_kind not in {"modernbert", "qwen3-embedding"}:
            raise ValueError(
                f"text_encoder_kind must be 'modernbert' or 'qwen3-embedding', got {text_encoder_kind}"
            )
        self.loss_type = loss_type
        self.text_encoder_kind = text_encoder_kind
        self.freeze_text = bool(freeze_text)

        self.gesture_backbone = GestureTransformerBackbone(
            feature_dim=feature_dim,
            max_frames=max_frames,
            embed_dim=gesture_embed_dim,
            depth=gesture_depth,
            num_heads=gesture_num_heads,
            mlp_ratio=gesture_mlp_ratio,
            dropout=projection_dropout,
        )
        if text_encoder_kind == "qwen3-embedding":
            self.text_encoder = Qwen3EmbeddingTextEncoder(
                text_model_name,
                max_text_length=max_text_length,
                dtype=text_encoder_dtype,
                attn_implementation=text_attn_implementation,
                apply_liger=apply_liger,
            )
        else:
            self.text_encoder = SentenceTransformerTextEncoder(
                text_model_name,
                max_text_length=max_text_length,
                gradient_checkpointing=gradient_checkpointing,
            )
        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
            self.text_encoder.eval()
        text_dim = self.text_encoder.embedding_dim
        self.embedding_dim = int(text_dim)
        # Gesture projection maps gesture_embed_dim -> text_dim. No text projection;
        # the text encoder's output dim is the shared CLIP embedding dim.
        self.gesture_projection = nn.Sequential(
            nn.LayerNorm(gesture_embed_dim),
            nn.Linear(gesture_embed_dim, text_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(projection_dropout),
            nn.Linear(text_dim, text_dim),
        )
        if loss_type == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor(float(sigmoid_logit_scale_init)))
            self.logit_bias = nn.Parameter(torch.tensor(float(sigmoid_bias_init)))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            self.logit_bias = None

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer

    def encode_gesture(
        self,
        gesture: torch.Tensor,
        *,
        gesture_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.gesture_backbone(gesture, attention_mask=gesture_attention_mask)
        return F.normalize(self.gesture_projection(pooled), dim=-1)

    def encode_text(self, text_features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_text:
            self.text_encoder.eval()
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_features)
            text_embeddings = text_embeddings.detach()
        else:
            text_embeddings = self.text_encoder(text_features)
        return F.normalize(text_embeddings, dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text:
            self.text_encoder.eval()
        return self

    def gesture_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict containing only the trainable gesture branch + logit params.

        Includes ``gesture_backbone.*``, ``gesture_projection.*``, ``logit_scale``,
        and (when applicable) ``logit_bias``. Excludes the text encoder.
        """
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("gesture_backbone.", self.gesture_backbone),
            ("gesture_projection.", self.gesture_projection),
        ):
            for k, v in module.state_dict().items():
                state[f"{prefix}{k}"] = v.detach().cpu().clone()
        state["logit_scale"] = self.logit_scale.detach().cpu().clone()
        if self.logit_bias is not None:
            state["logit_bias"] = self.logit_bias.detach().cpu().clone()
        return state

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        step: Optional[int] = None,
        save_text_encoder: bool = True,
        text_encoder_filename: str = "text_encoder.pt",
    ) -> dict[str, Path]:
        """Save the gesture branch (and optionally the text encoder) to disk.

        - ``gesture-{step:07d}.pt`` (or ``gesture.pt`` when step is None): contains
          gesture_backbone + gesture_projection + logit params.
        - ``text_encoder.pt``: full text encoder ``state_dict()``. Skipped if
          ``save_text_encoder`` is False.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gesture_name = f"gesture-{step:07d}.pt" if step is not None else "gesture.pt"
        gesture_path = output_dir / gesture_name
        torch.save(self.gesture_state_dict(), gesture_path)

        written: dict[str, Path] = {"gesture": gesture_path}
        if save_text_encoder:
            text_path = output_dir / text_encoder_filename
            text_state = {k: v.detach().cpu().clone() for k, v in self.text_encoder.state_dict().items()}
            torch.save(text_state, text_path)
            written["text_encoder"] = text_path
        return written

    def load_pretrained(
        self,
        source: str | Path,
        *,
        load_text_encoder: bool = True,
        text_encoder_filename: str = "text_encoder.pt",
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, list[str]]:
        """Load gesture (and optionally text encoder) weights into ``self``.

        ``source`` may be either a directory (we'll auto-pick the latest
        ``gesture-*.pt``, falling back to ``gesture.pt``) or a path to a
        specific gesture weight file. The text encoder, when available alongside
        the gesture file as ``text_encoder.pt``, is loaded as well unless
        ``load_text_encoder`` is False.
        """
        source = Path(source)
        if source.is_dir():
            gesture_path = self._find_latest_gesture_file(source)
            if gesture_path is None:
                raise FileNotFoundError(f"No gesture-*.pt or gesture.pt found in {source}")
            search_dir = source
        else:
            gesture_path = source
            search_dir = source.parent

        gesture_state = torch.load(str(gesture_path), map_location=map_location, weights_only=False)
        info: dict[str, list[str]] = {}

        backbone_sd = {
            k.removeprefix("gesture_backbone."): v
            for k, v in gesture_state.items()
            if k.startswith("gesture_backbone.")
        }
        proj_sd = {
            k.removeprefix("gesture_projection."): v
            for k, v in gesture_state.items()
            if k.startswith("gesture_projection.")
        }

        backbone_incompat = self.gesture_backbone.load_state_dict(backbone_sd, strict=strict)
        proj_incompat = self.gesture_projection.load_state_dict(proj_sd, strict=strict)
        info["gesture_backbone_missing"] = list(backbone_incompat.missing_keys)
        info["gesture_backbone_unexpected"] = list(backbone_incompat.unexpected_keys)
        info["gesture_projection_missing"] = list(proj_incompat.missing_keys)
        info["gesture_projection_unexpected"] = list(proj_incompat.unexpected_keys)

        if "logit_scale" in gesture_state:
            with torch.no_grad():
                self.logit_scale.copy_(
                    gesture_state["logit_scale"].to(self.logit_scale.device, self.logit_scale.dtype)
                )
        if "logit_bias" in gesture_state and self.logit_bias is not None:
            with torch.no_grad():
                self.logit_bias.copy_(
                    gesture_state["logit_bias"].to(self.logit_bias.device, self.logit_bias.dtype)
                )

        if load_text_encoder:
            text_path = search_dir / text_encoder_filename
            if text_path.exists():
                text_state = torch.load(str(text_path), map_location=map_location, weights_only=False)
                text_incompat = self.text_encoder.load_state_dict(text_state, strict=False)
                info["text_encoder_missing"] = list(text_incompat.missing_keys)
                info["text_encoder_unexpected"] = list(text_incompat.unexpected_keys)
            else:
                logger.info("text encoder file %s not found; skipping", text_path)

        return info

    @staticmethod
    def _find_latest_gesture_file(directory: Path) -> Optional[Path]:
        directory = Path(directory)
        if not directory.exists():
            return None
        stepped = []
        for p in directory.glob("gesture-*.pt"):
            m = _GESTURE_STEP_RE.search(p.name)
            if m:
                stepped.append((int(m.group(1)), p))
        if stepped:
            stepped.sort()
            return stepped[-1][1]
        plain = directory / "gesture.pt"
        return plain if plain.exists() else None

    def _contrastive_loss(
        self,
        gesture_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        *,
        target_texts: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        # Align dtypes: matmul needs matching dtypes. Promote both to the wider of the two
        # so we don't truncate fp32 gesture grads through a bf16 text bank when text is frozen.
        if gesture_embeddings.dtype != text_embeddings.dtype:
            common = torch.promote_types(gesture_embeddings.dtype, text_embeddings.dtype)
            gesture_embeddings = gesture_embeddings.to(common)
            text_embeddings = text_embeddings.to(common)
        local_bsz = gesture_embeddings.size(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        if world_size > 1:
            from torch.distributed.nn.functional import all_gather

            gathered_gesture = torch.cat(all_gather(gesture_embeddings), dim=0)
            gathered_text = torch.cat(all_gather(text_embeddings), dim=0)
            offset = torch.distributed.get_rank() * local_bsz
        else:
            gathered_gesture = gesture_embeddings
            gathered_text = text_embeddings
            offset = 0

        if self.loss_type == "sigmoid":
            logit_scale = self.logit_scale.exp()
            logits_per_gesture = logit_scale * (gesture_embeddings @ gathered_text.T) + self.logit_bias
            logits_per_text = logit_scale * (text_embeddings @ gathered_gesture.T) + self.logit_bias
        else:
            logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits_per_gesture = logit_scale * (gesture_embeddings @ gathered_text.T)
            logits_per_text = logit_scale * (text_embeddings @ gathered_gesture.T)
        targets = torch.arange(local_bsz, device=gesture_embeddings.device) + offset

        if target_texts is not None:
            local_texts = list(target_texts)
            if world_size > 1:
                gathered = [None] * world_size
                torch.distributed.all_gather_object(gathered, local_texts)
                all_texts = [text for sub in gathered for text in sub]
            else:
                all_texts = local_texts
            local_arr = np.asarray(local_texts, dtype=object).reshape(-1, 1)
            global_arr = np.asarray(all_texts, dtype=object).reshape(1, -1)
            dup = torch.from_numpy(local_arr == global_arr).to(gesture_embeddings.device)
            rows = torch.arange(local_bsz, device=gesture_embeddings.device)
            diag = torch.zeros_like(dup)
            diag[rows, targets] = True
            false_neg = dup & ~diag
        else:
            false_neg = None

        if self.loss_type == "sigmoid":
            global_bsz = gathered_text.size(0)
            labels = -torch.ones(local_bsz, global_bsz, device=gesture_embeddings.device, dtype=logits_per_gesture.dtype)
            rows = torch.arange(local_bsz, device=gesture_embeddings.device)
            labels[rows, targets] = 1.0
            if false_neg is not None:
                mask = (~false_neg).to(logits_per_gesture.dtype)
                loss_g = -(F.logsigmoid(labels * logits_per_gesture) * mask).sum() / mask.sum().clamp_min(1.0)
                loss_t = -(F.logsigmoid(labels * logits_per_text) * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss_g = -F.logsigmoid(labels * logits_per_gesture).mean()
                loss_t = -F.logsigmoid(labels * logits_per_text).mean()
            loss = 0.5 * (loss_g + loss_t) * gathered_text.size(0)
        else:
            if false_neg is not None:
                neg_inf = torch.finfo(logits_per_gesture.dtype).min
                logits_per_gesture = logits_per_gesture.masked_fill(false_neg, neg_inf)
                logits_per_text = logits_per_text.masked_fill(false_neg, neg_inf)
            loss = 0.5 * (
                F.cross_entropy(logits_per_gesture, targets) +
                F.cross_entropy(logits_per_text, targets)
            )
        return {
            "loss": loss,
            "logits_per_gesture": logits_per_gesture,
        }

    def forward(
        self,
        *,
        gesture: torch.Tensor,
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        gesture_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        gesture_embeddings = self.encode_gesture(gesture, gesture_attention_mask=gesture_attention_mask)
        text_embeddings = self.encode_text(text_features)
        out = self._contrastive_loss(
            gesture_embeddings, text_embeddings, target_texts=target_texts
        )
        out["gesture_embeddings"] = gesture_embeddings
        out["text_embeddings"] = text_embeddings
        return out

    def cached_forward(
        self,
        *,
        gesture: torch.Tensor,
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        gesture_attention_mask: Optional[torch.Tensor] = None,
        mini_batch_size: int = 32,
        no_sync_factory: Optional[Callable[[], ContextManager]] = None,
    ) -> dict[str, torch.Tensor]:
        """GradCache-style InfoNCE / sigmoid loss for the gesture branch.

        Mirrors the algorithm of
        :class:`sentence_transformers.losses.CachedMultipleNegativesRankingLoss`:

        1. Encode the *full* batch of gesture features in mini-batches under
           ``torch.no_grad()`` to obtain embeddings, while saving an RNG snapshot
           per mini-batch so dropout can be replayed exactly.
        2. Build a leaf cache (``requires_grad_()`` on the detached embeddings),
           encode the (frozen) text branch once, and compute the contrastive loss
           on the full batch. ``loss.backward()`` populates ``.grad`` on the cache
           tensors and on ``logit_scale``/``logit_bias``.
        3. Re-run the gesture encoder per mini-batch (with grad and the saved RNG
           context) and call ``torch.autograd.backward(emb, cache_grad)`` so the
           gradients flow through ``gesture_backbone`` / ``gesture_projection``.

        Backward is performed *inside* this method — the caller must NOT call
        ``.backward()`` on the returned ``loss`` tensor (it is detached).

        ``no_sync_factory`` should be a zero-arg callable returning a context
        manager (e.g. ``model.no_sync`` from a DDP wrapper) so all-reduce only
        fires on the final mini-batch's backward.
        """
        if not self.freeze_text:
            raise NotImplementedError(
                "cached_forward currently requires freeze_text=True; "
                "trainable text branch would need full GradCache on both towers."
            )

        device = gesture.device
        batch_size = gesture.size(0)
        mb = max(1, int(mini_batch_size))
        splits: list[tuple[int, int]] = [(s, min(batch_size, s + mb)) for s in range(0, batch_size, mb)]

        no_sync = no_sync_factory if no_sync_factory is not None else (lambda: contextlib.nullcontext())

        # 1. No-grad gesture encoding per mini-batch + save RNG state for replay.
        rand_states: list[_RandContext] = []
        detached_g: list[torch.Tensor] = []
        with torch.no_grad():
            for s, e in splits:
                g_chunk = gesture[s:e]
                am_chunk = gesture_attention_mask[s:e] if gesture_attention_mask is not None else None
                rand_states.append(_RandContext(g_chunk))
                detached_g.append(self.encode_gesture(g_chunk, gesture_attention_mask=am_chunk))

        # 2. Build leaf cache, encode text once (frozen), compute loss, backward through the
        #    cache + logit_scale. DDP no_sync defers all-reduce until the final mini-batch.
        cache = [d.detach().requires_grad_() for d in detached_g]
        full_gesture = torch.cat(cache, dim=0)
        text_embeddings = self.encode_text(text_features)

        with no_sync():
            out = self._contrastive_loss(
                full_gesture, text_embeddings, target_texts=target_texts
            )
            loss = out["loss"]
            loss.backward()

        # 3. Replay forward-with-grad per mini-batch and apply cached upstream grad.
        #    All but the last backward run under no_sync; the last one triggers DDP all-reduce
        #    over every gradient accumulated so far (cache loss grads + per-mini-batch grads).
        last_idx = len(splits) - 1
        for i, (s, e) in enumerate(splits):
            is_last = i == last_idx
            sync_ctx = contextlib.nullcontext() if is_last else no_sync()
            g_chunk = gesture[s:e]
            am_chunk = gesture_attention_mask[s:e] if gesture_attention_mask is not None else None
            with sync_ctx:
                with rand_states[i]:
                    emb = self.encode_gesture(g_chunk, gesture_attention_mask=am_chunk)
                if cache[i].grad is None:
                    # No grad reached this mini-batch (e.g. all masked out); skip.
                    continue
                torch.autograd.backward(emb, grad_tensors=cache[i].grad)

        return {
            "loss": loss.detach(),
            "logits_per_gesture": out["logits_per_gesture"].detach(),
            "gesture_embeddings": full_gesture.detach(),
            "text_embeddings": text_embeddings.detach(),
            "cached": True,
        }


__all__ = [
    "GestureSignCLIPModel",
    "GestureTransformerBackbone",
]
