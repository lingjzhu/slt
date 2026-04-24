# LeanVAE Encoder + Latent ViT CLIP Training Plan

Use LeanVAE's encoder (`/home/slimelab/Projects/Sign/LeanVAE`) as the pretrained front-end, then place a lighter latent ViT on top as the task head for CLIP training (code in `src/`).

Core idea:
raw video -> LeanVAE encoder features -> latent ViT aggregator -> projection head -> CLIP loss

## Architecture

1. Input raw video, not pre-exported latents.
2. Run `dwt + encoder.encode(...)` from LeanVAE to get the encoder feature tensor before stochastic latent sampling.
3. Treat that encoder feature tensor as a spatiotemporal token grid.
4. Feed those tokens into a latent ViT head that does task-specific temporal and spatial reasoning for recognition.
5. Apply a projection head and normalize for CLIP similarity against the text encoder.

## Why This Split

- LeanVAE encoder gives pretrained low and mid-level sign-video features.
- The latent ViT still gives a flexible discriminative head for CLIP alignment.
- This avoids training the whole visual stack from scratch.
- This avoids depending on sampled exported latents.

## Recommended Training Stages

### Stage 1

Freeze all LeanVAE encoder weights.

Train only:
- latent ViT
- video projection head
- text projection if needed

This checks whether pretrained encoder features are already useful.

### Stage 2

Unfreeze top LeanVAE blocks only.

Fine-tune:
- fusion layer
- high and low residual blocks near the output
- latent ViT

Keep lower encoder layers frozen.

### Stage 3

Optional full fine-tune.

Use a smaller learning rate on LeanVAE and a larger learning rate on the latent ViT head.

## Model Wiring

Add a new model class, for example `LeanVAECLIPModel`.

Visual path:
- raw video
- LeanVAE preprocessing
- `encoder.encode(...)`
- latent ViT
- pooling
- projection

Text path:
- keep the current ModernBERT text encoder

Loss:
- same CLIP InfoNCE loss already used
- optionally retain duplicate-text false-negative masking

## Feature Interface

- Best source feature is the encoder output `p` before `latent_bottleneck.sample(...)`.
- If that tensor shape is `[B, T, H, W, D]`, the latent ViT can consume it directly with small adjustments.
- If needed, add:
- linear adapter from encoder dimension to ViT dimension
- positional embeddings for `T`, `H`, and `W`
- mask handling for padded frames

## Initialization

- Initialize LeanVAE `dwt` and `encoder` from the VAE checkpoint.
- Initialize latent ViT randomly.
- Keep the latent ViT smaller than the current standalone latent model at first.

Suggested start:
- embed dim `512`
- depth `4-6`
- heads `8`

Add a small linear adapter before the ViT if feature dimensions do not match cleanly.

## Optimization

Use parameter groups:
- LeanVAE frozen or low LR, for example `1e-5` to `5e-5`
- latent ViT plus projection head, for example `1e-4` to `2e-4`
- text tower same as head or slightly lower than head

Warmup helps when unfreezing.

Gradient clipping should remain enabled.

## Ablations To Run

1. LeanVAE encoder plus mean pool plus projection, with no latent ViT.
2. LeanVAE encoder plus small latent ViT.
3. Freeze encoder versus partial unfreeze.
4. Pre-bottleneck encoder features versus sampled or exported latents.
5. Deterministic bottleneck output if available.

## Success Criteria

- First compare against the current latent baseline on the exact same test evaluation.
- If the encoder-only model already beats the current latent ViT-from-scratch setup, the pretrained front-end is helping.
- Then check whether the added latent ViT gives further gains over simple pooling.

## Suggested Implementation Order

1. Build a LeanVAE visual wrapper that returns encoder features before sampling.
2. Add a new CLIP model class using that wrapper plus a small latent ViT.
3. Add a new training script reusing the existing CLIP loss and evaluation code.
4. Run a frozen-encoder baseline first.
5. Then run partial unfreeze.
