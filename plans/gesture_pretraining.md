## Goal: Gesture Pretraining

Create a strategy to pretrain the SLT model on gesture data to improve its ability to understand and generate gestures.


### Plan
1. Use modern-bert base architecture for the gesture encoder. Initialize everything from scratch. Remove the embedding layer. Replace it with a lightweight Conv backbone that downsamples by a factor of 2. The size should be consistent with modernbert base. Provide a configuration file for it as well.
2. For the text encoder, use `jinaai/jina-embeddings-v5-text-nano`. Below is the code snippet to load this model. This model should stay frozen.
```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v5-text-nano",
    trust_remote_code=True,
    _attn_implementation="flash_attention_2",  # use sdpa if not available
    dtype=torch.bfloat16,  # Recommended for GPUs
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]
text_embeddings = model.encode(texts=texts, task="text-matching")
```
3. Use a light-weight swiglu nlp to project the output of the gesture encoder to the same dimension as the text encoder for contrastive learning using the info-nce loss function.
4. When computing the info-nce loss, compute it across the entire batch by gathering embeddings across devices. Each sample is a pair of gesture and text. Use a temperature parameter of 0.05.
5. Add a masked reconstruction loss to the encoder. The gestures should be masked at a rate of 50% and the model should reconstruct the masked gestures. The corruption should have a minimal span of 4. For corruption, you can consider replacing the input gesture with a learnable mask token before the conv layer.
6. For evaluation, you can select a subset of how2sign or csl_daily to compute recall@10, recall@50, and recall@100 for video-text retrieval.





### Guidelines
1. Place the code inside `src/gesture_pretraining`
2. Produce a `gesture_pretrain.sh` script to train the model.
3. Gesture data available at `/mnt/data2/sign_gestures`
4. Initial codebase for different modules can be found at: `slt/tmp_codebase`
5. You can refer to the PatchEmbed layer in `/home/slimelab/Projects/slt/src/mae_pretraining`.
6. Enable mixed precision training with bf16 or training in full bf16, sdpa, and liger kernels by default.
7. Make the code modular. Separate dataloader, model, and trainer into different files.
