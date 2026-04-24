import torch
import torch.nn as nn
import sys
import os

# Add vjepa2 to path to allow imports
VJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vjepa2"))
if VJ_PATH not in sys.path:
    sys.path.append(VJ_PATH)

from hubconf import vjepa2_1_vit_base_384, vjepa2_1_vit_large_384

class VJEPAISLR(nn.Module):
    def __init__(self, num_classes, variant='80m', pretrained=True):
        super().__init__()
        
        if variant == '80m':
            self.backbone, self.predictor = vjepa2_1_vit_base_384(pretrained=pretrained)
        elif variant == '300m':
            self.backbone, self.predictor = vjepa2_1_vit_large_384(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown variant {variant}")
            
        # We don't need the predictor for classification
        del self.predictor
        
        self.embed_dim = self.backbone.embed_dim
        
        # Improved classification head
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        # V-JEPA encoder expects (B, C, T, H, W)
        
        # backbone forward returns (B, L, D)
        # where L is the number of tokens (temporal * spatial)
        feat = self.backbone(x)
        
        # Global Average Pooling over tokens
        # feat: (B, L, D) -> (B, D)
        feat = feat.mean(dim=1)
        
        logits = self.head(feat)
        return logits

if __name__ == "__main__":
    # Test forward pass with dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VJEPAISLR(num_classes=1000, variant='80m', pretrained=False).to(device)
    
    # Dummy video: (Batch, Channels, Time, Height, Width)
    dummy_video = torch.randn(1, 3, 16, 224, 224).to(device)
    output = model(dummy_video)
    print(f"Output shape: {output.shape}")
