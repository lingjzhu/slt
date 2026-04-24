import torch
from torch.utils.data import DataLoader
from dataset import ISLRDataset, collate_fn
from vjepa_islr import VJEPAISLR
import os

def evaluate(checkpoint_path, variant='80m'):
    # Config
    batch_size = 4
    num_classes = 6707
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    test_dataset = ISLRDataset("/home/slime-base/projects/jian/islr/data/test.csv")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model
    model = VJEPAISLR(num_classes=num_classes, variant=variant, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    top5_correct = 0
    
    print(f"Starting evaluation on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            
            # Top-1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5
            _, top5_predicted = outputs.topk(5, 1, True, True)
            top5_correct += top5_predicted.eq(labels.view(-1, 1).expand_as(top5_predicted)).sum().item()
            
            if (i+1) % 10 == 0:
                print(f"Step [{i+1}/{len(test_loader)}], Top-1 Acc: {100.*correct/total:.2f}%")

    print(f"Final Results:")
    print(f"Top-1 Accuracy: {100.*correct/total:.2f}%")
    print(f"Top-5 Accuracy: {100.*top5_correct/total:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/slime-base/projects/jian/islr/checkpoints/best_model.pth")
    parser.add_argument("--variant", type=str, default="80m")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.variant)
