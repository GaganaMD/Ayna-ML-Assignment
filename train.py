import os
import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from utils.dataset import PolygonColorDataset
from model.unet_film import ConditionalUNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

# --- NEW: Perceptual Loss Class ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(weights='VGG16_Weights.DEFAULT').features.to(device).eval()
        self.loss_layers = [3, 8, 15, 22]  # VGG layers to use for feature extraction
        self.vgg_layers = []
        for i in range(len(self.vgg)):
            if isinstance(self.vgg[i], nn.MaxPool2d):
                self.vgg_layers.append(self.vgg[i])
            else:
                self.vgg_layers.append(self.vgg[i])
            if i in self.loss_layers:
                self.vgg_layers[-1].requires_grad_(False)
        self.vgg = nn.Sequential(*self.vgg_layers).to(device)

    def forward(self, pred, target):
        pred_features = []
        target_features = []
        
        # The VGG model expects 3-channel input
        if pred.shape[1] != 3:
            pred = pred.expand(-1, 3, -1, -1)
        if target.shape[1] != 3:
            target = target.expand(-1, 3, -1, -1)

        for i, layer in enumerate(self.vgg):
            pred = layer(pred)
            target = layer(target)
            if i in self.loss_layers:
                pred_features.append(pred)
                target_features.append(target)
        
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += F.l1_loss(pf, tf)
            
        return loss

# --- Main Training Function ---
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    import json
    with open(args.train_json, 'r') as f:
        mapping = json.load(f)
    color_names = sorted(set(rec['colour'] for rec in mapping))
    color2idx = {c: i for i, c in enumerate(color_names)}

    train_ds = PolygonColorDataset(args.train_root, args.train_json, color2idx)
    valid_ds = PolygonColorDataset(args.valid_root, args.valid_json, color2idx)
    tr_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    va_loader = DataLoader(valid_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    model = ConditionalUNet(n_channels=3, n_classes=3, n_colors=len(color2idx)).to(device)
    
    # NEW: Define a hybrid loss
    pixel_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    wandb.init(project=args.wandb_proj, config=vars(args))

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.
        for img, color, tgt in tr_loader:
            img, color, tgt = img.to(device), color.to(device), tgt.to(device)
            out = model(img, color)
            
            # NEW: Calculate both losses and combine them
            pixel_loss = pixel_loss_fn(out, tgt)
            perceptual_loss = perceptual_loss_fn(out, tgt)
            loss = pixel_loss + 0.1 * perceptual_loss # The weight 0.1 is a hyperparameter
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for img, color, tgt in va_loader:
                img, color, tgt = img.to(device), color.to(device), tgt.to(device)
                out = model(img, color)
                
                pixel_loss = pixel_loss_fn(out, tgt)
                perceptual_loss = perceptual_loss_fn(out, tgt)
                loss = pixel_loss + 0.1 * perceptual_loss
                
                val_loss += loss.item() * img.size(0)
        val_loss /= len(va_loader.dataset)
        scheduler.step(val_loss)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'color_to_idx': color2idx,
                'epoch': epoch,
                'val_loss': val_loss,
                'model_config': {
                    'n_channels': 3,
                    'n_classes': 3,
                    'n_colors': len(color2idx),
                    'bilinear': True
                }
            }, f"{args.out_dir}/best_model.pth")
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="dataset/training")
    parser.add_argument("--valid_root", type=str, default="dataset/validation")
    parser.add_argument("--train_json", type=str, default="dataset/training/data.json")
    parser.add_argument("--valid_json", type=str, default="dataset/validation/data.json")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4) 
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb_proj", type=str, default="unet-polygon-colorization-intern")
    parser.add_argument("--out_dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)