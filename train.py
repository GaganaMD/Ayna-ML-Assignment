import os
import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from utils.dataset import PolygonColorDataset
from model.unet_film import ConditionalUNet
import torch.nn.functional as F
import torch.nn as nn

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

    model = ConditionalUNet(num_colors=len(color2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # NEW: Define two loss functions
    image_loss_fn = lambda pred_img, target_img: F.l1_loss(F.interpolate(pred_img, size=target_img.shape[2:], mode='bilinear', align_corners=False), target_img)
    color_loss_fn = nn.CrossEntropyLoss()

    wandb.init(project=args.wandb_proj, config=vars(args))

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.
        for img, color, tgt in tr_loader:
            img, color, tgt = img.to(device), color.to(device), tgt.to(device)
            
            # NEW: Model returns two outputs
            pred_img, pred_color_logits = model(img, color)
            
            # Calculate and combine the losses
            image_loss = image_loss_fn(pred_img, tgt)
            color_loss = color_loss_fn(pred_color_logits, color)
            
            # NEW: Weighted combination of the two losses. Adjust the weight (e.g., 0.1) as needed.
            loss = image_loss + 0.1 * color_loss

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
                
                # NEW: Model returns two outputs
                pred_img, pred_color_logits = model(img, color)

                image_loss = image_loss_fn(pred_img, tgt)
                color_loss = color_loss_fn(pred_color_logits, color)
                loss = image_loss + 0.1 * color_loss

                val_loss += loss.item() * img.size(0)
        val_loss /= len(va_loader.dataset)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val:
            best_val = val_loss
            # NEW: Save the model
            torch.save({'model': model.state_dict(), 'color2idx': color2idx}, f"{args.out_dir}/best_model.pt")

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="dataset/training")
    parser.add_argument("--valid_root", type=str, default="dataset/validation")
    parser.add_argument("--train_json", type=str, default="dataset/training/data.json")
    parser.add_argument("--valid_json", type=str, default="dataset/validation/data.json")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb_proj", type=str, default="unet-polygon-colorization-intern")
    parser.add_argument("--out_dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)