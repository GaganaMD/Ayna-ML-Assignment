import os
import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from utils.dataset import PolygonColorDataset
from model.unet_film import PolygonUNetConditioned
import torch.nn.functional as F

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print ("Device:", device)

    # Load color2idx mapping from training json
    import json
    with open(args.train_json, 'r') as f:
        mapping = json.load(f)
    color_names = sorted(set(rec['colour'] for rec in mapping))
    color2idx = {c: i for i, c in enumerate(color_names)}
    
    # Datasets and loaders
    train_ds = PolygonColorDataset(args.train_root, args.train_json, color2idx)
    valid_ds = PolygonColorDataset(args.valid_root, args.valid_json, color2idx)
    tr_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    va_loader = DataLoader(valid_ds, batch_size=args.bs, shuffle=False, num_workers=2)
    
    # Model setup (huggingface UNet2DConditionModel wrapper)
    model = PolygonUNetConditioned(num_colors=len(color2idx), image_size=128, context_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss settings: combine L1 and MSE for sharper and smoother outputs
    def loss_fn(pred, target):
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return 0.7 * l1 + 0.3 * mse

    # wandb experiment tracking
    wandb.init(project=args.wandb_proj, config=vars(args))
    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.
        for img, color, tgt in tr_loader:
            img, color, tgt = img.to(device), color.to(device), tgt.to(device)
            out = model(img, color, timestep=torch.zeros(img.size(0), dtype=torch.long, device=img.device))
            loss = loss_fn(out, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(tr_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for img, color, tgt in va_loader:
                img, color, tgt = img.to(device), color.to(device), tgt.to(device)
                out = model(img, color, timestep=torch.zeros(img.size(0), dtype=torch.long, device=img.device))
                loss = loss_fn(out, tgt)
                val_loss += loss.item() * img.size(0)
        val_loss /= len(va_loader.dataset)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val:
            best_val = val_loss
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wandb_proj", type=str, default="unet-polygon-colorization-intern")
    parser.add_argument("--out_dir", type=str, default="checkpoints/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
