import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from model.unet_film import UNetFiLM
from utils.dataset import PolygonColorDataset

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from PIL import Image
import matplotlib.pyplot as plt

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'checkpoints/best_model.pt'
data_dir = 'dataset/validation'
json_path = 'dataset/validation/data.json'

color2idx = {
    'yellow': 0,
    'orange': 1,
    'purple': 2,
    'red': 3,
    'blue': 4,
    'magenta': 5,
    'cyan': 6,
    'green': 7
}

transform = transforms.Compose([transforms.ToTensor()])
val_dataset = PolygonColorDataset(data_dir, json_path, color2idx, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- Model ---
model = UNetFiLM(num_colors=len(color2idx))
checkpoint = torch.load(model_path, map_location=device)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# --- Evaluation ---
total_mse = 0
total_ssim = 0
num_samples = 0

with torch.no_grad():
    for batch in val_loader:
        imgs, color_idx, targets = batch
        imgs = imgs.to(device)
        color_idx = color_idx.to(device)
        targets = targets.to(device)

        outputs = model(imgs, color_idx)
        outputs = torch.clamp(outputs, 0, 1)  # Ensure valid range
        
        # Move to cpu, convert to numpy
        pred_np = outputs.squeeze().cpu().numpy().transpose(1,2,0)
        target_np = targets.squeeze().cpu().numpy().transpose(1,2,0)
        
        mse = mean_squared_error(pred_np.flatten(), target_np.flatten())
        ssim_score = ssim(pred_np, target_np, multichannel=True, data_range=1.0)
        
        total_mse += mse
        total_ssim += ssim_score
        num_samples += 1

        # OPTIONAL: visualize a few outputs
        if num_samples <= 5:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(imgs.squeeze().cpu().numpy().transpose(1,2,0))
            axs[0].set_title('Input Polygon')
            axs[1].imshow(target_np)
            axs[1].set_title('Target Output')
            axs[2].imshow(pred_np)
            axs[2].set_title('Model Output')
            plt.show()

avg_mse = total_mse / num_samples
avg_ssim = total_ssim / num_samples

print(f"Average MSE: {avg_mse:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
