import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from utils.dataset import PolygonColorDataset
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'checkpoints/best_model.pth'

# --- UNet Model Class Definitions ---
# This is needed to load the saved state_dict
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, n_colors=5, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels, self.n_classes, self.n_colors, self.bilinear = n_channels, n_classes, n_colors, bilinear
        self.color_embedding = nn.Embedding(n_colors, 64)
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3 = Down(64, 128), Down(128, 256), Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.color_proj = nn.Linear(64, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x, color_idx):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        color_emb = self.color_embedding(color_idx)
        color_proj = self.color_proj(color_emb)
        
        B, C, H, W = x5.shape
        color_proj = color_proj.view(B, C, 1, 1).expand(-1, -1, H, W)
        x5 = x5 + color_proj
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return torch.sigmoid(self.outc(x))

# --- Model Loading Function ---
def load_trained_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ConditionalUNet(
        n_channels=checkpoint['model_config']['n_channels'],
        n_classes=checkpoint['model_config']['n_classes'],
        n_colors=checkpoint['model_config']['n_colors'],
        bilinear=checkpoint['model_config']['bilinear']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint['color_to_idx']

# Load the best model
if os.path.exists(model_path):
    model, color_to_idx = load_trained_model(model_path, device)
    available_colors = list(color_to_idx.keys())
    print("Available colors:", available_colors)
else:
    print("Model checkpoint not found. Please train the model first.")
    exit()

# --- Inference and Wrapper Functions ---
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def inference(model, input_image_path, color_name, device, color_to_idx, transform):
    model.eval()
    
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    if color_name not in color_to_idx:
        print(f"Error: Color '{color_name}' not found. Available colors: {list(color_to_idx.keys())}")
        return None, input_image
    
    color_idx_tensor = torch.tensor([color_to_idx[color_name]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor, color_idx_tensor)
    
    pred_np = prediction.squeeze(0).cpu().permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0, 1)
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8))
    
    return pred_image, input_image

def color_polygon(input_image_path, desired_color):
    print(f"\n🎨 Processing: Coloring '{os.path.basename(input_image_path)}' with '{desired_color}'...")
    
    if not os.path.exists(input_image_path):
        print(f"❌ Error: File '{input_image_path}' not found!")
        return
        
    colored_image, original_image = inference(
        model, input_image_path, desired_color, device, 
        color_to_idx, inference_transform
    )
    
    if colored_image is None:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Input Polygon')
    axes[0].axis('off')
    
    axes[1].imshow(colored_image)
    axes[1].set_title(f'Colored Output: {desired_color.capitalize()}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# You can uncomment these lines and change the paths to test your images
# sample_image_path = "path/to/your/image.png"
# color_polygon(sample_image_path, 'yellow')
# ---
# Data preparation for batch processing
data_dir = 'dataset/validation'
json_path = 'dataset/validation/data.json'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
val_dataset = PolygonColorDataset(data_dir, json_path, color_to_idx, transform=transform)

total_mse, total_ssim, num_samples = 0, 0, 0
with torch.no_grad():
    for batch_idx in range(len(val_dataset)):
        # CORRECTED: Unpack the tuple returned by the dataset
        sample = val_dataset[batch_idx]
        imgs, color_idx, targets = sample
        
        imgs = imgs.unsqueeze(0).to(device)
        color_idx = color_idx.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)

        outputs = model(imgs, color_idx)
        outputs = torch.clamp(outputs, 0, 1)

        pred_np = outputs.squeeze().cpu().numpy().transpose(1, 2, 0)
        target_np = targets.squeeze().cpu().numpy().transpose(1, 2, 0)

        pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_img_resized = pred_img.resize((target_np.shape[1], target_np.shape[0]), Image.BILINEAR)
        resized_pred_np = np.array(pred_img_resized) / 255.0

        mse = mean_squared_error(resized_pred_np.flatten(), target_np.flatten())
        ssim_score = ssim(resized_pred_np, target_np, channel_axis=-1, data_range=1.0)
        total_mse += mse
        total_ssim += ssim_score
        num_samples += 1

        if num_samples <= 5:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(imgs.squeeze().cpu().numpy().transpose(1, 2, 0))
            axs[0].set_title('Input Polygon')
            axs[1].imshow(target_np)
            axs[1].set_title('Target Output')
            axs[2].imshow(resized_pred_np)
            axs[2].set_title('Model Output')
            
            output_filename = f"inference_output_{num_samples}.png"
            plt.savefig(output_filename)
            plt.close(fig)
            print(f"Plot saved to {output_filename}")
            
avg_mse = total_mse / num_samples
avg_ssim = total_ssim / num_samples
print(f"Average MSE: {avg_mse:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")