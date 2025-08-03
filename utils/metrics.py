import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def compute_ssim(img1, img2):
    img1 = img1.permute(1,2,0).cpu().numpy()
    img2 = img2.permute(1,2,0).cpu().numpy()
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def compute_psnr(img1, img2):
    img1 = img1.permute(1,2,0).cpu().numpy()
    img2 = img2.permute(1,2,0).cpu().numpy()
    return psnr(img1, img2, data_range=1.0)
