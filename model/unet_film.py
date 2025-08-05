import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class PolygonUNetConditioned(nn.Module):
    def __init__(self, num_colors, image_size=128, context_dim=32):
        super().__init__()
        # Learnable embedding for each color index
        self.color_emb = nn.Embedding(num_colors, context_dim)
        # Hugging Face conditional U-Net
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=3,       # RGB input
            out_channels=3,      # RGB output
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim=context_dim
        )
    def forward(self, x, color_idx, timestep=None):
        context = self.color_emb(color_idx)
        if timestep is None:
            # Use dummy timestep (e.g., 0) if not given
            timestep = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = self.unet(x, timestep=timestep, encoder_hidden_states=context.unsqueeze(1))
        return out.sample

