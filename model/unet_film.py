import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, n_feats, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, n_feats)
        self.beta = nn.Linear(cond_dim, n_feats)

    def forward(self, x, cond_vec):
        gamma = self.gamma(cond_vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond_vec).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.film = FiLM(out_ch, cond_dim)

    def forward(self, x, cond_vec):
        x = self.conv(x)
        x = self.film(x, cond_vec)
        return x

class UNetFiLM(nn.Module):
    def __init__(self, num_colors, color_embed_dim=64):
        super().__init__()
        self.color_emb = nn.Embedding(num_colors, color_embed_dim)
        filters = [32, 64, 128, 256]
        # Encoder
        self.enc1 = UNetBlock(3, filters[0], color_embed_dim)
        self.enc2 = UNetBlock(filters[0], filters[1], color_embed_dim)
        self.enc3 = UNetBlock(filters[1], filters[2], color_embed_dim)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(filters[2], filters[3], color_embed_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = UNetBlock(filters[3], filters[2], color_embed_dim)
        self.up1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = UNetBlock(filters[2], filters[1], color_embed_dim)
        self.final = nn.Conv2d(filters[1], 3, 1)

    def forward(self, img, color_idx):
        cond_vec = self.color_emb(color_idx)
        x1 = self.enc1(img, cond_vec)
        x2 = self.enc2(self.pool(x1), cond_vec)
        x3 = self.enc3(self.pool(x2), cond_vec)
        x4 = self.bottleneck(self.pool(x3), cond_vec)
        dec3 = self.up2(x4)
        dec3 = torch.cat([dec3, x3], dim=1)
        dec3 = self.dec3(dec3, cond_vec)
        dec2 = self.up1(dec3)
        dec2 = torch.cat([dec2, x2], dim=1)
        dec2 = self.dec2(dec2, cond_vec)
        out = self.final(dec2)
        out = torch.sigmoid(out)
        return out