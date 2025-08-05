import torch
import torch.nn as nn

class ConditionalUNet(nn.Module):
    def __init__(self, num_colors, color_embed_dim=64):
        super().__init__()
        self.num_colors = num_colors
        self.color_emb = nn.Embedding(num_colors, color_embed_dim)
        filters = [32, 64, 128, 256]

        # Adjust first conv input channels for image + color embedding map
        self.enc1 = nn.Sequential(
            nn.Conv2d(3 + color_embed_dim, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )
        self.enc2 = self._block(filters[0], filters[1])
        self.enc3 = self._block(filters[1], filters[2])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck + FiLM
        self.bottleneck_conv = self._block(filters[2], filters[3])
        self.color_proj_gamma = nn.Linear(color_embed_dim, filters[3])
        self.color_proj_beta = nn.Linear(color_embed_dim, filters[3])

        # Decoder
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = self._block(filters[3], filters[2])
        self.up1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = self._block(filters[2], filters[1])
        self.up0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec1 = self._block(filters[1], filters[0])

        # Image output head
        self.final = nn.Conv2d(filters[0], 3, 1)

        # NEW: Color classification head
        self.color_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[3], filters[3]),
            nn.ReLU(),
            nn.Linear(filters[3], num_colors)
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, color_idx):
        cond_vec = self.color_emb(color_idx)
        cond_map = cond_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, cond_map], dim=1)

        x1 = self.enc1(x_cond)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        b = self.bottleneck_conv(self.pool(x3))

        # NEW: Get color classification prediction before FiLM
        color_logits = self.color_classifier(b)

        gamma = self.color_proj_gamma(cond_vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.color_proj_beta(cond_vec).unsqueeze(-1).unsqueeze(-1)
        b = gamma * b + beta

        d3 = self.up2(b)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up1(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up0(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        out_img = self.final(d1)
        
        # NEW: Return both the image and the color logits
        return torch.sigmoid(out_img), color_logits