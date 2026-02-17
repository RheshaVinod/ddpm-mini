# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import timestep_embedding

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class SimpleUNet(nn.Module):
    """
    Minimal U-Net-like denoiser.
    forward(x, t) -> predicts noise for x_t.
    """
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, base_ch)
        )

        # encoder
        self.conv1 = ConvBlock(in_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, base_ch)
        # bottleneck
        self.conv3 = ConvBlock(base_ch, base_ch)
        # decoder
        self.conv4 = ConvBlock(base_ch + base_ch, base_ch)
        self.conv5 = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x, t):
        """
        x: [B, C, H, W]
        t: [B] int timesteps
        """
        # basic timestep embedding
        temb = timestep_embedding(t, dim=128)  # [B, 128]
        temb = self.time_mlp(temb)             # [B, base_ch]

        # encoder
        h1 = self.conv1(x)                     # [B, base_ch, H, W]
        h2 = self.conv2(h1)                    # [B, base_ch, H, W]

        # add time embedding (broadcast)
        B, C, H, W = h2.shape
        temb_b = temb[:, :, None, None].expand(-1, -1, H, W)
        h = h2 + temb_b[:, :C, :, :]           # mix time info

        # bottleneck
        h = self.conv3(h)

        # decoder (skip connection)
        h = torch.cat([h, h1], dim=1)
        h = self.conv4(h)
        out = self.conv5(h)                    # predict noise, same channels as input
        return out
