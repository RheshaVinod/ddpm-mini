# src/utils.py
import torch
import math

def timestep_embedding(timesteps, dim: int):
    """
    Create sinusoidal timestep embeddings.
    timesteps: [B] ints
    returns: [B, dim]
    """
    assert len(timesteps.shape) == 1
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device).float() * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # pad
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)], dim=1)
    return emb
