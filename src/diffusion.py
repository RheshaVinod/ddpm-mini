# src/diffusion.py
import torch
import torch.nn.functional as F

class Diffusion:
    """
    Minimal DDPM utilities: linear beta schedule, q_sample, and p_losses (noise prediction MSE).
    """
    def __init__(self, timesteps=200, device='cpu', beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)  # [T]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample q(x_t | x_0)
        x_start: [B, C, H, W]
        t: [B] long
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)
        # gather scalars per-batch
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x_start + sqrt_om * noise

    def p_losses(self, denoise_model, x_start, t):
        """
        Compute MSE between predicted noise and true noise for x_t.
        """
        noise = torch.randn_like(x_start, device=self.device)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted = denoise_model(x_noisy, t)  # model predicts noise
        return F.mse_loss(predicted, noise)
