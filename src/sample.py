# src/sample.py
"""
Simple DDPM-style sampling script. Not optimized for speed or best-quality sampling;
works as a demonstration to produce sample images given a trained checkpoint.
Usage:
    python src/sample.py --ckpt results/ckpt_final.pth
"""
import argparse
import torch
from torchvision.utils import save_image
import os

from src.model import SimpleUNet
from src.diffusion import Diffusion

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def sample_loop(model, diffusion, shape=(16,1,32,32), device='cpu'):
    model.eval()
    B, C, H, W = shape
    x = torch.randn(shape, device=device)
    T = diffusion.timesteps
    for i in reversed(range(T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        with torch.no_grad():
            predicted_noise = model(x, t)
        beta = diffusion.betas[i]
        alpha = diffusion.alphas[i]
        alpha_cum = diffusion.alphas_cumprod[i]
        # DDPM posterior mean simplified:
        # x_{t-1} = 1/sqrt(alpha) * (x_t - (beta / sqrt(1 - alpha_cum)) * predicted_noise) + sigma * z
        coef = beta / torch.sqrt(1 - alpha_cum)
        mean = (1.0 / torch.sqrt(alpha)) * (x - coef * predicted_noise)
        if i > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * z
        else:
            x = mean
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/ckpt_final.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_ch=1, base_ch=64).to(device)
    diffusion = Diffusion(timesteps=200, device=device)

    # load checkpoint
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("Loaded checkpoint", args.ckpt)
    else:
        print("Checkpoint not found:", args.ckpt)
        return

    # sample
    samples = sample_loop(model, diffusion, shape=(16,1,32,32), device=device)
    save_image((samples + 1) / 2, os.path.join(RESULTS_DIR, "samples.png"), nrow=4)
    print("Saved samples to results/samples.png")

if __name__ == "__main__":
    main()
