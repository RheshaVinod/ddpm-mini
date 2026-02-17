# src/train.py
"""
Train script (minimal). Usage:
    python src/train.py
"""
import os
import torch
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm

from src.model import SimpleUNet
from src.diffusion import Diffusion

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------- data ----------
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # range [-1,1]
    ])
    train_ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

    # ---------- model / diffusion ----------
    model = SimpleUNet(in_ch=1, base_ch=64).to(device)
    diffusion = Diffusion(timesteps=200, device=device)
    opt = Adam(model.parameters(), lr=1e-3)

    steps = 0
    for epoch in range(1):  # keep small for demo; increase as desired
        pbar = tqdm.tqdm(dl, desc=f"Epoch {epoch}")
        for x, _ in pbar:
            x = x.to(device)            # already normalized to [-1,1] by transform
            b = x.size(0)
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
            loss = diffusion.p_losses(model, x, t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if steps % 100 == 0:
                pbar.set_postfix(loss=loss.item())
            # save a checkpoint and a small sample occasionally
            if steps % 500 == 0:
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "ckpt_latest.pth"))
                # quick denoise-of-clean (reconstruction) for sanity:
                with torch.no_grad():
                    x_noisy = diffusion.q_sample(x, t)
                    pred_noise = model(x_noisy, t)
                    recon = x_noisy - pred_noise  # simple single-step denoise heuristic (not true sampling)
                    save_image((recon + 1) / 2, os.path.join(RESULTS_DIR, f"recon_{steps}.png"), nrow=8)
            steps += 1

    # final save
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "ckpt_final.pth"))
    print("Training finished. Check results/ for images and checkpoints.")

if __name__ == "__main__":
    main()
