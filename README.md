# DDPM-Mini: Diffusion-Based Image Generation in PyTorch

A minimal end-to-end implementation of a Denoising Diffusion Probabilistic Model (DDPM) built from scratch in PyTorch.  
This project demonstrates how diffusion models progressively add noise to images and learn to reverse that process to generate new samples from pure Gaussian noise.

---

## 🚀 Project Overview

Diffusion models are generative models that learn to transform random noise into structured data through iterative denoising. This implementation includes:

- Linear beta noise schedule
- Sinusoidal timestep embeddings
- U-Net style convolutional denoiser
- Noise prediction objective (MSE loss)
- Iterative reverse sampling procedure
- Training on MNIST (32×32) for fast experimentation

After training, the model can generate handwritten digits starting from random noise.

---

## 🧠 How It Works

### Forward Diffusion (Adding Noise)

An image is gradually corrupted with Gaussian noise over multiple timesteps:

x₀ → x₁ → x₂ → ... → xₜ → pure noise

At timestep `t`:

xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε

Where:
- ᾱₜ is the cumulative product of alphas
- ε ~ N(0, I)

---

### Reverse Process (Learning to Denoise)

The neural network is trained to predict the noise added at each timestep:

Loss = MSE(predicted_noise, true_noise)

During sampling:
1. Start with random noise
2. Iteratively remove predicted noise
3. Produce a generated image

This is the same foundational principle behind modern models like Stable Diffusion.

---

## 🏗 Architecture

- Lightweight U-Net-style CNN
- GroupNorm + SiLU activations
- Sinusoidal timestep embeddings
- Linear noise schedule
- Implemented entirely in PyTorch



## 📂 Project Structure

