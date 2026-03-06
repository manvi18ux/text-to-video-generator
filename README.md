---
title: Text-to-Video AI Generator
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: true
license: mit
---
# 🎬 Text-to-Video AI Generator

[

![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)

](https://huggingface.co/spaces/manvivek18ux/text-to-video)
[

![GitHub](https://img.shields.io/badge/GitHub-Repository-black)

](https://github.com/manvi18ux/text-to-video-generator)
Generate short videos from text prompts using Latent Diffusion Models.

> Built with ModelScope 1.7B · PyTorch · Diffusers · Gradio

---

## Features

- **Text to Video** — type any prompt, get a short AI generated video
- **Style Presets** — Cinematic, Nature, Dramatic, Minimal, Anime
- **Frame Interpolation** — 2x/3x smoother motion using frame blending
- **Anti-Flicker** — temporal blending + histogram matching for stable video
- **Generation History** — saves last 10 generations with prompts and timing
- **CFG Scale Slider** — control how strictly the model follows your prompt
- **Seed Control** — same seed = reproducible results

---

## How It Works

---
You type a prompt
↓
Style preset adds cinematic/nature/dramatic tags automatically
↓
CLIP encodes text → 77×768 dimensional embeddings
↓
U-Net runs 25 DDIM denoising steps conditioned on text via CFG
↓
VAE decoder converts latents → pixel frames
↓
Frame interpolation doubles/triples frame count for smooth motion
↓
Temporal blending + histogram matching reduces flickering
↓
MP4 video exported
---

## Architecture

---
Text Prompt
↓
CLIP Text Encoder (77×768 vectors)
↓
U-Net + Temporal Attention ←── DDIM Scheduler (25 steps)
↓                          CFG Scale (7.5)
VAE Decoder
↓
Raw Frames (16 frames)
↓
Frame Interpolation (16 → 32 frames)
↓
Temporal Blending + Histogram Matching
↓
MP4 Output (256×256 @ 16fps)
---

## Project Structure

---
text-to-video/
├── app.py               # Gradio UI + full pipeline
├── config.py            # All settings
├── generator.py         # Core generation logic
├── prompt_enhancer.py   # Style-based prompt enhancement
├── requirements.txt     # Dependencies
├── README.md
└── outputs/             # Generated videos
├── video_1.mp4
├── video_2.mp4
└── video_3.mp4
---

## Tech Stack

| Component     | Technology |
|-----------    |------------|
| Model         | ModelScope 1.7B (1.7B parameters) |
| Framework     | PyTorch + Diffusers |
| Text Encoder  | CLIP ViT-Large (768-dim) |
| Scheduler     | DDIM (25 steps) |
| VAE           | Variational Autoencoder |
| UI            | Gradio 5 |
|Post-processing| OpenCV (interpolation + blending) |

---
