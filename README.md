# 🎬 Text-to-Video AI Generator

Generate short videos from text prompts using Latent Diffusion Models.

## Demo


![Ocean Waves](outputs/video_1.mp4)



## What It Does
Type any text prompt → AI generates a short video using diffusion models.

## Tech Stack
- Python, PyTorch
- Diffusion Models (ModelScope 1.7B — 1.7 Billion parameters)
- CLIP Text Encoder (converts text → 768-dim vectors)
- Variational Autoencoder (VAE)
- DDIM Scheduler (25 inference steps)
- Gradio Web UI

## How It Works
1. CLIP encodes your text prompt into 77×768 dimensional embeddings
2. Random noise sampled in latent space (8× smaller than pixel space via VAE)
3. U-Net runs 25 DDIM denoising steps conditioned on text via CFG
4. VAE decoder converts clean latents back to pixel frames
5. Frames exported as MP4 at 8 FPS

## Project Structure
text-to-video/
├── config.py            # All settings (CFG, steps, seed, fps)
├── generator.py         # Core pipeline (load model + generate)
├── prompt_enhancer.py   # Improves prompts with style presets
├── app.py               # Gradio web interface
└── outputs/             # Generated videos
## Run It Yourself

### Install dependencies
```bash
pip install -r requirements.txt
Generate a video
from generator import load_pipeline, generate_video

pipe = load_pipeline()
generate_video(pipe, "Ocean waves at sunset")
Launch Web UI
python app.py
Sample Outputs
Prompt
Style
Ocean waves crashing at sunset
Nature
Fire burning at night
Dramatic
Rocket launching into space
Cinematic
Key Concepts
Diffusion Models — Learn to reverse a gradual noising process
CFG (Classifier-Free Guidance) — Controls prompt adherence
Latent Space — Runs diffusion 48× faster than pixel space
Temporal Attention — Ensures frame-to-frame consistency in video
Interview Topics Covered
Diffusion model forward/reverse process
Reparameterization trick in VAE
Why latent space instead of pixel space
How CFG works mathematically
DDIM vs DDPM schedulers