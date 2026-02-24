# generator.py
#The main engine. Loads the AI model and runs it to convert your text into videoframes.

import torch
import os
import time
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from config import CONFIG


def load_pipeline():
    print('Loading model...')
    pipe = DiffusionPipeline.from_pretrained(
        CONFIG['model_id'],
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    print('✅ Pipeline loaded!')
    print(f'  Text encoder : {type(pipe.text_encoder).__name__}')
    print(f'  U-Net        : {type(pipe.unet).__name__}')
    print(f'  VAE          : {type(pipe.vae).__name__}')
    print(f'  Scheduler    : {type(pipe.scheduler).__name__}')
    params = sum(p.numel() for p in pipe.unet.parameters()) / 1e9
    print(f'  U-Net params : {params:.2f}B')
    return pipe


def generate_video(pipe, prompt, output_filename=None, cfg_scale=None,
                   steps=None, seed=None):
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    cfg_scale = cfg_scale or CONFIG['guidance_scale']
    steps     = steps     or CONFIG['num_inference_steps']
    seed      = seed      or CONFIG['seed']

    if output_filename is None:
        safe = prompt[:25].replace(' ', '_').replace(',', '')
        output_filename = safe + '.mp4'

    output_path = os.path.join(CONFIG['output_dir'], output_filename)

    print(f'\nPrompt : {prompt}')
    print(f'CFG: {cfg_scale}  |  Steps: {steps}  |  Seed: {seed}')
    print('Generating...')

    start = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=CONFIG['negative_prompt'],
        num_frames=CONFIG['num_frames'],
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=CONFIG['width'],
        height=CONFIG['height'],
        generator=torch.manual_seed(seed),
    )

    elapsed = time.time() - start

    # Fix: convert PIL frames to numpy if needed
    frames = result.frames[0]
    if hasattr(frames[0], 'convert'):
        frames = [np.array(frame) for frame in frames]

    export_to_video(frames, output_path, fps=CONFIG['fps'])

    dur = CONFIG['num_frames'] / CONFIG['fps']
    print(f'✅ Done in {elapsed:.1f}s → {output_path} ({dur:.1f}s video)')
    return output_path