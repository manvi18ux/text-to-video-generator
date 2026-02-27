import gradio as gr
import torch
import cv2
import os
import time
import json
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    'model_id': 'damo-vilab/text-to-video-ms-1.7b',
    'num_frames': 16,
    'fps': 8,
    'width': 256,
    'height': 256,
    'num_inference_steps': 25,
    'guidance_scale': 7.5,
    'seed': 42,
    'negative_prompt': 'blurry, low quality, distorted, flickering, watermark',
    'output_dir': 'outputs',
    'history_file': 'history.json',
}

# ─────────────────────────────────────────
# STYLE PRESETS
# ─────────────────────────────────────────
STYLE_PRESETS = {
    'Cinematic': 'cinematic lighting, professional camera, smooth motion, high quality',
    'Nature':    'golden hour lighting, natural colors, peaceful atmosphere',
    'Dramatic':  'dramatic lighting, high contrast, intense atmosphere, epic scale',
    'Minimal':   'clean composition, soft lighting, minimalist style',
    'Anime':     'anime style, vibrant colors, smooth animation, Studio Ghibli',
}

def enhance_prompt(raw, style):
    if len(raw.split()) > 15:
        return raw
    style_tags = STYLE_PRESETS.get(style, '')
    return f'{raw}, {style_tags}' if style_tags else raw

# ─────────────────────────────────────────
# FRAME INTERPOLATION
# ─────────────────────────────────────────
def interpolate_frames(frames, multiplier=2):
    if multiplier == 1:
        return frames
    interpolated = []
    for i in range(len(frames) - 1):
        frame_current = np.array(frames[i]).astype(np.float32)
        frame_next    = np.array(frames[i + 1]).astype(np.float32)
        interpolated.append(frames[i])
        for j in range(1, multiplier):
            alpha   = j / multiplier
            blended = cv2.addWeighted(
                frame_current, 1 - alpha,
                frame_next,    alpha, 0
            ).astype(np.uint8)
            interpolated.append(blended)
    interpolated.append(frames[-1])
    return interpolated

# ─────────────────────────────────────────
# TEMPORAL BLENDING — fixes motion flicker
# ─────────────────────────────────────────
def smooth_frames(frames):
    smoothed = []
    frames_f = [np.array(f).astype(np.float32) for f in frames]
    for i in range(len(frames_f)):
        if i == 0:
            blended = frames_f[i] * 0.7 + frames_f[i+1] * 0.3
        elif i == len(frames_f) - 1:
            blended = frames_f[i] * 0.7 + frames_f[i-1] * 0.3
        else:
            blended = (
                frames_f[i-1] * 0.15 +
                frames_f[i]   * 0.70 +
                frames_f[i+1] * 0.15
            )
        blended = blended.astype(np.uint8)
        blended = cv2.GaussianBlur(blended, (3, 3), 0.5)
        smoothed.append(blended)
    return smoothed

# ─────────────────────────────────────────
# HISTOGRAM MATCHING — fixes color flicker
# ─────────────────────────────────────────
def match_histograms(frames):
    reference = np.array(frames[0]).astype(np.uint8)
    matched   = [reference]
    for frame in frames[1:]:
        frame         = np.array(frame).astype(np.uint8)
        matched_frame = np.zeros_like(frame)
        for c in range(3):
            ref_hist = cv2.calcHist([reference], [c], None, [256], [0, 256])
            src_hist = cv2.calcHist([frame],     [c], None, [256], [0, 256])
            ref_cdf  = ref_hist.cumsum() / ref_hist.sum()
            src_cdf  = src_hist.cumsum() / src_hist.sum()
            lookup   = np.zeros(256, dtype=np.uint8)
            j = 0
            for k in range(256):
                while j < 255 and src_cdf[j] < ref_cdf[k]:
                    j += 1
                lookup[k] = j
            matched_frame[:, :, c] = cv2.LUT(frame[:, :, c], lookup)
        matched.append(matched_frame)
    return matched

# ─────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────
def load_history():
    if os.path.exists(CONFIG['history_file']):
        with open(CONFIG['history_file'], 'r') as f:
            return json.load(f)
    return []

def save_to_history(prompt, style, enhanced, path, duration):
    history = load_history()
    entry   = {
        'prompt':   prompt,
        'style':    style,
        'enhanced': enhanced,
        'path':     path,
        'duration': f'{duration:.1f}s',
        'time':     time.strftime('%Y-%m-%d %H:%M'),
    }
    history.insert(0, entry)
    history = history[:10]
    with open(CONFIG['history_file'], 'w') as f:
        json.dump(history, f, indent=2)
    return history

def format_history(history):
    if not history:
        return 'No generations yet.'
    lines = []
    for i, h in enumerate(history):
        lines.append(
            f"#{i+1} [{h['time']}]\n"
            f"  Prompt : {h['prompt']}\n"
            f"  Style  : {h['style']}\n"
            f"  Time   : {h['duration']}\n"
        )
    return '\n'.join(lines)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print('Loading model...')
        pipe = DiffusionPipeline.from_pretrained(
            CONFIG['model_id'],
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        print('✅ Model ready!')

# ─────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────
def generate_video(prompt, style, cfg_scale, steps, seed, interpolation):
    if not prompt.strip():
        return None, 'Please enter a prompt!', format_history(load_history())

    load_model()

    enhanced    = enhance_prompt(prompt, style)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    safe        = prompt[:20].replace(' ', '_').replace(',', '')
    filename    = f'{safe}_{int(time.time())}.mp4'
    output_path = os.path.join(CONFIG['output_dir'], filename)

    print(f'Prompt : {enhanced}')
    print(f'Style  : {style}')
    print(f'CFG    : {cfg_scale} | Steps: {steps} | Seed: {seed}')

    start  = time.time()
    result = pipe(
        prompt=enhanced,
        negative_prompt=CONFIG['negative_prompt'],
        num_frames=CONFIG['num_frames'],
        num_inference_steps=int(steps),
        guidance_scale=cfg_scale,
        width=CONFIG['width'],
        height=CONFIG['height'],
        generator=torch.manual_seed(int(seed)),
    )

    # Convert frames correctly
    frames = result.frames[0]
    if hasattr(frames[0], 'convert'):
        frames = [np.array(f.convert('RGB')) for f in frames]
    else:
        frames = [np.array(f) for f in frames]

    # Ensure 0-255 range
    frames = [
        (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        for f in frames
    ]

    # Step 1 — interpolation
    multiplier = {'None': 1, '2x Smoother': 2, '3x Smoother': 3}[interpolation]
    if multiplier > 1:
        frames = interpolate_frames(frames, multiplier)
        print(f'Frames : 16 → {len(frames)}')

    # Step 2 — histogram matching
    frames = match_histograms(frames)

    # Step 3 — temporal blending
    frames = smooth_frames(frames)

    # Step 4 — export
    output_fps = CONFIG['fps'] * multiplier
    export_to_video(frames, output_path, fps=output_fps)

    elapsed = time.time() - start
    history = save_to_history(prompt, style, enhanced, output_path, elapsed)

    status = (
        f'✅ Done in {elapsed:.1f}s\n'
        f'Frames: {len(frames)} | FPS: {output_fps} | Style: {style}'
    )
    return output_path, status, format_history(history)

# ─────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────
with gr.Blocks(title='Text-to-Video AI', theme=gr.themes.Monochrome()) as demo:

    gr.Markdown('# 🎬 Text-to-Video AI Generator')
    gr.Markdown('Generate short videos from text using Latent Diffusion Models · ModelScope 1.7B')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### ✍️ Prompt')
            prompt_in = gr.Textbox(
                label='Describe your video',
                placeholder='A cat playing in autumn leaves...',
                lines=3
            )
            gr.Markdown('### 🎨 Style')
            style_in  = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value='Cinematic',
                label='Style Preset',
                info='Automatically added to your prompt'
            )
            gr.Markdown('### 🎞️ Interpolation')
            interp_in = gr.Dropdown(
                choices=['None', '2x Smoother', '3x Smoother'],
                value='2x Smoother',
                label='Frame Interpolation',
                info='Adds frames for smoother motion'
            )
            gr.Markdown('### ⚙️ Settings')
            cfg_in    = gr.Slider(1, 15, value=7.5, step=0.5,
                                  label='CFG Scale',
                                  info='Low = creative | High = strict')
            steps_in  = gr.Slider(10, 50, value=25, step=5,
                                  label='Inference Steps',
                                  info='More = better quality, slower')
            seed_in   = gr.Number(value=42,
                                  label='Seed (same seed = same output)')
            btn       = gr.Button('🎬 Generate Video', variant='primary', size='lg')

            gr.Markdown('### 💡 Examples')
            gr.Examples(
                examples=[
                    ['Ocean waves on a beach at sunset'],
                    ['A hummingbird near a red flower'],
                    ['Clouds moving across the sky'],
                    ['A campfire in a forest at night'],
                    ['Snow falling in a quiet forest'],
                ],
                inputs=prompt_in
            )

        with gr.Column(scale=1):
            gr.Markdown('### 🎥 Output')
            video_out   = gr.Video(label='Generated Video')
            status_out  = gr.Textbox(
                label='Status',
                interactive=False,
                lines=3
            )
            gr.Markdown('### 📋 History')
            history_out = gr.Textbox(
                label='Past Generations (last 10)',
                interactive=False,
                lines=12,
                value='No generations yet.'
            )

    btn.click(
        fn=generate_video,
        inputs=[prompt_in, style_in, cfg_in, steps_in, seed_in, interp_in],
        outputs=[video_out, status_out, history_out]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)