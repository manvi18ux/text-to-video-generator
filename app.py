# app.py
#Builds the website interface with butons and sliders so anyone can use your project without touching code.

import gradio as gr
import torch
from generator import load_pipeline, generate_video
from prompt_enhancer import load_enhancer, enhance_prompt

# Load models once at startup
pipe     = load_pipeline()
enhancer = load_enhancer()


def run_generation(prompt, use_enhancer, cfg_scale, num_steps, seed):
    if not prompt.strip():
        return None, 'Please enter a prompt!'
    if use_enhancer:
        prompt = enhance_prompt(enhancer, prompt)
        status = f'Enhanced: {prompt}'
    else:
        status = 'Using original prompt'
    path = generate_video(
        pipe, prompt,
        cfg_scale=cfg_scale,
        steps=int(num_steps),
        seed=int(seed)
    )
    return path, status


with gr.Blocks(title='Text-to-Video AI', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🎬 Text-to-Video AI Generator')
    gr.Markdown('Generate short videos from text using Latent Diffusion Models.')

    with gr.Row():
        with gr.Column():
            prompt_in  = gr.Textbox(
                label='Your prompt',
                placeholder='A cat playing in autumn leaves...',
                lines=3
            )
            use_enh    = gr.Checkbox(label='Auto-enhance prompt', value=True)
            cfg        = gr.Slider(1, 15, value=7.5, step=0.5, label='CFG Scale')
            steps      = gr.Slider(10, 50, value=25, step=5, label='Steps')
            seed       = gr.Number(value=42, label='Seed')
            btn        = gr.Button('🎬 Generate', variant='primary')
            gr.Examples(
                examples=[
                    ['Ocean waves on a beach at sunset'],
                    ['A hummingbird near a red flower'],
                    ['Clouds moving across the sky'],
                    ['A campfire in a forest at night'],
                ],
                inputs=prompt_in
            )
        with gr.Column():
            video_out  = gr.Video(label='Generated Video')
            status_out = gr.Textbox(label='Status', interactive=False)

    btn.click(
        run_generation,
        inputs=[prompt_in, use_enh, cfg, steps, seed],
        outputs=[video_out, status_out]
    )

demo.launch(share=True)