# config.py
#Settings file. How many frames? What size video? Change settings here, everything else follows.

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
}