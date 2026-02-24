# prompt_enhancer.py
#Takes your simple prompt like "a dog running" and makes it richer: "a golden retriever running through a meadow, cinematic lighting, 4K"

STYLE_PRESETS = {
    'cinematic': 'cinematic lighting, professional camera, smooth motion, high quality',
    'nature':    'golden hour lighting, natural colors, peaceful atmosphere',
    'dramatic':  'dramatic lighting, high contrast, intense atmosphere, epic scale',
    'minimal':   'clean composition, soft lighting, minimalist style',
}

def enhance_prompt(raw, style='cinematic'):
    # Don't enhance if already detailed
    if len(raw.split()) > 15:
        return raw
    style_tags = STYLE_PRESETS.get(style, STYLE_PRESETS['cinematic'])
    enhanced = f'{raw}, {style_tags}'
    return enhanced