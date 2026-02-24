# test_generate.py
# Run this to test your pipeline with multiple prompts

from generator import load_pipeline, generate_video

# Load model once
pipe = load_pipeline()

# Test prompts — start with simple ones
test_prompts = [
    "A fire burning, close up shot",
    "Ocean waves on a beach",
    "Clouds moving across the sky",
]

print("\n=== Starting generation tests ===\n")

for i, prompt in enumerate(test_prompts):
    print(f"\n--- Test {i+1}/{len(test_prompts)} ---")
    path = generate_video(pipe, prompt)
    print(f"✅ Done: {path}")

print("\n=== All videos generated! Check your outputs/ folder ===")