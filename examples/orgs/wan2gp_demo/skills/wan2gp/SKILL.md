---
name: wan2gp
description: Wan2GP generation parameters (low-end-PC image generation).
---

# Wan2GP Generation Parameters

Wan2GP generates images on low-end GPUs. A generation form has these fields:

- `prompt` (str, required): the text prompt.
- `negative_prompt` (str): terms to avoid.
- `width` / `height` (int): image dimensions; 512x512 is the safe default for
  low-end hardware, 1024x1024 only when the user asks for detail.
- `steps` (int): 20 default; 12 for speed, 35 for quality.
- `guidance_scale` (float): 7.5 default.
- `seed` (int | null): null = random.

Always emit a complete form. Never omit `prompt`.
