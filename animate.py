import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

step = 8
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "sam749/RealCartoon3D-V15"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
adapter = adapter.to(device, dtype)
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()

negative_prompt = "close-up, portrait, detailed face, visible face, front view, zoomed in, photorealistic, blurry, distorted face, alien face, deformed hands, modern clothing, modern weapons, sci-fi armor, fantasy armor, empty room, no people"

prompts = [
    "Dozens of senator figures in white togas, all seen from behind with their backs turned to the camera, gathered in a grand marble Roman senate hall, silhouette figures, minimal facial detail, full wide shot showing the entire crowd and the whole room, ancient Rome architecture, 3D cartoon style, Pixar style",
]

for i, prompt in enumerate(prompts):
    print(f"Generating clip {i+1}/{len(prompts)}: {prompt}")
    output = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], "clip_caesar_v6.gif")
    print("Saved clip_caesar_v6.gif")