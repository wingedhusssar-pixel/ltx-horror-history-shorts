import platform as _platform
_platform._wmi = None  # WMI hang workaround, same fix as pipeline.py/stitch.py
_platform.uname()

import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

OUTPUT_DIR = "tv_frame_candidates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target final video frame is 448x640 (9:16-ish). We want a TV image where
# the screen opening takes up almost the full width, with cabinet space
# (wood frame, knobs, speaker grille) showing mainly above and below, only
# a little on the sides. This is an unusual proportion for a real TV, so
# the prompt leans hard on "tall," "narrow," and "portrait" to push the
# model away from a typical wide TV silhouette.

PROMPTS = {
    "living_room": (
        "A large flat black TV screen filling almost the whole image, "
        "thin antique carved wood border on all sides, perfectly "
        "rectangular, shot straight on, small sliver of an elegant old "
        "room visible at the top edge, photorealistic, cinematic, "
        "detailed texture, sharp focus"
    ),
    "battlefield": (
        "A large flat black TV screen filling almost the whole image, "
        "thin scorched wood border on all sides, perfectly rectangular, "
        "shot straight on, small sliver of a war-torn battlefield visible "
        "at the top edge, cratered ground, barbed wire, smoke, "
        "photorealistic, cinematic, detailed texture, sharp focus"
    ),
    "creepy_backrooms": (
        "A large flat black TV screen filling almost the whole image, "
        "thin grimy yellowed border on all sides, perfectly rectangular, "
        "shot straight on, small sliver of yellow-green backrooms "
        "wallpaper visible at the top edge, fluorescent light, "
        "photorealistic, detailed texture, sharp focus"
    ),
}

NEGATIVE_PROMPT = (
    "wide screen, landscape orientation, horizontal television, modern flat "
    "screen TV, bad hands, bad anatomy, deformed, blurry, low quality, text, watermark"
)


def setup_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    print("Loading RealVisXL V5.0 (this takes a moment)...")
    pipe = setup_pipeline()

    for name, prompt in PROMPTS.items():
        print(f"\nGenerating candidate: {name}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=1216,
            width=832,
            num_inference_steps=35,
            guidance_scale=7.0,
        ).images[0]
        out_path = os.path.join(OUTPUT_DIR, f"tv_{name}.png")
        image.save(out_path)
        print(f"Saved {out_path}")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\nDone. Check the {OUTPUT_DIR}/ folder for candidates.")


if __name__ == "__main__":
    main()