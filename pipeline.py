import platform as _platform
_platform._wmi = None  # WMI (Win32_OperatingSystem query) hangs on this system; force fast fallback to getwindowsversion()
_platform.uname()  # pre-warm the cache now using the fast fallback path

import os

# Serialize torch's CPU intra-op thread pool. The VAE fp32-upcast and image
# post-processing at the end of the still pipeline run as CPU ATen kernels that
# fan out across OpenMP/MKL threads. That pool has an intermittent first-use /
# teardown race on Windows that null-derefs inside torch_cpu.dll (0xc0000005
# reading 0x0) ~once every N beats -- confirmed via Windows Event Log as a
# signature distinct from the c10.dll allocator and kaiming_uniform_ crashes.
# These env vars must be set BEFORE `import torch` to bind the OpenMP/MKL pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import sys
import json
import random
import difflib
import subprocess
import torch
torch.set_num_threads(1)  # belt-and-suspenders: also pin ATen's own intra-op pool
import anthropic
import map_fetch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
# NOTE: ChatterboxTTS / torchaudio are intentionally NOT imported here. The
# narration pass runs in a separate process (tts_pass.py) because loading the
# TTS stack after the diffusers still pass in the same process segfaults on
# this machine. See the narration audio pass section in run_pipeline().

# ---------- CONFIG ----------
OUTPUT_DIR = "pipeline_output"
NUM_SCENES = 4

# Topic history: a running log of topics already used, so the pipeline does not
# keep landing on the same famous case (the Mary Celeste, etc.) run after run.
# The recent entries are fed back into the prompt as a do-not-repeat list, and
# each new topic is appended after it is chosen. Lives outside OUTPUT_DIR so it
# survives the per-run cleanup that wipes the beat files.
TOPIC_HISTORY_PATH = "topic_history.txt"
TOPIC_HISTORY_LOOKBACK = 40   # how many recent topics to show the model


def load_topic_history(limit=TOPIC_HISTORY_LOOKBACK):
    """Return the most recent used topics, newest last, or [] if none yet."""
    if not os.path.exists(TOPIC_HISTORY_PATH):
        return []
    with open(TOPIC_HISTORY_PATH, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[-limit:]


def append_topic_history(topic):
    """Append a chosen topic to the history log."""
    with open(TOPIC_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(topic.strip() + "\n")


STILL_HEIGHT = 1216  # RealVisXL native vertical resolution, ~9:16-ish
STILL_WIDTH = 832    # within SDXL's native ~1MP training range
STILL_STEPS = 35     # RealVisXL recommended: 30+ steps with DPM++ SDE/2M Karras

# ---------- SCRIPT GENERATION ----------

client = anthropic.Anthropic()

# The full category taxonomy is withheld from the public repo (it is part of the
# channel's tuned content mix). A few representative entries are kept so the
# selection logic below reads clearly; generate_topic() picks one at random.
MYSTERY_CATEGORIES = [
    "a doomed or vanished expedition, ship, or flight (e.g. the Franklin Expedition, the Mary Celeste, Flight 19)",
    "an undeciphered text, code, cipher, or signal (e.g. the Voynich manuscript, the Zodiac ciphers, the Wow! signal)",
    "a broadcast or transmission intrusion that was never explained (e.g. the 1977 Vrillon intrusion, the Max Headroom hijack)",
    # ... additional categories withheld from public repo
]

# Full format list withheld from the public repo. generate_scenes() injects the
# real list into the system prompt; the model picks one format per video.
VIDEO_FORMATS = [
    "an unsolved case laid out clue by clue, ending on the piece that still does not fit",
    "two or three competing theories for one unexplained event, weighed against the evidence",
    "a real place and the unexplained thing that happened there, told through what was left behind",
    # ... additional formats withheld from public repo
]

# NOTE: The full system prompts below are withheld from the public repo.
# They are the tuned instructions that give the channel its specific voice and
# selection behavior, and are the main copyable asset in this project. The
# stubs preserve each prompt's role and output contract so the pipeline logic
# stays readable; the production wording lives outside version control.
TOPIC_SYSTEM_PROMPT = """[Production prompt withheld from public repo.]

Role: generate one specific, real, documented historical mystery for a given
category (unsolved cases, disappearances, strange artifacts, undeciphered texts,
hoaxes, eerie true events).

Enforces: real documented cases only; favor lesser-known over overexposed;
concrete and specific over vague; never invent a case, person, detail, or
explanation.

Output: a single topic sentence, nothing else."""

SYSTEM_PROMPT = """[Production prompt withheld from public repo.]

Role: turn one topic into a scene-by-scene short-video script for a channel
about real, documented, unexplained history.

Picks one video format from the provided list ({formats}), then returns a
structured plan. Internally enforces (full rules withheld):
- a strong factual hook on the first beat, mystery left open on the last;
- concrete verifiable nouns only, no invented resolutions;
- a photorealistic visual_prompt per beat, composed for a vertical
  letterboxed frame, with a special framing rule for flat documents;
- optional "map" beats that supply ordered search phrases for a real antique
  map, with a non-map fallback image;
- a short on-screen title and one mood tag drawn from a fixed set.

Output contract (this part is real, the pipeline parses it):
Return ONLY a JSON object with four fields:
  "format_used": the chosen format string,
  "title":       the short title-card text,
  "mood":        one mood tag from the fixed MOOD list,
  "scenes":      an array of scene objects, each with a "beats" array, where
                 each beat has "narration" and "visual_prompt" (and optionally
                 "map"). No markdown code fences."""


def _is_duplicate_topic(topic, history, threshold=0.8):
    """True if `topic` is an exact or near-exact match of something already used.

    The prompt already tells the model not to repeat prior topics, but that's
    an instruction, not a guarantee -- a narrow category (e.g. "unsolved theft")
    can have so few strong real candidates that the model picks the same one
    again anyway. difflib's ratio is a cheap way to catch both verbatim repeats
    and lightly reworded near-duplicates without an extra API call or embedding
    model; a real repeat shares almost all its proper nouns/dates/structure
    with the prior entry, so it scores well above unrelated topics in practice.
    """
    normalized = topic.strip().lower()
    for prior in history:
        if difflib.SequenceMatcher(None, normalized, prior.strip().lower()).ratio() >= threshold:
            return True
    return False


def generate_topic(max_attempts=3):
    history = load_topic_history()
    category = topic = None
    for attempt in range(1, max_attempts + 1):
        category = random.choice(MYSTERY_CATEGORIES)
        user_content = f"Category: {category}"
        if history:
            recent = "\n".join(f"- {t}" for t in history)
            user_content += (
                "\n\nYou have ALREADY covered the topics below. Do NOT repeat any of "
                "them or pick an obvious near-duplicate. Choose a DIFFERENT real, "
                "documented case, and favor a lesser-known one over the single most "
                "famous example in the category:\n" + recent
            )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=TOPIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            timeout=60.0,  # explicit timeout so a network hang raises a clear error instead of freezing silently forever
        )
        topic = response.content[0].text.strip()
        if not _is_duplicate_topic(topic, history):
            return category, topic
        print(f"  Duplicate topic on attempt {attempt}/{max_attempts}, retrying: {topic[:80]}...", flush=True)

    # Every attempt collided with history. Proceed with the last one rather
    # than fail the whole run over a repeated topic -- worst case is one
    # duplicate video, not a broken pipeline.
    return category, topic


def generate_scenes(topic, num_scenes=NUM_SCENES):
    system = SYSTEM_PROMPT.format(formats="\n".join(f"- {f}" for f in VIDEO_FORMATS))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5000,
        system=system,
        messages=[{"role": "user", "content": f"Topic: {topic}. Generate {num_scenes} scenes."}],
        timeout=90.0,  # explicit timeout, same reasoning as generate_topic
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as parse_error:
        # Print real diagnostics instead of just crashing blind. stop_reason
        # "max_tokens" means the response got cut off mid-JSON because the
        # model's reply was longer than the 3500 token budget allowed, the
        # most likely cause given this fails more often on longer/more
        # detailed topics. Printing the raw text also catches any other
        # cause (empty response, non-JSON reply, etc.) instead of leaving
        # it a total mystery.
        print(f"\n  JSON parse failed: {parse_error}")
        print(f"  stop_reason: {response.stop_reason}")
        print(f"  raw response text ({len(text)} chars):")
        print(f"  {text!r}\n")
        raise


# ---------- STILL IMAGE GENERATION SETUP (RealVisXL V5.0) ----------

def setup_still_pipeline():
    device = "cuda"
    dtype = torch.float16
    model_id = "SG161222/RealVisXL_V5.0"

    # IMPORTANT: do NOT call .to(device) here. enable_model_cpu_offload()
    # needs the pipeline on CPU so it can install its own offload hooks.
    # Calling .to("cuda") first caused a confirmed Windows access violation
    # when a second diffusers pipeline loaded in the same process later.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # Decode the VAE in PERMANENT float32 instead of letting diffusers upcast it
    # per-beat. The SDXL pipeline (pipeline_stable_diffusion_xl.py ~line 1254)
    # does, on EVERY decode:  upcast_vae() -> self.vae.to(float32)  ... decode ...
    # then  self.vae.to(float16)  to cast it back. Each of those two casts
    # allocates a fresh set of GPU buffers for the new dtype and frees the old
    # set. On the FIRST beat of a fresh process the CUDA caching allocator is
    # still cold (it has no cached blocks to hand back), so that fp16<->fp32
    # free is a real cudaFree against uninitialized allocator state -- which is
    # exactly the confirmed c10.dll access violation (Windows Event Log:
    # 0xc0000005, faulting offset 0x7f804, allocator free path). That is why the
    # crash is deterministic on scene_0_beat_0 and lands right AFTER the
    # upcast_vae FutureWarning prints (the warning is emitted inside upcast_vae,
    # immediately before its self.vae.to(float32) cast) and BEFORE "Saving..."
    # ever runs. On later beats the allocator is warm so the same casts reuse
    # cached blocks and never hit cudaFree -- but beat 0 dies every time, so we
    # never get there. NOTE: torch.cuda.synchronize() in the generation loop
    # cannot prevent this, because the crash happens INSIDE still_pipe(...)
    # during VAE decode, which is upstream of the synchronize/save lines.
    #
    # Pinning the VAE to float32 once removes both per-decode casts entirely.
    # The VAE was already running in float32 during decode (that's all the
    # upcast did), so output quality is identical; we just stop thrashing its
    # dtype. force_upcast=False stops the pipeline from re-triggering the upcast;
    # because the pipeline then no longer casts the latents up to the VAE dtype,
    # we bridge that one gap by casting the latents inside a thin decode wrapper.
    pipe.vae.to(torch.float32)
    pipe.vae.config.force_upcast = False
    _orig_vae_decode = pipe.vae.decode

    def _vae_decode_fp32(latents, *args, **kwargs):
        return _orig_vae_decode(latents.to(pipe.vae.dtype), *args, **kwargs)

    pipe.vae.decode = _vae_decode_fp32

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    # Sever the offload chain's link from unet to vae. model_cpu_offload_seq is
    # "text_encoder->text_encoder_2->image_encoder->unet->vae", and accelerate's
    # CpuOffload.pre_forward (accelerate/hooks.py) unconditionally forces the
    # PREVIOUS model in that chain back to CPU before the current one runs --
    # so every single vae.decode() call forces the entire unet (~2.6GB) off the
    # GPU first. Confirmed via faulthandler (2026-07-18): that forced unet
    # eviction, not any move of the vae itself, is what segfaults (Windows
    # access violation inside torch's per-tensor .to("cpu") copy, reached via
    # accelerate hooks.py pre_forward -> offload -> init_hook). It survives a
    # couple of decode cycles before corrupting, which is why testing only beat
    # 0 didn't catch it originally. Clearing prev_module_hook leaves vae's own
    # hook (and execution_device) intact -- pipeline._execution_device depends
    # on it being present -- it just stops that hook from evicting unet.
    #
    # CORRECTION (2026-07-19): the comment here originally claimed unet+vae
    # together are "well within this 8GB card" -- measured directly via
    # torch.cuda.mem_get_info() instrumentation, that was wrong. A CUDA
    # context alone costs ~1.1GB before any model is even loaded, and because
    # unet+vae now NEVER get evicted, every later beat's encode_prompt() still
    # has to fit the ~2.6GB text encoders in ALONGSIDE that permanently
    # resident ~3.9GB. Measured: mem_get_info() hit 0.000GB free after just
    # the first beat. That's what caused two real (non-segfault)
    # torch.OutOfMemoryError / CUDA OOM crashes during an actual batch run,
    # and a related numpy MemoryError during CPU-side postprocessing
    # (starved pinned-memory pool) -- not the access-violation class fixed
    # above, but a genuine, separate memory-budget problem this same fix
    # introduced. See enable_attention_slicing() below for the fix: cut peak
    # VRAM directly instead of re-introducing unet eviction (which would risk
    # the exact segfault this fix exists to avoid, since that crash was in
    # the low-level device-transfer itself, not particular to how it fired).
    pipe.vae._hf_hook.prev_module_hook = None

    # Disable the pipeline's own end-of-__call__ hook reset. StableDiffusionXL-
    # Pipeline.__call__ finishes with self.maybe_free_model_hooks(), which --
    # since _all_hooks is populated by enable_model_cpu_offload() above -- is
    # NOT a no-op: it fully reruns enable_model_cpu_offload() from scratch on
    # EVERY call. That rerun (a) does self.to("cpu", ...), moving every
    # component of the whole pipeline to CPU in one call, which is an even
    # bigger version of the same crashing device-transfer, and (b) reinstalls
    # a fresh hook chain, silently undoing the prev_module_hook fix above after
    # the very first beat. Confirmed via faulthandler (2026-07-18): with the
    # chain fix alone (before adding this), the crash reappeared on beat 0,
    # inside this exact self.to("cpu") call from maybe_free_model_hooks. This
    # reset-and-reapply is meant for one-off pipeline calls that want to hand
    # VRAM back afterward; it's actively harmful in our tight per-beat loop,
    # where we want unet+vae to just stay resident across beats and rely on
    # our own explicit unload_pipeline() for teardown once all beats are done.
    pipe.maybe_free_model_hooks = lambda: None

    # Directly cut peak VRAM during the unet forward pass, to buy back the
    # headroom that keeping unet+vae permanently resident gives up (see the
    # measured numbers above). Attention slicing processes attention in
    # chunks instead of one large batched op -- pure computation, no device
    # transfers, so it carries none of the crash risk that evicting unet did.
    pipe.enable_attention_slicing("max")

    return pipe


STILL_NEGATIVE_PROMPT = (
    "bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, "
    "deformed eyes, deformed mouth, open mouth, cartoon, anime, illustration, "
    "painting, 3d render, blurry, low quality, extra fingers, extra limbs"
)


def unload_pipeline(pipe):
    """Free a diffusers pipeline from GPU memory before loading the next one.

    enable_model_cpu_offload() installs accelerate hooks on the pipeline's
    component nn.Modules (unet, vae, text encoders). A DiffusionPipeline is
    NOT itself an nn.Module, so the previous remove_hook_from_module(pipe)
    call silently failed ('object has no attribute children') and left every
    offload hook installed. Use the pipeline's own remove_all_hooks(), which
    iterates the component modules and removes their hooks correctly.

    Deliberately does NOT move each component to CPU first (an earlier
    version did). Confirmed via faulthandler (2026-07-18) that the manual
    per-component `.to("cpu")` loop itself segfaults here (same class of
    device-transfer access violation as every other crash diagnosed tonight),
    and adding synchronize() around it did not help. That loop was also
    unnecessary: nothing in this process loads another diffusers pipeline
    afterward (the narration pass runs in its own subprocess), so once hooks
    are removed and `del pipe` drops the last reference, gc.collect() +
    empty_cache() below reclaim the GPU memory without ever touching the
    tensors' device placement directly.
    """
    try:
        pipe.remove_all_hooks()
    except Exception as cleanup_error:
        print(f"  (hook cleanup warning, continuing anyway: {cleanup_error})")

    del pipe
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # release cached CUDA IPC memory handles, beyond
                               # what empty_cache() alone frees, before the TTS
                               # subprocess tries to claim a fresh CUDA context
    import gc
    gc.collect()


# ---------- MAIN PIPELINE ----------

def run_pipeline():
    print("Pipeline starting...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear generated artifacts from any previous run BEFORE rendering. Otherwise
    # leftover beats from a run with a different beat count (for example after
    # switching channels) stay on disk and get stitched into the new video with
    # their old audio. Only the per-beat renders, narration wavs, and the old
    # script.json are removed; music, the TV photo, and anything else are left.
    import glob
    stale = []
    for pat in ("scene_*_still.png", "scene_*_video.gif", "scene_*_audio.wav", "script.json"):
        stale.extend(glob.glob(os.path.join(OUTPUT_DIR, pat)))
    for f in stale:
        try:
            os.remove(f)
        except OSError:
            pass
    if stale:
        print(f"Cleared {len(stale)} stale file(s) from a previous run.", flush=True)

    print("Generating topic...", flush=True)
    category, topic = generate_topic()
    print(f"Category: {category}")
    print(f"Topic: {topic}\n")
    append_topic_history(topic)  # record now, so it counts even if the render later fails

    print("Generating scenes...")
    script = generate_scenes(topic)
    print(f"Format used: {script['format_used']}\n")
    print(f"Title: {script.get('title', '(none)')}\n")
    print(f"Mood: {script.get('mood', '(none)')}\n")

    with open(os.path.join(OUTPUT_DIR, "script.json"), "w") as f:
        json.dump({"category": category, "topic": topic, **script}, f, indent=2)

    all_beats = []
    for scene_idx, scene in enumerate(script["scenes"]):
        for beat_idx, beat in enumerate(scene["beats"]):
            tag = f"scene_{scene_idx}_beat_{beat_idx}"
            all_beats.append((tag, beat))

    print(f"{len(all_beats)} beat(s) total, all still images\n")

    # ---- Still image generation pass ----
    print("Loading still image model (RealVisXL V5.0, this takes a moment)...")
    still_pipe = setup_still_pipeline()

    # Pre-encode every beat's prompt up front, in one pass, while
    # text_encoder/text_encoder_2 are still in their normal first-use state.
    # Measured via torch.cuda.mem_get_info() instrumentation (2026-07-19):
    # SDXL's unet alone is ~5.2GB in fp16 (not ~2.6GB as assumed when
    # prev_module_hook was severed above to stop it being evicted before
    # every vae decode). With unet now permanently resident, every beat's
    # internal encode_prompt() call reloading the ~2.6GB text encoders back
    # on top of that pushed the card to 0.000GB free during an actual batch
    # run -- two real CUDA OutOfMemoryErrors, not access violations, plus a
    # related CPU-side numpy MemoryError. encode_prompt() (called internally
    # by __call__) skips the text encoders entirely whenever *_embeds are
    # already supplied, so doing this once here means beats 1+ never need
    # the text encoders on GPU again at all -- they evict, via their own
    # ordinary hook chain, the first time unet's hook fires for beat 0
    # (the same transition that already happens safely in every prior run
    # tonight), and simply never come back.
    print("Pre-encoding prompts for all beats...", flush=True)
    beat_embeds = {}
    with torch.no_grad():
        for tag, beat in all_beats:
            if beat.get("map"):
                continue  # may not need a generated image at all; encode lazily below if it does
            beat_embeds[tag] = still_pipe.encode_prompt(
                prompt=beat["visual_prompt"],
                negative_prompt=STILL_NEGATIVE_PROMPT,
            )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for tag, beat in all_beats:
        print(f"\n--- {tag} ---", flush=True)
        print(f"Narration: {beat['narration']}", flush=True)

        image_path = os.path.join(OUTPUT_DIR, f"{tag}_still.png")

        # Map beat: fetch a real antique map instead of generating one. If every
        # search tier fails (nothing found, or no network), fall through to the
        # normal generated still from visual_prompt, so the beat is never empty.
        if beat.get("map"):
            print(f"Map beat, searching antique archives: {beat['map']}", flush=True)
            result = map_fetch.fetch_map(beat["map"], image_path)
            if result:
                print(f"Saved map still {image_path}", flush=True)
                continue
            print("  no antique map found, falling back to a generated still", flush=True)

        print(f"Visual: {beat['visual_prompt']}", flush=True)
        print("Generating still image...", flush=True)
        if tag in beat_embeds:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = beat_embeds[tag]
            output = still_pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                height=STILL_HEIGHT,
                width=STILL_WIDTH,
                num_inference_steps=STILL_STEPS,
                guidance_scale=7.0,
            )
        else:
            # Map beat that fell back to generation: never pre-encoded above
            # since whether it needs one at all depends on map_fetch's result.
            output = still_pipe(
                prompt=beat["visual_prompt"],
                negative_prompt=STILL_NEGATIVE_PROMPT,
                height=STILL_HEIGHT,
                width=STILL_WIDTH,
                num_inference_steps=STILL_STEPS,
                guidance_scale=7.0,
            )
        # The consistent c10.dll crash (Windows Event Log, offset 0x7f804,
        # 0xc0000005) is the CUDA caching allocator's free path being called
        # from an unsynchronized context: chaining .images[0] directly onto
        # the pipeline call lets Python GC the StableDiffusionXLPipelineOutput
        # temporary immediately, which triggers CUDA tensor frees while
        # model-offload stream callbacks may still be in flight. Synchronizing
        # here drains all pending CUDA work on the main thread before any
        # tensor destructors run, putting the allocator in a known-idle state.
        torch.cuda.synchronize()
        image = output.images[0]
        del output

        print(f"Saving {image_path}...", flush=True)
        image.save(image_path)
        print(f"Saved {image_path}", flush=True)

        torch.cuda.empty_cache()

    print("\nUnloading still image model...")
    unload_pipeline(still_pipe)

    # ---- Narration audio pass (isolated subprocess) ----
    # ROOT CAUSE (reproduced deterministically, 2026-06-25): loading the
    # ChatterboxTTS model stack into a process that has already loaded and run
    # a diffusers/accelerate CUDA pipeline (the RealVisXL still pass above)
    # segfaults (native exit 139) inside ChatterboxTTS.from_pretrained() --
    # no Python traceback, process dies silently, zero wav files. A fresh
    # process loads TTS, generates, and saves wav files perfectly; the crash
    # only happens on the still-pass -> TTS handoff in one process, and
    # persists even after correctly removing all accelerate offload hooks. It
    # is the same class of cross-model native crash already seen on this box.
    #
    # FIX: run the narration pass in its own clean subprocess (tts_pass.py),
    # which reads the beats back from pipeline_output/script.json (written
    # above) and writes one {tag}_audio.wav per beat. A nonzero/segfault exit
    # is turned into a loud RuntimeError instead of silently continuing.
    print("\nStarting narration audio pass in isolated subprocess...", flush=True)
    tts_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_pass.py")
    # -X faulthandler: if this subprocess still segfaults after the GPU cleanup
    # above, it prints a named native stack trace instead of a bare exit code,
    # so a future failure is diagnosable instead of just "exit code 3221225477".
    result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", tts_script])
    if result.returncode != 0:
        raise RuntimeError(
            f"Narration audio pass failed (exit code {result.returncode}). "
            "See the subprocess output above."
        )

    print(f"\nDone. All beats saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    run_pipeline()