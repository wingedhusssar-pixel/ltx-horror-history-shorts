# ltx-shorts (This is not my historical documentary channel pipeline, which remains private)

An automated pipeline that generates short-form videos about real, documented
historical mysteries, rendered in a VHS / found-footage aesthetic, and runs the
whole thing on a single 8GB laptop GPU (RTX 4070 Laptop, Windows).

It chains a language model, a diffusion image model, a text-to-speech model, and
a compositing engine into one batch run: topic -> script -> still images ->
narration -> analog-horror video assembly -> upload.

> **Note on scope.** This is a working personal system. Some content-specific
> pieces are withheld from the public repo and marked inline: the tuned LLM
> prompts, the full category/format taxonomy, and the music library. Every part
> that shows how the system is engineered (architecture, GPU memory handling,
> the native-crash fixes, DSP, compositing) is intact. Secrets and local assets
> are gitignored, not committed.

## Architecture

Three processes, deliberately separated:

1. **`pipeline.py`** — generates a non-repeating topic (Claude Haiku, with a
   difflib near-duplicate guard against a running topic history), expands it
   into a scene/beat script with narration and per-beat image prompts, then
   generates a still per beat with **RealVisXL V5.0** via `diffusers`. Map beats
   fetch a real public-domain antique map (`map_fetch.py`) instead of generating
   one, walking a specific-to-broad search ladder.
2. **`tts_pass.py`** — runs **ChatterboxTTS** narration in its own process.
3. **`stitch.py`** — assembles the final video with `moviepy`: TV-screen
   compositing, camcorder grain, analog-horror overlays, mood-matched music, and
   Whisper word-level caption alignment.

`pipeline.py` writes `script.json` and the per-beat stills/wavs; `stitch.py`
reads them back and renders. The stages talk through disk, so any stage can be
rerun alone.

## The interesting problem: two CUDA models on an 8GB card, on Windows

Most of the engineering here is about making heavy models coexist on a small GPU
without crashing. All of the following are real, reproduced-and-fixed issues,
documented inline where they are handled:

- **TTS-after-diffusers segfault.** Loading ChatterboxTTS into a process that
  has already run a `diffusers` CUDA pipeline segfaults natively (exit 139) with
  no Python traceback. Fix: run narration in a clean subprocess
  (`tts_pass.py`), invoked with `-X faulthandler` so any future native crash
  prints a named stack instead of a bare exit code.
- **VAE dtype-thrash access violation.** The SDXL pipeline upcasts the VAE to
  fp32 and back to fp16 on *every* decode. On a cold CUDA allocator (beat 0)
  that `cudaFree` hits uninitialized allocator state and access-violates in
  `c10.dll`. Fix: pin the VAE to fp32 once, set `force_upcast=False`, and bridge
  the latent dtype in a thin decode wrapper. Deterministic crash, gone.
- **Offload chain evicting the UNet.** `enable_model_cpu_offload()` forces the
  previous model in the offload sequence back to CPU before each call, so every
  `vae.decode()` was evicting the ~2.6GB UNet, and that device transfer is what
  segfaulted. Fix: sever `vae._hf_hook.prev_module_hook`, neutralize
  `maybe_free_model_hooks`, and buy the VRAM back with attention slicing instead
  of eviction. VRAM was measured directly via `torch.cuda.mem_get_info()`, not
  assumed.
- **Allocator free-path crash on output GC.** Chaining `.images[0]` straight off
  the pipeline call let Python free the CUDA output tensor while offload stream
  callbacks were still in flight. Fix: `torch.cuda.synchronize()` before any
  tensor destructor runs.
- **OpenMP/MKL first-use race.** The CPU-side VAE and post-processing kernels
  hit an intermittent thread-pool teardown race that null-derefs in
  `torch_cpu.dll`. Fix: pin thread counts *before* `import torch`.
- **Windows WMI hang.** `platform.uname()`'s WMI query hangs on this box; forced
  to the fast fallback at the top of the entry-point scripts.
- **Dependency break.** `setuptools` 82+ removed `pkg_resources`, which breaks
  `librosa`, which breaks `chatterbox-tts` at import. Pinned to `setuptools==81`.

## The analog-horror render (`stitch.py`)

The video look is built from signal-processing effects, not stock overlays:

- FFT-based seamless drifting fog/smoke noise fields (precomputed per clip to
  stay off the per-frame path and avoid memory blowups)
- screen-only vertical-hold tears on a precomputed event schedule, with audio
  dropouts fired on a fraction of tear events (visual-without-audio allowed,
  audio-without-visual never)
- 60Hz carrier hum and pitch-shifted tape slowdowns for deep voice drops
- per-frame lighting (tube flicker, candle flicker) with radial glow masks
- distance-based room-darkness masking, full-frame scan lines
- compositing onto the traced screen region of a real vintage-TV photo
- mood-matched background music, loudness-normalized *relative to* the measured
  narration RMS so the mix is consistent across tracks
- Whisper word-level caption timing

## Stack

Python 3.10 · PyTorch (CUDA) · diffusers (RealVisXL V5.0) · ChatterboxTTS ·
faster-whisper · moviepy · Anthropic API · Wikimedia Commons API

## Running it

This targets one specific machine, so treat this as reference, not a one-command
setup:

1. `pip install -r requirements.txt` (note the `setuptools==81` pin)
2. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
3. Drop mood-tagged tracks in `music/` and a caption font in `fonts/` (see
   `MOOD_MUSIC_MAP` in `stitch.py` for the expected names)
4. `python pipeline.py` then `python stitch.py`

## Layout

```
pipeline.py    topic + script (LLM) and still generation (diffusion)
tts_pass.py    narration TTS, isolated subprocess
stitch.py      video compositing, effects, music, captions
map_fetch.py   antique-map retrieval (Wikimedia Commons)
```
