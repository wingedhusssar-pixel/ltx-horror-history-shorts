"""Narration audio pass, run as an ISOLATED subprocess.

Why this is a separate process and not part of pipeline.py:
On this machine, loading the ChatterboxTTS model stack into a process that has
already loaded and run a diffusers/accelerate CUDA pipeline (the RealVisXL still
pass) causes a hard native segfault (exit 139) inside
ChatterboxTTS.from_pretrained() -- no Python traceback, process dies, zero wav
files. This was reproduced deterministically: TTS load + generate + save works
perfectly in a fresh process, but segfaults every time when run after the still
pass in the same process, even after correctly removing all accelerate offload
hooks. It is the same class of cross-model native crash already documented for
this box. Running the narration pass in its own clean process sidesteps the
corrupted CUDA/native state entirely.

This script reads the beats from pipeline_output/script.json (written by
pipeline.py before the still pass) and writes one {tag}_audio.wav per beat.
"""

import platform as _platform
_platform._wmi = None  # WMI query hangs on this system; force fast fallback
_platform.uname()      # pre-warm the platform cache via the fast path

import os
import sys
import json
import math
import struct

import torch
import torchaudio as ta
import chatterbox.tts as _chatterbox_tts
from chatterbox.tts import ChatterboxTTS

# chatterbox.tts.from_local() calls safetensors.torch.load_file(path) to load
# ve.safetensors / t3_cfg.safetensors / s3gen.safetensors. Confirmed via
# faulthandler (2026-07-18) that BOTH of safetensors' own loading entry points
# are unsafe on this machine for these specific checkpoint files:
#   - load_file() reads via safe_open(..., framework="pt"), which memory-maps
#     the file -- Windows access violation in torch/storage.py's __getitem__.
#   - An attempted fix routing through load(bytes) instead (no mmap) hit a
#     DIFFERENT native crash: "SystemError: deallocated bytearray object has
#     exported buffers" followed by a PyO3 panic, inside the shared Rust
#     deserialize() call that both entry points funnel through.
# Since the fragility is in safetensors' own Rust extension regardless of
# entry point, the fix is to never call into it. The safetensors format is a
# simple, fully documented layout: an 8-byte little-endian header length, a
# JSON header of {tensor_name: {dtype, shape, data_offsets}}, then one
# contiguous data blob -- trivial to parse by hand with the stdlib + torch.
_SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _load_file_via_bytes(path, device="cpu"):
    with open(path, "rb") as f:
        raw = f.read()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    data_start = 8 + header_len

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFETENSORS_DTYPE_MAP[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        if start == end:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            # frombuffer's offset/count read directly out of `raw` -- no
            # intermediate slice/bytearray copy. A MemoryError surfaced
            # (2026-07-18) from the previous version's `bytearray(raw[a:b])`,
            # which made TWO extra full copies of each tensor's bytes on top
            # of the .clone() below, tripling peak memory for large tensors
            # (the T3 LLM backbone checkpoint is the biggest of the three).
            count = math.prod(shape) if shape else 1
            tensor = torch.frombuffer(
                raw, dtype=dtype, count=count, offset=data_start + start
            ).reshape(shape).clone()
        tensors[name] = tensor.to(device) if device != "cpu" else tensor
    return tensors


_chatterbox_tts.load_file = _load_file_via_bytes

OUTPUT_DIR = "pipeline_output"


def main():
    script_path = os.path.join(OUTPUT_DIR, "script.json")
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found; nothing to narrate.", flush=True)
        return 2

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    all_beats = []
    for scene_idx, scene in enumerate(script["scenes"]):
        for beat_idx, beat in enumerate(scene["beats"]):
            tag = f"scene_{scene_idx}_beat_{beat_idx}"
            all_beats.append((tag, beat))

    print(f"Narration pass: {len(all_beats)} beat(s) to generate.", flush=True)

    print("Loading TTS model...", flush=True)
    # ChatterboxTTS's T3 Llama backbone calls torch.nn.init.kaiming_uniform_ to
    # randomly init every linear layer; that op triggers a native access
    # violation in torch_cpu.dll on this machine's hybrid-core CPU. Skip it:
    # from_pretrained() overwrites every one of those weights with the real
    # checkpoint immediately after construction, so the random values are
    # always discarded unused anyway.
    import torch.nn.init as _init
    _original_kaiming_uniform = _init.kaiming_uniform_
    _init.kaiming_uniform_ = lambda tensor, *a, **k: tensor
    try:
        tts_model = ChatterboxTTS.from_pretrained(device="cuda")
    finally:
        _init.kaiming_uniform_ = _original_kaiming_uniform

    # Safety check: confirm no parameter was left as uninitialized memory
    # (NaN/Inf) -- i.e. that every weight really was overwritten by the
    # checkpoint and the kaiming skip above was safe.
    _bad_params = [
        name for name, p in tts_model.t3.named_parameters()
        if torch.isnan(p).any() or torch.isinf(p).any()
    ]
    if _bad_params:
        raise RuntimeError(
            "ChatterboxTTS loaded with uninitialized weights in: "
            f"{_bad_params}. The kaiming_uniform_ skip assumption was wrong; "
            "these parameters were never overwritten by the checkpoint."
        )
    print("  TTS weight integrity check passed, all parameters loaded from checkpoint", flush=True)

    failures = 0
    for tag, beat in all_beats:
        print(f"\nGenerating narration audio for {tag}...", flush=True)
        wav = tts_model.generate(beat["narration"])
        # Drain all pending CUDA work on the main thread before any tensor
        # destructors run (mirrors the still-image pass fix).
        torch.cuda.synchronize()
        audio_path = os.path.join(OUTPUT_DIR, f"{tag}_audio.wav")
        ta.save(audio_path, wav, tts_model.sr)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Saved {audio_path}", flush=True)
        else:
            failures += 1
            print(f"ERROR: {audio_path} was not written.", flush=True)

    if failures:
        print(f"\nNarration pass finished with {failures} failure(s).", flush=True)
        return 1
    print(f"\nNarration pass done. {len(all_beats)} audio file(s) in ./{OUTPUT_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
