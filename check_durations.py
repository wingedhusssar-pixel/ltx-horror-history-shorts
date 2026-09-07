import os
import soundfile as sf
from PIL import Image

OUTPUT_DIR = "pipeline_output"

files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith("_video.gif"))

for video_file in files:
    tag = video_file.replace("_video.gif", "")
    gif_path = os.path.join(OUTPUT_DIR, video_file)
    wav_path = os.path.join(OUTPUT_DIR, f"{tag}_audio.wav")

    if not os.path.exists(wav_path):
        print(f"{tag}: missing wav, skip")
        continue

    gif = Image.open(gif_path)
    frame_count = gif.n_frames
    frame_duration_ms = gif.info.get("duration", 0)
    gif_duration_sec = (frame_count * frame_duration_ms) / 1000

    info = sf.info(wav_path)
    wav_duration_sec = info.frames / float(info.samplerate)

    print(f"{tag}: gif={gif_duration_sec:.2f}s ({frame_count} frames), wav={wav_duration_sec:.2f}s")