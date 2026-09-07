import platform as _platform
_platform._wmi = None  # WMI (Win32_OperatingSystem query) hangs on this system; force fast fallback to getwindowsversion()
_platform.uname()  # pre-warm the cache now using the fast fallback path

import os
import re
import json
import random
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import butter, lfilter
import whisper
from PIL import Image, ImageFont, ImageDraw
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.video.fx import MultiplyColor
from moviepy.audio.fx import MultiplyVolume

import tv_overlay
import frame_overlay

MUSIC_DIR = "music"
MUSIC_TO_NARRATION_RATIO = 0.405  # lowered 10% from 0.45 per user feedback, music was a bit too loud relative to narration
NARRATION_VOLUME = 0.85
HOOK_AUDIO_DELAY = 0.5

TAPE_OUTRO_PATH = "tape_outro.mp3"
TAPE_OUTRO_VOLUME = 0.9

ENABLE_CAMCORDER_LOOK = True
GRAIN_STRENGTH = 18
VIGNETTE_STRENGTH = 0.0  # disabled, room darkness mask handles darkening instead
FLICKER_STRENGTH = 0.06
DESATURATION_AMOUNT = 0.75
SICKLY_TINT_STRENGTH = 0.35  # 0 = pure neutral gray desaturation, 1 = fully shifted toward yellow-green, the core Backrooms color signature
CHROMATIC_ABERRATION_STRENGTH = 3  # pixel offset for RGB channel split at frame edges, subtle "filmed not rendered" unease
IRREGULAR_FLICKER_DOUBLE_CHANCE = 0.12  # probability per second of a brief double-flicker burst instead of the normal smooth flicker, mimics real arrhythmic fluorescent light
LETTERBOX_HEIGHT_RATIO = 0.06
GLITCH_FLASH_CHANCE_PER_SECOND = 0.15
GLITCH_FLASH_DURATION = 0.2

ENABLE_TV_OVERLAY = True

# Which screen world the broadcast plays in. "curved_tv" uses tv_overlay.py (the
# CRT). "backrooms" and "living_room" use the flat frames in frame_overlay.py.
ACTIVE_FRAME = "curved_tv"   # "curved_tv" | "backrooms" | "living_room"
TV_OVERLAY_CAMCORDER_SCREEN_ONLY = True

AMBIENT_GLOW_ENABLED = True
AMBIENT_GLOW_STRENGTH = 0.18

# Distance-based room darkening: the room/cabinet area falls toward near-
# black the further it sits from the screen, rather than being uniformly
# dim or uniformly bright. Per user request: the TV should "barely light
# up an otherwise dark room, only really lighting up the direct vicinity
# significantly". 1.0 = pixel rendered at full original brightness,
# 0.0 = pixel rendered fully black.
ROOM_NEAR_SCREEN_BRIGHTNESS = 0.9  # multiplier for room pixels immediately adjacent to the screen
ROOM_FAR_DARKNESS = 0.04  # multiplier for room pixels far from the screen, near-black

# Curated by track title/character (not audible review). Two of these
# filenames were truncated with a literal "..." (a copy-paste artifact),
# which silently failed pick_music_track's os.path.exists() check and made
# those two tracks (Silent Hill 2, molina) permanently unreachable -- the
# real pool was 8 of 10 files, not 10. Fixed here, which alone adds real
# variety back. Also lightly rebalanced so no single track (REPULSIVE was in
# 4 mood buckets) carries too much of the rotation, and "tense" -- the
# mood the SYSTEM_PROMPT in pipeline.py explicitly favors most -- gets the
# widest candidate pool.
# Music filenames are withheld from the public repo. The real library is a set
# of mood-tagged tracks in MUSIC_DIR; placeholder names below keep the mood
# system's structure intact. Drop your own .mp3 files in music/ and rename these
# keys' values to match. pick_music_track() only plays files that exist on disk.
MOOD_MUSIC_MAP = {
    "dramatic": ["track_dramatic_01.mp3", "track_dramatic_02.mp3"],
    "tense": ["track_tense_01.mp3", "track_tense_02.mp3", "track_tense_03.mp3", "track_tense_04.mp3"],
    "triumphant": ["track_triumphant_01.mp3"],
    "calm": ["track_calm_01.mp3", "track_calm_02.mp3"],
    "celtic": ["track_celtic_01.mp3", "track_celtic_02.mp3"],
    "indian": ["track_indian_01.mp3", "track_indian_02.mp3"],
    "medieval": ["track_medieval_01.mp3", "track_medieval_02.mp3"],
    "middle_eastern": ["track_middle_eastern_01.mp3", "track_middle_eastern_02.mp3"],
    "asian_tense": ["track_asian_tense_01.mp3", "track_asian_tense_02.mp3"],
    "asian_calm": ["track_asian_calm_01.mp3", "track_asian_calm_02.mp3"],
}

FALLBACK_MUSIC = ["track_fallback_01.mp3", "track_fallback_02.mp3", "track_fallback_03.mp3"]

# How many of the most recently used tracks to avoid repeating, mirroring
# pipeline.py's topic_history.txt pattern so a batch week doesn't lean on the
# same 1-2 tracks. Only enforced when a mood has enough real alternatives;
# a mood backed by a single track (e.g. "triumphant") will still repeat.
MUSIC_HISTORY_PATH = "music_history.txt"
MUSIC_HISTORY_LOOKBACK = 5


def load_music_history(limit=MUSIC_HISTORY_LOOKBACK):
    if not os.path.exists(MUSIC_HISTORY_PATH):
        return []
    with open(MUSIC_HISTORY_PATH, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[-limit:]


def append_music_history(filename):
    with open(MUSIC_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(filename.strip() + "\n")

OUTPUT_DIR = "pipeline_output"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_video.mp4")

WHISPER_MODEL_SIZE = "medium"

CAPTION_FONT = os.environ.get("CAPTION_FONT", "fonts/georgiab.ttf")  # override via env; ship your own font in fonts/
CAPTION_FONTSIZE = 48
CAPTION_COLOR = "#F5F0E6"  # cream highlight color for the currently-spoken word within a phrase
CAPTION_STROKE_COLOR = "#2B2118"  # stroke for the highlighted word
PHRASE_COLOR = "white"  # the rest of the phrase, not currently being spoken
PHRASE_STROKE_COLOR = "black"  # stroke for the non-highlighted phrase text, classic bold caption look per user's reference image
HIGHLIGHT_SIZE_MULTIPLIER = 1.25  # the currently-spoken word renders this much bigger than the rest of the phrase, not just a different color

# ---- Title card ----
TITLE_CARD_DURATION = 1.0  # seconds the title card shows before the first beat
TITLE_CARD_FONTSIZE = 128
TITLE_CARD_COLOR = "white"
TITLE_CARD_STROKE_COLOR = "black"
TITLE_CARD_STROKE_WIDTH = 7
TITLE_CARD_MAX_WIDTH_RATIO = 0.92
TITLE_CARD_TOP_MARGIN = 60  # pixels from the top of the frame, top-center positioning per user request

# ---- Static transition between beats ----
STATIC_TRANSITION_DURATION = 0.15  # seconds of heavy static shown at every beat-to-beat cut

CAPTION_STROKE_WIDTH = 3
CAPTION_MAX_WIDTH_RATIO = 0.85

# ---- Room fog effect ----
ENABLE_ROOM_FOG = True
ROOM_FOG_BLUR_RADIUS = 6    # gaussian blur sigma applied to the room region, higher = hazier

# ---- Analog horror overlay (full-frame, applied before camcorder effects) ----
ENABLE_ANALOG_HORROR_OVERLAY = True

# Scan lines: slow-drifting horizontal bands darkening the full frame
SCANLINE_BAND_HEIGHT = 4        # pixels tall per dark band
SCANLINE_DARK_STRENGTH = 0.0    # set to 0 to disable scanlines
SCANLINE_DRIFT_SPEED = 12.0     # pixels per second the bands drift downward

# Vertical hold failure: the whole frame tears sideways for a brief moment
VERTICAL_HOLD_CHANCE_PER_SECOND = 0.12   # probability per second a tear event starts
VERTICAL_HOLD_DURATION = 0.10            # seconds the tear lasts
VERTICAL_HOLD_BAND_HEIGHT = 60           # pixel height of the tearing band
VERTICAL_HOLD_MAX_SHIFT = 40             # max horizontal pixel shift during a tear

# ---- Ambient "living" layer: fan-chopped light flicker, slow breathing, drifting fumes ----
# Built to match the reference frames: a dim scene with rhythmic light flicker
# (as if a fan blade keeps crossing a ceiling light), a slow overall brightness
# swell, and slow smoke fumes drifting through the room. Flicker and breathing
# stay on the TV footage; the fumes drift in the viewer's dark room only.
ENABLE_FAN_FLICKER = True
FAN_FLICKER_FREQUENCY = 3.2      # Hz, blade-pass rhythm
FAN_FLICKER_DEPTH = 0.18         # how far the light dips at its darkest
FAN_FLICKER_IRREGULARITY = 0.22  # 0 = perfectly periodic, 1 = very uneven
FAN_FLICKER_SHARPNESS = 2.0      # >1 = mostly bright with quick dark dips

BREATHING_ENABLED = True
BREATHING_PERIOD = 11.0          # seconds for one slow brightness swell
BREATHING_DEPTH = 0.08           # gentle overall rise and fall

# Drifting smoke fumes: soft low-opacity billows rolling slowly through the dark
# room around the tube, NOT on the screen. The field is built in the frequency
# domain so it is seamless under np.roll drift (no wrap seam, unlike sparse
# dust). Kept faint and slow so it reads as smoke haze, not fog filling the room.
ENABLE_FUMES = True
FUME_LAYERS = 3                  # parallax depth layers, each its own billow pattern
FUME_BRIGHTNESS = 32.0           # additive glow of the nearest layer (0-255), keep faint
FUME_SOFTNESS = 1.8             # low-pass power: higher = larger, smoother billows
FUME_CONTRAST = 4.5             # >1 thins the smoke into wisps instead of a flat wash
FUME_DRIFT_Y = 5.0              # pixels/sec base vertical drift, reads as motion within seconds
FUME_DRIFT_X = 6.0              # pixels/sec base horizontal drift
# Per-layer drift angles (radians) so billows roll in different directions and
# cross, reading as smoke wandering rather than one sheet sliding across.
FUME_LAYER_ANGLES = [0.5, 2.3, 4.2]
FUME_SWAY_AMPLITUDE = 9.0        # pixels of slow side-to-side wander per layer
FUME_SWAY_PERIOD = 17.0          # seconds per sway cycle
FUME_FEATHER_SIGMA = 26.0        # px, softness of the fade-out band as fumes approach the screen

# How fast a room light's glow dissipates from its center. Lower = broader,
# softer spread; higher = tighter pool. Used to build the radial light mask.
LIGHT_FALLOFF = 1.2

# If a flat frame is selected, apply its room-treatment overrides so a bright
# world (backrooms) is not crushed dark like the curved-TV viewer room. The
# curved TV keeps the tuned defaults above. These reassign the module globals
# that the camcorder pass reads at render time.
if ACTIVE_FRAME != "curved_tv":
    _room = frame_overlay.room_settings(ACTIVE_FRAME)
    ROOM_FAR_DARKNESS = _room.get("far_darkness", ROOM_FAR_DARKNESS)
    ROOM_NEAR_SCREEN_BRIGHTNESS = _room.get("near_brightness", ROOM_NEAR_SCREEN_BRIGHTNESS)
    ENABLE_ROOM_FOG = _room.get("fog", ENABLE_ROOM_FOG)
    ROOM_FOG_BLUR_RADIUS = _room.get("fog_radius", ROOM_FOG_BLUR_RADIUS)
    ENABLE_FUMES = _room.get("fumes", ENABLE_FUMES)

# ---- Analog horror audio distortion (applied to the full mixed output) ----
ENABLE_ANALOG_HORROR_AUDIO = True

# Carrier hum: a constant low-frequency sine tone underneath the whole mix
HUM_FREQUENCY = 60.0        # Hz, matches US mains frequency for authenticity
HUM_AMPLITUDE = 0.018       # relative to full scale

# Tape slowdown: brief pitch/speed drop on the full mix
TAPE_SLOWDOWN_CHANCE_PER_BEAT = 0.12    # probability per beat-length window
TAPE_SLOWDOWN_DURATION = 0.5            # seconds the slowdown lasts
TAPE_SLOWDOWN_FACTOR = 0.82             # speed multiplier during slowdown (1.0 = normal)
TAPE_PITCH_SHIFT_MIN = 0.72             # lightest pitch drop
TAPE_PITCH_SHIFT_MAX = 0.55             # deepest pitch drop (heavy)

# Audio dropout: full mix briefly cuts or distorts
AUDIO_DROPOUT_CHANCE_PER_BEAT = 0.25   # probability per beat-length window
AUDIO_DROPOUT_ON_TEAR_CHANCE = 0.40    # fraction of visual tear events that also trigger an audio dropout
AUDIO_DROPOUT_DURATION = 0.18          # seconds of dropout
AUDIO_DROPOUT_VOLUME = 0.04            # volume during dropout (near-silence)

# Full-mix degradation folded into apply_analog_horror_audio. These run on the
# whole mix BEFORE the carrier hum is added, so the bandpass does not strip the
# 60Hz hum tone. Each effect is gated and blended so it can be tuned by ear.
ENABLE_MIX_WOW_FLUTTER = True
MIX_WOW_DEPTH = 0.0030      # slow pitch wobble depth
MIX_WOW_RATE = 0.5          # Hz, slow wow
MIX_FLUTTER_DEPTH = 0.0012  # fast pitch wobble depth
MIX_FLUTTER_RATE = 7.3      # Hz, fast flutter

ENABLE_MIX_BROADCAST_EQ = True
MIX_EQ_LOW = 180.0          # Hz, low corner of the band
MIX_EQ_HIGH = 5200.0        # Hz, high corner, thins the top end
MIX_EQ_BLEND = 0.6          # 0 = original full range, 1 = fully band-limited

ENABLE_MIX_BITCRUSH = True
MIX_BITCRUSH_BITS = 11      # lower = grittier
MIX_BITCRUSH_BLEND = 0.5    # 0 = clean, 1 = fully crushed

FRAME_WIDTH = 448
FRAME_HEIGHT = 640

ZOOM_FACTOR = 1.15


def active_screen_composite(content_img, W, H):
    """Composite the content still into the active screen, routing to the curved
    TV (tv_overlay) or a flat frame (frame_overlay) based on ACTIVE_FRAME."""
    if ACTIVE_FRAME == "curved_tv":
        return tv_overlay.composite_into_screen(content_img, W, H)
    return frame_overlay.composite_into_screen(ACTIVE_FRAME, content_img, W, H)


def active_screen_mask(W, H):
    """Return the active screen's content mask, from the curved TV or flat frame."""
    if ACTIVE_FRAME == "curved_tv":
        return tv_overlay.get_screen_content_mask(W, H)
    return frame_overlay.get_screen_content_mask(ACTIVE_FRAME, W, H)


def load_valid_beat_keys():
    """Return the set of (scene_idx, beat_idx) beats defined in the current
    script.json. Used to ignore orphaned beat files left over from a previous
    run with a different beat count, which would otherwise get stitched into the
    video with their old audio. Returns None if script.json is missing, in which
    case find_beats falls back to using whatever renders are on disk."""
    script_path = os.path.join(OUTPUT_DIR, "script.json")
    if not os.path.exists(script_path):
        return None
    with open(script_path) as f:
        script = json.load(f)
    keys = set()
    for scene_idx, scene in enumerate(script.get("scenes", [])):
        for beat_idx, _ in enumerate(scene.get("beats", [])):
            keys.add((scene_idx, beat_idx))
    return keys


def find_beats():
    valid = load_valid_beat_keys()
    pattern = re.compile(r"scene_(\d+)_beat_(\d+)_(still|video)\.(png|gif)$")
    beats = {}
    skipped = 0
    for fname in os.listdir(OUTPUT_DIR):
        match = pattern.match(fname)
        if match:
            scene_idx = int(match.group(1))
            beat_idx = int(match.group(2))
            render_type = match.group(3)
            if valid is not None and (scene_idx, beat_idx) not in valid:
                skipped += 1  # orphan from an earlier run, not in the current script
                continue
            beats[(scene_idx, beat_idx)] = render_type
    if skipped:
        print(f"Skipped {skipped} stale beat file(s) not in the current script.json.")
    return sorted(beats.items())


def freeze_extend(video_clip, target_duration):
    if video_clip.duration >= target_duration:
        return video_clip.subclipped(0, target_duration)
    gap = target_duration - video_clip.duration
    last_frame = video_clip.get_frame(video_clip.duration - 0.04)
    freeze_clip = ImageClip(last_frame).with_duration(gap)
    freeze_clip = freeze_clip.with_fps(video_clip.fps)
    return concatenate_videoclips([video_clip, freeze_clip])


def pick_punch_time(words, duration):
    if not words:
        return None
    middle_words = [w for w in words if duration * 0.3 <= w[1] <= duration * 0.7]
    candidates = middle_words if middle_words else words
    chosen = random.choice(candidates)
    return chosen[1]


def ease_in_out(progress):
    return progress * progress * (3 - 2 * progress)


def build_panzoom_clip(image_path, duration, punch_time=None):
    base_clip = ImageClip(image_path).with_duration(duration)
    src_w, src_h = base_clip.size
    SAFETY_OVERSCAN = 1.03
    scale_to_cover = max(FRAME_WIDTH / src_w, FRAME_HEIGHT / src_h) * SAFETY_OVERSCAN
    covered_w = int(src_w * scale_to_cover) + 2
    covered_h = int(src_h * scale_to_cover) + 2
    base_clip = base_clip.resized((covered_w, covered_h))

    zoom_direction = random.choice(["in", "out"])
    pan_direction = random.choice(["left", "right", "none"])

    PUNCH_DURATION = 0.18
    PUNCH_STRENGTH = 0.06

    def punch_offset(t):
        if punch_time is None:
            return 0
        delta = abs(t - punch_time)
        if delta > PUNCH_DURATION:
            return 0
        punch_progress = 1 - (delta / PUNCH_DURATION)
        return PUNCH_STRENGTH * (punch_progress ** 2)

    def make_frame_resizer(t):
        raw_progress = t / duration if duration > 0 else 0
        progress = ease_in_out(raw_progress)
        if zoom_direction == "in":
            base_scale = 1.0 + (ZOOM_FACTOR - 1.0) * progress
        else:
            base_scale = ZOOM_FACTOR - (ZOOM_FACTOR - 1.0) * progress
        return base_scale + punch_offset(t)

    def current_scale(t):
        raw_progress = t / duration if duration > 0 else 0
        progress = ease_in_out(raw_progress)
        if zoom_direction == "in":
            return 1.0 + (ZOOM_FACTOR - 1.0) * progress
        else:
            return ZOOM_FACTOR - (ZOOM_FACTOR - 1.0) * progress

    def make_position(t):
        raw_progress = t / duration if duration > 0 else 0
        progress = ease_in_out(raw_progress)
        scale_now = current_scale(t)
        max_shift = (covered_w * scale_now - FRAME_WIDTH) / 2
        max_shift = max(max_shift, 0)
        if pan_direction == "left":
            x_offset = max_shift - (2 * max_shift * progress)
        elif pan_direction == "right":
            x_offset = -max_shift + (2 * max_shift * progress)
        else:
            x_offset = 0
        return (x_offset, "center")

    zoomed_clip = base_clip.resized(make_frame_resizer)
    positioned_clip = zoomed_clip.with_position(make_position)
    final = CompositeVideoClip(
        [positioned_clip], size=(FRAME_WIDTH, FRAME_HEIGHT)
    ).with_duration(duration)
    final = final.with_effects([MultiplyColor(1.06)])
    punch_note = f", punch at {punch_time:.2f}s" if punch_time is not None else ""
    print(f"  still motion: zoom_{zoom_direction}, pan_{pan_direction}, eased{punch_note}")
    return final


def build_tv_clip(image_path, duration):
    content_img = Image.open(image_path)
    composite_arr = active_screen_composite(content_img, FRAME_WIDTH, FRAME_HEIGHT)
    static_clip = ImageClip(composite_arr).with_duration(duration)
    final = CompositeVideoClip(
        [static_clip], size=(FRAME_WIDTH, FRAME_HEIGHT)
    ).with_duration(duration)
    print(f"  TV overlay composite: still fit into traced screen region, no pan/zoom")
    return final


def get_word_timestamps(whisper_model, wav_path):
    result = whisper_model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word_text = word_info["word"].strip()
            if word_text:
                words.append((word_text, word_info["start"], word_info["end"]))
    return words


def align_known_text_to_timing(known_text, whisper_words):
    known_words = known_text.split()
    if not whisper_words:
        return [(w, i, i + 1) for i, w in enumerate(known_words)]
    if len(known_words) == len(whisper_words):
        return [
            (known_words[i], whisper_words[i][1], whisper_words[i][2])
            for i in range(len(known_words))
        ]
    total_start = whisper_words[0][1]
    total_end = whisper_words[-1][2]
    total_span = max(total_end - total_start, 0.01)
    per_word = total_span / len(known_words)
    aligned = []
    for i, w in enumerate(known_words):
        start = total_start + (i * per_word)
        end = total_start + ((i + 1) * per_word)
        aligned.append((w, start, end))
    return aligned


def group_words_into_phrases(words, max_words_per_phrase=4):
    """Group a flat word list into phrase-level chunks, splitting at
    natural clause boundaries (commas, periods) rather than a fixed word
    count, since that reads more naturally as a caption unit. Each
    original word's trailing punctuation determines where a phrase ends.
    As a safety cap, also force-breaks any run longer than
    max_words_per_phrase even without punctuation, since an unusually
    long clause could otherwise produce a phrase too wide to fit on
    screen at any allowed font size. Returns a list of phrase groups,
    where each group is a list of (word_text, start, end) tuples from
    the original words list."""
    phrases = []
    current_phrase = []
    for word_text, start, end in words:
        current_phrase.append((word_text, start, end))
        stripped = word_text.rstrip()
        ends_clause = stripped.endswith((",", ".", "!", "?", ";", ":"))
        hit_max_length = len(current_phrase) >= max_words_per_phrase
        if ends_clause or hit_max_length:
            phrases.append(current_phrase)
            current_phrase = []
    if current_phrase:
        phrases.append(current_phrase)
    return phrases


def render_phrase_frame(words_upper, highlight_index, font_size, video_width, video_height):
    """Render ONE phrase as a single PIL image, with all words at
    font_size in PHRASE_COLOR/PHRASE_STROKE_COLOR, EXCEPT the word at
    highlight_index, which renders bigger (HIGHLIGHT_SIZE_MULTIPLIER) in
    CAPTION_COLOR/CAPTION_STROKE_COLOR. This replaces the previous
    approach of compositing two separately-positioned moviepy TextClips
    (a normal-size base phrase plus a separately-sized highlight word),
    which required the two clips' positions to agree pixel-for-pixel and
    broke twice in testing (off-screen crash, then visible double-text
    misalignment when the highlight word's size changed). Drawing
    everything by hand into ONE image guarantees correct relative
    positioning by construction, since every word is placed using the
    same running x-offset in the same coordinate system, rather than
    relying on two independent clips to land in agreement.

    Returns a PIL Image (RGBA, transparent background) sized to fit the
    rendered text with margin, along with the rendered width and height,
    so the caller can center/position the resulting single clip."""
    from PIL import ImageFont, ImageDraw

    normal_font = ImageFont.truetype(CAPTION_FONT, font_size)
    highlight_font_size = int(font_size * HIGHLIGHT_SIZE_MULTIPLIER)
    highlight_font = ImageFont.truetype(CAPTION_FONT, highlight_font_size)

    # Measure each word's rendered width at its actual font (normal or
    # highlighted), plus a single space width, to lay out the whole line.
    probe_img = Image.new("RGBA", (10, 10))
    probe_draw = ImageDraw.Draw(probe_img)

    space_width = probe_draw.textlength(" ", font=normal_font)

    word_metrics = []  # (text, font, width, is_highlight)
    for i, word in enumerate(words_upper):
        is_highlight = (i == highlight_index)
        font = highlight_font if is_highlight else normal_font
        width = probe_draw.textlength(word, font=font)
        word_metrics.append((word, font, width, is_highlight))

    total_width = sum(w for _, _, w, _ in word_metrics) + space_width * (len(word_metrics) - 1)
    # Generous height to fit the taller (highlighted) font plus stroke and descenders
    line_height = int(highlight_font_size * 1.5) + (CAPTION_STROKE_WIDTH * 4)

    margin = CAPTION_STROKE_WIDTH + 6
    canvas_width = int(total_width) + margin * 2
    canvas_height = line_height + margin * 2

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # All words share the same baseline (bottom-aligned), computed from
    # the tallest (highlighted) font, so a bigger highlighted word grows
    # upward from the same line the other words sit on, rather than
    # words appearing to float at different heights.
    baseline_y = canvas_height - margin

    x_cursor = margin
    for word, font, width, is_highlight in word_metrics:
        fill_color = CAPTION_COLOR if is_highlight else PHRASE_COLOR
        stroke_color = CAPTION_STROKE_COLOR if is_highlight else PHRASE_STROKE_COLOR

        # anchor="ls" = left, baseline-aligned. This is what guarantees
        # every word, regardless of its own font size, sits on the same
        # visual line rather than top-aligning (which would make a bigger
        # word's top edge match instead of its baseline, looking wrong).
        draw.text(
            (x_cursor, baseline_y),
            word,
            font=font,
            fill=fill_color,
            stroke_fill=stroke_color,
            stroke_width=CAPTION_STROKE_WIDTH,
            anchor="ls",
        )
        x_cursor += width + space_width

    return canvas, canvas_width, canvas_height


def build_caption_clips(words, video_size):
    """Build caption clips grouped into natural phrases (split at commas/
    periods). Each phrase is rendered as a SINGLE image (see
    render_phrase_frame) per active-word state, so the currently-spoken
    word appears bigger and in the cream highlight color while the rest
    of the phrase stays a consistent size in white/black, all correctly
    positioned by construction since it's one rendered image, not
    multiple composited clips."""
    width, height = video_size
    max_text_width = width * CAPTION_MAX_WIDTH_RATIO

    def measure_phrase_width(words_upper, font_size):
        _, canvas_width, _ = render_phrase_frame(words_upper, -1, font_size, width, height)
        return canvas_width

    def split_oversized_phrase(phrase_words, font_size):
        """If a phrase doesn't fit within max_text_width even at the
        smallest allowed font, split it into two halves and recurse."""
        words_upper = [w.upper() for w, _, _ in phrase_words]
        if measure_phrase_width(words_upper, font_size) <= max_text_width or len(phrase_words) <= 1:
            return [phrase_words]
        mid = len(phrase_words) // 2
        left_half = phrase_words[:mid]
        right_half = phrase_words[mid:]
        return split_oversized_phrase(left_half, font_size) + split_oversized_phrase(right_half, font_size)

    phrases = group_words_into_phrases(words)

    # Safety pass: guarantee every phrase fits at the SMALLEST allowed
    # font size before doing any rendering.
    safe_phrases = []
    for phrase_words in phrases:
        safe_phrases.extend(split_oversized_phrase(phrase_words, 16))
    phrases = safe_phrases

    caption_clips = []
    previous_phrase_end = None

    for phrase_words in phrases:
        if not phrase_words:
            continue

        words_upper = [w.upper() for w, _, _ in phrase_words]
        phrase_start = phrase_words[0][1]
        phrase_end = phrase_words[-1][2]

        # Prevent overlap with the previous phrase.
        CAPTION_GAP_SECONDS = 0.02
        if previous_phrase_end is not None and phrase_start < previous_phrase_end + CAPTION_GAP_SECONDS:
            phrase_start = previous_phrase_end + CAPTION_GAP_SECONDS
            phrase_start = min(phrase_start, phrase_end - 0.05)
        previous_phrase_end = phrase_end

        # Find the font size that fits the WIDEST state of this phrase
        # (i.e. whichever word being highlighted produces the widest
        # render, since the highlighted word is bigger), so the chosen
        # font size is safe across every highlight-state this phrase will
        # cycle through, not just the unhighlighted state.
        font_size = CAPTION_FONTSIZE
        while font_size > 16:
            widest_this_size = max(
                measure_phrase_width(words_upper, font_size) if hi == -1
                else render_phrase_frame(words_upper, hi, font_size, width, height)[1]
                for hi in range(-1, len(words_upper))
            )
            if widest_this_size <= max_text_width:
                break
            font_size -= 4

        letterbox_bar_height_px = int(height * LETTERBOX_HEIGHT_RATIO)
        bottom_margin_px = int(height * 0.10) + letterbox_bar_height_px

        # Render one clip per (sub-duration, highlighted word) pair: the
        # phrase highlights word 0 from its start until word 1 starts,
        # then word 1 until word 2 starts, and so on, then the LAST
        # word's highlight holds until the phrase's true end.
        for i, (word_text, w_start, w_end) in enumerate(phrase_words):
            seg_start = max(w_start, phrase_start) if i == 0 else w_start
            seg_end = phrase_words[i + 1][1] if i + 1 < len(phrase_words) else phrase_end
            seg_duration = max(seg_end - seg_start, 0.05)

            frame_img, frame_w, frame_h = render_phrase_frame(words_upper, i, font_size, width, height)
            frame_arr = np.array(frame_img)

            target_bottom_y = height - bottom_margin_px
            y_position = target_bottom_y - frame_h
            max_allowed_y = height - letterbox_bar_height_px - frame_h - 5
            y_position = min(y_position, max_allowed_y)
            y_position = max(y_position, 0)
            x_position = (width - frame_w) / 2

            img_clip = ImageClip(frame_arr).with_start(seg_start).with_duration(seg_duration)
            img_clip = img_clip.with_position((x_position, y_position))
            caption_clips.append(img_clip)

    return caption_clips


def build_beat_clip(scene_idx, beat_idx, render_type, whisper_model, narration_text, apply_punch=False, audio_delay=0):
    tag = f"scene_{scene_idx}_beat_{beat_idx}"
    wav_path = os.path.join(OUTPUT_DIR, f"{tag}_audio.wav")
    audio = AudioFileClip(wav_path)
    audio = audio.with_effects([MultiplyVolume(NARRATION_VOLUME)])

    print(f"{tag}: running Whisper alignment...")
    whisper_words = get_word_timestamps(whisper_model, wav_path)
    words = align_known_text_to_timing(narration_text, whisper_words)
    print(f"{tag}: {len(words)} words aligned (whisper detected {len(whisper_words)})")

    total_duration = audio.duration + audio_delay
    if audio_delay > 0:
        words = [(w, start + audio_delay, end + audio_delay) for w, start, end in words]

    if render_type == "video":
        gif_path = os.path.join(OUTPUT_DIR, f"{tag}_video.gif")
        video = VideoFileClip(gif_path)
        visual_clip = freeze_extend(video, total_duration)
        print(f"{tag} (video): video {video.duration:.2f}s -> {visual_clip.duration:.2f}s, audio {audio.duration:.2f}s, delay {audio_delay}s")
    else:
        still_path = os.path.join(OUTPUT_DIR, f"{tag}_still.png")
        if ENABLE_TV_OVERLAY:
            visual_clip = build_tv_clip(still_path, total_duration)
            print(f"{tag} (still): {audio.duration:.2f}s, TV overlay composite, delay {audio_delay}s")
        else:
            punch_time = pick_punch_time(words, total_duration) if apply_punch else None
            visual_clip = build_panzoom_clip(still_path, total_duration, punch_time)
            print(f"{tag} (still): {audio.duration:.2f}s, panzoom applied, punch at {punch_time}, delay {audio_delay}s")

    if audio_delay > 0:
        audio = audio.with_start(audio_delay)

    # Composite the narration audio against a silent base track spanning
    # the FULL total_duration. This fixes a real crash found in testing:
    # "Accessing time t=5.90-5.94 seconds, with clip duration=5.880000
    # seconds", a read request landing a fraction of a second past the
    # narration clip's own internal end, most likely from fps-rounding
    # during the final write step. A silent full-length base track
    # guarantees any read within [0, total_duration] finds valid (silent)
    # samples, rather than the narration clip alone erroring on a
    # boundary read it can't actually satisfy.
    from moviepy.audio.AudioClip import AudioClip as _AudioClip
    silent_base = _AudioClip(lambda t: 0.0, duration=total_duration, fps=audio.fps if hasattr(audio, "fps") else 44100)
    audio = CompositeAudioClip([silent_base, audio])

    visual_clip = visual_clip.with_audio(audio)
    visual_clip = visual_clip.with_duration(total_duration)
    # IMPORTANT: captions are NOT composited here anymore. They used to be
    # baked into each beat's clip before concatenation, which meant the
    # camcorder look (grain, vignette, flicker, glitch flashes) applied
    # AFTER concatenation was unintentionally also degrading the caption
    # text itself, causing a visible flicker/glitch on the captions that
    # was never supposed to touch them. Captions are now returned
    # separately (as a list of (clip, beat_offset_seconds) pairs) so the
    # caller can composite them onto the final video AFTER the camcorder
    # look has already been applied, keeping caption text clean and
    # unaffected by that effect.
    raw_words = words  # already absolute-timed relative to this beat's own clip start at 0
    return visual_clip, raw_words


def pick_music_track(mood):
    """Pick a background music file matching the script's mood tag
    (generated by pipeline.py, see MOOD_MUSIC_MAP above). Falls back to
    FALLBACK_MUSIC if mood is missing/unrecognized (e.g. an older
    script.json from before mood tagging existed) or if none of the
    mapped files exist on disk.

    Avoids repeating a track used in the last MUSIC_HISTORY_LOOKBACK videos
    when the mood has another real candidate, so a batch week doesn't lean
    on the same 1-2 tracks. Only falls back to a recent repeat when every
    matching candidate has been used recently (e.g. a mood backed by a
    single track)."""
    candidates = MOOD_MUSIC_MAP.get(mood, [])
    existing = [f for f in candidates if os.path.exists(os.path.join(MUSIC_DIR, f))]
    if not existing:
        existing = [f for f in FALLBACK_MUSIC if os.path.exists(os.path.join(MUSIC_DIR, f))]
    if not existing:
        return None

    recent = set(load_music_history())
    fresh = [f for f in existing if f not in recent]
    pool = fresh if fresh else existing

    chosen = random.choice(pool)
    append_music_history(chosen)
    return os.path.join(MUSIC_DIR, chosen)


def find_strongest_section(audio_clip, section_duration, window_size=2.0):
    sample_rate = 22050
    samples = audio_clip.to_soundarray(fps=sample_rate)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    window_samples = int(window_size * sample_rate)
    if window_samples <= 0 or len(samples) <= window_samples:
        return 0.0
    hop = max(window_samples // 2, 1)
    rms_values = []
    rms_times = []
    for start_sample in range(0, len(samples) - window_samples, hop):
        window = samples[start_sample:start_sample + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
        rms_times.append(start_sample / sample_rate)
    if not rms_values:
        return 0.0
    windows_per_section = max(int(section_duration / (window_size / 2)), 1)
    num_candidates = len(rms_values) - windows_per_section + 1
    best_start_time = 0.0
    best_score = -1
    for i in range(num_candidates):
        loudness_score = sum(rms_values[i:i + windows_per_section]) / windows_per_section
        position_fraction = i / max(num_candidates - 1, 1)
        position_bonus = 1.0 + (position_fraction * 0.35)
        score = loudness_score * position_bonus
        if score > best_score:
            best_score = score
            best_start_time = rms_times[i]
    max_start = max(audio_clip.duration - section_duration, 0)
    return min(best_start_time, max_start)


def measure_rms_loudness(audio_clip):
    """Measure a clip's RMS (root mean square) loudness, which tracks
    perceived volume much better than peak amplitude. Peak-based
    normalization (the old approach) was confirmed by the user to behave
    inconsistently across tracks: a track with one loud transient spike
    but mostly quiet content gets the same peak-normalized treatment as a
    track that's consistently loud throughout, even though the second one
    is genuinely louder to listen to. RMS reflects sustained loudness,
    which is what actually matters for "does this sound too loud/quiet
    relative to the narration"."""
    sample_rate = 22050
    samples = audio_clip.to_soundarray(fps=sample_rate)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def normalize_to_target_rms(audio_clip, target_rms):
    """Scale audio so its RMS loudness matches target_rms exactly,
    regardless of the source track's own mastering loudness. Replaces
    the old peak-based normalize_to_target_peak for the reason described
    in measure_rms_loudness."""
    current_rms = measure_rms_loudness(audio_clip)
    if current_rms <= 0:
        return audio_clip
    scale_factor = target_rms / current_rms
    return audio_clip.with_effects([MultiplyVolume(scale_factor)])


def build_background_music(track_path, target_duration, narration_rms):
    """Build the background music bed, with its loudness set RELATIVE TO
    THE ACTUAL NARRATION'S MEASURED LOUDNESS for this specific video,
    rather than a fixed absolute target. This is the direct fix for the
    user's report that the old fixed-volume approach sounded right for
    some tracks/videos but too quiet for others, since narration
    recordings and music tracks both vary in their own natural loudness,
    a single fixed number can't track that relationship correctly.

    MUSIC_TO_NARRATION_RATIO controls how far below the narration's
    loudness the music sits."""
    from moviepy import concatenate_audioclips
    full_track = AudioFileClip(track_path)
    if full_track.duration <= target_duration:
        loops_needed = int(target_duration / full_track.duration) + 1
        music = concatenate_audioclips([AudioFileClip(track_path) for _ in range(loops_needed)])
        music = music.subclipped(0, target_duration)
    else:
        section_start = find_strongest_section(full_track, target_duration)
        print(f"  strongest section starts at {section_start:.1f}s")
        music = full_track.subclipped(section_start, section_start + target_duration)

    target_music_rms = narration_rms * MUSIC_TO_NARRATION_RATIO
    music = normalize_to_target_rms(music, target_music_rms)
    print(f"  narration RMS: {narration_rms:.4f}, music target RMS: {target_music_rms:.4f} (ratio {MUSIC_TO_NARRATION_RATIO})")
    return music


def build_tear_event_schedule(duration):
    """Precompute all vertical hold tear events for the full video duration.
    Returns a list of (start_time, end_time) tuples. Used by both the video
    and audio passes so tears and dropouts fire at the same moments."""
    events = []
    for second in range(int(duration) + 1):
        rng = np.random.RandomState(second + 7777)
        if rng.random() < VERTICAL_HOLD_CHANCE_PER_SECOND:
            offset = rng.uniform(0, max(1.0 - VERTICAL_HOLD_DURATION, 0.01))
            start = second + offset
            end = start + VERTICAL_HOLD_DURATION
            if start < duration:
                events.append((start, min(end, duration)))
    return events


def apply_analog_horror_scanlines(frame, t):
    """Full-frame scan line bands. Runs before the content mask blend so
    the effect covers the whole frame including the room."""
    height = frame.shape[0]
    drift_offset = int(t * SCANLINE_DRIFT_SPEED) % (SCANLINE_BAND_HEIGHT * 2)
    for row in range(height):
        if ((row + drift_offset) // SCANLINE_BAND_HEIGHT) % 2 == 0:
            frame[row] = frame[row] * (1.0 - SCANLINE_DARK_STRENGTH)
    return frame


def apply_analog_horror_vertical_hold(frame, t, tear_events):
    """Screen-only vertical hold tear using precomputed event schedule so
    tears and audio dropouts fire at the same times."""
    height, width = frame.shape[0], frame.shape[1]
    tear_active = any(start <= t <= end for start, end in tear_events)
    if tear_active:
        current_second = int(t)
        hold_rng = np.random.RandomState(current_second + 7777)
        hold_rng.random()  # consume the chance roll to keep band/shift deterministic
        hold_rng.uniform(0, max(1.0 - VERTICAL_HOLD_DURATION, 0.01))  # consume offset roll
        band_start = hold_rng.randint(0, max(height - VERTICAL_HOLD_BAND_HEIGHT, 1))
        band_end = min(band_start + VERTICAL_HOLD_BAND_HEIGHT, height)
        shift = int(hold_rng.uniform(-VERTICAL_HOLD_MAX_SHIFT, VERTICAL_HOLD_MAX_SHIFT))
        if shift != 0:
            band = frame[band_start:band_end]
            if shift > 0:
                padded = np.pad(band, ((0, 0), (shift, 0), (0, 0)), mode="edge")
                frame[band_start:band_end] = padded[:, :width]
            else:
                padded = np.pad(band, ((0, 0), (0, -shift), (0, 0)), mode="edge")
                frame[band_start:band_end] = padded[:, -shift:]
    return frame


def make_fume_layers(width, height, seed=1234):
    """Precompute parallax smoke-fume layers: soft low-frequency billows, one
    field per depth layer. Built in the frequency domain (random spectrum with a
    low-pass falloff, inverse FFT) so each field is naturally periodic and drifts
    seamlessly under np.roll with no wrap seam, unlike a sparse point field.
    FUME_SOFTNESS sets billow size, FUME_CONTRAST thins the smoke into wisps.
    Built once in apply_camcorder_look and passed in."""
    layers = []
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    radius[0, 0] = 1.0  # avoid divide-by-zero at the DC term
    falloff = 1.0 / (radius ** FUME_SOFTNESS)  # emphasize low frequencies -> big soft billows
    for i in range(FUME_LAYERS):
        depth = (i + 1) / FUME_LAYERS  # 1/N (far) .. 1.0 (near)
        rng = np.random.RandomState(seed + i * 101)
        spectrum = np.fft.fft2(rng.randn(height, width)) * falloff
        field = np.real(np.fft.ifft2(spectrum))
        field = field - field.min()
        if field.max() > 0:
            field = field / field.max()
        field = field ** FUME_CONTRAST  # thin the wash into denser wisps and clearer gaps
        layers.append((field, depth))
    return layers


def _wrap_shift(field, sy, sx):
    """Shift a periodic field by fractional (sy, sx) pixels with wrap-around,
    using bilinear interpolation built from np.roll. Sub-pixel offsets let the
    smoke glide continuously instead of the old int() truncation, which froze
    the slow layers at zero for seconds and then jumped a whole pixel."""
    iy = int(np.floor(sy)); fy = sy - iy
    ix = int(np.floor(sx)); fx = sx - ix
    a = np.roll(field, iy, axis=0)
    b = np.roll(field, iy + 1, axis=0)
    row_interp = a * (1.0 - fy) + b * fy
    a = np.roll(row_interp, ix, axis=1)
    b = np.roll(row_interp, ix + 1, axis=1)
    return a * (1.0 - fx) + b * fx


def compute_fumes(fume_layers, t, width, height):
    """Additive smoke-brightness field for time t. Each billow layer drifts in
    its OWN direction with a slow sway, so the smoke wanders and crosses rather
    than sliding as one sheet. Drift uses fractional sub-pixel offsets (see
    _wrap_shift) so slow layers glide smoothly. The fields are periodic, so the
    wrap-around drift is seamless."""
    total = np.zeros((height, width), dtype=np.float64)
    for i, (field, depth) in enumerate(fume_layers):
        angle = FUME_LAYER_ANGLES[i % len(FUME_LAYER_ANGLES)]
        sway = FUME_SWAY_AMPLITUDE * np.sin(2 * np.pi * t / FUME_SWAY_PERIOD + i)
        sy = FUME_DRIFT_Y * np.sin(angle) * depth * t
        sx = FUME_DRIFT_X * np.cos(angle) * depth * t + sway
        scrolled = _wrap_shift(field, sy, sx)
        total += scrolled * (FUME_BRIGHTNESS * depth)
    return total[:, :, np.newaxis]


def fan_flicker_multiplier(t):
    """Blade-pass light flicker: mostly bright with quick dark dips, at a fan
    rhythm, nudged per second so it never feels mechanical. Distinct from the
    existing IRREGULAR_FLICKER (random fluorescent jitter); this one is the
    periodic sweep of a fan crossing a light, per the reference frames."""
    phase = 2 * np.pi * FAN_FLICKER_FREQUENCY * t
    wave = (0.5 + 0.5 * np.sin(phase)) ** FAN_FLICKER_SHARPNESS  # 0..1, biased bright
    sec_rng = np.random.RandomState(int(t) + 4200)
    jitter = 1.0 + (sec_rng.uniform(-1, 1) * FAN_FLICKER_IRREGULARITY)
    dip = FAN_FLICKER_DEPTH * jitter * (1.0 - wave)
    return 1.0 - dip


def breathing_multiplier(t):
    """Slow overall brightness swell, the scene inhaling and exhaling."""
    return 1.0 + BREATHING_DEPTH * np.sin(2 * np.pi * t / BREATHING_PERIOD)


# Fluorescent-tube flicker for a room light source (e.g. the backrooms tube).
# A failing fluorescent sits mostly lit with a faint buzz, then stutters and
# drops out at irregular moments. Distinct from the fan blade sweep.
FLUOR_BUZZ_RATE = 9.0        # Hz, faint constant flutter
FLUOR_BUZZ_DEPTH = 0.05      # amplitude of the buzz
FLUOR_JITTER = 0.04          # per-frame random unevenness
FLUOR_DROPOUT_CHANCE = 0.5   # probability per second of a stutter/dropout
FLUOR_DROPOUT_DUR = 0.14     # seconds a dropout lasts
FLUOR_DROPOUT_MIN = 0.20     # darkest the tube falls during a dropout
FLUOR_DROPOUT_MAX = 0.70     # lightest a dropout dip reaches


def fluorescent_flicker(t):
    """Brightness multiplier for a failing fluorescent light at time t: a faint
    steady buzz with occasional irregular stutters and dropouts."""
    r = np.random.RandomState(int(t * 1000) % (2 ** 31))
    m = 1.0 + FLUOR_BUZZ_DEPTH * np.sin(2 * np.pi * FLUOR_BUZZ_RATE * t)
    m += r.uniform(-1, 1) * FLUOR_JITTER
    sec = int(t)
    sr = np.random.RandomState(sec + 51234)
    if sr.random() < FLUOR_DROPOUT_CHANCE:
        offset = sr.uniform(0, max(1.0 - FLUOR_DROPOUT_DUR, 0.01))
        ti = t - sec
        if offset <= ti <= offset + FLUOR_DROPOUT_DUR:
            m *= sr.uniform(FLUOR_DROPOUT_MIN, FLUOR_DROPOUT_MAX)
    return max(m, 0.0)


# Candlelight flicker: a STEADY flame with slow, gentle sway and the occasional
# soft gutter that eases in and out. Deliberately low frequencies so it never
# strobes at 24fps, unlike a fluorescent, and the draft dip uses a raised-cosine
# envelope so the flame dims and recovers smoothly rather than snapping.
CANDLE_DRAFT_CHANCE = 0.25    # probability per second of a gentle gutter
CANDLE_DRAFT_DUR = 0.35       # seconds a gutter lasts (eased in and out)
CANDLE_DRAFT_DEPTH = 0.22     # how far the flame softly ducks in a gutter


def candle_flicker(t):
    """Brightness multiplier for a steady candle flame at time t: slow gentle
    sway most of the time, with occasional soft gutters that ease in and out."""
    m = (1.0
         + 0.025 * np.sin(2 * np.pi * 1.3 * t)
         + 0.018 * np.sin(2 * np.pi * 2.1 * t + 1.1)
         + 0.012 * np.sin(2 * np.pi * 0.7 * t + 2.4))
    r = np.random.RandomState(int(t * 2) % (2 ** 31))  # slow drift, ~2/sec
    m += r.uniform(-1, 1) * 0.015
    sec = int(t)
    sr = np.random.RandomState(sec + 9182)
    if sr.random() < CANDLE_DRAFT_CHANCE:
        offset = sr.uniform(0, max(1.0 - CANDLE_DRAFT_DUR, 0.01))
        ti = t - sec
        if offset <= ti <= offset + CANDLE_DRAFT_DUR:
            phase = (ti - offset) / CANDLE_DRAFT_DUR
            envelope = 0.5 - 0.5 * np.cos(2 * np.pi * phase)  # smooth 0 -> 1 -> 0
            m *= (1.0 - CANDLE_DRAFT_DEPTH * envelope)
    return max(m, 0.0)


def camcorder_transform_pixels(frame, t, vignette_mask, content_mask=None, room_darkness_mask=None, tear_events=None, fog_mask=None, fume_layers=None, fume_gate=None, light_mask=None, light_type="fluorescent"):
    height, width = frame.shape[0], frame.shape[1]
    original_frame = frame.astype(np.float64).copy()
    frame = frame.astype(np.float64)

    if ENABLE_ANALOG_HORROR_OVERLAY:
        frame = apply_analog_horror_scanlines(frame, t)

    # Plain neutral desaturation, the yellow-green Backrooms tint was
    # removed per user feedback, it wasn't adding anything worth keeping.
    gray = frame.mean(axis=2, keepdims=True)
    frame = frame * DESATURATION_AMOUNT + gray * (1 - DESATURATION_AMOUNT)

    frame = frame * vignette_mask

    # Irregular flicker: most of the time a normal smooth flicker, but
    # with some probability per second, a brief double-flicker burst
    # (two quick brightness dips close together) instead, mimicking real
    # arrhythmic fluorescent light rather than a perfectly periodic effect.
    rng = np.random.RandomState(int(t * 1000) % (2**31))
    current_second_for_flicker = int(t)
    flicker_second_rng = np.random.RandomState(current_second_for_flicker + 9000)  # offset seed so this doesn't correlate with the glitch-flash second_rng below
    is_double_flicker_second = flicker_second_rng.random() < IRREGULAR_FLICKER_DOUBLE_CHANCE
    if is_double_flicker_second:
        # Two sharp dips within this second, rather than one smooth wave
        time_in_sec = t - current_second_for_flicker
        dip_1 = abs(((time_in_sec * 6) % 1.0) - 0.5)
        flicker = 1 - (FLICKER_STRENGTH * 2.5 * (1 - dip_1))
    else:
        flicker = 1 + (rng.uniform(-1, 1) * FLICKER_STRENGTH)
    frame = frame * flicker

    noise = rng.normal(0, GRAIN_STRENGTH, frame.shape)
    frame = frame + noise

    # Living-light layer (screen-only when TV overlay is on, since this runs
    # before the content_mask blend that restores the room from the untouched
    # original_frame). Fan flicker and breathing modulate the SCREEN brightness.
    # Fumes are deliberately NOT here: they drift in the dark ROOM, added after
    # the mask blend below, so they land on the background and never the screen.
    if ENABLE_FAN_FLICKER:
        frame = frame * fan_flicker_multiplier(t)
    if BREATHING_ENABLED:
        frame = frame * breathing_multiplier(t)

    current_second = int(t)
    second_rng = np.random.RandomState(current_second)
    flash_roll = second_rng.random()
    flash_offset_in_second = second_rng.uniform(0, max(1 - GLITCH_FLASH_DURATION, 0))
    time_in_second = t - current_second
    is_flash_active = (
        flash_roll < GLITCH_FLASH_CHANCE_PER_SECOND
        and flash_offset_in_second <= time_in_second <= flash_offset_in_second + GLITCH_FLASH_DURATION
    )
    if is_flash_active:
        num_lines = 6
        for _ in range(num_lines):
            row = rng.randint(0, height)
            line_height = rng.randint(2, 7)
            shift = int(rng.uniform(-35, 35))
            row_end = min(row + line_height, height)
            frame[row:row_end] = np.roll(frame[row:row_end], shift, axis=1)
        frame = frame + rng.normal(0, 50, frame.shape)

    frame = np.clip(frame, 0, 255)

    # Chromatic aberration: shift the R and B channels in opposite
    # directions horizontally, more so toward the frame edges, mimicking
    # cheap lens color fringing. A subtle, "filmed not rendered" unease
    # rather than an obvious filter effect.
    #
    # IMPORTANT: do NOT use np.roll here. np.roll wraps pixels around from
    # the opposite edge of the frame, which is exactly what produced a
    # visible stray blue/teal patch in the corner during testing, content
    # from the far edge bleeding into the near edge. Edge-padded shifting
    # (np.pad with mode="edge", then slice back to original size) instead
    # duplicates the nearest real edge pixel into the vacated space, which
    # reads as a natural soft fringe rather than a visible artifact.
    if CHROMATIC_ABERRATION_STRENGTH > 0:
        shift = CHROMATIC_ABERRATION_STRENGTH

        r_channel = frame[:, :, 0]
        r_padded = np.pad(r_channel, ((0, 0), (shift, 0)), mode="edge")
        frame[:, :, 0] = r_padded[:, :width]  # shifted right, edge-padded on the left

        b_channel = frame[:, :, 2]
        b_padded = np.pad(b_channel, ((0, 0), (0, shift)), mode="edge")
        frame[:, :, 2] = b_padded[:, shift:]  # shifted left, edge-padded on the right

        frame = np.clip(frame, 0, 255)

    if content_mask is not None:
        mask_norm = content_mask.astype(np.float64)
        if mask_norm.max() > 1.0:
            mask_norm = mask_norm / 255.0
        mask_norm = mask_norm[:, :, np.newaxis]
        frame = frame * mask_norm + original_frame * (1 - mask_norm)

        if ENABLE_TV_OVERLAY and AMBIENT_GLOW_ENABLED:
            screen_pixels = original_frame * mask_norm
            screen_pixel_count = mask_norm.sum() * 3
            if screen_pixel_count > 0:
                screen_avg_brightness = screen_pixels.sum() / screen_pixel_count
            else:
                screen_avg_brightness = 128.0

            brightness_norm = (screen_avg_brightness - 128.0) / 128.0
            glow_multiplier = 1.0 + (brightness_norm * AMBIENT_GLOW_STRENGTH)

            room_region = frame * (1 - mask_norm)
            # The ambient glow reacts to screen brightness, so it shifts the room
            # at every scene change. Keep a room light source (candle/tube) OUT of
            # that reaction, so its brightness does not jump when the still cuts.
            glow_field = glow_multiplier
            if light_mask is not None:
                glow_field = glow_multiplier * (1.0 - light_mask) + 1.0 * light_mask
            room_region_glowed = room_region * glow_field
            frame = frame * mask_norm + room_region_glowed

        # Distance-based room darkening: applied AFTER ambient glow, so
        # the room still reacts to screen brightness (glow above) but is
        # also pulled toward darkness the further it sits from the
        # screen, rather than the whole room brightening/dimming
        # uniformly. room_darkness_mask is 1.0 inside the screen
        # (untouched here) and falls toward ROOM_FAR_DARKNESS outside it.
        if room_darkness_mask is not None:
            frame = frame * room_darkness_mask

    # Flicker a room light source (backrooms fluorescent tube, or the living
    # room candle). Applied after the room darkening, which already exempted
    # this region so the light stays lit; here it buzzes or sways by its type.
    if light_mask is not None:
        flick = candle_flicker(t) if light_type == "candle" else fluorescent_flicker(t)
        frame = frame * (1.0 - light_mask * (1.0 - flick))

    if ENABLE_ANALOG_HORROR_OVERLAY and tear_events is not None:
        frame = apply_analog_horror_vertical_hold(frame, t, tear_events)

    if ENABLE_ROOM_FOG and fog_mask is not None:
        # Blur the full frame, then composite sharp original back over screen.
        # fog_mask is precomputed in apply_camcorder_look, not rebuilt per-frame.
        blurred = gaussian_filter(frame, sigma=[ROOM_FOG_BLUR_RADIUS, ROOM_FOG_BLUR_RADIUS, 0])
        frame = frame * fog_mask + blurred * (1 - fog_mask)

    # Drifting smoke fumes in the dark ROOM around the tube, NOT on the screen.
    # Added after the content_mask blend and the room fog so the room does not
    # wipe them. fume_gate (precomputed in apply_camcorder_look) is a FEATHERED
    # room mask: hard zero on the screen so no screen pixel is ever touched, and
    # a soft fade-out band as the smoke nears the screen so it dissolves into
    # the tube rather than stopping at a hard edge. When there is no screen mask
    # (TV overlay off), fumes cover the full frame.
    if ENABLE_FUMES and fume_layers is not None:
        fume = compute_fumes(fume_layers, t, width, height)
        if fume_gate is not None:
            fume = fume * fume_gate
        # Keep smoke off the light itself. (1 - light_mask) is 0 at the flame
        # core and rises to 1 out in the room, so the candle stays honestly
        # clear while its outer glow and the room beyond can still catch haze.
        if light_mask is not None:
            fume = fume * (1.0 - light_mask)
        frame = frame + fume
        frame = np.clip(frame, 0, 255)

    bar_height = int(height * LETTERBOX_HEIGHT_RATIO)
    if bar_height > 0:
        frame[:bar_height] = 0
        frame[height - bar_height:] = 0

    # Final clamp before the 8-bit cast. A light region with light_level > 1.0
    # (a brightened candle) can push pixels past 255; without this they would
    # wrap to dark garbage on the uint8 conversion.
    frame = np.clip(frame, 0, 255)
    return frame


def make_vignette_mask(width, height):
    y_coords, x_coords = np.ogrid[0:height, 0:width]
    center_x, center_y = width / 2, height / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    normalized_dist = dist_from_center / max_dist
    vignette_mask = 1 - (normalized_dist**2 * VIGNETTE_STRENGTH)
    vignette_mask = np.clip(vignette_mask, 0, 1)
    return vignette_mask[:, :, np.newaxis]


def make_room_darkness_mask(content_mask, width, height, light_mask=None, light_level=1.0):
    """Build a per-pixel darkness multiplier for the room/cabinet area
    (everything outside the TV screen), based on actual distance from the
    screen edge, not a flat uniform value. Per user request: 'the TV
    should barely light up an otherwise dark room, only really lighting
    up the direct vicinity significantly', so pixels right next to the
    screen stay relatively bright (close to ROOM_NEAR_SCREEN_BRIGHTNESS),
    while pixels far from the screen fall toward ROOM_FAR_DARKNESS,
    using a real distance transform rather than the existing radial
    vignette (which is centered on the FRAME, not the screen, and
    darkens uniformly by distance from frame-center regardless of where
    the actual light source, the screen, sits).

    If light_mask is given (a room light source such as the backrooms
    fluorescent tube), that region is kept lit (multiplier toward 1.0) so
    the light is not crushed by the distance falloff; its own flicker is
    applied later in the transform.

    Returns a (height, width, 1) array of multipliers, 1.0 inside the
    screen (untouched, since this only affects the room), falling off
    smoothly outside it."""
    if content_mask is None:
        return np.ones((height, width, 1))

    mask_norm = content_mask.astype(np.float64)
    if mask_norm.max() > 1.0:
        mask_norm = mask_norm / 255.0

    # distance_transform_edt measures, for each pixel OUTSIDE the screen
    # (where mask_norm is 0), the real Euclidean distance to the nearest
    # screen pixel. This is what makes the falloff follow the screen's
    # actual traced shape, rather than a generic frame-centered radius.
    is_outside_screen = mask_norm < 0.5
    distance_from_screen = distance_transform_edt(is_outside_screen)

    max_relevant_distance = max(width, height) * 0.5  # distances beyond this are already at full darkness
    normalized_distance = np.clip(distance_from_screen / max_relevant_distance, 0, 1)

    darkness_multiplier = ROOM_NEAR_SCREEN_BRIGHTNESS - (normalized_distance * (ROOM_NEAR_SCREEN_BRIGHTNESS - ROOM_FAR_DARKNESS))
    # Inside the screen itself, force the multiplier to 1.0 (untouched),
    # this mask only ever affects the room/cabinet region.
    darkness_multiplier = np.where(mask_norm >= 0.5, 1.0, darkness_multiplier)

    # Keep a room light source lit against the falloff, but only up to
    # light_level, so a dimmed light reads as struggling rather than fully lit.
    if light_mask is not None:
        lm = light_mask[:, :, 0] if light_mask.ndim == 3 else light_mask
        darkness_multiplier = darkness_multiplier * (1.0 - lm) + light_level * lm

    return darkness_multiplier[:, :, np.newaxis]


def apply_camcorder_look(clip, tear_events=None):
    width, height = clip.size
    vignette_mask = np.ones((height, width, 1))  # vignette disabled, room darkness mask handles darkening
    fume_layers = make_fume_layers(width, height) if ENABLE_FUMES else None
    fume_gate = None

    # Soft mask for a flickering room light source (backrooms tube, candle).
    # Built as a radial glow that dissipates smoothly from the center of the
    # light_region outward, rather than a filled box, so no hard lit shape shows.
    light_mask = None
    light_type = "fluorescent"
    light_level = 1.0
    if ACTIVE_FRAME != "curved_tv":
        lr = frame_overlay.light_region(ACTIVE_FRAME)
        if lr is not None:
            light_type = frame_overlay.light_type(ACTIVE_FRAME)
            light_level = frame_overlay.light_level(ACTIVE_FRAME)
            x0, y0, x1, y1 = lr
            cx = (x0 + x1) / 2 * width
            cy = (y0 + y1) / 2 * height
            rx = max((x1 - x0) / 2 * width, 1.0)
            ry = max((y1 - y0) / 2 * height, 1.0)
            yy, xx = np.ogrid[0:height, 0:width]
            d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            lm = np.exp(-LIGHT_FALLOFF * d2)  # 1.0 at center, smooth dissipation outward
            lm = lm / lm.max()
            light_mask = lm[:, :, np.newaxis]

    content_mask = None
    room_darkness_mask = None
    fog_mask = None
    if ENABLE_TV_OVERLAY:
        screen_mask = active_screen_mask(width, height)
        room_darkness_mask = make_room_darkness_mask(screen_mask, width, height, light_mask, light_level)
        if TV_OVERLAY_CAMCORDER_SCREEN_ONLY:
            content_mask = screen_mask
        if ENABLE_FUMES:
            # Feathered room gate for the fumes: hard zero on the screen so no
            # screen pixel is ever touched, blurred outward into a soft band so
            # the smoke fades out as it approaches the tube instead of hitting a
            # hard mask edge. Precomputed once here, not rebuilt per frame.
            fm = screen_mask.astype(np.float64)
            if fm.max() > 1.0:
                fm = fm / 255.0
            feather = np.clip(gaussian_filter(fm, sigma=FUME_FEATHER_SIGMA), 0.0, 1.0)
            gate = np.clip(1.0 - feather, 0.0, 1.0)
            gate = np.where(fm >= 0.5, 0.0, gate)  # absolute zero on the screen itself
            fume_gate = gate[:, :, np.newaxis]
            # Keep fumes off a room light's core too, fading them back in as the
            # glow falls off, so the candle stays clear of haze while the room
            # around it can still fog.
            if light_mask is not None:
                fume_gate = fume_gate * (1.0 - light_mask)
        if ENABLE_ROOM_FOG:
            # Vertical gradient: 1.0 (sharp) near the TV screen rows,
            # falling to 0.0 (full blur) at the top and bottom edges.
            # Find the vertical extent of the screen from the screen mask.
            m = screen_mask.astype(np.float64)
            if m.max() > 1.0:
                m = m / 255.0
            row_coverage = m.max(axis=1)  # 1.0 for rows that contain screen pixels
            screen_rows = np.where(row_coverage > 0.5)[0]
            if len(screen_rows) > 0:
                screen_top = screen_rows[0]
                screen_bot = screen_rows[-1]
                screen_mid = (screen_top + screen_bot) / 2.0
                screen_half = max((screen_bot - screen_top) / 2.0, 1.0)
            else:
                screen_mid = height / 2.0
                screen_half = height / 4.0
            # Distance from screen center in vertical pixels, normalized.
            row_idx = np.arange(height, dtype=np.float64)
            vert_dist = np.abs(row_idx - screen_mid) / (screen_mid + 1e-6)
            # Clamp to 0-1 and invert so near = 1.0, far = 0.0
            fog_row = np.clip(1.0 - vert_dist, 0.0, 1.0)
            fog_mask = fog_row[:, np.newaxis, np.newaxis] * np.ones((height, width, 1))
            # Keep a room light source sharp against the fog, so the candle
            # stays fully clear; only the drifting fumes pass over it.
            if light_mask is not None:
                fog_mask = np.maximum(fog_mask, light_mask)

    def transform_frame(get_frame, t):
        frame = get_frame(t)
        frame = camcorder_transform_pixels(frame, t, vignette_mask, content_mask, room_darkness_mask, tear_events, fog_mask, fume_layers, fume_gate, light_mask, light_type)
        return frame.astype("uint8")

    return clip.transform(transform_frame)


def _mix_wow_flutter(samples, fps):
    """Continuous tape wow (slow) and flutter (fast) applied as a bounded
    time warp over the full mix. The read positions are normalized to span
    the clip exactly, so net timing lands at zero and captions stay synced
    to the audio. This runs UNDER the discrete tape slowdown events already
    fired in apply_analog_horror_audio, adding a constant subtle warble."""
    n = len(samples)
    t = np.arange(n) / fps
    rate = 1.0 + MIX_WOW_DEPTH * np.sin(2 * np.pi * MIX_WOW_RATE * t)
    rate += MIX_FLUTTER_DEPTH * np.sin(2 * np.pi * MIX_FLUTTER_RATE * t)
    pos = np.cumsum(rate)
    pos = (pos - pos[0]) / (pos[-1] - pos[0] + 1e-9) * (n - 1)
    src = np.arange(n)
    if samples.ndim > 1:
        return np.stack([np.interp(pos, src, samples[:, c]) for c in range(samples.shape[1])], axis=1)
    return np.interp(pos, src, samples)


def _mix_broadcast_eq(samples, fps):
    """Band-limit the full mix so it reads like a degraded broadcast, thin
    on both ends. Blended with the original so it never fully hollows out.
    Runs BEFORE the carrier hum is added so the low corner does not strip
    the 60Hz hum tone."""
    ny = fps / 2.0
    high = min(MIX_EQ_HIGH, ny - 1)
    if MIX_EQ_LOW <= 0 or MIX_EQ_LOW >= high:
        return samples
    b, a = butter(2, [MIX_EQ_LOW / ny, high / ny], btype="band")
    if samples.ndim > 1:
        filt = np.stack([lfilter(b, a, samples[:, c]) for c in range(samples.shape[1])], axis=1)
    else:
        filt = lfilter(b, a, samples)
    return samples * (1 - MIX_EQ_BLEND) + filt * MIX_EQ_BLEND


def _mix_bitcrush(samples):
    """Quantize the full mix to fewer amplitude levels for digital grit.
    Blended with the original so it stays a texture rather than a wall."""
    levels = 2 ** MIX_BITCRUSH_BITS
    crushed = np.round(samples * levels) / levels
    return samples * (1 - MIX_BITCRUSH_BLEND) + crushed * MIX_BITCRUSH_BLEND


def apply_analog_horror_audio(audio_clip, beat_duration=4.0, tear_events=None):
    """Apply analog horror audio distortion to the full mixed audio clip.
    Effects: continuous wow/flutter, degraded-broadcast bandpass, bitcrush,
    a constant carrier hum, randomized tape slowdowns, and dropouts
    synchronized to tear_events so audio and video glitch together."""
    from moviepy.audio.AudioClip import AudioClip as _AudioClip

    fps = 44100
    samples = audio_clip.to_soundarray(fps=fps).copy()
    is_stereo = samples.ndim > 1
    total_samples = len(samples)
    duration = audio_clip.duration

    # Full-mix degradation, applied BEFORE the carrier hum so the bandpass
    # does not strip the 60Hz tone added below. Each stage is individually
    # gated and blended so it can be dialed in or out by ear.
    if ENABLE_MIX_WOW_FLUTTER:
        samples = _mix_wow_flutter(samples, fps)
    if ENABLE_MIX_BROADCAST_EQ:
        samples = _mix_broadcast_eq(samples, fps)
    if ENABLE_MIX_BITCRUSH:
        samples = _mix_bitcrush(samples)

    # Carrier hum: a 60Hz sine wave added to every sample.
    t_arr = np.arange(total_samples) / fps
    hum = np.sin(2 * np.pi * HUM_FREQUENCY * t_arr) * HUM_AMPLITUDE
    if is_stereo:
        hum = hum[:, np.newaxis]
    samples = samples + hum

    # Determine event windows based on beat_duration.
    # Each window rolls independently for tape slowdown and dropout.
    num_windows = max(int(duration / beat_duration), 1)
    window_samples = total_samples // num_windows

    for w in range(num_windows):
        w_rng = np.random.RandomState(w * 3 + 1001)
        w_start = w * window_samples
        w_end = min(w_start + window_samples, total_samples)

        # Tape slowdown: pitch-shift a slice downward by resampling at a
        # fractional rate, then pad back to original length with silence.
        # Pitch shift factor is rolled randomly between min and max each
        # event so depth varies. Lower factor = deeper voice drop.
        if w_rng.random() < TAPE_SLOWDOWN_CHANCE_PER_BEAT:
            slow_samples = int(TAPE_SLOWDOWN_DURATION * fps)
            slow_offset = w_rng.randint(0, max(w_end - w_start - slow_samples, 1))
            s_start = w_start + slow_offset
            s_end = min(s_start + slow_samples, total_samples)
            segment = samples[s_start:s_end]
            seg_len = len(segment)
            pitch_factor = w_rng.uniform(TAPE_PITCH_SHIFT_MIN, TAPE_PITCH_SHIFT_MAX)
            # Downsample to lower pitch: take every 1/pitch_factor-th sample,
            # producing a shorter array at lower pitch, then pad with silence
            # back to the original segment length so timing is preserved.
            indices = np.arange(0, seg_len, 1.0 / pitch_factor).astype(int)
            indices = indices[indices < seg_len]
            pitched = segment[indices]
            if len(pitched) < seg_len:
                pad_len = seg_len - len(pitched)
                pad_shape = (pad_len, segment.shape[1]) if pitched.ndim > 1 else (pad_len,)
                pitched = np.concatenate([pitched, np.zeros(pad_shape)])
            samples[s_start:s_start + seg_len] = pitched[:seg_len]

        # No independent dropout rolls. Dropouts fire only on tear events below.
        pass

    # Dropouts fire on a subset of visual tear events so audio glitches are
    # sparser than visual ones. Every audio glitch has a matching visual.
    if tear_events:
        event_rng = np.random.RandomState(8888)
        for t_start, t_end in tear_events:
            if event_rng.random() < AUDIO_DROPOUT_ON_TEAR_CHANCE:
                s = int(t_start * fps)
                e = min(int(t_end * fps) + int(AUDIO_DROPOUT_DURATION * fps), total_samples)
                if s < total_samples:
                    samples[s:e] = samples[s:e] * AUDIO_DROPOUT_VOLUME

    samples = np.clip(samples, -1.0, 1.0)

    def make_frame(t):
        if np.isscalar(t):
            idx = min(int(t * fps), total_samples - 1)
            return samples[idx]
        idx = np.clip((np.array(t) * fps).astype(int), 0, total_samples - 1)
        return samples[idx]

    return _AudioClip(make_frame, duration=duration, fps=fps)


def find_peak_moment(audio_clip, window_size=0.15):
    sample_rate = 22050
    samples = audio_clip.to_soundarray(fps=sample_rate)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    window_samples = max(int(window_size * sample_rate), 1)
    if len(samples) <= window_samples:
        return audio_clip.duration / 2
    hop = max(window_samples // 4, 1)
    best_time = 0.0
    best_rms = -1
    for start_sample in range(0, len(samples) - window_samples, hop):
        window = samples[start_sample:start_sample + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > best_rms:
            best_rms = rms
            best_time = start_sample / sample_rate
    return best_time


def build_outro_extension(final_video):
    if not os.path.exists(TAPE_OUTRO_PATH):
        print(f"  (outro file not found at {TAPE_OUTRO_PATH}, skipping outro)")
        return None

    outro_audio = AudioFileClip(TAPE_OUTRO_PATH)
    outro_audio = outro_audio.with_effects([MultiplyVolume(TAPE_OUTRO_VOLUME)])

    last_frame = final_video.get_frame(final_video.duration - 0.04)
    fps = final_video.fps if hasattr(final_video, "fps") else 24
    duration = outro_audio.duration
    height, width = last_frame.shape[0], last_frame.shape[1]

    peak_time = find_peak_moment(outro_audio)
    print(f"  outro audio peak detected at {peak_time:.2f}s of {duration:.2f}s")
    glitch_end_time = peak_time

    y_coords, x_coords = np.ogrid[0:height, 0:width]
    center_x, center_y = width / 2, height / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    vignette_mask = make_vignette_mask(width, height)

    def make_frame(t):
        frame = last_frame.copy()
        frame = frame.astype(np.float64)

        rng = np.random.RandomState(int(t * 1000) % (2**31))
        glitch_progress = min(t / glitch_end_time, 1.0) if glitch_end_time > 0 else 1.0

        num_tracking_lines = int(glitch_progress * 8)
        for _ in range(num_tracking_lines):
            row = rng.randint(0, height)
            line_height = rng.randint(2, 6)
            shift = int(rng.uniform(-20, 20) * glitch_progress)
            row_end = min(row + line_height, height)
            frame[row:row_end] = np.roll(frame[row:row_end], shift, axis=1)

        static_strength = 40 * glitch_progress
        noise = rng.normal(0, static_strength, frame.shape)
        frame = frame + noise

        if glitch_progress > 0.7 and rng.random() < 0.15:
            flash_mask = rng.random(frame.shape[:2]) < 0.08
            frame[flash_mask] = 255

        frame = np.clip(frame, 0, 255)

        if t >= peak_time:
            frame = frame * 0
        else:
            iris_start_time = peak_time * 0.5
            if t > iris_start_time:
                iris_progress = (t - iris_start_time) / (peak_time - iris_start_time) if peak_time > iris_start_time else 1.0
                iris_progress = min(iris_progress, 1.0)
                current_radius = max_radius * (1 - iris_progress)
                iris_mask = (dist_from_center <= current_radius).astype(np.float64)
                iris_mask = iris_mask[:, :, np.newaxis]
                frame = frame * iris_mask

        return frame.astype("uint8")

    from moviepy import VideoClip
    outro_visual = VideoClip(make_frame, duration=duration)
    outro_visual = outro_visual.with_fps(fps)
    outro_visual = outro_visual.with_audio(outro_audio)
    return outro_visual


def load_narration_lookup():
    script_path = os.path.join(OUTPUT_DIR, "script.json")
    lookup = {}
    if not os.path.exists(script_path):
        return lookup
    with open(script_path) as f:
        script = json.load(f)
    for scene_idx, scene in enumerate(script.get("scenes", [])):
        for beat_idx, beat in enumerate(scene.get("beats", [])):
            lookup[(scene_idx, beat_idx)] = beat.get("narration", "")
    return lookup


def load_title():
    """Load the short title card text from script.json, written by
    pipeline.py's title generation. Returns None if the field is missing
    (e.g. an older script.json from before this feature existed), so
    the title card step can be safely skipped rather than crash."""
    script_path = os.path.join(OUTPUT_DIR, "script.json")
    if not os.path.exists(script_path):
        return None
    with open(script_path) as f:
        script = json.load(f)
    return script.get("title")


def build_title_card(title_text, duration=TITLE_CARD_DURATION):
    """Render the title-card BACKGROUND only: the TV background with static in
    the screen, shown for about 1 second before the first beat. The title TEXT
    is NOT drawn here anymore. It is composited separately on top of the finished
    video (see build_title_text_overlay), AFTER the camcorder look, fog, and
    fumes, so the text stays bright and sharp instead of being dimmed by them."""
    # Static noise composited into the TV screen only. The room/cabinet
    # stays dark around it, same as every other beat.
    rng = np.random.RandomState(42)
    noise = rng.randint(80, 200, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)
    if ENABLE_TV_OVERLAY:
        composite_arr = active_screen_composite(
            Image.fromarray(noise), FRAME_WIDTH, FRAME_HEIGHT
        )
        canvas = Image.fromarray(composite_arr).convert("RGB")
    else:
        canvas = Image.fromarray(noise).convert("RGB")

    title_clip = ImageClip(np.array(canvas)).with_duration(duration)
    print(f"  Title card: \"{title_text}\" ({duration}s)")
    return title_clip


def build_title_text_overlay(title_text, duration=TITLE_CARD_DURATION):
    """Render the title TEXT as a separate transparent overlay, composited on
    top of the finished video AFTER the camcorder look, fog, and fumes, so it
    stays bright and sharp (same approach as captions). Large, top-center,
    white with a heavy stroke. Wraps to at most two lines and only shrinks the
    font if a line still will not fit, so the text stays large."""
    from PIL import ImageFont, ImageDraw

    canvas = Image.new("RGBA", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    max_text_width = FRAME_WIDTH * TITLE_CARD_MAX_WIDTH_RATIO
    words = title_text.split()

    def wrap(font):
        lines = []
        current = ""
        for w in words:
            trial = (current + " " + w).strip()
            if not current or draw.textlength(trial, font=font) <= max_text_width:
                current = trial
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    font_size = TITLE_CARD_FONTSIZE
    font = ImageFont.truetype(CAPTION_FONT, font_size)
    lines = wrap(font)
    # Shrink only until the title fits in at most two lines with each line
    # inside the width, so a long title stays as large as it can rather than
    # collapsing to one tiny line.
    while font_size > 32:
        lines = wrap(font)
        widest = max(draw.textlength(line, font=font) for line in lines)
        if len(lines) <= 2 and widest <= max_text_width:
            break
        font_size -= 4
        font = ImageFont.truetype(CAPTION_FONT, font_size)

    line_height = int(font_size * 1.15)
    y_position = TITLE_CARD_TOP_MARGIN
    for line in lines:
        line_width = draw.textlength(line, font=font)
        x_position = (FRAME_WIDTH - line_width) / 2
        draw.text(
            (x_position, y_position),
            line,
            font=font,
            fill=TITLE_CARD_COLOR,
            stroke_fill=TITLE_CARD_STROKE_COLOR,
            stroke_width=TITLE_CARD_STROKE_WIDTH,
            anchor="la",  # left, ascender-aligned, so y_position is the true top
        )
        y_position += line_height

    overlay = ImageClip(np.array(canvas)).with_start(0).with_duration(duration)
    return overlay.with_position((0, 0))


def build_static_transition(duration=STATIC_TRANSITION_DURATION):
    """Build a brief, heavy static/noise burst clip, inserted between every beat
    cut. The static is composited into the SCREEN region of the active frame
    only, so the room/cabinet and any room light (the candle, the tube) stay lit
    and steady through the cut instead of the whole frame blinking to noise. That
    is why the candle no longer flickers at scene switches. When no screen
    overlay is active, static fills the full frame as before."""

    def make_frame(t):
        rng = np.random.RandomState(int(t * 10000) % (2**31))
        noise = rng.randint(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.float64)
        num_tear_lines = rng.randint(3, 8)
        for _ in range(num_tear_lines):
            row = rng.randint(0, FRAME_HEIGHT)
            line_height = rng.randint(2, 10)
            row_end = min(row + line_height, FRAME_HEIGHT)
            noise[row:row_end] = rng.randint(200, 256, (row_end - row, FRAME_WIDTH, 3))
        noise = np.clip(noise, 0, 255).astype("uint8")
        if ENABLE_TV_OVERLAY:
            # Static only inside the screen; the room comes from the frame photo
            # and gets its steady lighting from the camcorder pass downstream.
            return active_screen_composite(Image.fromarray(noise), FRAME_WIDTH, FRAME_HEIGHT)
        return noise

    from moviepy import VideoClip
    static_clip = VideoClip(make_frame, duration=duration)
    static_clip = static_clip.with_fps(24)
    return static_clip


def stitch_pipeline():
    beats = find_beats()
    if not beats:
        print("No beat files found in pipeline_output. Run pipeline.py first.")
        return

    print(f"Found {len(beats)} beats: {beats}")

    narration_lookup = load_narration_lookup()

    still_beat_keys = [key for key, render_type in beats if render_type == "still"]
    num_punch_beats = min(random.choice([1, 2]), len(still_beat_keys))
    punch_beat_keys = set(random.sample(still_beat_keys, num_punch_beats)) if still_beat_keys else set()
    print(f"Punch effect will apply to: {punch_beat_keys}\n")

    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    clips = []
    all_words_with_offsets = []  # list of (word_text, absolute_start, absolute_end)
    running_offset = 0.0

    title_text = load_title()
    if title_text:
        title_clip = build_title_card(title_text)
        clips.append(title_clip)
        running_offset += title_clip.duration
    else:
        print("  (no title found in script.json, skipping title card)")

    for (scene_idx, beat_idx), render_type in beats:
        apply_punch = (scene_idx, beat_idx) in punch_beat_keys
        is_hook_beat = (scene_idx, beat_idx) == beats[0][0]
        audio_delay = HOOK_AUDIO_DELAY if is_hook_beat else 0
        narration_text = narration_lookup.get((scene_idx, beat_idx), "")
        visual_clip, beat_words = build_beat_clip(scene_idx, beat_idx, render_type, whisper_model, narration_text, apply_punch, audio_delay)

        # Static transition before every beat that isn't the first thing
        # in the video (title card counts as "something before it" too,
        # so the title-to-first-beat cut also gets a static burst).
        if clips:
            static_clip = build_static_transition()
            clips.append(static_clip)
            running_offset += static_clip.duration

        clips.append(visual_clip)

        for word_text, start, end in beat_words:
            all_words_with_offsets.append((word_text, start + running_offset, end + running_offset))
        running_offset += visual_clip.duration

    print("\nConcatenating all beats into final video...")
    final_video = concatenate_videoclips(clips, method="compose")

    print("\nAdding tape-rolling outro...")
    outro_clip = build_outro_extension(final_video)
    if outro_clip:
        final_video = concatenate_videoclips([final_video, outro_clip], method="compose")

    tear_events = build_tear_event_schedule(final_video.duration) if ENABLE_ANALOG_HORROR_OVERLAY else []

    if ENABLE_CAMCORDER_LOOK:
        print("\nApplying old-timey camcorder look (grain, vignette, flicker, desaturation)...")
        final_video = apply_camcorder_look(final_video, tear_events=tear_events)

    # Captions composite onto final_video HERE, after the camcorder look
    # has already been baked in, so caption text is never touched by the
    # grain/vignette/flicker/glitch effects. This is the direct fix for
    # captions visibly flickering/glitching for a split second, which
    # happened because they used to be composited BEFORE the camcorder
    # pass ran over the whole frame, captions included.
    print("\nCompositing captions on top of the finished video (after camcorder effects)...")
    caption_clips = build_caption_clips(all_words_with_offsets, (FRAME_WIDTH, FRAME_HEIGHT))
    overlay_clips = list(caption_clips)
    # Title text rides on top of the finished frame too, at full brightness,
    # over the title-card background during its opening second.
    if title_text:
        overlay_clips.append(build_title_text_overlay(title_text))
    final_video = CompositeVideoClip([final_video] + overlay_clips, size=(FRAME_WIDTH, FRAME_HEIGHT))
    final_video = final_video.with_duration(running_offset + (outro_clip.duration if outro_clip else 0))

    script_path = os.path.join(OUTPUT_DIR, "script.json")
    mood = None
    if os.path.exists(script_path):
        with open(script_path) as f:
            mood = json.load(f).get("mood")

    track_path = pick_music_track(mood) if mood else pick_music_track(None)

    if track_path:
        print(f"\nAdding background music: {track_path} (mood: {mood})")
        narration_rms = measure_rms_loudness(final_video.audio)
        music = build_background_music(track_path, final_video.duration, narration_rms)
        mixed_audio = CompositeAudioClip([final_video.audio, music])
        final_video = final_video.with_audio(mixed_audio)
    else:
        print(f"\nNo music track found for mood '{mood}' and no fallback tracks exist in {MUSIC_DIR}/. Skipping music.")

    if ENABLE_ANALOG_HORROR_AUDIO and final_video.audio is not None:
        print("\nApplying analog horror audio distortion (wow/flutter, EQ, bitcrush, hum, tape slowdown, dropout)...")
        avg_beat_duration = final_video.duration / max(len(clips), 1)
        distorted_audio = apply_analog_horror_audio(final_video.audio, beat_duration=avg_beat_duration, tear_events=tear_events)
        final_video = final_video.with_audio(distorted_audio)

    print(f"Writing final video to {FINAL_OUTPUT} ...")
    final_video.write_videofile(
        FINAL_OUTPUT,
        fps=24,
        codec="libx264",
        audio_codec="aac",
    )

    for clip in clips:
        clip.close()
    final_video.close()

    print(f"\nDone. Final video saved at {FINAL_OUTPUT}")


if __name__ == "__main__":
    stitch_pipeline()