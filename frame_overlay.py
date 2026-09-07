"""Flat-frame screen overlay for the non-curved backdrops (backrooms, living
room, and any future flat frame). This is separate from tv_overlay.py, which is
specific to the curved CRT and its traced screen. Each frame here is a still
photograph with a flat rectangular screen region traced as four normalized
corners. The content still is cover-fit to the screen's aspect, then perspective
-warped into the traced quad. Exposes the same two calls stitch.py already uses,
composite_into_screen and get_screen_content_mask, with a frame name in front.

Corners are stored normalized (fractions of width/height) in TL, TR, BR, BL
order, so they scale to any output size regardless of the source image's own
resolution. Corners were measured from the frame pixels, not eyeballed.
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# Folder holding the frame photographs, relative to where stitch.py runs.
FRAMES_DIR = "frames"

FRAMES = {
    "backrooms": {
        "image": "tv_creepy_backrooms.png",
        # dark board on the yellow wall, detected at 99% fill
        "corners": [(0.162, 0.406), (0.837, 0.406), (0.837, 0.739), (0.162, 0.739)],
        # Bright yellow environment, pushed deep into darkness: the wall around
        # the board is dimmed hard, the corners fall to near-black, thick fog.
        # The screen should dominate; the room is nearly an abyss around it.
        "room": {"far_darkness": 0.03, "near_brightness": 0.14, "fog": True, "fog_radius": 22, "fumes": True},
        # The fluorescent tube at the top of the photo. stitch flickers this
        # region like a failing light. light_level holds it BELOW full brightness
        # so it reads as a dying light struggling against the dark, not a lit room.
        # Normalized bbox [x0, y0, x1, y1], expanded around the tube for glow.
        "light_region": [0.30, 0.050, 0.70, 0.160],
        "light_type": "fluorescent",
        "light_level": 0.36,
    },
    "living_room": {
        "image": "tv_living_room.png",
        # black opening inside the bronze frame, detected at 99% fill
        "corners": [(0.243, 0.146), (0.780, 0.146), (0.780, 0.391), (0.243, 0.391)],
        # Already dark panelling, crushed near the backrooms level, but with a
        # LIGHTER fog than backrooms, since the table and candle sit low in frame
        # and heavy fog smears them into the dark.
        "room": {"far_darkness": 0.02, "near_brightness": 0.15, "fog": True, "fog_radius": 9, "fumes": True},
        # The candle on the console table. The lit region is a warm POOL over the
        # whole table cluster, not just the flame, so the candle actually lights
        # the table it sits on and it stays visible against the crushed room.
        "light_region": [0.20, 0.480, 0.56, 0.720],
        "light_type": "candle",
        "light_level": 0.90,
    },
}


def _find_coeffs(dst_quad, src_rect):
    """Solve the 8 perspective coefficients that map OUTPUT (canvas) coordinates
    in dst_quad back to INPUT (content) coordinates in src_rect, for
    PIL Image.transform(..., Image.PERSPECTIVE, coeffs)."""
    matrix = []
    for (dx, dy), (sx, sy) in zip(dst_quad, src_rect):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src_rect, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


def _corners_px(frame_name, W, H):
    corners = FRAMES[frame_name]["corners"]
    return [(fx * W, fy * H) for fx, fy in corners]


def _load_bg(frame_name, W, H):
    path = os.path.join(FRAMES_DIR, FRAMES[frame_name]["image"])
    return Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)


def composite_into_screen(frame_name, content_img, W, H):
    """Warp the content still into the traced screen quad of the named frame and
    composite it over the frame photo. Returns an (H, W, 3) uint8 array, the same
    shape tv_overlay.composite_into_screen returns."""
    bg = _load_bg(frame_name, W, H)
    corners = _corners_px(frame_name, W, H)  # TL, TR, BR, BL in output space

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    sw = max(xs) - min(xs)
    sh = max(ys) - min(ys)
    if sw < 1 or sh < 1:
        return np.array(bg)

    # Cover-fit the still to the screen rectangle, then center-crop, so the still
    # fills the screen without distortion before the perspective warp.
    content = content_img.convert("RGB")
    cw, ch = content.size
    scale = max(sw / cw, sh / ch)
    rw, rh = max(int(cw * scale), 1), max(int(ch * scale), 1)
    content_r = content.resize((rw, rh), Image.LANCZOS)
    left = int((rw - sw) / 2)
    top = int((rh - sh) / 2)
    content_c = content_r.crop((left, top, left + int(sw), top + int(sh)))

    src_rect = [(0, 0), (content_c.width, 0), (content_c.width, content_c.height), (0, content_c.height)]
    coeffs = _find_coeffs(corners, src_rect)
    warped = content_c.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(corners, fill=255)

    out = bg.copy()
    out.paste(warped, (0, 0), mask)
    return np.array(out)


def get_screen_content_mask(frame_name, W, H):
    """Return a 2D (H, W) uint8 mask, 255 inside the traced screen quad and 0
    outside, matching tv_overlay.get_screen_content_mask."""
    corners = _corners_px(frame_name, W, H)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(corners, fill=255)
    return np.array(mask)


def room_settings(frame_name):
    """Per-frame room-treatment overrides, read by stitch.py so a bright frame
    (backrooms) is not crushed dark like the curved TV's viewer room."""
    return FRAMES.get(frame_name, {}).get("room", {})


def light_region(frame_name):
    """Normalized [x0, y0, x1, y1] bbox of a room light source that should
    flicker (e.g. the backrooms fluorescent tube), or None if the frame has no
    flickering light."""
    return FRAMES.get(frame_name, {}).get("light_region")


def light_type(frame_name):
    """Which flicker style the frame's light uses: "fluorescent" (buzzing,
    stuttering) or "candle" (gentle warm flutter). Defaults to fluorescent."""
    return FRAMES.get(frame_name, {}).get("light_type", "fluorescent")


def light_level(frame_name):
    """How lit the frame's light source stays against the edge dim, 0..1. Below
    1.0 keeps the light dimmed (struggling against the dark). Defaults to 1.0."""
    return FRAMES.get(frame_name, {}).get("light_level", 1.0)