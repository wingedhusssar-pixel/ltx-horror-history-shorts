import platform as _platform
_platform._wmi = None
_platform.uname()

import json
import numpy as np
from PIL import Image

TV_IMAGE_PATH = "tv_frame.png"
SCREEN_MASK_POINTS_PATH = "screen_mask_points.json"

_cache = {}


def _load_calibration(frame_width, frame_height):
    """Load the TV photo and traced screen polygon once, scale both to
    cover the target frame exactly the way build_panzoom_clip's
    scale_to_cover does for stills, and precompute per-row screen edges
    at that final scale. Cached per (frame_width, frame_height) since
    stitch_pipeline calls this once per beat, not once per pipeline run."""
    cache_key = (frame_width, frame_height)
    if cache_key in _cache:
        return _cache[cache_key]

    with open(SCREEN_MASK_POINTS_PATH) as f:
        calib = json.load(f)

    tv_img = Image.open(TV_IMAGE_PATH).convert("RGB")
    src_w, src_h = tv_img.size

    scale_to_cover = max(frame_width / src_w, frame_height / src_h)
    scaled_w = int(round(src_w * scale_to_cover))
    scaled_h = int(round(src_h * scale_to_cover))
    tv_img_scaled = tv_img.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Center-crop the scaled TV photo down to exactly frame_width x
    # frame_height, matching how a cover-fit crop normally behaves.
    crop_x = (scaled_w - frame_width) // 2
    crop_y = (scaled_h - frame_height) // 2
    tv_img_final = tv_img_scaled.crop(
        (crop_x, crop_y, crop_x + frame_width, crop_y + frame_height)
    )
    tv_background = np.array(tv_img_final)

    # Scale and shift the traced polygon by the same transform applied to
    # the photo, so the mask still lines up with the screen after the
    # cover-fit resize and center-crop.
    polygon = np.array(calib["polygon"], dtype=np.float64)
    polygon = polygon * scale_to_cover
    polygon[:, 0] -= crop_x
    polygon[:, 1] -= crop_y

    mask_img = Image.new("L", (frame_width, frame_height), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(p) for p in polygon], fill=255)
    mask_arr = np.array(mask_img)

    left_edge = np.full(frame_height, np.nan)
    right_edge = np.full(frame_height, np.nan)
    for y in range(frame_height):
        xs = np.where(mask_arr[y] > 0)[0]
        if len(xs) > 0:
            left_edge[y] = xs.min()
            right_edge[y] = xs.max()

    valid_rows = np.where(~np.isnan(left_edge))[0]
    if len(valid_rows) == 0:
        raise ValueError(
            "Screen mask produced no visible rows after scaling to "
            f"{frame_width}x{frame_height}. Check screen_mask_points.json."
        )
    row_min, row_max = int(valid_rows.min()), int(valid_rows.max())

    result = {
        "tv_background": tv_background,
        "mask_arr": mask_arr,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "row_min": row_min,
        "row_max": row_max,
    }
    _cache[cache_key] = result
    return result


def composite_into_screen(content_img, frame_width, frame_height):
    """Fit a still image (PIL Image, any size) into the traced screen
    region of the TV photo, scaled to frame_width x frame_height. Uses a
    real 2D Lanczos resize per row-band rather than 1D linear
    interpolation, so the result is as sharp as the source still allows.
    Returns an (frame_height, frame_width, 3) uint8 numpy array, content
    cropped to cover the full screen shape with no gaps at the curved
    edges. content_img must already be in a roughly matching aspect
    ratio to the screen region; it is cropped (cover-fit), not padded."""
    calib = _load_calibration(frame_width, frame_height)
    out = calib["tv_background"].copy()
    left_edge = calib["left_edge"]
    right_edge = calib["right_edge"]
    row_min, row_max = calib["row_min"], calib["row_max"]

    content_img = content_img.convert("RGB")
    cw, ch = content_img.size

    # Cover-fit the still to the screen's bounding box first (so content
    # fills the full screen height with no letterboxing), matching the
    # screen's overall aspect ratio before per-row cropping to the curve.
    screen_w = int(np.nanmax(right_edge[row_min:row_max + 1]) -
                   np.nanmin(left_edge[row_min:row_max + 1])) + 1
    screen_h = row_max - row_min + 1
    cover_scale = max(screen_w / cw, screen_h / ch)
    resized_w = max(int(round(cw * cover_scale)), screen_w)
    resized_h = max(int(round(ch * cover_scale)), screen_h)
    content_resized = content_img.resize((resized_w, resized_h), Image.LANCZOS)

    # Center-crop down to exactly the screen's bounding box.
    cx = (resized_w - screen_w) // 2
    cy = (resized_h - screen_h) // 2
    content_cropped = content_resized.crop((cx, cy, cx + screen_w, cy + screen_h))
    content_arr = np.array(content_cropped)

    # Find the widest row's left edge to use as the common x origin, so
    # the bounding-box crop above lines up the same way every row reads
    # its slice from content_arr.
    global_left = int(np.nanmin(left_edge[row_min:row_max + 1]))

    for y in range(row_min, row_max + 1):
        l, r = left_edge[y], right_edge[y]
        if np.isnan(l) or np.isnan(r) or r <= l:
            continue
        row_width = int(r - l) + 1
        content_row_idx = y - row_min
        content_row_idx = min(content_row_idx, content_arr.shape[0] - 1)
        local_l = int(l) - global_left
        local_l = max(0, min(local_l, content_arr.shape[1] - row_width))
        out[y, int(l):int(l) + row_width] = content_arr[content_row_idx, local_l:local_l + row_width]

    return out


def get_screen_content_mask(frame_width, frame_height):
    """Return the (frame_height, frame_width) uint8 mask array (255 inside
    the screen, 0 outside) for the given frame size, so the camcorder
    pass can apply grain/vignette/flicker only inside the screen and
    leave the TV cabinet, wood grain, and room background untouched."""
    calib = _load_calibration(frame_width, frame_height)
    return calib["mask_arr"]
