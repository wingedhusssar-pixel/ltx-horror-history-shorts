import platform as _platform
_platform._wmi = None
_platform.uname()

import json
import sys
import numpy as np
from PIL import Image, ImageDraw

MASK_POINTS_PATH = "screen_mask_points.json"
OUTPUT_MASK_PATH = "screen_mask.png"
OUTPUT_PREVIEW_PATH = "warp_preview.png"


def load_polygon(path):
    with open(path) as f:
        data = json.load(f)
    return data["image_path"], np.array(data["polygon"], dtype=np.float32)


def polygon_to_mask(polygon, width, height):
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(p) for p in polygon], fill=255)
    return mask_img


def corner_quad_from_polygon(polygon):
    cx, cy = polygon[:, 0].mean(), polygon[:, 1].mean()

    def farthest_in_direction(dx, dy):
        scores = (polygon[:, 0] - cx) * dx + (polygon[:, 1] - cy) * dy
        return polygon[np.argmax(scores)]

    top_left = farthest_in_direction(-1, -1)
    top_right = farthest_in_direction(1, -1)
    bottom_right = farthest_in_direction(1, 1)
    bottom_left = farthest_in_direction(-1, 1)
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def mask_edges_per_row(mask_arr):
    """For each row, find the leftmost and rightmost mask pixel. Returns
    arrays of shape (height,) with NaN for rows with no mask pixels."""
    h = mask_arr.shape[0]
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    for y in range(h):
        xs = np.where(mask_arr[y] > 0)[0]
        if len(xs) > 0:
            left[y] = xs.min()
            right[y] = xs.max()
    return left, right


def warp_content_into_screen(content_img, frame_img, screen_corners, screen_mask):
    """Fill the exact traced mask shape with content, stretched per-row to
    reach the mask's real left/right edge at every row, including the
    curved top and bottom sections a 4-point homography cannot reach."""
    fw, fh = frame_img.size
    cw, ch = content_img.size

    mask_arr = np.array(screen_mask)
    ys_idx, xs_idx = np.where(mask_arr > 0)
    if len(xs_idx) == 0:
        raise ValueError("Mask is empty, check polygon points")
    min_y, max_y = ys_idx.min(), ys_idx.max() + 1

    left_edge, right_edge = mask_edges_per_row(mask_arr)

    out = np.array(frame_img.convert("RGB"))
    content_arr = np.array(content_img.convert("RGB"))

    valid_rows = np.where(~np.isnan(left_edge[min_y:max_y]))[0] + min_y
    row_min, row_max = valid_rows.min(), valid_rows.max()

    for y in range(row_min, row_max + 1):
        l, r = left_edge[y], right_edge[y]
        if np.isnan(l) or np.isnan(r) or r <= l:
            continue
        row_width = int(r - l) + 1
        v = (y - row_min) / max(1, (row_max - row_min))
        src_y = int(round(v * (ch - 1)))
        src_row = content_arr[src_y]
        x_src_coords = np.linspace(0, cw - 1, row_width)
        x0 = np.floor(x_src_coords).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, cw - 1)
        frac = (x_src_coords - x0)[:, None]
        sampled_row = (src_row[x0] * (1 - frac) + src_row[x1] * frac).astype(np.uint8)
        out[y, int(l):int(l) + row_width] = sampled_row

    return Image.fromarray(out)


def make_placeholder_content(width, height):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    bar_h = height // 6
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    for i, color in enumerate(colors):
        y0 = i * bar_h
        y1 = height if i == len(colors) - 1 else (i + 1) * bar_h
        arr[y0:y1, :] = color
    return Image.fromarray(arr)


def main():
    polygon_path = sys.argv[1] if len(sys.argv) > 1 else MASK_POINTS_PATH
    frame_path_override = sys.argv[2] if len(sys.argv) > 2 else None

    image_path, polygon = load_polygon(polygon_path)
    frame_path = frame_path_override or image_path
    frame_img = Image.open(frame_path)
    fw, fh = frame_img.size

    mask_img = polygon_to_mask(polygon, fw, fh)
    mask_img.save(OUTPUT_MASK_PATH)
    print(f"Saved screen mask to {OUTPUT_MASK_PATH}")

    screen_corners = corner_quad_from_polygon(polygon)

    content_w, content_h = 720, 1280
    content_img = make_placeholder_content(content_w, content_h)

    result = warp_content_into_screen(content_img, frame_img, screen_corners, mask_img)
    result.save(OUTPUT_PREVIEW_PATH)
    print(f"Saved preview to {OUTPUT_PREVIEW_PATH}")
    print("Check that the color bars fill the screen exactly, with no overshoot past the curved edge.")


if __name__ == "__main__":
    main()