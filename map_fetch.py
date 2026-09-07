import platform as _platform
_platform._wmi = None  # WMI hang bypass, same as pipeline.py / stitch.py
_platform.uname()

import io
import os
import requests
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Antique map fetcher.
#
# Map beats do NOT go through RealVisXL. A diffusion model invents coastlines
# and garbles place names, which is fatal for a real-history channel. Instead a
# map beat fetches a genuine scanned antique map from Wikimedia Commons (public
# domain), walking a widen-out fallback ladder: it tries the tightest search
# first (a specific chart of a specific place) and steps out to broader regions
# until a real map is found, so the channel never shows a fake map, only a
# wider real one. If every tier fails (including no network), it returns None
# and pipeline.py falls back to a normal generated still from the beat's
# visual_prompt, so a map beat never leaves a hole in the video.
# ─────────────────────────────────────────────────────────────

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent with contact info. EDIT the
# contact before heavy use; a generic UA can get rate-limited or blocked.
USER_AGENT = "476MHz-Maps/1.0 (historical shorts pipeline; contact: wingedhusssar@gmail.com)"

NAMESPACE_FILE = 6      # Commons "File:" namespace
SEARCH_LIMIT = 20       # results to consider per tier
THUMB_WIDTH = 1600      # request a rasterized JPEG at this width, not the raw TIFF
MIN_WIDTH = 600         # reject tiny images
MIN_HEIGHT = 400
TIMEOUT = 30            # seconds per request

STILL_WIDTH = 832       # matches pipeline.py's STILL_WIDTH / STILL_HEIGHT
STILL_HEIGHT = 1216
PAD_COLOR = (10, 9, 8)  # near-black warm padding behind a map, blends into the dark frame

# Words that mark a result as an actual map, used to rank map-like titles first.
MAP_HINT_WORDS = ("map", "chart", "carte", "mappa", "karte", "plan", "atlas", "survey", "kaart")
# License hints. Public domain / CC0 preferred; explicitly non-free rejected.
PD_HINTS = ("public domain", "cc0", "pd-", "pdold", "pd-old", "pd-us", "pd-art", "no known copyright")
NONFREE_HINTS = ("fair use", "non-free", "all rights reserved")


def _get(params):
    p = {"action": "query", "format": "json", **params}
    r = requests.get(API, params=p, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def search_maps(query, limit=SEARCH_LIMIT):
    """One combined search + imageinfo call. Returns a list of candidate dicts
    (title, url, width, height, mime, license, index) in search-relevance order."""
    data = _get({
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": NAMESPACE_FILE,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": THUMB_WIDTH,
    })
    pages = data.get("query", {}).get("pages", {})
    items = []
    for pg in pages.values():
        info_list = pg.get("imageinfo")
        if not info_list:
            continue
        info = info_list[0]
        extmeta = info.get("extmetadata", {})
        license_name = ""
        if "LicenseShortName" in extmeta:
            license_name = extmeta["LicenseShortName"].get("value", "")
        elif "License" in extmeta:
            license_name = extmeta["License"].get("value", "")
        items.append({
            "title": pg.get("title", ""),
            "url": info.get("thumburl") or info.get("url"),
            "width": info.get("thumbwidth") or info.get("width") or 0,
            "height": info.get("thumbheight") or info.get("height") or 0,
            "mime": info.get("mime", ""),
            "license": license_name,
            "index": pg.get("index", 999),
        })
    items.sort(key=lambda d: d["index"])  # keep search relevance order
    return items


def _license_rank(license_name):
    low = license_name.lower()
    if any(h in low for h in NONFREE_HINTS) and not any(h in low for h in PD_HINTS):
        return 2  # avoid entirely
    if any(h in low for h in PD_HINTS):
        return 0  # public domain, best
    return 1      # unknown license, acceptable as a lower priority


def _usable(item):
    if not item["url"]:
        return False
    if item["mime"] and not item["mime"].startswith("image/"):
        return False
    if item["width"] < MIN_WIDTH or item["height"] < MIN_HEIGHT:
        return False
    if _license_rank(item["license"]) == 2:
        return False
    return True


def _score(item):
    """Sort key: public domain first, then map-like titles, then search
    relevance, then larger images."""
    title = item["title"].lower()
    map_like = any(w in title for w in MAP_HINT_WORDS)
    area = item["width"] * item["height"]
    return (_license_rank(item["license"]), 0 if map_like else 1, item["index"], -area)


def _download_image(url):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _fit_to_still(img, width=STILL_WIDTH, height=STILL_HEIGHT):
    """Contain the whole map on a portrait still canvas (padding near-black), so
    the full map survives the cover-fit into the landscape screen downstream
    rather than being cropped twice."""
    canvas = Image.new("RGB", (width, height), PAD_COLOR)
    scale = min(width / img.width, height / img.height)
    new_w = max(int(img.width * scale), 1)
    new_h = max(int(img.height * scale), 1)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def fetch_map(tiers, out_path, still_size=(STILL_WIDTH, STILL_HEIGHT)):
    """Walk the fallback ladder of search queries (tightest first, widest last).
    Save the first genuine antique map found to out_path, fitted to the still
    canvas. Returns a dict of source metadata on success, or None if every tier
    failed (nothing usable found, or no network)."""
    if isinstance(tiers, str):
        tiers = [tiers]

    for query in tiers:
        try:
            results = search_maps(query)
        except Exception as error:
            print(f"  map search failed for {query!r}: {error}")
            continue

        usable = [it for it in results if _usable(it)]
        if not usable:
            print(f"  no usable antique map for {query!r}, widening the search")
            continue

        usable.sort(key=_score)
        for pick in usable[:4]:  # try a few best candidates in case a download fails
            try:
                img = _download_image(pick["url"])
            except Exception as error:
                print(f"  download failed for {pick['title']!r}: {error}")
                continue
            canvas = _fit_to_still(img, *still_size)
            canvas.save(out_path)
            license_note = pick["license"] or "license unknown"
            print(f"  MAP: {pick['title']}  [{license_note}]  (matched {query!r})")
            return {
                "path": out_path,
                "title": pick["title"],
                "url": pick["url"],
                "license": pick["license"],
                "query": query,
            }

    return None
