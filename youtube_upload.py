import platform as _platform
_platform._wmi = None  # WMI hang bypass, consistent with pipeline.py / stitch.py
_platform.uname()

import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ─────────────────────────────────────────────────────────────
# YouTube upload with scheduled publish time.
#
# One-time setup (see README_YOUTUBE_SETUP.md): create an OAuth client in
# Google Cloud Console for the YouTube Data API v3, download it as
# client_secret.json next to this file. The FIRST run opens a browser for you
# to grant access once; after that a token is cached in token.pickle and every
# later run is silent (auto-refreshed), no repeated login.
#
# Scheduling mechanics: the YouTube API does not have a separate "schedule"
# call. A video is uploaded with privacyStatus="private" and a publishAt
# timestamp; YouTube itself flips it to public at that moment. publishAt MUST
# be in the future and MUST be RFC 3339 UTC (a trailing "Z", not a local
# offset), or the upload is rejected outright.
# ─────────────────────────────────────────────────────────────

CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.pickle"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    # list_existing_publish_times() below calls search().list(forMine=True)
    # and videos().list() to check for slot collisions -- read operations
    # that youtube.upload alone (upload/write only) does not cover. Without
    # this, that call fails with "insufficient authentication scopes" and
    # batch_run.py silently falls back to no collision avoidance.
    "https://www.googleapis.com/auth/youtube.readonly",
]

DEFAULT_CATEGORY_ID = "27"  # "Education". Change to "24" for "Entertainment" if preferred.


def get_authenticated_service():
    """Return an authenticated YouTube API client, using a cached token if
    present and valid, refreshing it silently if expired, or running the
    one-time browser consent flow if no token exists yet."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET_FILE):
                raise FileNotFoundError(
                    f"{CLIENT_SECRET_FILE} not found. Download your OAuth client "
                    "JSON from Google Cloud Console and place it next to this "
                    "script. See README_YOUTUBE_SETUP.md."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # opens a browser once for consent
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


def upload_scheduled(video_path, title, description, publish_at_utc_iso, tags=None,
                      category_id=DEFAULT_CATEGORY_ID, made_for_kids=False):
    """Upload video_path to YouTube, private, scheduled to go public at
    publish_at_utc_iso (an RFC 3339 UTC string ending in 'Z'). Returns the new
    video's id and URL on success. Raises on failure, caller decides how to
    handle a single video's upload failing without stopping a whole batch."""
    if not publish_at_utc_iso.endswith("Z"):
        raise ValueError(
            f"publish_at_utc_iso must be RFC 3339 UTC ending in 'Z', got {publish_at_utc_iso!r}"
        )

    youtube = get_authenticated_service()

    body = {
        "snippet": {
            "title": title[:100],           # YouTube's title length limit
            "description": description[:5000],
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": "private",     # required for a scheduled video
            "publishAt": publish_at_utc_iso,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}"}


def list_existing_publish_times(max_results=50):
    """Return a list of RFC3339 UTC datetime strings for the channel's own
    videos that have a publishAt set: scheduled-private videos, whether
    uploaded by this script on a prior run OR manually scheduled by hand in
    YouTube Studio. Used to avoid double-booking a timeslot.

    Implementation note: search.list with forMine=True is the standard way to
    list the authenticated user's own videos INCLUDING private/scheduled ones
    (a plain channel listing only shows public videos). It's a heavier call
    than an uploads-playlist listing, but the uploads playlist frequently does
    NOT include not-yet-public scheduled videos, which is exactly the case
    that matters here."""
    youtube = get_authenticated_service()

    search_resp = youtube.search().list(
        part="id", forMine=True, type="video", order="date", maxResults=max_results,
    ).execute()
    video_ids = [item["id"]["videoId"] for item in search_resp.get("items", []) if "videoId" in item.get("id", {})]
    if not video_ids:
        return []

    publish_times = []
    # videos.list accepts at most 50 ids per call.
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        resp = youtube.videos().list(part="status", id=",".join(chunk)).execute()
        for item in resp.get("items", []):
            publish_at = item.get("status", {}).get("publishAt")
            if publish_at:
                publish_times.append(publish_at)
    return publish_times
