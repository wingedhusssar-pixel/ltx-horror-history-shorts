import platform as _platform
_platform._wmi = None  # WMI hang bypass, consistent with pipeline.py / stitch.py
_platform.uname()

import os

# Same thread-pinning as pipeline.py, must land before torch is imported by
# anything this script pulls in transitively.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, date, timedelta, timezone

import youtube_upload

# ─────────────────────────────────────────────────────────────
# Weekly batch: generates one week of episodes (2 per day, 12pm and 7pm local
# time), runs pipeline.py then stitch.py for each, and uploads every finished
# video to YouTube as a scheduled private upload that YouTube itself flips
# public at the assigned time. Meant to be triggered once a week (Task
# Scheduler), not run continuously.
#
# One video failing (a bad still run, a network hiccup on upload) is logged
# and the batch moves on to the next slot rather than aborting the whole week.
# ─────────────────────────────────────────────────────────────

HOUR_SLOTS = [12, 19]          # 12:00 and 19:00 local time
DAYS_AHEAD = 7                 # one slot pair per day for a week
VIDEOS_PER_RUN = DAYS_AHEAD * len(HOUR_SLOTS)

OUTPUT_DIR = "pipeline_output"
ARCHIVE_DIR = "batch_archive"  # finished videos copied here before the next iteration
BATCH_LOG_PATH = "batch_log.jsonl"
STATE_PATH = "batch_state.json"

DESCRIPTION_TEMPLATE = (
    "{title}\n\n"
    "A short look at a real, documented case from history.\n\n"
    "#history #mystery #shorts"
)


def next_monday(from_dt):
    """The date of the upcoming Monday strictly after from_dt. If from_dt is
    itself a Monday, returns the FOLLOWING Monday, since this batch always
    covers a full week ahead, not slots later the same day. This is what makes
    a Sunday-morning trigger land on tomorrow (Monday) 12:00 as the first slot,
    and a late/catch-up trigger (e.g. the PC was off Sunday and Task Scheduler
    only runs this Wednesday) still land on the correct upcoming Monday rather
    than any day already in the past."""
    days_ahead = (0 - from_dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (from_dt + timedelta(days=days_ahead)).date()


def build_week_slots(anchor_monday, days_ahead=DAYS_AHEAD, hour_slots=HOUR_SLOTS):
    """All local datetimes for the week starting at anchor_monday: 12:00 and
    19:00 every day for days_ahead days, in order. Used only as a fallback if
    fetching existing publish times fails (see resolve_conflict_free_slots)."""
    slots = []
    for d in range(days_ahead):
        day = anchor_monday + timedelta(days=d)
        for hour in hour_slots:
            slots.append(datetime(day.year, day.month, day.day, hour, 0))
    return slots


SLOT_COLLISION_TOLERANCE_MINUTES = 45  # how close to an existing video counts as "occupied"


def _to_utc(local_dt):
    return local_dt.astimezone().astimezone(timezone.utc)


def _parse_existing_times(iso_list):
    return [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in iso_list]


def _is_occupied(local_dt, occupied_utc, tolerance_minutes=SLOT_COLLISION_TOLERANCE_MINUTES):
    target_utc = _to_utc(local_dt)
    return any(abs((target_utc - o).total_seconds()) <= tolerance_minutes * 60 for o in occupied_utc)


def resolve_conflict_free_slots(anchor_monday, needed, hour_slots=HOUR_SLOTS):
    """Build `needed` local publish datetimes at hour_slots starting from
    anchor_monday, SKIPPING any that collide with a video already scheduled or
    published on the channel (whether by a previous batch run or uploaded
    manually), and extending forward day by day past the nominal week if
    necessary until `needed` clean slots are found. This is what makes a
    manually-scheduled upload in one of the batch's usual slots simply push
    that slot's content later, rather than double-booking the timeslot.

    If fetching existing times from YouTube fails for any reason (auth not
    set up yet, network hiccup), falls back to the plain, non-conflict-aware
    week so the batch can still run rather than blocking entirely on this."""
    try:
        occupied_iso = youtube_upload.list_existing_publish_times()
        occupied_utc = _parse_existing_times(occupied_iso)
        if occupied_iso:
            print(f"  found {len(occupied_iso)} existing scheduled/timed video(s) on the channel "
                  f"to check for slot collisions.")
    except Exception as error:
        print(f"  could not check for existing scheduled videos ({error}); "
              f"proceeding without collision avoidance for this run.")
        return build_week_slots(anchor_monday, needed // len(hour_slots) + 1, hour_slots)[:needed]

    slots = []
    day_offset = 0
    guard = 0
    while len(slots) < needed:
        day = anchor_monday + timedelta(days=day_offset)
        for hour in hour_slots:
            if len(slots) >= needed:
                break
            candidate = datetime(day.year, day.month, day.day, hour, 0)
            if not _is_occupied(candidate, occupied_utc):
                slots.append(candidate)
            else:
                print(f"  slot {candidate.strftime('%a %Y-%m-%d %H:%M')} already occupied on "
                      f"the channel, pushing to the next available slot.")
        day_offset += 1
        guard += 1
        if guard > 60:  # safety valve, should never trigger
            break
    return slots


def week_key(anchor_monday):
    """A stable identifier for 'this week's batch', so resuming after a crash
    or a computer being off knows which week's progress to check."""
    return anchor_monday.isoformat()


def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_slot_done(state, wk, slot_iso):
    return state.get(wk, {}).get(slot_iso, {}).get("status") == "uploaded"


def find_incomplete_week(state, slots_per_week=VIDEOS_PER_RUN):
    """Return the week_key of an in-progress (not fully uploaded) batch, if
    one exists, else None. Checked BEFORE computing a fresh next-Monday
    anchor, so a re-run after the computer was off resumes the SAME week
    that was already started rather than rolling forward to the following
    Monday and abandoning unfinished slots. Assumes at most one week is ever
    in progress at a time, which holds as long as batch_run.py is only
    triggered on its own schedule."""
    for wk, slot_map in state.items():
        uploaded = sum(1 for v in slot_map.values() if v.get("status") == "uploaded")
        if uploaded < slots_per_week:
            return wk
    return None


def ensure_future(publish_dt, min_lead_minutes=15):
    """YouTube's API rejects a publishAt that is not strictly in the future. If
    the machine was off long enough that a slot's original time has already
    passed, push it to a few minutes from now instead of letting the upload
    fail outright. Returns (possibly adjusted datetime, was_adjusted)."""
    now = datetime.now()
    if publish_dt > now + timedelta(minutes=1):
        return publish_dt, False
    adjusted = now + timedelta(minutes=min_lead_minutes)
    return adjusted, True


def mark_slot(state, wk, slot_iso, result):
    """Record a slot's outcome and persist immediately, so progress survives
    even if the process is killed on the very next slot (power loss, crash)."""
    state.setdefault(wk, {})[slot_iso] = result
    save_state(state)


def to_rfc3339_utc(local_dt):
    """Convert a naive local datetime to an RFC 3339 UTC string ending in 'Z',
    which is the exact format the YouTube API requires for publishAt."""
    aware_local = local_dt.astimezone()  # attach the machine's local tzinfo
    utc_dt = aware_local.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


BATCH_STEP_LOG_DIR = "batch_step_logs"


def run_step(args, label, log_name):
    """Run a pipeline subprocess step, returning True on success. Streams
    output live to the console exactly as before, AND saves the full output
    to disk at batch_step_logs/<log_name>. This batch is meant to run
    unattended (Task Scheduler); a native crash during pipeline.py or
    stitch.py previously only ever printed a bare exit code with no named
    trace to whoever happened to be watching the console at that moment --
    the exact blind spot that cost hours of guessing before -X faulthandler
    was added to manual runs. -X faulthandler is added to these subprocess
    invocations below (in run_one_video) for the same reason: so a segfault
    prints a named native stack trace instead of silence. Saving to disk
    means that trace survives even if no one is watching, or the console
    scrolls past it, or the window is closed."""
    os.makedirs(BATCH_STEP_LOG_DIR, exist_ok=True)
    log_path = os.path.join(BATCH_STEP_LOG_DIR, log_name)
    print(f"  [{label}] running: {' '.join(args)}", flush=True)
    print(f"  [{label}] log: {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        process.wait()
    if process.returncode != 0:
        print(f"  [{label}] FAILED (exit code {process.returncode}). See {log_path}", flush=True)
        return False
    return True


def load_script_metadata():
    """Pull title/mood/topic back out of script.json for the video description."""
    path = os.path.join(OUTPUT_DIR, "script.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def log_batch_event(event):
    event["logged_at"] = datetime.now().isoformat()
    with open(BATCH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run_one_video(slot_index, publish_local_dt, total, state, wk):
    """Generate one episode (pipeline.py -> stitch.py), archive the finished
    file, and upload it scheduled for publish_local_dt. Returns a result dict;
    never raises, so one bad video does not stop the week's batch. Every
    outcome is written to the persistent state file immediately, so a re-run
    after a crash or the computer being off resumes at the next incomplete
    slot instead of restarting or re-uploading."""
    label = f"video {slot_index + 1}/{total}"
    slot_iso = publish_local_dt.isoformat()
    print(f"\n=== {label}  (scheduled for {publish_local_dt.strftime('%a %Y-%m-%d %H:%M')}) ===", flush=True)

    python_exe = sys.executable
    log_stamp = publish_local_dt.strftime("%Y%m%d_%H%M")

    if not run_step(
        [python_exe, "-u", "-X", "faulthandler", "pipeline.py"],
        f"{label} pipeline",
        f"{log_stamp}_pipeline.log",
    ):
        result = {"status": "failed", "stage": "pipeline"}
        mark_slot(state, wk, slot_iso, result)
        log_batch_event({"slot": slot_index, "stage": "pipeline", "status": "failed"})
        return result

    if not run_step(
        [python_exe, "-u", "-X", "faulthandler", "stitch.py"],
        f"{label} stitch",
        f"{log_stamp}_stitch.log",
    ):
        result = {"status": "failed", "stage": "stitch"}
        mark_slot(state, wk, slot_iso, result)
        log_batch_event({"slot": slot_index, "stage": "stitch", "status": "failed"})
        return result

    final_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    if not os.path.exists(final_path):
        result = {"status": "failed", "stage": "missing_output"}
        mark_slot(state, wk, slot_iso, result)
        log_batch_event({"slot": slot_index, "stage": "stitch", "status": "missing_output"})
        return result

    meta = load_script_metadata()
    title = meta.get("title") or "Untitled"
    topic = meta.get("topic", "")

    # Archive BEFORE the next iteration's pipeline.py run wipes pipeline_output.
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = publish_local_dt.strftime("%Y%m%d_%H%M")
    archived_path = os.path.join(ARCHIVE_DIR, f"{stamp}_{title[:40]}.mp4".replace("/", "-").replace("\\", "-"))
    shutil.copy2(final_path, archived_path)
    print(f"  archived to {archived_path}", flush=True)

    try:
        safe_publish_dt, was_adjusted = ensure_future(publish_local_dt)
        if was_adjusted:
            print(f"  WARNING: original slot time has passed, publishing at "
                  f"{safe_publish_dt.strftime('%a %H:%M')} instead of "
                  f"{publish_local_dt.strftime('%a %H:%M')}", flush=True)
        publish_at = to_rfc3339_utc(safe_publish_dt)
        description = DESCRIPTION_TEMPLATE.format(title=topic or title)
        upload_result = youtube_upload.upload_scheduled(
            video_path=archived_path,
            title=title,
            description=description,
            publish_at_utc_iso=publish_at,
        )
        print(f"  uploaded: {upload_result['url']}  (publishes {publish_local_dt.strftime('%a %H:%M')} local)", flush=True)
        result = {
            "status": "uploaded", "title": title, "video_id": upload_result["video_id"],
            "url": upload_result["url"], "archived_path": archived_path,
        }
        mark_slot(state, wk, slot_iso, result)
        log_batch_event({"slot": slot_index, "publish_local": slot_iso, "publish_utc": publish_at, **result})
        return result
    except Exception:
        print(f"  UPLOAD FAILED for {label}:", flush=True)
        traceback.print_exc()
        # The video was still generated and archived even though the upload
        # failed, so nothing is lost; it can be uploaded manually or retried.
        # NOT marked "uploaded" in state, so a re-run will retry the upload
        # (pipeline.py/stitch.py are skipped since a video already exists here,
        # see run_weekly_batch).
        result = {"status": "upload_failed", "archived_path": archived_path, "title": title}
        mark_slot(state, wk, slot_iso, result)
        log_batch_event({"slot": slot_index, "status": "upload_failed", "title": title, "archived_path": archived_path})
        return result


def run_weekly_batch():
    trigger_dt = datetime.now()
    state = load_state()

    resuming_wk = find_incomplete_week(state)
    if resuming_wk:
        anchor = date.fromisoformat(resuming_wk)
        wk = resuming_wk
        print(f"Found an unfinished batch for week {wk}. Resuming it instead of "
              f"starting a new week.")
    else:
        anchor = next_monday(trigger_dt)
        wk = week_key(anchor)

    slots = resolve_conflict_free_slots(anchor, VIDEOS_PER_RUN)

    print(f"Weekly batch triggered {trigger_dt.strftime('%a %Y-%m-%d %H:%M')} local.")
    print(f"This week's batch: {wk}  ({len(slots)} slots, Monday {anchor.strftime('%b %d')} through Sunday)")

    already_done = [s for s in slots if is_slot_done(state, wk, s.isoformat())]
    remaining = [s for s in slots if not is_slot_done(state, wk, s.isoformat())]
    if already_done:
        print(f"Resuming: {len(already_done)} slot(s) already uploaded this week, skipping them.")
    print(f"{len(remaining)} slot(s) remaining:")
    for s in remaining:
        print(f"  - {s.strftime('%a %Y-%m-%d %H:%M')}")

    results = []
    for slot in remaining:
        idx = slots.index(slot)

        # If this slot already generated a video but the upload failed on a
        # previous attempt, don't burn GPU time regenerating it: reuse the
        # archived file already sitting on disk and just retry the upload.
        prior = state.get(wk, {}).get(slot.isoformat())
        if prior and prior.get("status") == "upload_failed" and os.path.exists(prior.get("archived_path", "")):
            print(f"\n=== video {idx + 1}/{len(slots)}  (retrying upload only, already generated) ===", flush=True)
            try:
                safe_publish_dt, was_adjusted = ensure_future(slot)
                if was_adjusted:
                    print(f"  WARNING: original slot time has passed, publishing at "
                          f"{safe_publish_dt.strftime('%a %H:%M')} instead of "
                          f"{slot.strftime('%a %H:%M')}", flush=True)
                publish_at = to_rfc3339_utc(safe_publish_dt)
                result = youtube_upload.upload_scheduled(
                    video_path=prior["archived_path"],
                    title=prior.get("title", "Untitled"),
                    description=DESCRIPTION_TEMPLATE.format(title=prior.get("title", "Untitled")),
                    publish_at_utc_iso=publish_at,
                )
                print(f"  uploaded: {result['url']}", flush=True)
                outcome = {"status": "uploaded", **result, "archived_path": prior["archived_path"]}
                mark_slot(state, wk, slot.isoformat(), outcome)
                results.append(outcome)
            except Exception:
                print("  retry upload FAILED again:", flush=True)
                traceback.print_exc()
                results.append(prior)
            continue

        results.append(run_one_video(idx, slot, len(slots), state, wk))

    succeeded = sum(1 for r in results if r["status"] == "uploaded")
    failed = len(results) - succeeded
    total_done_this_week = sum(1 for s in slots if is_slot_done(state, wk, s.isoformat()))
    print(f"\nThis run: {succeeded} uploaded, {failed} failed.")
    print(f"Week total: {total_done_this_week}/{len(slots)} uploaded so far.")
    if total_done_this_week < len(slots):
        print("Not all slots are complete. Re-running this script will resume the remaining ones.")
    print(f"See {BATCH_LOG_PATH} and {STATE_PATH} for details.")


if __name__ == "__main__":
    run_weekly_batch()
