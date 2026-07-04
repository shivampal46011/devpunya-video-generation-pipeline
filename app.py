"""
Automated Video Content Generation Pipeline (Streamlit)

Queue multiple jobs — each job is a text prompt plus an optional reference
image — and batch-generate videos with the Gemini interactions API
(gemini-omni-flash-preview). Generated videos are saved to ./generated_videos
and shown inline with download buttons.

Run:
    streamlit run app.py
"""

import base64
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "generated_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "gemini-omni-flash-preview"
API_REVISION = "2026-05-20"

VIDEO_MODELS = ["gemini-omni-flash-preview"]
IMAGE_MODELS = ["imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"]
VIDEO_ASPECTS = ["9:16", "16:9", "1:1"]
IMAGE_ASPECTS = ["9:16", "16:9", "1:1", "3:4", "4:3"]

PLATFORM_GUIDELINES = {
    "Instagram Reels": "Platform: Instagram Reels — vertical full-frame, bold text-overlay "
        "safe zones top/bottom, fast-paced cuts, must work with sound OFF.",
    "YouTube Shorts": "Platform: YouTube Shorts — vertical full-frame, first frame must be "
        "instantly readable as a thumbnail, punchy pacing, strong loop-back ending.",
    "TikTok": "Platform: TikTok — vertical, raw energetic feel, immediate movement in "
        "frame one, trend-aware visual language.",
    "YouTube (long-form)": "Platform: YouTube long-form — cinematic widescreen framing, "
        "title-card quality composition, documentary polish.",
    "Facebook": "Platform: Facebook feed — high-contrast visuals readable inline, "
        "emotionally warm and shareable.",
    "WhatsApp Status": "Platform: WhatsApp Status — vertical, warm personal tone, "
        "clear even on small screens and low bandwidth.",
}

PURPOSE_GUIDELINES = {
    "Devotional storytelling": "Purpose: devotional storytelling — a narrative arc of "
        "longing/seeking that resolves into darshan or divine grace; evoke deep bhakti.",
    "Motivation / inspiration": "Purpose: motivation — frame the spiritual message as "
        "empowering life wisdom; imagery of rising, overcoming, awakening.",
    "Festival greeting": "Purpose: festival greeting — celebratory mood with diyas, "
        "rangoli, flowers and warm golden glow that clearly evokes the occasion.",
    "Bhajan / mantra visual": "Purpose: bhajan/mantra visual — meditative rhythmic "
        "imagery fit for chanting: repeating sacred motifs, slow powerful zooms, "
        "hypnotic symmetry.",
    "Teaching / discourse clip": "Purpose: teaching clip — calm dignified framing that "
        "keeps full attention on the message; minimal visual distraction.",
}

SPIRITUAL_DIRECTIVE = (
    "Context (mandatory): deeply SPIRITUAL and devotional — rooted in Indian sacred "
    "aesthetics: temples, deities, diyas, incense smoke curling in light beams, sacred "
    "geometry, golden-hour divine light, Ganga aarti, Himalayan ashrams, om symbols. "
    "Tone: reverent, awe-inspiring, emotionally uplifting; always respectful and "
    "accurate to tradition."
)

HOOK_DIRECTIVE_VIDEO = (
    "Retention engineering (CRITICAL): the FIRST 1.5 seconds must be an irresistible "
    "hook — open mid-action at the single most stunning moment (divine reveal, dramatic "
    "light burst, extreme close-up of eyes opening) — never a slow fade-in or "
    "establishing shot. Escalate visual interest every 2-3 seconds with a new angle, "
    "reveal or transformation so there is no static lull. Build to an emotional payoff "
    "in the final 2 seconds and END on a frame that loops seamlessly back to the "
    "opening shot to drive rewatches."
)

HOOK_DIRECTIVE_IMAGE = (
    "Scroll-stopping composition: one dominant subject, extreme light/dark contrast, "
    "dramatic divine lighting, depth and scale that reads instantly at thumbnail size."
)


def build_final_prompt(user_prompt: str, is_video: bool, aspect_ratio: str,
                       platform: str, purpose: str, spiritual: bool,
                       high_retention: bool, duration_s: int) -> str:
    """Layer the style directives onto the user's prompt."""
    parts = [user_prompt.strip()]
    if spiritual:
        parts.append(SPIRITUAL_DIRECTIVE)
    if high_retention:
        parts.append(HOOK_DIRECTIVE_VIDEO if is_video else HOOK_DIRECTIVE_IMAGE)
    if platform in PLATFORM_GUIDELINES:
        parts.append(PLATFORM_GUIDELINES[platform])
    if purpose in PURPOSE_GUIDELINES:
        parts.append(PURPOSE_GUIDELINES[purpose])
    tail = f"STRICT {aspect_ratio} aspect ratio — the frame must be exactly {aspect_ratio}."
    if is_video:
        tail += f" Target duration ~{duration_s} seconds."
    parts.append(tail)
    return "\n\n".join(parts)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_PATH = Path(__file__).parent / "client_secret.json"
DRIVE_TOKEN_PATH = Path(__file__).parent / "drive_token.json"
SA_KEY_PATH = Path(__file__).parent / "service_account.json"
HISTORY_FILENAME = "pipeline_history.json"
SETTINGS_PATH = Path(__file__).parent / "pipeline_settings.json"
DEFAULT_FOLDER_NAME = "AI Video Generation - Gemini"
CRED_MAX_AGE_DAYS = 90


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_settings(settings: dict):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=1))


def cred_age_ok(path: Path) -> bool:
    """Credentials are kept for 90 days; older files are removed (re-link needed)."""
    if not path.exists():
        return False
    if time.time() - path.stat().st_mtime > CRED_MAX_AGE_DAYS * 86400:
        path.unlink(missing_ok=True)
        return False
    return True


def cred_days_left(path: Path) -> int:
    return max(0, int(CRED_MAX_AGE_DAYS - (time.time() - path.stat().st_mtime) / 86400))

st.set_page_config(page_title="Video Content Pipeline", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Generation core (adapted from your snippet)
# ---------------------------------------------------------------------------

def build_input(prompt_text: str, image_bytes: bytes | None, image_mime: str | None):
    """Build the interaction input: plain text, or text + inline image parts."""
    if not image_bytes:
        return prompt_text
    return [
        {"type": "text", "text": prompt_text},
        {
            "type": "image",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": image_mime or "image/png",
        },
    ]


def _pget(obj, key, default=None):
    """Read a field from a step/part that may be a dict or an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def download_from_gcs(uri: str) -> bytes:
    """Download a gs:// object (video delivered to Cloud Storage)."""
    from google.cloud import storage
    bucket_name, blob_name = uri[len("gs://"):].split("/", 1)
    return storage.Client().bucket(bucket_name).blob(blob_name).download_as_bytes()


def extract_video_bytes(interaction) -> tuple[bytes | None, str]:
    """Pull video bytes (and any model text) out of the interaction steps."""
    text_out = []
    video_bytes = None
    for step in interaction.steps:
        if _pget(step, "type") == "model_output" and _pget(step, "content"):
            for part in _pget(step, "content"):
                ptype = _pget(part, "type")
                if ptype == "text" and _pget(part, "text"):
                    text_out.append(_pget(part, "text"))
                elif ptype == "video":
                    data = _pget(part, "data")
                    uri = _pget(part, "uri")
                    if data:
                        video_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
                    elif uri:
                        video_bytes = download_from_gcs(uri)
    return video_bytes, "\n".join(text_out)


def has_adc() -> bool:
    """Is a local gcloud application-default credential available?"""
    import os
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).exists()
    return (Path.home() / ".config/gcloud/application_default_credentials.json").exists()


def make_client(auth: dict) -> genai.Client:
    """Build a Vertex AI client (bills the GCP project).

    Locally, uses gcloud application-default credentials. When deployed,
    a `gcp_service_account` table in Streamlit secrets takes over.
    """
    kwargs = {}
    sa_info = None
    if cred_age_ok(SA_KEY_PATH):
        sa_info = json.loads(SA_KEY_PATH.read_text())
    elif _secret("gcp_service_account"):
        sa_info = dict(_secret("gcp_service_account"))
    if sa_info:
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    elif not has_adc():
        # Without this check the library probes the GCE metadata server and
        # fails with a confusing timeout on deployed machines.
        raise RuntimeError(
            "No GCP credentials on this machine. Open '🔑 Credentials setup' in "
            "the sidebar and upload a service account key JSON (create one at "
            "GCP Console → IAM → Service Accounts with the 'Vertex AI User' "
            "role → Keys → Add key → JSON).")
    return genai.Client(
        vertexai=True,
        project=auth["project"],
        location=auth["location"],
        http_options=types.HttpOptions(headers={"Api-Revision": API_REVISION}),
        **kwargs,
    )


def generate_video(auth: dict, model: str, prompt_text: str,
                   image_bytes: bytes | None, image_mime: str | None,
                   duration_s: int, thinking_level: str, aspect_ratio: str = "9:16",
                   max_retries: int = 3) -> dict:
    """Generate one video. Returns {ok, path, text, error, elapsed}."""
    client = make_client(auth)

    # API requires an explicit task: text_to_video, image_to_video,
    # reference_to_video, edit, or extend.
    rich_kwargs = {
        "generation_config": {
            "max_output_tokens": 65536,
            "thinking_level": thinking_level,
            "video_config": {
                "task": "image_to_video" if image_bytes else "text_to_video",
            },
        },
        "response_modalities": ["video"],
        "response_format": {
            "type": "video",
            "duration": f"{duration_s}s",
            "aspect_ratio": aspect_ratio,
        },
    }

    started = time.time()
    last_err = None
    use_rich = True  # fall back to a minimal model+input request on 400s
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            interaction = client.interactions.create(
                model=model,
                input=build_input(prompt_text, image_bytes, image_mime),
                **(rich_kwargs if use_rich else {}),
            )
            video_bytes, text = extract_video_bytes(interaction)
            if not video_bytes:
                raise RuntimeError("Model returned no video data" + (f" (text: {text[:200]})" if text else ""))

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"video_{stamp}_{uuid.uuid4().hex[:6]}.mp4"
            path.write_bytes(video_bytes)
            return {"ok": True, "path": str(path), "text": text,
                    "error": None, "elapsed": time.time() - started}
        except Exception as e:  # noqa: BLE001 — surface any API failure to the UI
            last_err = e
            msg = str(e).lower()
            if use_rich and ("invalid" in msg or "400" in msg):
                # Server rejected the extra params — retry with the minimal
                # request shape (model + input only), which is known to work.
                use_rich = False
                attempt -= 1  # the fallback attempt is free
                continue
            if "429" in msg or "quota" in msg or "too_many_requests" in msg:
                if attempt < max_retries:
                    time.sleep(65)  # per-minute quota: wait out the window
                continue
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return {"ok": False, "path": None, "text": None,
            "error": str(last_err), "elapsed": time.time() - started}


def generate_image(auth: dict, model: str, prompt_text: str,
                   aspect_ratio: str, max_retries: int = 3) -> dict:
    """Generate one image with Imagen. Aspect ratio is enforced by the API."""
    client = make_client(auth)
    started = time.time()
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = client.models.generate_images(
                model=model,
                prompt=prompt_text,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                ),
            )
            if not res.generated_images:
                raise RuntimeError("Model returned no image (possibly safety-filtered)")
            img_bytes = res.generated_images[0].image.image_bytes
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"image_{stamp}_{uuid.uuid4().hex[:6]}.png"
            path.write_bytes(img_bytes)
            return {"ok": True, "path": str(path), "text": None,
                    "error": None, "elapsed": time.time() - started}
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg:
                if attempt < max_retries:
                    time.sleep(65)
            elif attempt < max_retries:
                time.sleep(2 * attempt)
    return {"ok": False, "path": None, "text": None,
            "error": str(last_err), "elapsed": time.time() - started}


# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------

def _secret(key):
    """Read a Streamlit secret; returns None when no secrets are configured."""
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001 — missing secrets.toml or key
        return None


def get_drive_creds(interactive: bool = False):
    """Load cached Drive OAuth credentials; optionally run the sign-in flow.

    Uses the user's own OAuth client (client_secret.json) because Google
    blocks the shared gcloud client for the Drive scope. The token is cached
    in drive_token.json, so sign-in happens only once.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if cred_age_ok(DRIVE_TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(str(DRIVE_TOKEN_PATH), DRIVE_SCOPES)
        except Exception:  # noqa: BLE001 — corrupt token file, re-auth
            creds = None
    if creds is None and _secret("drive_token"):
        # Deployed mode: token pasted into Streamlit secrets (link locally
        # first, then copy drive_token.json contents into the secret).
        import json
        try:
            creds = Credentials.from_authorized_user_info(
                json.loads(_secret("drive_token")), DRIVE_SCOPES)
        except Exception:  # noqa: BLE001
            creds = None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            DRIVE_TOKEN_PATH.write_text(creds.to_json())
        except Exception:  # noqa: BLE001
            creds = None
    if (not creds or not creds.valid) and interactive and CLIENT_SECRET_PATH.exists():
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), DRIVE_SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        DRIVE_TOKEN_PATH.write_text(creds.to_json())
    return creds if creds and creds.valid else None


def get_drive_service():
    from googleapiclient.discovery import build
    creds = get_drive_creds()
    if not creds:
        raise RuntimeError("Google Drive is not linked yet.")
    return build("drive", "v3", credentials=creds)


@st.cache_data(ttl=300)
def list_drive_folders():
    """Folders the user can pick as an upload destination."""
    service = get_drive_service()
    res = service.files().list(
        q="mimeType='application/vnd.google-apps.folder' and trashed=false",
        pageSize=100,
        fields="files(id, name)",
        orderBy="name",
    ).execute()
    return res.get("files", [])


def parse_drive_folder_id(text: str) -> str:
    """Accept a raw folder ID or a drive.google.com/.../folders/<id> link."""
    text = text.strip()
    if "/folders/" in text:
        text = text.split("/folders/", 1)[1]
    return text.split("?")[0].split("/")[0]


def upload_to_drive(path: str, folder_id: str) -> dict:
    """Upload one media file to Drive; returns {id, link}."""
    from googleapiclient.http import MediaFileUpload
    service = get_drive_service()
    mime = "video/mp4" if path.endswith(".mp4") else "image/png"
    meta = {"name": Path(path).name, "parents": [folder_id]}
    media = MediaFileUpload(path, mimetype=mime, resumable=True)
    created = service.files().create(
        body=meta, media_body=media, fields="id, webViewLink",
        supportsAllDrives=True,
    ).execute()
    return {"id": created["id"], "link": created["webViewLink"]}


def drive_download_file(file_id: str, dest: Path):
    """Download a Drive file to a local path (restore media for preview)."""
    import io as _io
    from googleapiclient.http import MediaIoBaseDownload
    request = get_drive_service().files().get_media(fileId=file_id)
    buf = _io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buf.getvalue())


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def fetch_drive_bytes(drive_id: str) -> bytes:
    """Drive is the source of truth: media is streamed from it for preview."""
    import io as _io
    from googleapiclient.http import MediaIoBaseDownload
    request = get_drive_service().files().get_media(fileId=drive_id)
    buf = _io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def result_bytes(result: dict) -> bytes | None:
    """Media bytes for a result — from Drive first, local file as fallback."""
    if result.get("drive_id"):
        try:
            return fetch_drive_bytes(result["drive_id"])
        except Exception:  # noqa: BLE001
            pass
    p = result.get("path")
    if p and Path(p).exists():
        return Path(p).read_bytes()
    return None


def upload_result_to_drive(result: dict, folder_id: str):
    """Move a generated file to Drive: upload, then remove the local copy."""
    up = upload_to_drive(result["path"], folder_id)
    result["drive_link"] = up["link"]
    result["drive_id"] = up["id"]
    try:
        p = Path(result["path"])
        if p.exists():
            p.unlink()
        result["path"] = p.name  # keep only the filename for display/downloads
    except Exception:  # noqa: BLE001
        pass


def get_or_create_default_folder() -> str:
    """Find (or create) the app's default Drive folder and return its id."""
    service = get_drive_service()
    res = service.files().list(
        q=f"name='{DEFAULT_FOLDER_NAME}' and "
          "mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)", pageSize=1,
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    created = service.files().create(
        body={"name": DEFAULT_FOLDER_NAME,
              "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return created["id"]


# --- Drive folder as persistent job-history DB ---

def _serialize_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("image_bytes", "image_b64")}


def _find_history_file(service, folder_id: str):
    res = service.files().list(
        q=f"name='{HISTORY_FILENAME}' and '{folder_id}' in parents and trashed=false",
        fields="files(id)", pageSize=1, supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def drive_save_history(folder_id: str, jobs: list):
    """Write the job history JSON into the Drive folder (create or update)."""
    from googleapiclient.http import MediaInMemoryUpload
    service = get_drive_service()
    payload = json.dumps([_serialize_job(j) for j in jobs], indent=1).encode()
    media = MediaInMemoryUpload(payload, mimetype="application/json")
    file_id = _find_history_file(service, folder_id)
    if file_id:
        service.files().update(fileId=file_id, media_body=media,
                               supportsAllDrives=True).execute()
    else:
        service.files().create(
            body={"name": HISTORY_FILENAME, "parents": [folder_id]},
            media_body=media, supportsAllDrives=True,
        ).execute()


def drive_load_history(folder_id: str) -> list:
    """Read the job history JSON from the Drive folder ([] if none yet)."""
    service = get_drive_service()
    file_id = _find_history_file(service, folder_id)
    if not file_id:
        return []
    content = service.files().get_media(fileId=file_id).execute()
    jobs = json.loads(content)
    for j in jobs:
        j["image_b64"] = None
        j.setdefault("image_mime", None)
        j.setdefault("results", [])
        j.setdefault("variations", 1)
        j.setdefault("mode", "video")
        # Restored jobs from another machine are display-only, never re-run.
        if j.get("status") in ("queued", "running"):
            j["status"] = "done" if j.get("results") else "failed"
    return jobs


DRIVE_SETUP_HELP = f"""\
**One-time setup (your own OAuth app — Google blocks the shared gcloud one):**
1. Enable the Drive API: [console link](https://console.cloud.google.com/apis/library/drive.googleapis.com?project=devpunya-c7c68)
2. [OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent?project=devpunya-c7c68): \
choose **External**, fill the required fields, and add your own email under **Test users**.
3. [Credentials](https://console.cloud.google.com/apis/credentials?project=devpunya-c7c68) → \
**Create credentials → OAuth client ID → Desktop app**, then **download the JSON** and save it as:
   `{CLIENT_SECRET_PATH}`
4. Reload this page and click **Link Google Drive**.
"""


# ---------------------------------------------------------------------------
# Persistent job store + background worker
#
# Jobs live in a server-side store (survives page refreshes) backed by a
# local JSON DB (survives server restarts) and mirrored to Drive history.
# A daemon worker thread processes the queue; the page only renders state.
# ---------------------------------------------------------------------------

JOBS_DB_PATH = OUTPUT_DIR / "jobs_db.json"


def load_jobs_db() -> list:
    try:
        jobs = json.loads(JOBS_DB_PATH.read_text())
    except Exception:  # noqa: BLE001 — first run / corrupt db
        return []
    for j in jobs:
        if j.get("status") == "running":
            j["status"] = "queued"  # resume work interrupted by a restart
    return jobs


def save_jobs_db(store):
    with store["lock"]:
        JOBS_DB_PATH.write_text(json.dumps(store["jobs"], indent=1))


def _job_image(job):
    return base64.b64decode(job["image_b64"]) if job.get("image_b64") else None


def process_job(store, job):
    """Full lifecycle for one job: generate all variations → upload → record."""
    while len(job["results"]) < job.get("variations", 1):
        if job["mode"] == "video":
            result = generate_video(
                job["auth"], job["model"], job["final_prompt"],
                _job_image(job), job.get("image_mime"),
                job.get("duration_s") or 12, job.get("thinking_level", "medium"),
                job.get("aspect_ratio", "9:16"))
        else:
            result = generate_image(
                job["auth"], job["model"], job["final_prompt"],
                job.get("aspect_ratio", "9:16"))

        if result["ok"] and job.get("drive_upload"):
            folder = load_settings().get("drive_folder_id")
            if folder and get_drive_creds():
                try:
                    upload_result_to_drive(result, folder)
                except Exception as e:  # noqa: BLE001
                    result["drive_error"] = str(e)[:200]

        with store["lock"]:
            job["results"].append(result)
        save_jobs_db(store)

    job["status"] = "done" if any(r["ok"] for r in job["results"]) else "failed"
    save_jobs_db(store)

    folder = load_settings().get("drive_folder_id")
    if job.get("drive_upload") and folder:
        try:
            drive_save_history(folder, store["jobs"])
        except Exception:  # noqa: BLE001 — history mirror is best-effort
            pass


def worker_loop(store):
    while True:
        job = None
        with store["lock"]:
            for j in store["jobs"]:
                if j["status"] == "queued":
                    j["status"] = "running"
                    job = j
                    break
        if job is None:
            time.sleep(2)
            continue
        try:
            process_job(store, job)
        except Exception as e:  # noqa: BLE001 — job must never kill the worker
            job["status"] = "failed"
            job["error"] = str(e)[:300]
            save_jobs_db(store)


@st.cache_resource
def get_store():
    store = {"lock": threading.RLock(), "jobs": load_jobs_db()}
    worker = threading.Thread(target=worker_loop, args=(store,), daemon=True)
    worker.start()
    store["worker"] = worker
    return store


store = get_store()


def add_job(prompt: str, image_file=None):
    """Snapshot ALL current settings into the job — the full request body."""
    job = {
        "id": uuid.uuid4().hex[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt.strip(),
        "final_prompt": build_final_prompt(
            prompt, is_video, aspect_ratio, platform, purpose,
            spiritual_mode, high_retention, int(duration_s)),
        "mode": "video" if is_video else "image",
        "model": model,
        "aspect_ratio": aspect_ratio,
        "duration_s": int(duration_s),
        "thinking_level": thinking_level,
        "variations": int(variations),
        "auth": dict(auth),
        "drive_upload": bool(drive_enabled and drive_folder_id),
        "image_name": image_file.name if image_file else None,
        "image_b64": base64.b64encode(image_file.getvalue()).decode()
                     if image_file else None,
        "image_mime": image_file.type if image_file else None,
        "status": "queued",
        "results": [],
    }
    with store["lock"]:
        store["jobs"].append(job)
    save_jobs_db(store)


# ---------------------------------------------------------------------------
# Sidebar — pipeline settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Pipeline Settings")

    import os
    st.caption("Backend: **Vertex AI** (bills your GCP project). Requires "
               "`gcloud auth application-default login` on this machine.")
    project = st.text_input(
        "GCP Project ID",
        value=os.environ.get("GOOGLE_CLOUD_PROJECT", "devpunya-c7c68"),
    )
    location = st.text_input(
        "Location",
        value=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )
    auth = {"mode": "vertex", "project": project.strip(), "location": location.strip()}

    st.divider()
    mode = st.radio("Generate", ["🎬 Video", "🖼️ Image"], horizontal=True)
    is_video = mode.startswith("🎬")

    if is_video:
        model = st.selectbox("Model", VIDEO_MODELS)
        aspect_ratio = st.selectbox("Aspect ratio", VIDEO_ASPECTS, index=0)
        duration_s = st.slider("Video duration (seconds)", 4, 30, 12)
    else:
        model = st.selectbox("Model", IMAGE_MODELS)
        aspect_ratio = st.selectbox("Aspect ratio", IMAGE_ASPECTS, index=0)
        duration_s = 0
    thinking_level = st.selectbox("Thinking level", ["high", "medium", "low"], index=1)
    variations = st.number_input("Variations per prompt", 1, 5, 1,
                                 help="Generate N outputs for each prompt. The "
                                      "background worker processes them one at a "
                                      "time (avoids per-minute quota 429s).")

    st.divider()
    st.subheader("🎯 Content style")
    platform = st.selectbox("Platform (optional)", ["None"] + list(PLATFORM_GUIDELINES))
    purpose = st.selectbox("Purpose (optional)", ["None"] + list(PURPOSE_GUIDELINES))
    spiritual_mode = st.toggle("🕉️ Spiritual context", value=True,
                               help="Injects devotional aesthetic direction into every prompt.")
    high_retention = st.toggle("🔥 High hook + view-through", value=True,
                               help="Injects retention engineering: 1.5s hook, escalation "
                                    "every 2-3s, seamless loop ending.")

    st.divider()
    with st.expander("🔑 Credentials setup"):
        st.caption("Upload any credential JSON — the app detects what it is: "
                   "a **service account key** (used for Vertex video generation), "
                   "an **OAuth client secret**, or a **Drive token**.")
        cred_files = st.file_uploader(
            "Credential JSON file(s)", type=["json"],
            accept_multiple_files=True, key="cred_upload",
        )
        for cf in cred_files or []:
            try:
                info = json.loads(cf.getvalue())
            except Exception:  # noqa: BLE001
                st.error(f"{cf.name}: not valid JSON")
                continue
            if "private_key" in info and "client_email" in info:
                SA_KEY_PATH.write_text(json.dumps(info))
                st.success(f"{cf.name} → saved as **service account key** "
                           f"(`{info['client_email']}`)")
            elif "installed" in info or "web" in info:
                CLIENT_SECRET_PATH.write_text(json.dumps(info))
                st.success(f"{cf.name} → saved as **OAuth client secret**")
            elif "refresh_token" in info:
                DRIVE_TOKEN_PATH.write_text(json.dumps(info))
                st.success(f"{cf.name} → saved as **Drive token** (Drive is linked)")
            else:
                st.error(f"{cf.name}: unrecognized credential format")

        active = []
        for label, p in [("service account", SA_KEY_PATH),
                         ("client secret", CLIENT_SECRET_PATH),
                         ("drive token", DRIVE_TOKEN_PATH)]:
            if cred_age_ok(p):
                active.append(f"{label} ✅ ({cred_days_left(p)}d left)")
        st.caption("Stored: " + (", ".join(active) if active else "none") +
                   f" · Credentials are kept for {CRED_MAX_AGE_DAYS} days, then "
                   "removed automatically. Generation uses the service account "
                   "if present, else your gcloud login.")
        if SA_KEY_PATH.exists() and st.button("Remove service account key"):
            SA_KEY_PATH.unlink()
            st.rerun()

    if not cred_age_ok(SA_KEY_PATH) and not _secret("gcp_service_account") and not has_adc():
        st.warning("⚠️ No GCP credentials — generation will fail. Upload a "
                   "service account key in **🔑 Credentials setup** above.")

    st.divider()
    st.subheader("☁️ Google Drive")
    _settings = load_settings()
    drive_enabled = st.toggle("Upload generated videos to Drive",
                              value=_settings.get("drive_upload_enabled", True))
    if drive_enabled != _settings.get("drive_upload_enabled", True):
        _settings["drive_upload_enabled"] = drive_enabled
        save_settings(_settings)
    drive_folder_id = None
    if drive_enabled:
        linked = get_drive_creds() is not None
        if not linked:
            if CLIENT_SECRET_PATH.exists():
                st.info("Click below — a Google sign-in page will open in your browser. "
                        "On the warning screen choose **Advanced → Go to app (unsafe)** "
                        "(it's your own app, this is safe).")
                if st.button("🔗 Link Google Drive", type="primary", use_container_width=True):
                    try:
                        get_drive_creds(interactive=True)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Sign-in failed: {str(e)[:200]}")
                        st.caption("No browser on this server? Use the manual link below.")

                with st.expander("Manual link (for deployed apps)"):
                    from google_auth_oauthlib.flow import Flow
                    if "oauth_manual" not in st.session_state:
                        flow = Flow.from_client_secrets_file(
                            str(CLIENT_SECRET_PATH), scopes=DRIVE_SCOPES,
                            redirect_uri="http://localhost")
                        auth_url, _ = flow.authorization_url(
                            access_type="offline", prompt="consent")
                        st.session_state.oauth_manual = {
                            "url": auth_url, "verifier": flow.code_verifier}
                    st.markdown(
                        f"1. [Open the Google sign-in page]({st.session_state.oauth_manual['url']}) "
                        "and approve access.\n"
                        "2. Your browser will land on a `localhost` page that fails to "
                        "load — **that's expected**. Copy that page's full URL from the "
                        "address bar and paste it here:")
                    pasted = st.text_input("Redirect URL (or just the code= value)",
                                           key="oauth_code_paste")
                    if pasted.strip():
                        try:
                            code = pasted.strip()
                            if "code=" in code:
                                from urllib.parse import parse_qs, urlparse
                                code = parse_qs(urlparse(code).query)["code"][0]
                            flow = Flow.from_client_secrets_file(
                                str(CLIENT_SECRET_PATH), scopes=DRIVE_SCOPES,
                                redirect_uri="http://localhost")
                            flow.code_verifier = st.session_state.oauth_manual["verifier"]
                            flow.fetch_token(code=code)
                            DRIVE_TOKEN_PATH.write_text(flow.credentials.to_json())
                            del st.session_state.oauth_manual
                            st.success("Drive linked!")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Could not exchange the code: {str(e)[:200]}")
                    st.caption("Alternative: upload your local `drive_token.json` in "
                               "the Credentials setup section above.")
                drive_enabled = False
            else:
                st.warning("Drive is not linked yet.")
                st.markdown(DRIVE_SETUP_HELP)
                drive_enabled = False
        else:
            try:
                settings = load_settings()
                drive_folder_id = settings.get("drive_folder_id")
                if not drive_folder_id:
                    drive_folder_id = get_or_create_default_folder()
                    settings["drive_folder_id"] = drive_folder_id
                    settings["drive_folder_name"] = DEFAULT_FOLDER_NAME
                    save_settings(settings)
                folder_name = settings.get("drive_folder_name", DEFAULT_FOLDER_NAME)
                st.caption(f"Folder: **{folder_name}**")

                with st.expander("Change folder"):
                    folders = list_drive_folders()
                    options = {"My Drive (root)": "root"}
                    options.update({f["name"]: f["id"] for f in folders})
                    chosen = st.selectbox("Drive folder", list(options.keys()))
                    custom = st.text_input(
                        "…or paste a folder link / ID (overrides the picker)",
                        placeholder="https://drive.google.com/drive/folders/…",
                    )
                    if st.button("Use this folder", use_container_width=True):
                        if custom.strip():
                            settings["drive_folder_id"] = parse_drive_folder_id(custom)
                            settings["drive_folder_name"] = "custom folder"
                        else:
                            settings["drive_folder_id"] = options[chosen]
                            settings["drive_folder_name"] = chosen
                        save_settings(settings)
                        st.session_state.history_synced = False
                        st.rerun()

                if st.button("↻ Sync history with Drive", use_container_width=True,
                             help="Merges the history saved in this Drive folder with "
                                  "the current session, then saves it back."):
                    st.session_state.history_synced = False
                    st.rerun()
                if st.button("Unlink Drive"):
                    DRIVE_TOKEN_PATH.unlink(missing_ok=True)
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                drive_enabled = False
                st.error(f"Drive error: {str(e)[:200]}")

    st.divider()
    st.caption(f"Output folder: `{OUTPUT_DIR}`")
    if st.button("🗑️ Clear finished jobs", use_container_width=True):
        with store["lock"]:
            store["jobs"][:] = [j for j in store["jobs"]
                                if j["status"] in ("queued", "running")]
        save_jobs_db(store)
        st.rerun()


# ---------------------------------------------------------------------------
# Main — job builder
# ---------------------------------------------------------------------------

st.title("🎬 Automated Video Content Pipeline")
st.caption("Queue text + image prompts, then batch-generate videos with Gemini.")

# The Drive folder doubles as a persistent history DB: load it once per session.
if drive_enabled and drive_folder_id and not st.session_state.get("history_synced"):
    try:
        loaded = drive_load_history(drive_folder_id)
        with store["lock"]:
            existing_ids = {j["id"] for j in store["jobs"]}
            restored = [j for j in loaded if j["id"] not in existing_ids]
            store["jobs"][:] = restored + store["jobs"]
        save_jobs_db(store)
        drive_save_history(drive_folder_id, store["jobs"])
        st.session_state.history_synced = True
        if restored:
            st.toast(f"Restored {len(restored)} job(s) from Drive history.")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not sync Drive history: {str(e)[:150]}")

with st.form("single_job", clear_on_submit=True):
    prompt = st.text_area("Prompt", height=120,
                          placeholder="Shiva meditating on Mount Kailash as dawn light "
                                      "breaks through the clouds...")
    image = st.file_uploader("Reference image (optional, video mode only)",
                             type=["png", "jpg", "jpeg", "webp"])
    if st.form_submit_button("Add to queue", type="primary"):
        if prompt.strip():
            add_job(prompt, image)
            st.success("Job added.")
        else:
            st.warning("Prompt is empty.")


# ---------------------------------------------------------------------------
# Queue view + run
# ---------------------------------------------------------------------------

st.divider()
with store["lock"]:
    jobs = list(store["jobs"])
queued = [j for j in jobs if j["status"] == "queued"]
running = [j for j in jobs if j["status"] == "running"]

st.subheader(f"Queue — {len(jobs)} job(s) · {len(queued)} queued · "
             f"{len(running)} running")
st.caption("Jobs process automatically in the background — you can refresh or "
           "close this page without losing progress. Results and Drive uploads "
           "appear here as each variation finishes.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

ok_results = [r for j in jobs for r in j.get("results") or [] if r["ok"]]
if ok_results and st.toggle("Prepare ZIP of all files", value=False,
                            help="Streams every file from Drive and bundles "
                                 "them into one download."):
    import io
    import zipfile
    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w") as zf:
        for r in ok_results:
            data = result_bytes(r)
            if data:
                zf.writestr(Path(r["path"]).name, data)
                added += 1
    st.download_button(
        f"⬇️ Download all {added} file(s) as ZIP",
        data=buf.getvalue(),
        file_name=f"media_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
        mime="application/zip",
    )

for job in jobs:
    icon = {"queued": "🕓", "running": "⏳", "done": "✅", "failed": "❌"}[job["status"]]
    kind_icon = "🎬" if job.get("mode", "video") == "video" else "🖼️"
    with st.expander(f"{icon} {kind_icon} [{job['status']}] {job['prompt'][:90]}",
                     expanded=job["status"] in ("done", "running")):
        if job["status"] == "running":
            done_n = len(job.get("results") or [])
            total_n = max(job.get("variations", 1), 1)
            st.progress(min(done_n / total_n, 0.95),
                        text=f"Generating variation {min(done_n + 1, total_n)}/{total_n}…")
        if job["status"] == "failed" and job.get("error"):
            st.error(job["error"])
        meta_col, media_col = st.columns([1, 2])
        with meta_col:
            st.write(f"**Job ID:** `{job['id']}`")
            st.caption(f"{job.get('mode', 'video')} · {job.get('model', '?')} · "
                       f"{job.get('aspect_ratio', '?')}"
                       + (f" · {job.get('duration_s')}s" if job.get("mode") == "video" else "")
                       + f" · {job.get('variations', 1)} variation(s)"
                       + (" · ☁️ Drive" if job.get("drive_upload") else ""))
            if job.get("image_b64"):
                st.image(base64.b64decode(job["image_b64"]),
                         caption=job.get("image_name"), width=220)
            else:
                st.caption("No reference image.")
            if st.button("Remove", key=f"rm_{job['id']}",
                         disabled=job["status"] == "running"):
                with store["lock"]:
                    store["jobs"][:] = [j for j in store["jobs"]
                                        if j["id"] != job["id"]]
                save_jobs_db(store)
                st.rerun()

        with media_col:
            if job.get("final_prompt"):
                with st.popover("View full prompt sent to the model"):
                    st.text(job["final_prompt"])
            for i, result in enumerate(job.get("results") or []):
                if result["ok"]:
                    is_mp4 = (result.get("path") or "").endswith(".mp4")
                    data = result_bytes(result)
                    if data:
                        if is_mp4:
                            st.video(data)
                        else:
                            st.image(data)
                    else:
                        st.caption("Media unavailable — not on Drive and no local copy.")
                    src = "☁️ Drive" if result.get("drive_id") else "local"
                    st.caption(f"Variation {i + 1} · {result.get('elapsed') or 0:.0f}s "
                               f"· {Path(result.get('path') or '?').name} · {src}")
                    dl_col, drive_col = st.columns(2)
                    with dl_col:
                        if data:
                            st.download_button(
                                "⬇️ Download " + ("MP4" if is_mp4 else "PNG"),
                                data=data,
                                file_name=Path(result["path"]).name,
                                mime="video/mp4" if is_mp4 else "image/png",
                                key=f"dl_{job['id']}_{i}",
                            )
                    with drive_col:
                        if result.get("drive_link"):
                            st.link_button("☁️ View on Drive", result["drive_link"])
                        elif result.get("drive_error"):
                            st.caption(f"Drive upload failed: {result['drive_error']}")
                        elif drive_enabled and drive_folder_id:
                            if st.button("☁️ Upload to Drive", key=f"up_{job['id']}_{i}"):
                                try:
                                    upload_result_to_drive(result, drive_folder_id)
                                    save_jobs_db(store)
                                    drive_save_history(drive_folder_id, store["jobs"])
                                except Exception as e:  # noqa: BLE001
                                    result["drive_error"] = str(e)[:200]
                                st.rerun()
                    if result["text"]:
                        st.caption(f"Model notes: {result['text'][:300]}")
                else:
                    st.error(f"Variation {i + 1} failed: {result['error']}")

# ---------------------------------------------------------------------------
# Live refresh while the background worker is busy
# ---------------------------------------------------------------------------

if any(j["status"] in ("queued", "running") for j in jobs):
    time.sleep(3)
    st.rerun()
