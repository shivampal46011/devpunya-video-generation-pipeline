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
import math
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
        parts.append(get_prompt("spiritual_directive"))
    if high_retention:
        parts.append(get_prompt("hook_video" if is_video else "hook_image"))
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


# ---------------------------------------------------------------------------
# Editable system prompts — defaults live here; overrides in settings
# ---------------------------------------------------------------------------

AGENT1_CORE_DEFAULT = """YOUR 2 CORE OBJECTIVES — every creative decision serves them:
1. EXTREMELY HIGH HOOK RATE — frame 1 must stop the scroll within 1.5 seconds.
2. EXTREMELY HIGH VIEW-THROUGH RATE — every frame must end on a pull that forces the
   viewer into the next frame; the final frame delivers the emotional payoff.

MANDATORY CONSISTENCY RULES (apply across ALL frames):
1. Tone/theme must be IDENTICAL in every frame.
2. Personas/characters: define every character ONCE in the style bible (age, face, hair,
   clothing, build) and reuse those exact descriptions in every frame they appear in.
3. Continuity: each frame must begin exactly where the previous frame ended — state it.
4. Visual style, color grade and lighting language must be the same in every frame."""

AGENT2_CORE_DEFAULT = """Rules for the prompt you write:
- Re-state every visible character with their FULL locked description from the style bible.
- Open mid-action (no fade-ins, no dead air) and escalate visual interest every 2-3 seconds.
- End on a moment that flows seamlessly into the next frame.
- Specify camera, lighting, color grade and mood explicitly, matching the style bible.
- Do NOT mention audio, narration, subtitles or text overlays."""

DEFAULT_PROMPTS = {
    "spiritual_directive": SPIRITUAL_DIRECTIVE,
    "hook_video": HOOK_DIRECTIVE_VIDEO,
    "hook_image": HOOK_DIRECTIVE_IMAGE,
    "agent1_core": AGENT1_CORE_DEFAULT,
    "agent2_core": AGENT2_CORE_DEFAULT,
}


def get_prompt(key: str) -> str:
    """Effective prompt text: user override from settings, else the default."""
    return (load_settings().get("prompts") or {}).get(key) or DEFAULT_PROMPTS[key]

st.set_page_config(page_title="Video Content Pipeline", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Generation core (adapted from your snippet)
# ---------------------------------------------------------------------------

def build_input(prompt_text: str, media: list | None):
    """Build the interaction input: plain text, or text + inline media parts.

    media is a list of (bytes, mime) tuples — any mix of images and videos."""
    if not media:
        return prompt_text
    parts = [{"type": "text", "text": prompt_text}]
    for data, mime in media:
        kind = "video" if (mime or "").startswith("video") else "image"
        parts.append({
            "type": kind,
            "data": base64.b64encode(data).decode("utf-8"),
            "mime_type": mime or "image/png",
        })
    return parts


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
                   media: list | None,
                   duration_s: int, thinking_level: str, aspect_ratio: str = "9:16",
                   max_retries: int = 3, video_task: str | None = None) -> dict:
    """Generate one video. media is a list of (bytes, mime) reference files.
    Returns {ok, path, text, error, elapsed}."""
    client = make_client(auth)

    # API requires an explicit task: text_to_video, image_to_video,
    # reference_to_video, edit, or extend.
    has_video_ref = any((m or "").startswith("video") for _, m in media or [])
    if media and video_task:
        task = video_task
    elif media and (has_video_ref or len(media) > 1):
        task = "reference_to_video"
    elif media:
        task = "image_to_video"
    else:
        task = "text_to_video"
    rich_kwargs = {
        "generation_config": {
            "max_output_tokens": 65536,
            "thinking_level": thinking_level,
            "video_config": {
                "task": task,
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
                input=build_input(prompt_text, media),
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
# Script-narration pipeline (Tab 2)
#
# Script → ElevenLabs Hindi TTS → clips = RoundUp(audio/10) → Agent 1 breaks
# the script into frame definitions (+ locked style bible) → Agent 2 writes a
# detailed prompt per frame → one video job per frame → ffmpeg stitches the
# clips and lays the SAME narration audio over the whole film.
# ---------------------------------------------------------------------------

ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"  # supports Hindi
AGENT_TEXT_MODEL = "gemini-2.5-flash"
MAX_CLIP_SECONDS = 10
MIN_CLIP_SECONDS = 4
STITCH_RESOLUTIONS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}


def elevenlabs_api_key() -> str | None:
    """The key lives in local settings (never mirrored to Drive) or secrets."""
    return load_settings().get("elevenlabs_api_key") or _secret("elevenlabs_api_key")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_elevenlabs_voices(api_key: str) -> list:
    import requests
    r = requests.get(ELEVENLABS_VOICES_URL, headers={"xi-api-key": api_key}, timeout=30)
    r.raise_for_status()
    return r.json().get("voices", [])


ELEVENLABS_TTS_TS_URL = ("https://api.elevenlabs.io/v1/text-to-speech/"
                         "{voice_id}/with-timestamps")


def elevenlabs_tts(api_key: str, voice_id: str, text: str, out_path: Path) -> dict | None:
    """Generate the narration MP3. Uses the with-timestamps endpoint so we get
    character-level timing (returned as the alignment dict); falls back to the
    plain endpoint (returns None) if timestamps are unavailable."""
    import requests
    payload = {
        "text": text,
        "model_id": ELEVENLABS_TTS_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    r = requests.post(ELEVENLABS_TTS_TS_URL.format(voice_id=voice_id),
                      headers=headers, json=payload, timeout=300)
    if r.status_code == 200:
        data = r.json()
        out_path.write_bytes(base64.b64decode(data["audio_base64"]))
        return data.get("alignment") or data.get("normalized_alignment")
    r = requests.post(ELEVENLABS_TTS_URL.format(voice_id=voice_id),
                      headers=headers, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS failed ({r.status_code}): {r.text[:300]}")
    out_path.write_bytes(r.content)
    return None


SENTENCE_ENDERS = "।॥.!?\n"


def build_scenes(alignment: dict, audio_len: float) -> list:
    """Chunk the narration into scenes using the TTS character timestamps.

    Sentences (split on Hindi/Latin sentence enders) are grouped into scenes of
    at most MAX_CLIP_SECONDS; overlong sentences are split at word boundaries.
    Scenes tile the audio exactly (pauses attach to the preceding scene), so
    each scene knows the precise [start, end] of the narration it covers."""
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    if not chars or len(chars) != len(starts) or len(chars) != len(ends):
        return []

    sentences, buf, t0 = [], [], None
    for ch, s, e in zip(chars, starts, ends):
        if t0 is None:
            t0 = s
        buf.append(ch)
        if ch in SENTENCE_ENDERS:
            text = "".join(buf).strip()
            if text:
                sentences.append({"text": text, "start": t0, "end": e})
            buf, t0 = [], None
    if buf:
        text = "".join(buf).strip()
        if text:
            sentences.append({"text": text, "start": t0, "end": ends[-1]})
    if not sentences:
        return []

    # Split any single sentence longer than one clip at word boundaries.
    pieces = []
    for s in sentences:
        dur = s["end"] - s["start"]
        if dur <= MAX_CLIP_SECONDS:
            pieces.append(s)
            continue
        parts = max(2, math.ceil(dur / MAX_CLIP_SECONDS))
        words = s["text"].split()
        per = max(1, math.ceil(len(words) / parts))
        seg = dur / parts
        for k in range(parts):
            wtext = " ".join(words[k * per:(k + 1) * per]).strip()
            if wtext:
                pieces.append({"text": wtext,
                               "start": s["start"] + k * seg,
                               "end": min(s["end"], s["start"] + (k + 1) * seg)})

    # Group consecutive pieces into scenes of at most MAX_CLIP_SECONDS.
    scenes, cur = [], None
    for s in pieces:
        if cur is None:
            cur = dict(s)
        elif s["end"] - cur["start"] <= MAX_CLIP_SECONDS:
            cur["text"] += " " + s["text"]
            cur["end"] = s["end"]
        else:
            scenes.append(cur)
            cur = dict(s)
    if cur:
        scenes.append(cur)

    # Tile the timeline exactly: scene i runs until scene i+1 starts.
    for i, sc in enumerate(scenes):
        sc["end"] = scenes[i + 1]["start"] if i + 1 < len(scenes) else audio_len
    scenes[0]["start"] = 0.0
    out = []
    for sc in scenes:
        sc["start"], sc["end"] = round(sc["start"], 2), round(sc["end"], 2)
        sc["duration"] = round(sc["end"] - sc["start"], 2)
        if sc["duration"] > 0.2:
            out.append(sc)
    return out


def audio_duration_seconds(path: str | Path) -> float:
    from mutagen import File as MutagenFile
    info = MutagenFile(str(path))
    if not info or not getattr(info, "info", None):
        raise RuntimeError("Could not read the narration audio duration")
    return float(info.info.length)


def clip_durations(audio_len: float) -> list[int]:
    """Number of clips = RoundUp(audio/10); each clip ≤10s (e.g. 53s → 6)."""
    n = max(1, math.ceil(audio_len / MAX_CLIP_SECONDS))
    durs = [MAX_CLIP_SECONDS] * (n - 1)
    last = math.ceil(audio_len - MAX_CLIP_SECONDS * (n - 1))
    durs.append(min(MAX_CLIP_SECONDS, max(MIN_CLIP_SECONDS, last)))
    return durs


def run_text_agent(auth: dict, prompt: str, max_retries: int = 3) -> str:
    """One call to the text model (used by Agent 1 and Agent 2)."""
    client = make_client(auth)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(model=AGENT_TEXT_MODEL, contents=prompt)
            if not resp.text:
                raise RuntimeError("Agent returned an empty response")
            return resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                msg = str(e).lower()
                time.sleep(65 if ("429" in msg or "quota" in msg) else 2 * attempt)
    raise RuntimeError(f"Text agent failed: {last_err}")


def parse_agent_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError("Agent did not return JSON: " + text[:200])
    return json.loads(text[start:end + 1])


def strip_md_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def style_bible_md(sb: dict) -> str:
    lines = ["# Style Bible (locked across ALL frames)", ""]
    for key, label in [("tone_theme", "Tone / theme"), ("visual_style", "Visual style"),
                       ("color_grade_lighting", "Color grade & lighting"),
                       ("setting", "Setting")]:
        if sb.get(key):
            lines += [f"**{label}:** {sb[key]}", ""]
    chars = sb.get("characters") or []
    if chars:
        lines.append("**Characters / personas (reuse these descriptions verbatim):**")
        for c in chars:
            lines.append(f"- **{c.get('name', '?')}** — {c.get('description', '')}")
    return "\n".join(lines)


def frame_definition_md(fdef: dict, index: int, n: int, duration: int) -> str:
    lines = [f"# Frame {index} of {n} — {duration}s clip", ""]
    for key, label in [("narration_text", "Narration covered"), ("scene", "Scene"),
                       ("camera", "Camera"), ("emotion", "Emotion"),
                       ("continuity_from_previous", "Continuity from previous frame"),
                       ("hook_or_retention_device", "Hook / retention device")]:
        if fdef.get(key):
            lines += [f"**{label}:** {fdef[key]}", ""]
    return "\n".join(lines)


def agent1_prompt(job: dict) -> str:
    """Agent 1 — break the script into N frame definitions + a style bible."""
    nar = job["narration"]
    n, durs = nar["n_clips"], nar["clip_durations"]
    scenes = nar.get("scenes") or []
    if scenes and scenes[0].get("text"):
        windows_txt = "\n".join(
            f'frame {i + 1}: {sc["start"]:.2f}s → {sc["end"]:.2f}s '
            f'({sc["duration"]:.2f}s) — narration: "{sc["text"]}"'
            for i, sc in enumerate(scenes))
        cut_note = ("The frames were PRE-CUT from the narration's real timestamps — each "
                    "frame covers exactly the narration text shown for its window. Use that "
                    "text VERBATIM as narration_text and design the visuals for that text.")
    else:
        windows, t = [], 0.0
        for i, d in enumerate(durs):
            windows.append(f"frame {i + 1}: {t:.0f}s → "
                           f"{min(t + d, nar['audio_duration']):.0f}s (clip length {d}s)")
            t += d
        windows_txt = "\n".join(windows)
        cut_note = ""
    spiritual = ("\n\n" + get_prompt("spiritual_directive")) if job.get("spiritual") else ""
    return f"""You are AGENT 1 — the "AI Script Breaker" for short-form narrated videos.{spiritual}

The narration audio is {nar['audio_duration']:.1f} seconds long, so the film is split into
EXACTLY {n} frames. Time windows:
{windows_txt}
{cut_note}

{get_prompt("agent1_core")}

Return STRICT JSON only — no markdown fences, no commentary:
{{
  "style_bible": {{
    "tone_theme": "...",
    "visual_style": "...",
    "color_grade_lighting": "...",
    "setting": "...",
    "characters": [{{"name": "...", "description": "locked physical description reused verbatim"}}]
  }},
  "frames": [
    {{
      "index": 1,
      "start_s": 0,
      "end_s": 10,
      "narration_text": "the exact part of the script this frame covers",
      "scene": "what happens on screen, moment by moment",
      "camera": "shot types and movement",
      "emotion": "the emotional beat",
      "continuity_from_previous": "how this frame picks up where the previous ended ('none' for frame 1)",
      "hook_or_retention_device": "the specific device keeping the viewer watching"
    }}
  ]
}}
The "frames" array must contain EXACTLY {n} entries.

SCRIPT (Hindi narration):
\"\"\"{job['script']}\"\"\"
"""


def agent2_prompt(job: dict, i: int) -> str:
    """Agent 2 — turn ONE frame definition into a detailed video prompt."""
    nar = job["narration"]
    fr = nar["frames"][i]
    n = nar["n_clips"]
    if i == 0:
        prev_note = "This is the OPENING frame — it IS the hook."
    else:
        prev_note = ("PREVIOUS frame ended with: "
                     + json.dumps(nar["frames"][i - 1].get("definition") or {},
                                  ensure_ascii=False))
    if i == n - 1:
        next_note = ("This is the FINAL frame — land the emotional payoff and end on a "
                     "composition that loops back to frame 1 to drive rewatches.")
    else:
        next_scene = (nar["frames"][i + 1].get("definition") or {}).get("scene") or ""
        next_note = "NEXT frame will show: " + next_scene
    return f"""You are AGENT 2 — an elite text-to-video prompt engineer.

Write ONE detailed generation prompt for FRAME {i + 1} of {n} of a narrated short film.
The clip is {fr['duration']} seconds, {job.get('aspect_ratio', '9:16')} aspect ratio, and SILENT
(the same narration audio track is laid over all frames afterwards — never include
speech, lip-sync, captions or on-screen text).

2 CORE OBJECTIVES: extremely high hook rate and extremely high view-through rate.

STYLE BIBLE — obey it EXACTLY so tone/theme, personas/characters, color grade and
lighting are identical across all {n} frames:
{nar['style_bible_md']}

{prev_note}
{next_note}

FRAME DEFINITION to convert into the prompt:
{fr['definition_md']}

{get_prompt("agent2_core")}

Return ONLY the final video-generation prompt text — no headers, no commentary, no markdown.
"""


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        import shutil
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not available — `pip install imageio-ffmpeg`")
        return exe


def stitch_clips(clip_paths: list, audio_path: str, out_path: Path, aspect_ratio: str,
                 exact_durations: list | None = None):
    """Concat all clips (normalized to one resolution) + the narration audio.

    When exact_durations (from the narration timestamps) are given, every clip
    is trimmed — or freeze-frame-padded — to EXACTLY its scene's length, so
    each scene stays in sync with the narration it belongs to. -shortest then
    trims any sub-frame remainder.
    """
    import subprocess
    w, h = STITCH_RESOLUTIONS.get(aspect_ratio, (1080, 1920))
    n = len(clip_paths)
    cmd = [get_ffmpeg_exe(), "-y"]
    for p in clip_paths:
        cmd += ["-i", str(p)]
    cmd += ["-i", str(audio_path)]
    filters = []
    for i in range(n):
        chain = (f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                 f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24")
        d = (exact_durations or [None] * n)[i]
        if d:
            # pad with a freeze frame in case the clip is shorter, then cut
            # to the scene's exact narration duration
            chain += (f",tpad=stop_mode=clone:stop_duration={MAX_CLIP_SECONDS},"
                      f"trim=duration={d:.3f},setpts=PTS-STARTPTS")
        filters.append(chain + f"[v{i}]")
    filters.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", f"{n}:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError("ffmpeg stitch failed: " + (proc.stderr or "")[-400:])


def process_narration_job(store, job):
    """Full narration pipeline. Every stage is idempotent, so a failed job can
    be re-queued and resumes exactly where it stopped."""
    nar = job["narration"]
    proj = Path(nar["dir"])
    proj.mkdir(parents=True, exist_ok=True)

    def set_stage(stage: str):
        with store["lock"]:
            job["stage"] = stage
        save_jobs_db(store)

    # -- Step 1: Text-to-speech ------------------------------------------------
    if not nar.get("audio_path") or not Path(nar["audio_path"]).exists():
        set_stage("🎙️ Step 1/5 — text-to-speech + narration timestamps")
        api_key = elevenlabs_api_key()
        if not api_key:
            raise RuntimeError("No ElevenLabs API key — add it in the "
                               "'Video from Script Narration' tab.")
        audio = proj / "narration.mp3"
        alignment = elevenlabs_tts(api_key, job["voice_id"], job["script"], audio)
        audio_len = round(audio_duration_seconds(audio), 2)
        scenes = []
        if alignment:
            (proj / "alignment.json").write_text(json.dumps(alignment))
            scenes = build_scenes(alignment, audio_len)
        if scenes:
            durs = [int(min(MAX_CLIP_SECONDS,
                            max(MIN_CLIP_SECONDS, math.ceil(sc["duration"]))))
                    for sc in scenes]
        else:  # no timestamps available — fall back to fixed 10s windows
            durs = clip_durations(audio_len)
            t = 0.0
            for d in durs:
                scenes.append({"text": "", "start": round(t, 2),
                               "end": round(min(t + d, audio_len), 2),
                               "duration": round(min(t + d, audio_len) - t, 2)})
                t += d
        (proj / "scenes.json").write_text(
            json.dumps(scenes, ensure_ascii=False, indent=1))
        with store["lock"]:
            nar["audio_path"] = str(audio)
            nar["audio_duration"] = audio_len
            nar["scenes"] = scenes
            nar["clip_durations"] = durs
            nar["n_clips"] = len(scenes)
        save_jobs_db(store)

    n = nar["n_clips"]

    # -- Step 2: Agent 1 — script breaker → frame definitions -------------------
    if not nar.get("frames"):
        set_stage(f"🧠 Step 2/5 — Agent 1 breaking script into {n} frames")
        base_prompt = agent1_prompt(job)
        data = parse_agent_json(run_text_agent(job["auth"], base_prompt))
        frames_raw = data.get("frames") or []
        if len(frames_raw) != n:
            data = parse_agent_json(run_text_agent(
                job["auth"],
                base_prompt + f"\n\nIMPORTANT: a previous answer had {len(frames_raw)} "
                              f"frames — return EXACTLY {n} frames."))
            frames_raw = data.get("frames") or []
        if len(frames_raw) != n:
            raise RuntimeError(f"Agent 1 returned {len(frames_raw)} frames instead of "
                               f"{n} — hit Retry to run it again.")
        sb_md = style_bible_md(data.get("style_bible") or {})
        (proj / "style_bible.md").write_text(sb_md)
        frames = []
        for i, fdef in enumerate(frames_raw):
            md = frame_definition_md(fdef, i + 1, n, nar["clip_durations"][i])
            (proj / f"frame_{i + 1:02d}_definition.md").write_text(md)
            scene = (nar.get("scenes") or [{}] * n)[i]
            frames.append({"index": i + 1, "duration": nar["clip_durations"][i],
                           "exact_duration": scene.get("duration"),
                           "scene_text": scene.get("text"),
                           "definition": fdef, "definition_md": md,
                           "prompt": None, "result": None})
        with store["lock"]:
            nar["style_bible_md"] = sb_md
            nar["frames"] = frames
        save_jobs_db(store)

    # -- Step 3: Agent 2 — one detailed prompt per frame (run separately) -------
    for i, fr in enumerate(nar["frames"]):
        if fr.get("prompt"):
            continue
        set_stage(f"✍️ Step 3/5 — Agent 2 writing prompt for frame {i + 1}/{n}")
        prompt = strip_md_fences(run_text_agent(job["auth"], agent2_prompt(job, i)))
        (proj / f"frame_{i + 1:02d}_prompt.md").write_text(prompt)
        with store["lock"]:
            fr["prompt"] = prompt
        save_jobs_db(store)

    # -- Step 4: one video job per frame PROMPT file ----------------------------
    for i, fr in enumerate(nar["frames"]):
        if fr.get("result") and fr["result"].get("ok"):
            continue
        set_stage(f"🎬 Step 4/5 — generating clip {i + 1}/{n}")
        job_media = _job_media(job)
        result = generate_video(
            job["auth"], job["model"], fr["prompt"], job_media,
            fr["duration"], job.get("thinking_level", "medium"),
            job.get("aspect_ratio", "9:16"),
            video_task="reference_to_video" if job_media else None)
        if (not result["ok"] and job_media
                and ("invalid" in str(result.get("error", "")).lower()
                     or "400" in str(result.get("error", "")))):
            # The API sometimes rejects the references for a specific clip —
            # regenerate this clip without them rather than failing the project.
            set_stage(f"🎬 Step 4/5 — clip {i + 1}/{n} retry without reference")
            result = generate_video(
                job["auth"], job["model"], fr["prompt"], None,
                fr["duration"], job.get("thinking_level", "medium"),
                job.get("aspect_ratio", "9:16"))
            if result["ok"]:
                result["note"] = "reference rejected for this clip — generated without it"
        with store["lock"]:
            fr["result"] = result
        save_jobs_db(store)
        if not result["ok"]:
            raise RuntimeError(f"Clip {i + 1}/{n} failed: {result['error']}")

    # -- Step 5: stitch clips + the SAME narration audio ------------------------
    if not nar.get("final_path") or not Path(nar["final_path"]).exists():
        set_stage("🧵 Step 5/5 — stitching clips + narration audio")
        final = proj / f"narrated_video_{job['id']}.mp4"
        stitch_clips([f["result"]["path"] for f in nar["frames"]],
                     nar["audio_path"], final, job.get("aspect_ratio", "9:16"),
                     [f.get("exact_duration") for f in nar["frames"]])
        with store["lock"]:
            nar["final_path"] = str(final)
        save_jobs_db(store)

    # -- Save into Drive (local copy is kept for the download button) -----------
    if job.get("drive_upload") and not nar.get("final_drive_link"):
        set_stage("☁️ Uploading final video to Drive")
        folder = load_settings().get("drive_folder_id")
        if folder and get_drive_creds():
            try:
                up = upload_to_drive(nar["final_path"], folder)
                with store["lock"]:
                    nar["final_drive_link"] = up["link"]
                    nar["final_drive_id"] = up["id"]
            except Exception as e:  # noqa: BLE001
                nar["final_drive_error"] = str(e)[:200]

    with store["lock"]:
        job["status"] = "done"
        job["stage"] = "✅ Complete"
    save_jobs_db(store)

    folder = load_settings().get("drive_folder_id")
    if job.get("drive_upload") and folder:
        try:
            drive_save_history(folder, store["jobs"])
        except Exception:  # noqa: BLE001 — history mirror is best-effort
            pass


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


def _job_media(job) -> list:
    """All reference media on a job as (bytes, mime) tuples. Reads the new
    multi-file ref_media list, falling back to the legacy single-image fields."""
    out = []
    for m in job.get("ref_media") or []:
        if m.get("b64"):
            out.append((base64.b64decode(m["b64"]), m.get("mime")))
    if not out and job.get("image_b64"):
        out.append((base64.b64decode(job["image_b64"]), job.get("image_mime")))
    return out


def process_job(store, job):
    """Full lifecycle for one job: generate all variations → upload → record."""
    if job.get("mode") == "narration":
        process_narration_job(store, job)
        return
    while len(job["results"]) < job.get("variations", 1):
        if job["mode"] == "video":
            result = generate_video(
                job["auth"], job["model"], job["final_prompt"],
                _job_media(job),
                job.get("duration_s") or 12, job.get("thinking_level", "medium"),
                job.get("aspect_ratio", "9:16"),
                video_task=job.get("video_task"))
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


def _pack_ref_files(files) -> list:
    return [{"name": f.name, "mime": f.type,
             "b64": base64.b64encode(f.getvalue()).decode()}
            for f in (files or [])]


def add_job(prompt: str, ref_files=None, video_task: str | None = None):
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
        "ref_media": _pack_ref_files(ref_files),
        "video_task": video_task,
        "status": "queued",
        "results": [],
    }
    with store["lock"]:
        store["jobs"].append(job)
    save_jobs_db(store)


def add_narration_job(script: str, voice_id: str, voice_name: str, gender: str,
                      ref_files=None):
    """Queue a full script→narration→frames→stitched-video pipeline job."""
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "narration",
        "ref_media": _pack_ref_files(ref_files),
        "script": script.strip(),
        "prompt": script.strip()[:120],  # for shared queue displays
        "voice_id": voice_id,
        "voice_name": voice_name,
        "voice_gender": gender,
        "model": model if is_video else VIDEO_MODELS[0],
        "aspect_ratio": aspect_ratio if is_video else "9:16",
        "thinking_level": thinking_level,
        "spiritual": bool(spiritual_mode),
        "auth": dict(auth),
        "drive_upload": bool(drive_enabled and drive_folder_id),
        "status": "queued",
        "stage": "🕓 queued",
        "results": [],
        "narration": {"dir": str(OUTPUT_DIR / f"narration_{job_id}"), "frames": []},
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

tab_single, tab_script, tab_prompts = st.tabs([
    "🎬 Single Video / Frame Generation",
    "🎙️ Video from Script Narration",
    "🧠 System Prompts",
])

with tab_prompts:
    st.caption("Every system prompt the pipeline injects, editable. Dynamic values "
               "(your script, time windows, style bible, aspect ratio, durations…) "
               "are inserted around these automatically. Edits apply to NEW jobs; "
               "saved in `pipeline_settings.json` on this machine.")
    _prompt_overrides = load_settings().get("prompts") or {}
    PROMPT_METAS = [
        ("spiritual_directive", "🕉️ Spiritual context directive",
         "Appended to every generation prompt when the Spiritual toggle is on "
         "(single tab), and to Agent 1's briefing in the narration pipeline."),
        ("hook_video", "🔥 Hook + view-through directive — video",
         "Appended to single-tab video prompts when the High hook toggle is on."),
        ("hook_image", "🔥 Hook directive — image",
         "Appended to single-tab image prompts when the High hook toggle is on."),
        ("agent1_core", "🧠 Agent 1 — script breaker core instructions",
         "The objectives + consistency rules inside Agent 1's prompt (narration "
         "tab). The time windows, scene narration text and JSON output format "
         "are added around this automatically."),
        ("agent2_core", "✍️ Agent 2 — prompt writer rules",
         "The rules block inside Agent 2's prompt (narration tab). The style "
         "bible, frame definition and continuity notes are added automatically."),
    ]
    for _pkey, _ptitle, _pdesc in PROMPT_METAS:
        _edited = _pkey in _prompt_overrides
        with st.expander(_ptitle + ("  ·  ✏️ edited" if _edited else ""),
                         expanded=False):
            st.caption(_pdesc)
            _pval = st.text_area(
                "Prompt text", value=_prompt_overrides.get(_pkey) or DEFAULT_PROMPTS[_pkey],
                height=240, key=f"prompt_edit_{_pkey}", label_visibility="collapsed")
            _c1, _c2 = st.columns(2)
            if _c1.button("💾 Save", key=f"prompt_save_{_pkey}",
                          disabled=_pval.strip() == (_prompt_overrides.get(_pkey)
                                                     or DEFAULT_PROMPTS[_pkey]).strip()):
                _s = load_settings()
                _s.setdefault("prompts", {})[_pkey] = _pval.strip()
                save_settings(_s)
                st.rerun()
            if _edited and _c2.button("↩️ Reset to default", key=f"prompt_reset_{_pkey}"):
                _s = load_settings()
                (_s.get("prompts") or {}).pop(_pkey, None)
                save_settings(_s)
                st.rerun()

with store["lock"]:
    all_jobs = list(store["jobs"])
single_jobs = [j for j in all_jobs if j.get("mode") != "narration"]
narration_jobs = [j for j in all_jobs if j.get("mode") == "narration"]


# ---------------------------------------------------------------------------
# Tab 1 — Single Video / Frame Generation (job form + queue + results)
# ---------------------------------------------------------------------------

with tab_single:
    st.caption("Queue text + image prompts, then batch-generate videos with Gemini.")
    with st.form("single_job", clear_on_submit=True):
        prompt = st.text_area("Prompt", height=120,
                              placeholder="Shiva meditating on Mount Kailash as dawn light "
                                          "breaks through the clouds...")
        ref_files = st.file_uploader(
            "Reference images / videos (optional, video mode only — attach several)",
            type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"],
            accept_multiple_files=True)
        VIDEO_TASKS = {
            "🎨 Style / content reference (reference_to_video)": "reference_to_video",
            "➕ Continue it (extend)": "extend",
            "✏️ Modify it per the prompt (edit)": "edit",
        }
        video_task_label = st.selectbox(
            "If the reference is a VIDEO, use it to…", list(VIDEO_TASKS),
            help="Only applies when the uploaded reference is a video. Images "
                 "always use image_to_video.")
        if st.form_submit_button("Add to queue", type="primary"):
            oversized = [f.name for f in (ref_files or [])
                         if f.size > 25 * 1024 * 1024]
            if not prompt.strip():
                st.warning("Prompt is empty.")
            elif oversized:
                st.warning(f"Too large (keep each under 25 MB): "
                           f"{', '.join(oversized)} — files are sent inline "
                           "with the API request.")
            else:
                add_job(prompt, ref_files, VIDEO_TASKS[video_task_label])
                st.success("Job added.")

    st.divider()
    jobs = single_jobs
    queued = [j for j in jobs if j["status"] == "queued"]
    running = [j for j in jobs if j["status"] == "running"]

    st.subheader(f"Queue — {len(jobs)} job(s) · {len(queued)} queued · "
                 f"{len(running)} running")
    st.caption("Jobs process automatically in the background — you can refresh or "
               "close this page without losing progress. Results and Drive uploads "
               "appear here as each variation finishes.")

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
                ref_items = list(job.get("ref_media") or [])
                if not ref_items and job.get("image_b64"):
                    ref_items = [{"name": job.get("image_name"),
                                  "mime": job.get("image_mime"),
                                  "b64": job["image_b64"]}]
                if ref_items:
                    st.caption(f"{len(ref_items)} reference file(s) · task: "
                               f"{job.get('video_task') or 'auto'}")
                    for m in ref_items:
                        if not m.get("b64"):
                            continue
                        ref_bytes = base64.b64decode(m["b64"])
                        if (m.get("mime") or "").startswith("video"):
                            st.video(ref_bytes)
                            st.caption(m.get("name") or "video")
                        else:
                            st.image(ref_bytes, caption=m.get("name"), width=220)
                else:
                    st.caption("No reference media.")
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
# Tab 2 — Video from Script Narration
# ---------------------------------------------------------------------------

def _scene_table_md(scenes: list) -> str:
    rows = ["| # | start | end | length | narration |",
            "|---|-------|-----|--------|-----------|"]
    for i, sc in enumerate(scenes, 1):
        text = (sc.get("text") or "—").replace("|", "\\|")
        rows.append(f"| {i} | {sc['start']:.2f}s | {sc['end']:.2f}s "
                    f"| {sc['duration']:.2f}s | {text} |")
    return "\n".join(rows)


def render_narration_job(job):
    icon = {"queued": "🕓", "running": "⏳", "done": "✅", "failed": "❌"}.get(
        job["status"], "❔")
    nar = job.get("narration") or {}
    frames = nar.get("frames") or []
    scenes = nar.get("scenes") or []
    running_stage = job.get("stage", "")

    def mark(done: bool, active_key: str) -> str:
        if done:
            return "✅"
        return "⏳" if (job["status"] == "running" and active_key in running_stage) else "🕓"

    with st.container(border=True):
        head, rm = st.columns([6, 1])
        head.markdown(f"**{icon} 🎙️ [{job['status']}] {job.get('script', '')[:80]}**")
        if rm.button("Remove", key=f"nrm_{job['id']}",
                     disabled=job["status"] == "running"):
            with store["lock"]:
                store["jobs"][:] = [j for j in store["jobs"] if j["id"] != job["id"]]
            save_jobs_db(store)
            st.rerun()

        if job["status"] == "running":
            st.info(f"**{running_stage or 'working…'}**")
        elif job["status"] == "queued":
            st.caption("Waiting for the background worker…")
        if job["status"] == "failed":
            st.error(job.get("error") or "Failed")
            if st.button("🔁 Retry (resumes where it stopped)",
                         key=f"nretry_{job['id']}"):
                with store["lock"]:
                    job["status"] = "queued"
                    job["error"] = None
                save_jobs_db(store)
                st.rerun()

        info = (f"**Job ID:** `{job['id']}` · voice: {job.get('voice_name', '?')} "
                f"({job.get('voice_gender', '?')}) · {job.get('model', '?')} · "
                f"{job.get('aspect_ratio', '?')}")
        if nar.get("audio_duration"):
            info += (f" · audio {nar['audio_duration']:.1f}s → "
                     f"**{nar.get('n_clips')} scenes**")
        if job.get("drive_upload"):
            info += " · ☁️ Drive"
        ref_names = [m.get("name") for m in job.get("ref_media") or [] if m.get("name")]
        if not ref_names and job.get("image_name"):
            ref_names = [job["image_name"]]
        if ref_names:
            info += f" · 🎞️ refs: {', '.join(ref_names)}"
        st.caption(info)

        # ---- Step 1: narration audio + timestamps -----------------------------
        s1_done = bool(nar.get("audio_path"))
        with st.expander(f"{mark(s1_done, 'Step 1')} Step 1 — Narration audio & "
                         "scene timestamps", expanded=not s1_done):
            audio_path = nar.get("audio_path")
            if audio_path and Path(audio_path).exists():
                st.audio(Path(audio_path).read_bytes(), format="audio/mp3")
            if scenes:
                if scenes[0].get("text"):
                    st.caption(f"{len(scenes)} scenes cut at sentence boundaries "
                               "from the real TTS timestamps:")
                else:
                    st.caption(f"{len(scenes)} scenes (fixed windows — timestamps "
                               "were unavailable for this run):")
                st.markdown(_scene_table_md(scenes))
            elif not s1_done:
                st.caption("pending…")

        # ---- Step 2: Agent 1 — style bible + frame definitions ----------------
        s2_done = bool(nar.get("style_bible_md"))
        with st.expander(f"{mark(s2_done, 'Step 2')} Step 2 — Agent 1: style bible "
                         "& frame definitions", expanded=False):
            if nar.get("style_bible_md"):
                st.markdown(nar["style_bible_md"])
                st.divider()
            if frames:
                for ftab, fr in zip(st.tabs([f"Frame {f['index']}" for f in frames]),
                                    frames):
                    with ftab:
                        st.markdown(fr.get("definition_md") or "_pending…_")
            elif not s2_done:
                st.caption("pending…")

        # ---- Step 3: Agent 2 — video prompts ----------------------------------
        s3_done = bool(frames) and all(f.get("prompt") for f in frames)
        with st.expander(f"{mark(s3_done, 'Step 3')} Step 3 — Agent 2: video "
                         "prompts (one per frame)", expanded=False):
            if frames:
                for ftab, fr in zip(st.tabs([f"Frame {f['index']}" for f in frames]),
                                    frames):
                    with ftab:
                        if fr.get("prompt"):
                            st.text(fr["prompt"])
                        else:
                            st.caption("pending…")
            else:
                st.caption("pending…")

        # ---- Step 4: generated clips ------------------------------------------
        s4_done = bool(frames) and all(
            (f.get("result") or {}).get("ok") for f in frames)
        with st.expander(f"{mark(s4_done, 'Step 4')} Step 4 — Generated clips",
                         expanded=False):
            if frames:
                done_clips = sum(1 for f in frames if (f.get('result') or {}).get('ok'))
                st.caption(f"{done_clips}/{len(frames)} clips generated · each clip is "
                           "trimmed to its scene's exact narration window at stitch time")
                for ftab, fr in zip(st.tabs([f"Clip {f['index']}" for f in frames]),
                                    frames):
                    with ftab:
                        if fr.get("exact_duration"):
                            st.caption(f"scene length {fr['exact_duration']:.2f}s "
                                       f"(generated at {fr.get('duration')}s)")
                        r = fr.get("result")
                        if r and r.get("ok") and r.get("path") and Path(r["path"]).exists():
                            st.video(Path(r["path"]).read_bytes())
                        elif r and not r.get("ok"):
                            st.error(f"Clip failed: {r.get('error')}")
                        else:
                            st.caption("pending…")
            else:
                st.caption("pending…")

        # ---- Step 5: final stitched video --------------------------------------
        final_path = nar.get("final_path")
        s5_done = bool(final_path and Path(final_path).exists())
        with st.expander(f"{mark(s5_done, 'Step 5')} Step 5 — Final stitched video "
                         "(clips + narration audio)", expanded=s5_done):
            if s5_done:
                final_data = Path(final_path).read_bytes()
                st.video(final_data)
                dl_col, drive_col = st.columns(2)
                with dl_col:
                    st.download_button("⬇️ Download final MP4", data=final_data,
                                       file_name=Path(final_path).name,
                                       mime="video/mp4", key=f"ndl_{job['id']}")
                with drive_col:
                    if nar.get("final_drive_link"):
                        st.link_button("☁️ View on Drive", nar["final_drive_link"])
                    else:
                        if nar.get("final_drive_error"):
                            st.caption(f"Drive upload failed: "
                                       f"{nar['final_drive_error']}")
                        if st.button("☁️ Upload to Drive", key=f"nup_{job['id']}"):
                            folder = load_settings().get("drive_folder_id")
                            if not folder or not get_drive_creds():
                                st.error("Drive is not linked or no folder is set "
                                         "— check the sidebar.")
                            else:
                                try:
                                    with st.spinner("Uploading final video…"):
                                        up = upload_to_drive(final_path, folder)
                                    with store["lock"]:
                                        nar["final_drive_link"] = up["link"]
                                        nar["final_drive_id"] = up["id"]
                                        nar["final_drive_error"] = None
                                    save_jobs_db(store)
                                    st.rerun()
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"Upload failed: {str(e)[:200]}")
            else:
                st.caption("pending…")


with tab_script:
    st.caption("Script → 🎙️ ElevenLabs Hindi narration → clips = RoundUp(audio/10) → "
               "🧠 Agent 1 breaks the script into frame definitions → ✍️ Agent 2 writes "
               "a detailed prompt per frame → 🎬 one video job per frame → 🧵 stitched "
               "into one film over the same narration audio → ⬇️ download + ☁️ Drive.")

    _nsettings = load_settings()
    with st.expander("🎙️ ElevenLabs voice settings",
                     expanded=not elevenlabs_api_key()):
        key_in = st.text_input("ElevenLabs API key", type="password",
                               value=_nsettings.get("elevenlabs_api_key", ""),
                               help="Stored locally in pipeline_settings.json — "
                                    "never mirrored to Drive.")
        if key_in.strip() != _nsettings.get("elevenlabs_api_key", ""):
            _nsettings["elevenlabs_api_key"] = key_in.strip()
            save_settings(_nsettings)

        voice_gender = st.radio("Narrator voice", ["Male", "Female"], horizontal=True)
        st.caption("Target: **Indian, Hindi, 35–55 y/o**. Add such a voice to your "
                   "ElevenLabs account from the Voice Library if none is listed — "
                   "narration uses the multilingual model, which speaks Hindi.")

        voices, voice_options = [], {}
        api_key_now = elevenlabs_api_key()
        if api_key_now:
            try:
                voices = fetch_elevenlabs_voices(api_key_now)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not fetch voices: {str(e)[:150]}")
        matching = [v for v in voices
                    if (v.get("labels") or {}).get("gender", "").lower()
                    == voice_gender.lower()] or voices
        for v in matching:
            labels = ", ".join(str(x) for x in (v.get("labels") or {}).values() if x)
            voice_options[f"{v['name']}" + (f" — {labels}" if labels else "")] = \
                (v["voice_id"], v["name"])
        chosen_voice = st.selectbox(
            "Voice", list(voice_options) or ["— add an API key to list voices —"])
        manual_voice = st.text_input("…or paste a voice ID (overrides the picker)",
                                     placeholder="e.g. 21m00Tcm4TlvDq8ikWAM")
        if manual_voice.strip():
            narration_voice_id, narration_voice_name = manual_voice.strip(), "custom voice"
        elif chosen_voice in voice_options:
            narration_voice_id, narration_voice_name = voice_options[chosen_voice]
        else:
            narration_voice_id, narration_voice_name = None, None

    script_text = st.text_area(
        "Script (Hindi narration)", height=200, key="narration_script",
        placeholder="अपनी कहानी यहाँ लिखें… (the full narration script — the voice-over "
                    "is generated from this text exactly)")
    narration_refs = st.file_uploader(
        "Reference images / videos (optional — style/persona anchors for ALL clips)",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm"],
        key="narration_ref", accept_multiple_files=True,
        help="All attached files are sent with every clip generation as "
             "reference_to_video anchors so characters, style and setting stay "
             "consistent across all scenes.")
    st.caption("Example: a 53s narration → RoundUp(53/10) = **6 clips** of ≤10s each, "
               "generated with a consistent style bible and stitched over the audio. "
               "Video model / aspect ratio / Drive upload come from the sidebar.")

    if st.button("🚀 Generate narrated video", type="primary"):
        if not script_text.strip():
            st.warning("Script is empty.")
        elif not elevenlabs_api_key():
            st.warning("Add your ElevenLabs API key in the voice settings above.")
        elif not narration_voice_id:
            st.warning("Pick a voice (or paste a voice ID) in the voice settings above.")
        elif any(f.size > 25 * 1024 * 1024 for f in (narration_refs or [])):
            st.warning("A reference file is too large — keep each under 25 MB.")
        else:
            add_narration_job(script_text, narration_voice_id,
                              narration_voice_name, voice_gender, narration_refs)
            st.success("Narration pipeline queued — it runs in the background.")
            st.rerun()

    st.divider()
    st.subheader(f"Narration projects — {len(narration_jobs)}")
    for _njob in reversed(narration_jobs):
        render_narration_job(_njob)

# ---------------------------------------------------------------------------
# Live refresh while the background worker is busy
# ---------------------------------------------------------------------------

with store["lock"]:
    _busy = any(j["status"] in ("queued", "running") for j in store["jobs"])
if _busy:
    time.sleep(3)
    st.rerun()
