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
import concurrent.futures
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


def make_client(auth: dict) -> genai.Client:
    """Build a Vertex AI client (bills the GCP project)."""
    return genai.Client(
        vertexai=True,
        project=auth["project"],
        location=auth["location"],
        http_options=types.HttpOptions(headers={"Api-Revision": API_REVISION}),
    )


def generate_video(auth: dict, model: str, prompt_text: str,
                   image_bytes: bytes | None, image_mime: str | None,
                   duration_s: int, thinking_level: str,
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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "jobs" not in st.session_state:
    # each job: {id, prompt, image_name, image_bytes, image_mime,
    #            status: queued|running|done|failed, result}
    st.session_state.jobs = []
if "running" not in st.session_state:
    st.session_state.running = False


def add_job(prompt: str, image_file=None):
    job = {
        "id": uuid.uuid4().hex[:8],
        "prompt": prompt.strip(),
        "image_name": image_file.name if image_file else None,
        "image_bytes": image_file.getvalue() if image_file else None,
        "image_mime": image_file.type if image_file else None,
        "status": "queued",
        "result": None,
    }
    st.session_state.jobs.append(job)


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

    model = st.text_input("Model", value=DEFAULT_MODEL)
    duration_s = st.slider("Video duration (seconds)", 4, 30, 10)
    thinking_level = st.selectbox("Thinking level", ["high", "medium", "low"], index=0)
    variations = st.number_input("Variations per prompt", 1, 5, 1,
                                 help="Generate N videos for each prompt.")
    max_workers = st.slider("Parallel generations", 1, 4, 1,
                            help="Keep at 1 unless your Vertex per-minute quota "
                                 "allows more — parallel requests hit 429s.")

    st.divider()
    st.caption(f"Output folder: `{OUTPUT_DIR}`")
    if st.button("🗑️ Clear all jobs", use_container_width=True):
        st.session_state.jobs = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main — job builder
# ---------------------------------------------------------------------------

st.title("🎬 Automated Video Content Pipeline")
st.caption("Queue text + image prompts, then batch-generate videos with Gemini.")

tab_single, tab_batch = st.tabs(["➕ Add job", "📋 Batch add (multiple prompts)"])

with tab_single:
    with st.form("single_job", clear_on_submit=True):
        prompt = st.text_area("Prompt", height=120,
                              placeholder="A cinematic drone shot over a neon city at night...")
        image = st.file_uploader("Reference image (optional)",
                                 type=["png", "jpg", "jpeg", "webp"])
        if st.form_submit_button("Add to queue", type="primary"):
            if prompt.strip():
                add_job(prompt, image)
                st.success("Job added.")
            else:
                st.warning("Prompt is empty.")

with tab_batch:
    with st.form("batch_jobs", clear_on_submit=True):
        prompts_text = st.text_area(
            "One prompt per line", height=160,
            placeholder="A golden retriever surfing a wave\nTimelapse of a city skyline at dusk\nMacro shot of coffee being poured",
        )
        batch_images = st.file_uploader(
            "Reference images (optional, multiple)",
            type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
            help="Matched to prompts in order. If you upload exactly one image, it is used for every prompt.",
        )
        if st.form_submit_button("Add all to queue", type="primary"):
            lines = [ln for ln in prompts_text.splitlines() if ln.strip()]
            for i, line in enumerate(lines):
                img = None
                if batch_images:
                    img = batch_images[0] if len(batch_images) == 1 else (
                        batch_images[i] if i < len(batch_images) else None)
                add_job(line, img)
            st.success(f"Added {len(lines)} job(s).")


# ---------------------------------------------------------------------------
# Queue view + run
# ---------------------------------------------------------------------------

st.divider()
jobs = st.session_state.jobs
queued = [j for j in jobs if j["status"] == "queued"]

left, right = st.columns([3, 1])
with left:
    st.subheader(f"Queue — {len(jobs)} job(s), {len(queued)} pending")
with right:
    run_clicked = st.button(
        f"🚀 Generate {len(queued) * variations} video(s)",
        type="primary", use_container_width=True,
        disabled=not queued or st.session_state.running,
    )

if run_clicked:
    if not auth["project"]:
        st.error("No GCP project ID — enter one in the sidebar.")
    else:
        st.session_state.running = True

        # Expand queued jobs by requested variation count.
        work = [(job, v) for job in queued for v in range(int(variations))]
        progress = st.progress(0.0, text="Starting…")
        done_count = 0

        for job, _ in work:
            job["status"] = "running"

        with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            futures = {
                pool.submit(
                    generate_video, auth, model, job["prompt"],
                    job["image_bytes"], job["image_mime"],
                    int(duration_s), thinking_level,
                ): job
                for job, _ in work
            }
            for fut in concurrent.futures.as_completed(futures):
                job = futures[fut]
                result = fut.result()
                done_count += 1
                # A job may have several variations; collect all results.
                job.setdefault("results", []).append(result)
                if len(job.get("results", [])) >= int(variations):
                    job["status"] = "done" if any(r["ok"] for r in job["results"]) else "failed"
                progress.progress(
                    done_count / len(work),
                    text=f"Generated {done_count}/{len(work)} — last: "
                         f"{'✅ ok' if result['ok'] else '❌ ' + str(result['error'])[:80]}",
                )

        st.session_state.running = False
        st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

for job in jobs:
    icon = {"queued": "🕓", "running": "⏳", "done": "✅", "failed": "❌"}[job["status"]]
    with st.expander(f"{icon} [{job['status']}] {job['prompt'][:90]}", expanded=job["status"] == "done"):
        meta_col, media_col = st.columns([1, 2])
        with meta_col:
            st.write(f"**Job ID:** `{job['id']}`")
            if job["image_bytes"]:
                st.image(job["image_bytes"], caption=job["image_name"], width=220)
            else:
                st.caption("No reference image (text-to-video).")
            if st.button("Remove", key=f"rm_{job['id']}"):
                st.session_state.jobs = [j for j in st.session_state.jobs if j["id"] != job["id"]]
                st.rerun()

        with media_col:
            for i, result in enumerate(job.get("results") or []):
                if result["ok"]:
                    st.video(result["path"])
                    st.caption(f"Variation {i + 1} · {result['elapsed']:.0f}s · `{result['path']}`")
                    st.download_button(
                        "⬇️ Download MP4",
                        data=Path(result["path"]).read_bytes(),
                        file_name=Path(result["path"]).name,
                        mime="video/mp4",
                        key=f"dl_{job['id']}_{i}",
                    )
                    if result["text"]:
                        st.caption(f"Model notes: {result['text'][:300]}")
                else:
                    st.error(f"Variation {i + 1} failed: {result['error']}")
