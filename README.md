# devpunya-video-generation-pipeline

Automated video content generation pipeline built with Streamlit and Google Vertex AI
(`gemini-omni-flash-preview` via the interactions API).

Queue multiple jobs — each a text prompt plus an optional reference image — and
batch-generate videos. Generated MP4s are saved to `generated_videos/` and shown
inline with download buttons.

## Features

- Single and batch job entry (one prompt per line, images matched in order)
- Text-to-video and image-to-video (task auto-selected per job)
- N variations per prompt, configurable parallel workers
- Automatic retry with backoff on per-minute quota (429) errors
- Fallback to the minimal request shape if the API rejects extra parameters

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Vertex AI auth (bills your GCP project)
gcloud auth application-default login
```

The GCP project needs the Vertex AI API enabled:

```bash
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

## Run

```bash
.venv/bin/streamlit run app.py
```

Then open http://localhost:8501, set your GCP Project ID and Location in the
sidebar, queue prompts (with optional reference images), and hit Generate.

## Tabs

### 🎬 Single Video / Frame Generation

The original flow: queue text (+ optional reference image) prompts and
batch-generate videos or images.

### 🎙️ Video from Script Narration

Turn a full Hindi script into one narrated film:

1. **Text-to-speech** — ElevenLabs (`eleven_multilingual_v2`, speaks Hindi).
   Pick a Male/Female Indian voice (target 35–55 y/o) from your ElevenLabs
   voice library, or paste a voice ID. The API key is stored locally in
   `pipeline_settings.json` (or a `elevenlabs_api_key` Streamlit secret).
2. **Clip plan** — number of clips = `RoundUp(audio seconds / 10)`; each clip
   is at most 10s (e.g. a 53s narration → 6 clips).
3. **Agent 1 (script breaker)** — breaks the script into exactly N frame
   definitions plus a locked *style bible* (tone/theme, visual style, color
   grade, setting, character/persona descriptions). Optimized for extremely
   high hook rate and view-through rate. Saved as
   `frame_XX_definition.md` + `style_bible.md`.
4. **Agent 2 (prompt writer)** — runs separately for each frame definition and
   writes a detailed video-generation prompt that restates the locked
   character descriptions and continuity from the previous frame. Saved as
   `frame_XX_prompt.md`.
5. **Generation** — one video job per frame *prompt* file (sequential, quota
   friendly), then all clips are stitched with ffmpeg and the **same
   narration audio** is laid over the full film (`-shortest` trims the video
   to the exact audio length).
6. **Output** — final MP4 available as a local download and (if enabled)
   uploaded to Google Drive.

Consistency guarantees across frames: identical tone/theme, locked
personas/characters reused verbatim, explicit frame-to-frame continuity, one
shared audio track.

The whole pipeline runs in the background worker — refresh-safe, and a failed
project can be retried and resumes from the step where it stopped. All
artifacts live in `generated_videos/narration_<job id>/`.
