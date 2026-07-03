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
