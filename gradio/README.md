---
title: ENEAS
emoji: ⚡
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 6.0.2
app_file: spaces.py
pinned: false
---

# ENEAS — Gradio demo

This folder contains the Gradio demo for **ENEAS** (Embedding-guided Neural Ensemble for Adaptive Segmentation).


### Prerequisites
* **Python 3.10 - 3.12**
* **Ollama**: Recommended to install system-wide from [ollama.com](https://ollama.com). If not installed, the script will attempt to use a local binary if provided in `bin/`.
* **FFmpeg**: Required for video processing. 

This packages should be installed in the system `ffmpeg` `libgl1` defined in `packages.txt` and `ollama`

---

## Install and run gradio demo

### uv
```bash
uv sync --extras ui
uv run demo_gradio.py
```

### pip and venv

```bash
# 1. From repo root
# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the project in editable mode + UI extras (gradio/spaces)
pip install -e .[ui]
python demo_gradio.py
```

---

## Environment variables the demo uses
You can found in `.env.sample` the environment variables needed when running the demo.
```bash
# Ollama Configuration 
# Host where Ollama is running the VLM. 
# OLLAMA_HOST=127.0.0.1:11434

# If you want the app to use a specific binary, set the full path here.
# OLLAMA_BIN=./bin/ollama

#  Hugging Face
# require authentication for model access, provide your token here.
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

You should do (gradio demo automatically loads .env file vars)
```bash
cp .env.samples .env
```
