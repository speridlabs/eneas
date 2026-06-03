import os

import spaces  # HF ZeroGPU — must be imported before torch.

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server
from gradio.data_classes import FileData
from hydra.core.global_hydra import GlobalHydra
from PIL import Image

load_dotenv()

# ===========================================
# LOGGING
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===========================================
# PATHS
# ===========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from eneas.segmentation import UniqueInstanceSegmenter, GenericCategorySegmenter
    from eneas.segmentation.model_manager import ModelManager
except ImportError as e:
    logger.error(f"Error importing ENEAS: {e}")
    raise e


# ===========================================
# HYDRA CONTEXT MANAGER
# ===========================================
@contextmanager
def hydra_environment(config_module: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    try:
        logger.info(f"Initializing Hydra for module: {config_module}")
        hydra.initialize_config_module(config_module=config_module, version_base="1.2")
        yield
    except Exception as e:
        logger.error(f"Hydra initialization error: {e}")
        raise e
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()


# ===========================================
# CONSTANTS & ENV
# ===========================================
SPACE_SAMPLING = "1 FPS (Space limit)"
# Locally the user picks the sampling rate and there is no frame cap.
# The HF Space sets SAMPLING_LOCKED = True (forces 1 FPS) and MAX_FRAMES = 150.
SAMPLING_LOCKED = True
MAX_FRAMES = 150
PREFETCH_MODELS = True
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"
OLLAMA_BIN = os.getenv("OLLAMA_BIN", os.path.join(project_root, "bin", "ollama"))
OLLAMA_LOG_PATH = os.path.join(tempfile.gettempdir(), "ollama_serve.log")
HF_READY_FLAG = os.path.join(tempfile.gettempdir(), "eneas_hf_ready")
OLLAMA_READY_FLAG = os.path.join(tempfile.gettempdir(), "eneas_ollama_ready")

OUTPUT_BASE_DIR = str(Path(current_dir) / "gradio_outputs")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

VLM_MODELS = [
    "qwen3-vl:4b-instruct-q8_0",
    "qwen3-vl:2b-instruct-q8_0"
]


class CompletedStartupTask:
    def is_alive(self):
        return False

    def join(self):
        return None


# ===========================================
# OLLAMA FUNCTIONS (run inside the GPU worker)
# ===========================================
def get_ollama_env():
    """Get environment variables for Ollama process with GPU support."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_ORIGINS"] = "*"
    env["HOME"] = os.getcwd()

    # Add local lib path for the extracted binary
    cwd = os.getcwd()
    lib_path = f"{cwd}/lib"
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] += f":{lib_path}"
    else:
        env["LD_LIBRARY_PATH"] = lib_path

    return env


def is_ollama_server_running() -> bool:
    """Check if Ollama server is responding."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", OLLAMA_URL],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() == "200"
    except Exception:
        return False


def start_ollama_server_gpu():
    """
    Start Ollama server inside the GPU worker.
    This ensures Ollama detects and uses the GPU.

    Returns:
        bool: True if server started successfully
    """
    if is_ollama_server_running():
        logger.info("Ollama server is already running.")
        return True

    logger.info("Starting Ollama server inside GPU context...")

    try:
        env = get_ollama_env()

        # Start server as background process; capture its logs to a file so we
        # can inspect GPU detection / layer offload decisions on failure.
        ollama_log = open(OLLAMA_LOG_PATH, "w")
        process = subprocess.Popen(
            [OLLAMA_BIN, "serve"],
            env=env,
            stdout=ollama_log,
            stderr=subprocess.STDOUT
        )

        # Wait for server to be ready (max 30 seconds)
        max_retries = 30
        for i in range(max_retries):
            if is_ollama_server_running():
                logger.info(f"Ollama server started successfully in {i+1} seconds.")
                return True
            time.sleep(1)

        logger.error("Ollama server failed to start within 30 seconds.")
        return False

    except Exception as e:
        logger.error(f"Failed to start Ollama server: {e}")
        return False


def load_model_into_vram(model_name: str) -> bool:
    """
    Load model into VRAM for faster inference.
    Uses keep_alive=-1 to keep model loaded.

    Args:
        model_name: Name of the Ollama model to load

    Returns:
        bool: True if model loaded successfully
    """
    logger.info(f"Loading model {model_name} into VRAM (num_ctx=8192)...")
    t0 = time.time()

    try:
        # Send a minimal request to trigger model loading
        result = subprocess.run(
            [
                "curl", "-s", f"{OLLAMA_URL}/api/generate",
                "-d", f'{{"model": "{model_name}", "prompt": "hi", "stream": false, "options": {{"num_ctx": 8192}}}}'
            ],
            capture_output=True,
            text=True,
            timeout=240  # Model load into VRAM on ZeroGPU can be slow on first call
        )
        logger.info(f"VLM /api/generate returned in {time.time()-t0:.1f}s.")

        if "error" in result.stdout.lower():
            logger.error(f"Error loading model: {result.stdout}")
            dump_ollama_log("VLM load error")
            return False

        # Set keep_alive to -1 to keep model in VRAM
        subprocess.run(
            [
                "curl", "-s", f"{OLLAMA_URL}/api/generate",
                "-d", f'{{"model": "{model_name}", "keep_alive": -1, "options": {{"num_ctx": 8192}}}}'
            ],
            capture_output=True,
            timeout=10
        )

        logger.info(f"Model {model_name} loaded into VRAM successfully in {time.time()-t0:.1f}s.")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while loading model into VRAM after {time.time()-t0:.1f}s.")
        dump_ollama_log("VLM load timeout")
        return False
    except Exception as e:
        logger.error(f"Error loading model into VRAM: {e}")
        dump_ollama_log("VLM load exception")
        return False


def log_active_models():
    """Log which models are currently loaded in VRAM (not just on disk)."""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{OLLAMA_URL}/api/ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        logger.info(f"Active models in VRAM: {result.stdout}")
    except Exception as e:
        logger.warning(f"Could not get active models: {e}")


def dump_ollama_log(tag: str, n: int = 60):
    """Log the tail of the Ollama server log (shows GPU detection / offload)."""
    try:
        with open(OLLAMA_LOG_PATH) as f:
            tail = f.readlines()[-n:]
        logger.warning(f"[{tag}] ollama server log (last {len(tail)} lines):\n" + "".join(tail))
    except Exception as e:
        logger.warning(f"[DEBUG {tag}] could not read ollama log: {e}")


def wait_for_startup_flag(path: str, label: str, timeout: int = 600):
    """Wait for a startup-ready sentinel file. Files survive the ZeroGPU fork
    (threads do NOT), so this replaces the ineffective t_*.join(). No-op locally
    (PREFETCH_MODELS False), where models load on demand."""
    if not PREFETCH_MODELS:
        return
    t0 = time.time()
    while not os.path.exists(path) and time.time() - t0 < timeout:
        logger.info(f"Waiting for {label} startup to finish...")
        time.sleep(3)


def ensure_model_pulled(model_name: str) -> bool:
    """Make sure the model is on disk before loading. The startup pull runs in a
    main-process thread, but ZeroGPU forks the worker (threads don't survive the
    fork) so t_ollama.join() can't guarantee it. /api/pull is idempotent: fast if
    the model is already present, waits/dedupes if a startup pull is in flight."""
    try:
        r = subprocess.run(
            ["curl", "-s", f"{OLLAMA_URL}/api/pull",
             "-d", f'{{"model": "{model_name}", "stream": false}}'],
            capture_output=True, text=True, timeout=600,
        )
        if '"error"' in r.stdout.lower():
            logger.error(f"Pull failed for {model_name}: {r.stdout[:300]}")
            return False
        logger.info(f"Model {model_name} present/pulled.")
        return True
    except Exception as e:
        logger.warning(f"Pull check failed for {model_name}: {e}")
        return False


def ensure_ollama_ready_gpu(model_name: str) -> bool:
    """
    Main function to ensure Ollama is fully ready with GPU support.
    MUST be called inside the GPU worker.

    This function:
    1. Starts Ollama server (which will detect GPU)
    2. Loads the specified model into VRAM
    3. Logs which model is active

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        bool: True if ready

    Raises:
        RuntimeError: If setup fails
    """
    logger.info(f"Ensuring Ollama is ready with GPU for model: {model_name}")

    # Step 1: Start server (will detect GPU since we're inside the GPU worker)
    if not start_ollama_server_gpu():
        raise RuntimeError("Failed to start Ollama server with GPU")

    # Step 1b: ensure the model is on disk (cold-start race: the startup pull
    # thread does NOT survive the ZeroGPU fork, so don't rely on t_ollama.join()).
    if not ensure_model_pulled(model_name):
        raise RuntimeError(f"Failed to pull model {model_name}")

    # Step 2: Load model into VRAM
    if not load_model_into_vram(model_name):
        raise RuntimeError(f"Failed to load model {model_name} into VRAM")

    # Step 3: Log which model is actually active in VRAM
    log_active_models()

    logger.info("Ollama is ready with GPU support!")
    return True


# ===========================================
# STARTUP: DOWNLOAD BINARY AND MODELS (CPU)
# ===========================================
OLLAMA_VERSION = "0.30.2"  # pinned for reproducible Space builds; no effect locally (only downloaded on the Space, with ENEAS_PREFETCH_MODELS=1)


def download_ollama_binary():
    """Download Ollama binary if not present."""
    if os.path.exists(OLLAMA_BIN):
        logger.info("Ollama binary already exists.")
        return True

    logger.info("Downloading Ollama binary (ZST)...")
    try:
        subprocess.run(
            ["curl", "-L", f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst", "-o", "ollama.tar.zst"],
            check=True,
            timeout=300
        )
        subprocess.run(["tar", "--zstd", "-xf", "ollama.tar.zst"], check=True)
        subprocess.run(["chmod", "+x", OLLAMA_BIN], check=True)
        os.remove("ollama.tar.zst")  # Cleanup
        logger.info("Ollama binary downloaded and extracted successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download Ollama binary: {e}")
        return False


def pull_ollama_models():
    """
    Pull Ollama models at startup (runs on CPU).
    This pre-downloads the models so they're ready when GPU is available.
    """
    logger.info("Pre-downloading Ollama models...")

    # Need to temporarily start server to pull models
    env = get_ollama_env()

    # Start server temporarily
    server_process = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server
    time.sleep(5)
    for _ in range(20):
        if is_ollama_server_running():
            break
        time.sleep(1)

    # Pull each model
    for model in VLM_MODELS:
        logger.info(f"Pulling model: {model}")
        try:
            subprocess.run(
                [OLLAMA_BIN, "pull", model],
                env=env,
                timeout=600,
                capture_output=True
            )
            logger.info(f"Model {model} pulled successfully.")
        except Exception as e:
            logger.warning(f"Failed to pull model {model}: {e}")

    # Stop server (we'll restart it inside GPU context later)
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

    logger.info("Ollama models pre-download complete.")


def setup_ollama_startup():
    """Setup Ollama at startup: download binary and pull models."""
    try:
        download_ollama_binary()
        pull_ollama_models()
    finally:
        open(OLLAMA_READY_FLAG, "w").close()


def setup_hf_models():
    """
    Downloads heavy HuggingFace models to disk at startup.
    This prevents ZeroGPU timeouts during the first inference.
    """
    logger.info("Starting HuggingFace models download (Warm-up)...")
    try:
        manager = ModelManager()

        # 1. SeC-4B (Heavy, ~15GB)
        logger.info("Downloading SeC-4B...")
        manager.download("OpenIXCLab/SeC-4B")

        # 2. Florence-2 (Grounding)
        logger.info("Downloading Florence-2...")
        manager.download("microsoft/Florence-2-large")

        # 3. SigLIP (For Generic Category)
        logger.info("Downloading SigLIP...")
        manager.download("google/siglip2-base-patch16-naflex")

        # 4. SAM2 Checkpoint (Direct URL)
        logger.info("Downloading SAM2 checkpoint...")
        manager.download_url(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            "sam2.1_hiera_large.pt"
        )

        logger.info("All HuggingFace models downloaded successfully.")
    except Exception as e:
        logger.error(f"Error during HF model download: {e}")
    finally:
        open(HF_READY_FLAG, "w").close()


# ===========================================
# STARTUP: PARALLEL MODEL DOWNLOADS
# ===========================================
if not PREFETCH_MODELS:
    logger.info("Skipping background model downloads. Set ENEAS_PREFETCH_MODELS=1 to prefetch at startup.")
    t_hf = CompletedStartupTask()
    t_ollama = CompletedStartupTask()
else:
    logger.info("Starting parallel model downloads at startup...")
    for _flag in (HF_READY_FLAG, OLLAMA_READY_FLAG):
        try:
            os.remove(_flag)
        except FileNotFoundError:
            pass
    t_hf = threading.Thread(target=setup_hf_models, daemon=True)
    t_ollama = threading.Thread(target=setup_ollama_startup, daemon=True)

    t_hf.start()
    t_ollama.start()

# NOTE: never touch torch.cuda at module level. On ZeroGPU the GPU is only
# attached inside @spaces.GPU workers; initializing CUDA in the main process
# corrupts NVML and crashes later .to(cuda) calls. Use pick_device() at call time.


def pick_device() -> str:
    """Resolve the device at call time. On HF ZeroGPU the GPU is only attached
    inside the worker, so module-level detection would wrongly read 'cpu' —
    always detect when segmentation actually runs."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================
# FRAME PROCESSING
# ===========================================
def target_fps_from_sampling(sampling_mode: str) -> float | None:
    if not sampling_mode or sampling_mode == SPACE_SAMPLING:
        return None
    if sampling_mode.endswith(" FPS"):
        try:
            return float(sampling_mode.split(" ", 1)[0])
        except ValueError:
            return None
    return None


def resolve_target_fps(sampling_mode: str, detected_fps: float) -> float | None:
    """None → keep every frame (Native). A number → resample to that FPS.
    Unknown / Space limit → 1 FPS."""
    if sampling_mode and sampling_mode.lower().startswith("native"):
        return None
    fps = target_fps_from_sampling(sampling_mode)
    return 1.0 if fps is None else fps


def sampling_widget_html(prefix: str) -> str:
    """Per-environment frame-sampling control injected into index.html."""
    if SAMPLING_LOCKED:
        return '<div class="sampling-note">1 FPS &middot; up to 150 frames.</div>'
    return (
        f'<select id="{prefix}-sampling" class="sampling-select" aria-label="Frame sampling">'
        '<option value="Native FPS" selected>Native FPS</option>'
        '<option value="10 FPS">10 FPS</option>'
        '<option value="5 FPS">5 FPS</option>'
        '<option value="2 FPS">2 FPS</option>'
        '</select>'
    )


def sampling_metadata(
    sampling_mode: str,
    source_fps: float,
    output_fps: float,
    source_frames: int,
    saved_frames: int,
    is_video: bool,
) -> dict:
    return {
        "sampling_mode": sampling_mode or SPACE_SAMPLING,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "source_frames": source_frames,
        "saved_frames": saved_frames,
        "is_video": is_video,
        "sampled": saved_frames < source_frames,
    }


def frame_extraction_message(meta: dict) -> str:
    saved = meta["saved_frames"]
    output_fps = meta["output_fps"]
    if meta["sampled"]:
        source = meta["source_frames"]
        if meta["is_video"]:
            return f"Using first {saved:,} sampled frames from {source:,} source frames · {output_fps:.2f} FPS."
        return f"Using first {saved:,} images from {source:,} uploaded files."
    if meta["is_video"]:
        return f"{saved:,} frames extracted at {output_fps:.2f} FPS."
    return f"{saved:,} images loaded."


def uploaded_file_path(file_obj: Any) -> str:
    """Resolve Gradio Blocks uploads, @gradio/client FileData, and raw paths."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name") or ""
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None) or str(file_obj)


def uploaded_file_name(file_obj: Any) -> str:
    """Return the user-facing/original filename when Gradio keeps it separate."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return os.path.basename(file_obj)
    if isinstance(file_obj, dict):
        return (
            file_obj.get("orig_name")
            or os.path.basename(file_obj.get("name") or "")
            or os.path.basename(file_obj.get("path") or "")
        )
    return (
        getattr(file_obj, "orig_name", None)
        or os.path.basename(getattr(file_obj, "name", None) or "")
        or os.path.basename(getattr(file_obj, "path", None) or "")
        or str(file_obj)
    )


def uploaded_file_mime(file_obj: Any) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, dict):
        return file_obj.get("mime_type") or ""
    return getattr(file_obj, "mime_type", None) or ""


def uploaded_file_is_video(file_obj: Any, video_extensions: set[str]) -> bool:
    mime = uploaded_file_mime(file_obj).lower()
    if mime.startswith("video/"):
        return True
    candidates = (uploaded_file_name(file_obj), uploaded_file_path(file_obj))
    return any(os.path.splitext(candidate)[1].lower() in video_extensions for candidate in candidates if candidate)


def file_content_looks_like_video(path: str) -> bool:
    """Fallback for browser uploads that arrive without filename or mime."""
    if not path or not os.path.exists(path):
        return False
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False
        ret_first, _ = cap.read()
        if not ret_first:
            return False
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count and not np.isnan(frame_count) and frame_count > 1:
            return True
        ret_second, _ = cap.read()
        return bool(ret_second)
    finally:
        cap.release()


def uploaded_file_sort_key(file_obj: Any) -> str:
    return uploaded_file_name(file_obj) or uploaded_file_path(file_obj)


def normalize_uploaded_files(input_data: Any) -> list:
    if input_data is None:
        return []
    if isinstance(input_data, (list, tuple)):
        return [f for f in input_data if f is not None]
    return [input_data]


def generic_completion_message(detections: int, frames: int, meta: dict) -> str:
    if meta["sampled"]:
        if meta["is_video"]:
            return (
                f"Completed · {detections:,} detections across first {frames:,} sampled frames "
                f"from {meta['source_frames']:,} source frames."
            )
        return (
            f"Completed · {detections:,} detections across first {frames:,} images "
            f"from {meta['source_frames']:,} uploaded files."
        )
    return f"Completed · {detections:,} detections across {frames:,} frames."


def process_inputs_to_frames(input_data, output_folder: str, sampling_mode: str = SPACE_SAMPLING) -> tuple:
    """
    Extract frames for the Hugging Face Space.

    Videos are sampled at 1 FPS and limited to MAX_FRAMES sampled frames
    to keep demo runtime predictable. Image uploads keep their sequence
    order and use the same maximum.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    if SAMPLING_LOCKED:
        sampling_mode = SPACE_SAMPLING

    frame_paths = []
    detected_fps = 30.0
    output_fps = 1.0
    source_frames = 0
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    input_list = normalize_uploaded_files(input_data)
    if not input_list:
        meta = sampling_metadata(SPACE_SAMPLING, detected_fps, output_fps, 0, 0, False)
        return output_folder, [], output_fps, meta

    first_input = input_list[0]
    first_file = uploaded_file_path(first_input)
    is_video = uploaded_file_is_video(first_input, video_extensions) or file_content_looks_like_video(first_file)
    logger.info(
        "Resolved input: path=%s name=%s mime=%s is_video=%s",
        first_file,
        uploaded_file_name(first_input),
        uploaded_file_mime(first_input),
        is_video,
    )

    if is_video:
        cap = cv2.VideoCapture(first_file)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_original = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if video_fps > 0 and not np.isnan(video_fps):
            detected_fps = video_fps
        if total_frames_original > 0 and not np.isnan(total_frames_original):
            source_frames = int(total_frames_original)

        target_fps = resolve_target_fps(sampling_mode, detected_fps)
        if target_fps is None:
            frame_interval = 1
            output_fps = detected_fps
        else:
            frame_interval = max(1, int(round(detected_fps / target_fps)))
            output_fps = float(target_fps)
        logger.info(f"Sampling '{sampling_mode}': {detected_fps:.2f} FPS source -> interval {frame_interval}, output {output_fps:.2f} FPS")
        source_count = 0
        saved_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if source_count % frame_interval == 0:
                filename = f"frame_{saved_count:05d}.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, frame)
                frame_paths.append(filepath)
                saved_count += 1

                if saved_count >= MAX_FRAMES:
                    logger.warning(f"Max frame limit ({MAX_FRAMES}) reached.")
                    break

            source_count += 1

        cap.release()
        source_frames = source_frames or source_count
    else:
        detected_fps = 5.0
        output_fps = detected_fps
        input_list.sort(key=uploaded_file_sort_key)
        source_frames = len(input_list)
        for i, f in enumerate(input_list[:MAX_FRAMES]):
            path = uploaded_file_path(f)
            try:
                img = Image.open(path).convert("RGB")
                filename = f"frame_{i:05d}.jpg"
                filepath = os.path.join(output_folder, filename)
                img.save(filepath)
                frame_paths.append(filepath)
            except Exception as e:
                logger.warning(f"Skipping file {path}: {e}")

    meta = sampling_metadata(
        sampling_mode,
        detected_fps,
        output_fps,
        source_frames,
        len(frame_paths),
        is_video,
    )
    return output_folder, frame_paths, output_fps, meta


def transcode_to_h264(src: str, dst: str) -> bool:
    """Re-encode a video to browser-playable H.264 (yuv420p) via ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", src,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                dst,
            ],
            check=True,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"H.264 transcode failed ({e}); falling back to raw output.")
        return False


def create_video_overlay(frames_folder: str, masks_dict: dict, output_path: str, fps: float) -> str:
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    if not frame_files:
        return None

    first = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    h, w, _ = first.shape

    logger.info(f"Constructing output video at {fps} FPS...")

    # OpenCV can only encode MPEG-4 Part 2 here, which browsers won't play —
    # write a raw file then transcode to H.264 below.
    raw_path = f"{output_path}.raw.mp4"
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    mask_color = np.array([255, 100, 0], dtype=np.uint8)

    for i, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frames_folder, fname))
        mask_overlay = np.zeros_like(frame)

        if i in masks_dict:
            masks = masks_dict[i]
            if isinstance(masks, np.ndarray):
                mask_overlay[masks > 0] = mask_color
            elif isinstance(masks, list):
                for m in masks:
                    mask_overlay[m > 0] = mask_color

        if np.any(mask_overlay):
            frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)
        out.write(frame)

    out.release()

    if transcode_to_h264(raw_path, output_path):
        os.remove(raw_path)
    else:
        shutil.move(raw_path, output_path)
    return output_path


@spaces.GPU(duration=180)
def run_unique_segmentation(input_files, points, prompt, encoder_size, offload, frame_idx, frames_dir_cache, original_fps):
    if not frames_dir_cache or not os.path.exists(frames_dir_cache):
        if not input_files:
            return None, status_html("idle", "Process input first."), unique_stats_html()
        temp_dir = tempfile.mkdtemp()
        frames_dir_cache, _, original_fps, _ = process_inputs_to_frames(input_files, temp_dir, SPACE_SAMPLING)

    wait_for_startup_flag(HF_READY_FLAG, "Hugging Face models")

    sec_config_module = "eneas.vendor.SeC.inference.sam2.configs"

    try:
        with hydra_environment(sec_config_module):
            device = pick_device()
            logger.info(f"Starting Unique Segmentation on {device}...")

            segmenter = UniqueInstanceSegmenter(
                sam_encoder=encoder_size,
                device=device
            )

            annotation_frame = f"frame_{int(frame_idx):05d}.jpg"

            t_start = time.time()
            if points:
                result = segmenter.segment(
                    frames_path=frames_dir_cache,
                    points=points,
                    annotation_frame=annotation_frame,
                    offload_frames_to_gpu=offload
                )
            elif prompt and prompt.strip():
                result = segmenter.segment(
                    frames_path=frames_dir_cache,
                    text=prompt,
                    annotation_frame=annotation_frame,
                    offload_frames_to_gpu=offload
                )
            else:
                return None, status_html("idle", "Provide points (click image) or text."), unique_stats_html()
            elapsed = time.time() - t_start

            out_vid = os.path.join(OUTPUT_BASE_DIR, f"unique_result_{int(time.time() * 1000)}.mp4")
            final_path = create_video_overlay(frames_dir_cache, result.masks, out_vid, fps=original_fps)
            mask_count = len(result.masks)
            return (
                final_path,
                status_html("ok", f"Completed · {result.num_frames} frames segmented."),
                unique_stats_html(result.num_frames, mask_count, elapsed),
            )

    except Exception as e:
        logger.error(f"Error in unique seg: {e}")
        import traceback
        traceback.print_exc()
        return None, status_html("err", str(e)), unique_stats_html()


@spaces.GPU(duration=300)
def run_generic_segmentation(input_files, category, accept, reject, vlm_model, sampling_mode):
    if not input_files:
        return None, status_html("idle", "No input."), stats_html()
    if not category or not category.strip():
        return None, status_html("idle", "Specify a category."), stats_html()

    wait_for_startup_flag(HF_READY_FLAG, "Hugging Face models")
    wait_for_startup_flag(OLLAMA_READY_FLAG, "Ollama setup")

    sam2_config_module = "eneas.vendor.sam2"

    try:
        ensure_ollama_ready_gpu(vlm_model)

        temp_dir = tempfile.mkdtemp()
        frames_dir, frame_paths, output_fps, meta = process_inputs_to_frames(input_files, temp_dir, SPACE_SAMPLING)
        n_frames = len(frame_paths)

        with hydra_environment(sam2_config_module):
            logger.info(f"Starting Generic Segmentation with {vlm_model}...")
            segmenter = GenericCategorySegmenter(
                device=pick_device(),
                vlm_model=vlm_model
            )

            t_start = time.time()
            result = segmenter.segment(
                frames_path=frames_dir,
                category=category,
                accept_threshold=accept,
                reject_threshold=reject
            )
            elapsed = time.time() - t_start

            out_vid = os.path.join(OUTPUT_BASE_DIR, f"generic_result_{int(time.time() * 1000)}.mp4")
            final_path = create_video_overlay(frames_dir, result.masks, out_vid, fps=output_fps)

            count = sum(len(v) for v in result.metadata['detections'].values())

            return (
                final_path,
                status_html("ok", generic_completion_message(count, n_frames, meta)),
                stats_html(n_frames, count, elapsed),
            )

    except Exception as e:
        logger.error(f"Error generic seg: {e}")
        dump_ollama_log("generic error")
        import traceback
        traceback.print_exc()
        return None, status_html("err", str(e)), stats_html()


# ===========================================
# HTML SNIPPETS (status/stats)
# ===========================================
def status_html(state: str, msg: str) -> str:
    """state: 'idle' | 'live' | 'ok' | 'err'."""
    dot_cls = state if state in ("idle", "live", "ok") else "idle"
    color_override = ' style="background:var(--sl-signal)"' if state == "err" else ""
    safe = (msg or "").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="status-row">'
        f'<span class="status-dot {dot_cls}"{color_override}></span>'
        f'<span class="status-msg">{safe}</span>'
        f'<span class="status-time">{time.strftime("%H:%M:%S")}</span>'
        f'</div>'
    )


def _latency_cell(frames: int, secs: float) -> str:
    lat = (secs / frames) if frames > 0 else 0.0
    return f'<div class="cell"><div class="k">Latency / frame</div><div class="v">{lat:.3f} s</div></div>'


def stats_html(frames: int = 0, dets: int = 0, secs: float = 0.0) -> str:
    return (
        f'<div class="stats">'
        f'<div class="cell"><div class="k">Frames</div><div class="v">{frames}</div></div>'
        f'<div class="cell"><div class="k">Detections</div><div class="v">{dets}</div></div>'
        f'{_latency_cell(frames, secs)}'
        f'<div class="cell"><div class="k">Inference</div><div class="v">{secs:.1f} s</div></div>'
        f'</div>'
    )


def unique_stats_html(frames: int = 0, masks: int = 0, secs: float = 0.0) -> str:
    return (
        f'<div class="stats">'
        f'<div class="cell"><div class="k">Frames</div><div class="v">{frames}</div></div>'
        f'<div class="cell"><div class="k">Masks</div><div class="v">{masks}</div></div>'
        f'{_latency_cell(frames, secs)}'
        f'<div class="cell"><div class="k">Inference</div><div class="v">{secs:.1f} s</div></div>'
        f'</div>'
    )


# ===========================================
# CUSTOM GRADIO SERVER APP
# ===========================================
def file_data(path: str | None) -> FileData | None:
    if not path or not os.path.exists(path):
        return None
    return FileData(path=path)


def normalize_points(points: Any) -> list[tuple[int, int]]:
    if not points:
        return []
    cleaned = []
    for point in points:
        if not point or len(point) < 2:
            continue
        try:
            cleaned.append((int(round(float(point[0]))), int(round(float(point[1])))))
        except (TypeError, ValueError):
            continue
    return cleaned


app = Server()

assets_dir = Path(current_dir) / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

outputs_dir = Path(current_dir) / OUTPUT_BASE_DIR
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.get("/")
async def homepage():
    html_path = Path(current_dir) / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html is missing</h1>", status_code=500)
    html = html_path.read_text(encoding="utf-8")
    # Cache-bust static assets by file mtime so browsers always pick up the
    # latest CSS/JS after a deploy (no stale-cache surprises).
    for rel in ("eneas.css", "eneas.js"):
        asset = assets_dir / rel
        if asset.exists():
            html = html.replace(f"/assets/{rel}", f"/assets/{rel}?v={int(asset.stat().st_mtime)}")
    html = html.replace("<!--SAMPLING:unique-->", sampling_widget_html("unique"))
    html = html.replace("<!--SAMPLING:generic-->", sampling_widget_html("generic"))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.api(name="extract_unique")
def extract_unique_api(input_files: Any = None, sampling: str = SPACE_SAMPLING) -> tuple[FileData | None, dict]:
    if not input_files:
        return None, {
            "success": False,
            "status_html": status_html("idle", "No files uploaded yet."),
            "stats_html": unique_stats_html(),
        }

    temp_dir = tempfile.mkdtemp()
    frames_dir, frame_paths, output_fps, meta = process_inputs_to_frames(input_files, temp_dir, sampling)
    if not frame_paths:
        return None, {
            "success": False,
            "frames_dir": frames_dir,
            "frame_count": 0,
            "fps": output_fps,
            "status_html": status_html("idle", "No frames extracted."),
            "stats_html": unique_stats_html(),
        }

    return file_data(frame_paths[0]), {
        "success": True,
        "frames_dir": frames_dir,
        "frame_count": len(frame_paths),
        "fps": output_fps,
        "status_html": status_html("ok", frame_extraction_message(meta)),
        "stats_html": unique_stats_html(),
    }


@app.api(name="reference_frame")
def reference_frame_api(frames_dir: str, frame_idx: int = 0) -> tuple[FileData | None, dict]:
    if not frames_dir:
        return None, {"success": False, "error": "Extract frames first."}
    frame_idx = max(0, int(frame_idx or 0))
    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
    if not os.path.exists(frame_path):
        return None, {"success": False, "error": "Reference frame not found."}
    return file_data(frame_path), {"success": True, "frame_idx": frame_idx}


@app.api(name="preview_generic")
def preview_generic_api(input_files: Any = None) -> tuple[FileData | None, dict]:
    """First-frame thumbnail for the Generic input, extracted server-side with
    OpenCV so it works for any codec the browser can't decode (mpeg4, HEVC...)."""
    input_list = normalize_uploaded_files(input_files)
    if not input_list:
        return None, {"success": False}
    first = input_list[0]
    path = uploaded_file_path(first)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    is_video = uploaded_file_is_video(first, video_extensions) or file_content_looks_like_video(path)
    if not is_video:
        return file_data(path), {"success": True}
    try:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None, {"success": False, "error": "Could not read first frame."}
        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "generic_preview.jpg")
        cv2.imwrite(out_path, frame)
        return file_data(out_path), {"success": True}
    except Exception as e:
        logger.error(f"Generic preview extraction failed: {e}")
        return None, {"success": False, "error": str(e)}


@app.api(name="run_unique")
def run_unique_api(
    input_files: Any = None,
    points: Any = None,
    prompt: str = "",
    encoder_size: str = "long-large",
    offload: bool = False,
    frame_idx: int = 0,
    frames_dir_cache: str | None = None,
    original_fps: float = 1.0,
) -> tuple[FileData | None, dict]:
    video_path, status, stats = run_unique_segmentation(
        input_files,
        normalize_points(points),
        prompt,
        encoder_size or "long-large",
        bool(offload),
        int(frame_idx or 0),
        frames_dir_cache,
        float(original_fps or 1.0),
    )
    return file_data(video_path), {
        "success": bool(video_path),
        "status_html": status,
        "stats_html": stats,
    }


@app.api(name="run_generic")
def run_generic_api(
    input_files: Any = None,
    category: str = "",
    accept: float = 0.30,
    reject: float = 0.10,
    vlm_model: str = VLM_MODELS[0],
    sampling: str = SPACE_SAMPLING,
) -> tuple[FileData | None, dict]:
    video_path, status, stats = run_generic_segmentation(
        input_files,
        category,
        float(accept),
        float(reject),
        vlm_model or VLM_MODELS[0],
        sampling,
    )
    return file_data(video_path), {
        "success": bool(video_path),
        "status_html": status,
        "stats_html": stats,
    }


if __name__ == "__main__":
    app.launch(show_error=True)
