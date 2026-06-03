import { client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

let clientInstance = null;

const state = {
  unique: {
    files: [],
    framesDir: null,
    frameCount: 0,
    fps: 1.0,
    frameIdx: 0,
    points: []
  },
  generic: {
    files: []
  }
};

const $ = (id) => document.getElementById(id);

function samplingMode(prefix) {
  const el = $(`${prefix}-sampling`);
  return el ? el.value : "1 FPS (Space limit)";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nowTime() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function statusMarkup(stateName, message) {
  const dot = stateName === "err" ? "idle" : stateName;
  const color = stateName === "err" ? ' style="background:var(--sl-signal)"' : "";
  return `
    <div class="status-row">
      <span class="status-dot ${dot}"${color}></span>
      <span class="status-msg">${escapeHtml(message)}</span>
      <span class="status-time">${nowTime()}</span>
    </div>
  `;
}

function setStatus(prefix, stateName, message) {
  $(`${prefix}-status`).innerHTML = statusMarkup(stateName, message);
}

function setBackendHtml(prefix, meta) {
  if (meta?.status_html) {
    $(`${prefix}-status`).innerHTML = meta.status_html;
  }
  if (meta?.stats_html) {
    $(`${prefix}-stats`).innerHTML = meta.stats_html;
  }
}

async function extractGenericServerPreview(file) {
  const img = $("generic-input-img");
  const empty = $("generic-result-empty");
  try {
    const api = await getClient();
    const result = await api.predict("/preview_generic", { input_files: wrapFiles([file]) });
    const [frameFile, meta] = result.data;
    const url = fileUrl(frameFile);
    if (meta && meta.success && url) {
      img.onload = () => empty.classList.add("hidden");
      img.src = url;
      img.classList.remove("hidden");
      empty.classList.add("hidden");
      return;
    }
  } catch (e) {
    console.error("Server preview failed:", e);
  }
  empty.textContent = `${file.name} — preview not available (it will still be processed).`;
  empty.classList.remove("hidden");
}

function setGenericPreview(file) {
  const video = $("generic-input");
  const img = $("generic-input-img");
  const empty = $("generic-result-empty");
  // leaving any previous result behind
  $("generic-result").classList.add("hidden");
  $("generic-download").classList.add("hidden");
  $("generic-stats").classList.add("hidden");

  empty.textContent = "Drop input to preview.";
  if (!file) {
    video.classList.add("hidden");
    img.classList.add("hidden");
    video.removeAttribute("src");
    img.removeAttribute("src");
    empty.classList.remove("hidden");
    return;
  }

  const url = URL.createObjectURL(file);
  if ((file.type || "").startsWith("video")) {
    video.muted = true;
    video.src = url;
    video.classList.remove("hidden");
    img.classList.add("hidden");
    video.load();
    // preload="metadata" won't paint a frame on its own — seek a hair to
    // force the browser to decode and show the first frame as a thumbnail.
    video.addEventListener("loadedmetadata", function onMeta() {
      try {
        video.currentTime = Math.min(0.1, (video.duration || 1) / 2);
      } catch (e) {}
    }, { once: true });
    // Some codecs (MPEG-4 Part 2, HEVC…) can't be decoded by the browser.
    // Fall back to a first frame extracted server-side with OpenCV (like Unique).
    video.addEventListener("error", function onErr() {
      video.classList.add("hidden");
      empty.textContent = "Loading preview…";
      empty.classList.remove("hidden");
      extractGenericServerPreview(file);
    }, { once: true });
  } else {
    img.src = url;
    img.classList.remove("hidden");
    video.classList.add("hidden");
  }
  empty.classList.add("hidden");
  $("generic-head-label").textContent = "Input · preview";
}

function showUniqueView(view) {
  const result = view === "result";
  $("unique-setup-view").classList.toggle("hidden", result);
  $("unique-result-view").classList.toggle("hidden", !result);
  if (!result) {
    requestAnimationFrame(drawUniquePoints);
  }
}

async function getClient() {
  if (!clientInstance) {
    clientInstance = await client(window.location.origin);
  }
  return clientInstance;
}

function wrapFiles(files) {
  return Array.from(files || []).map((file) => handle_file(file));
}

function fileUrl(fileObj) {
  return fileObj?.url || fileObj?.path || "";
}

function setButtonBusy(button, busy, label) {
  if (!button.dataset.label) {
    button.dataset.label = button.querySelector("span:last-child")?.textContent || button.textContent;
  }
  button.disabled = busy;
  const text = button.querySelector("span:last-child");
  if (text) {
    text.textContent = busy ? label : button.dataset.label;
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tab;
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab === button));
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `${target}-panel`);
      });
      requestAnimationFrame(drawUniquePoints);
    });
  });
}

function setupFileDrop(prefix) {
  const drop = $(`${prefix}-drop`);
  const input = $(`${prefix}-file`);
  const label = $(`${prefix}-file-label`);

  function assignFiles(files) {
    const list = Array.from(files || []);
    state[prefix].files = list;
    drop.classList.toggle("has-file", list.length > 0);
    if (list.length === 0) {
      label.textContent = "Drop input here or browse";
    } else if (list.length === 1) {
      label.textContent = list[0].name;
    } else {
      label.textContent = `${list.length} files selected`;
    }

    if (prefix === "unique") {
      clearUniqueReference();
      setStatus("unique", "idle", "Input loaded. Run Extract Frames.");
    } else {
      setGenericPreview(list[0]);
      setStatus("generic", "idle", "Input loaded. Ready to run segmentation.");
    }
  }

  input.addEventListener("change", () => assignFiles(input.files));

  ["dragenter", "dragover"].forEach((eventName) => {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.add("drag");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.remove("drag");
    });
  });

  drop.addEventListener("drop", (event) => {
    assignFiles(event.dataTransfer.files);
  });
}

function clearUniqueReference() {
  state.unique.framesDir = null;
  state.unique.frameCount = 0;
  state.unique.fps = 1.0;
  state.unique.frameIdx = 0;
  state.unique.points = [];
  $("unique-slider-wrap").classList.add("hidden");
  $("unique-ref-hint").classList.remove("hidden");
  $("unique-frame-slider").value = "0";
  $("unique-frame-slider").max = "0";
  $("unique-frame-count").textContent = "0 / 0";
  $("unique-reference").classList.add("empty");
  $("unique-reference").querySelector(".empty-state").classList.remove("hidden");
  $("unique-reference-img").classList.add("hidden");
  $("unique-reference-img").removeAttribute("src");
  $("unique-reference").classList.remove("running");
  showUniqueView("setup");
  drawUniquePoints();
}

function setReferenceImage(fileObj) {
  const url = fileUrl(fileObj);
  const image = $("unique-reference-img");
  const box = $("unique-reference");
  if (!url) {
    clearUniqueReference();
    return;
  }
  image.src = url;
  image.classList.remove("hidden");
  box.classList.remove("empty");
  box.querySelector(".empty-state").classList.add("hidden");
  image.onload = drawUniquePoints;
}

function updateFrameCounter() {
  const count = state.unique.frameCount || 0;
  $("unique-frame-count").textContent = count ? `${state.unique.frameIdx + 1} / ${count}` : "0 / 0";
}

async function extractUniqueFrames() {
  if (!state.unique.files.length) {
    setStatus("unique", "idle", "Drop input before extracting frames.");
    return;
  }

  const button = $("unique-extract");
  setButtonBusy(button, true, "Extracting");
  setStatus("unique", "live", "Extracting frames...");

  try {
    const api = await getClient();
    const result = await api.predict("/extract_unique", {
      input_files: wrapFiles(state.unique.files),
      sampling: samplingMode("unique")
    });
    const [frameFile, meta] = result.data;
    setBackendHtml("unique", meta);

    if (meta?.success) {
      state.unique.framesDir = meta.frames_dir;
      state.unique.frameCount = meta.frame_count || 0;
      state.unique.fps = meta.fps || 1.0;
      state.unique.frameIdx = 0;
      state.unique.points = [];
      $("unique-frame-slider").max = String(Math.max(0, state.unique.frameCount - 1));
      $("unique-frame-slider").value = "0";
      $("unique-slider-wrap").classList.remove("hidden");
      $("unique-ref-hint").classList.add("hidden");
      updateFrameCounter();
      setReferenceImage(frameFile);
    }
  } catch (error) {
    console.error(error);
    setStatus("unique", "err", error.message || "Frame extraction failed.");
  } finally {
    setButtonBusy(button, false);
  }
}

async function loadReferenceFrame(frameIdx) {
  if (!state.unique.framesDir) {
    return;
  }
  state.unique.frameIdx = Number(frameIdx) || 0;
  state.unique.points = [];
  showUniqueView("setup");
  updateFrameCounter();
  drawUniquePoints();

  try {
    const api = await getClient();
    const result = await api.predict("/reference_frame", {
      frames_dir: state.unique.framesDir,
      frame_idx: state.unique.frameIdx
    });
    const [frameFile, meta] = result.data;
    if (meta?.success) {
      setReferenceImage(frameFile);
    } else {
      setStatus("unique", "err", meta?.error || "Reference frame not found.");
    }
  } catch (error) {
    console.error(error);
    setStatus("unique", "err", error.message || "Could not load reference frame.");
  }
}

function displayedImageRect() {
  const image = $("unique-reference-img");
  const box = $("unique-reference");
  if (image.classList.contains("hidden") || !image.naturalWidth) {
    return null;
  }
  const imageRect = image.getBoundingClientRect();
  const boxRect = box.getBoundingClientRect();
  return {
    image,
    box,
    width: imageRect.width,
    height: imageRect.height,
    left: imageRect.left - boxRect.left,
    top: imageRect.top - boxRect.top,
    naturalWidth: image.naturalWidth,
    naturalHeight: image.naturalHeight
  };
}

function drawUniquePoints() {
  const canvas = $("unique-point-canvas");
  const rect = displayedImageRect();
  if (!rect) {
    canvas.width = 0;
    canvas.height = 0;
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  canvas.style.left = `${rect.left}px`;
  canvas.style.top = `${rect.top}px`;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#315DCE";
  ctx.lineWidth = 2;

  state.unique.points.forEach(([x, y]) => {
    const px = (x / rect.naturalWidth) * rect.width;
    const py = (y / rect.naturalHeight) * rect.height;
    ctx.beginPath();
    ctx.moveTo(px - 10, py);
    ctx.lineTo(px + 10, py);
    ctx.moveTo(px, py - 10);
    ctx.lineTo(px, py + 10);
    ctx.stroke();
  });
}

function addUniquePoint(event) {
  const rect = displayedImageRect();
  if (!rect) {
    return;
  }
  const imageRect = rect.image.getBoundingClientRect();
  const x = Math.round(((event.clientX - imageRect.left) / imageRect.width) * rect.naturalWidth);
  const y = Math.round(((event.clientY - imageRect.top) / imageRect.height) * rect.naturalHeight);
  state.unique.points.push([x, y]);
  $("unique-text").value = "";
  drawUniquePoints();
}

function showVideo(prefix, fileObj) {
  const url = fileUrl(fileObj);
  const video = $(`${prefix}-result`);
  const empty = $(`${prefix}-result-empty`);
  const download = $(`${prefix}-download`);
  if (!url) {
    video.classList.add("hidden");
    empty.classList.remove("hidden");
    download.classList.add("hidden");
    return;
  }
  video.src = url;
  video.classList.remove("hidden");
  empty.classList.add("hidden");
  download.href = url;
  download.classList.remove("hidden");
}

async function runUniqueSegmentation() {
  const button = $("unique-run");
  setButtonBusy(button, true, "Running");
  setStatus("unique", "live", "Running segmentation...");
  $("unique-reference").classList.add("running");

  try {
    const api = await getClient();
    const result = await api.predict("/run_unique", {
      input_files: wrapFiles(state.unique.files),
      points: state.unique.points,
      prompt: $("unique-text").value,
      encoder_size: $("unique-encoder").value,
      offload: $("unique-offload").checked,
      frame_idx: state.unique.frameIdx,
      frames_dir_cache: state.unique.framesDir,
      original_fps: state.unique.fps
    });
    const [videoFile, meta] = result.data;
    setBackendHtml("unique", meta);
    showVideo("unique", videoFile);
    showUniqueView("result");
  } catch (error) {
    console.error(error);
    setStatus("unique", "err", error.message || "Segmentation failed.");
  } finally {
    $("unique-reference").classList.remove("running");
    setButtonBusy(button, false);
  }
}

async function runGenericSegmentation() {
  if (!state.generic.files.length) {
    setStatus("generic", "idle", "Drop input before running segmentation.");
    return;
  }

  const button = $("generic-run");
  setButtonBusy(button, true, "Running");
  setStatus("generic", "live", "Running segmentation...");
  $("generic-result-box").classList.add("running");

  try {
    const api = await getClient();
    const result = await api.predict("/run_generic", {
      input_files: wrapFiles(state.generic.files),
      category: $("generic-category").value,
      accept: Number($("generic-accept").value),
      reject: Number($("generic-reject").value),
      vlm_model: $("generic-vlm").value,
      sampling: samplingMode("generic")
    });
    const [videoFile, meta] = result.data;
    setBackendHtml("generic", meta);
    $("generic-input").classList.add("hidden");
    $("generic-input-img").classList.add("hidden");
    $("generic-head-label").textContent = "Result · masks overlaid per frame";
    showVideo("generic", videoFile);
    $("generic-stats").classList.remove("hidden");
  } catch (error) {
    console.error(error);
    setStatus("generic", "err", error.message || "Segmentation failed.");
  } finally {
    $("generic-result-box").classList.remove("running");
    setButtonBusy(button, false);
  }
}

function setupRanges() {
  const accept = $("generic-accept");
  const reject = $("generic-reject");
  accept.addEventListener("input", () => {
    $("generic-accept-value").textContent = Number(accept.value).toFixed(2);
  });
  reject.addEventListener("input", () => {
    $("generic-reject-value").textContent = Number(reject.value).toFixed(2);
  });

  const frameSlider = $("unique-frame-slider");
  frameSlider.addEventListener("input", () => {
    state.unique.frameIdx = Number(frameSlider.value) || 0;
    updateFrameCounter();
  });
  frameSlider.addEventListener("change", () => loadReferenceFrame(frameSlider.value));
}

async function loadExample(btn) {
  const prefix = btn.dataset.prefix;
  const src = btn.dataset.src;
  let fields = {};
  try { fields = JSON.parse(btn.dataset.fields || "{}"); } catch (e) {}
  btn.disabled = true;
  try {
    const resp = await fetch(src);
    const blob = await resp.blob();
    const file = new File([blob], src.split("/").pop(), { type: blob.type || "video/mp4" });
    const input = $(`${prefix}-file`);
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));  // reuse the normal upload flow
    Object.entries(fields).forEach(([id, val]) => {
      const el = $(id);
      if (!el) return;
      el.value = val;
      el.dispatchEvent(new Event("input"));
    });
  } catch (e) {
    console.error("Example load failed:", e);
  } finally {
    btn.disabled = false;
  }
}

function setupActions() {
  $("unique-extract").addEventListener("click", extractUniqueFrames);
  $("unique-run").addEventListener("click", runUniqueSegmentation);
  $("generic-run").addEventListener("click", runGenericSegmentation);
  $("unique-reference-img").addEventListener("click", addUniquePoint);
  $("unique-clear-points").addEventListener("click", () => {
    state.unique.points = [];
    drawUniquePoints();
  });
  $("unique-back").addEventListener("click", () => showUniqueView("setup"));
  document.querySelectorAll(".example-card").forEach((b) => b.addEventListener("click", () => loadExample(b)));
  window.addEventListener("resize", drawUniquePoints);
}

setupTabs();
setupFileDrop("unique");
setupFileDrop("generic");
setupRanges();
setupActions();
