# ENEAS

**Embedding-guided Neural Ensemble for Adaptive Segmentation**

ENEAS is an open-vocabulary segmentation method and model with semantic understanding.
It can track and segment both **unique instances** and **generic semantic instance** across frame sequences. 

It works from points or natural-language descriptions and returns high-quality binary masks. And has been designed for robustness to avoid common failure modes of segmentation models, such as drifting, false positives, and false negatives.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

| 🎯 Unique Instance Segmentation | 🔮 Generic Semantic Category Segmentation |
| --- | --- |
| Track one specific object instance across frames.<br><br>• Point-based or text-based annotation (open-vocabulary, zero-click)<br>• Bidirectional temporal propagation<br>• High-quality binary masks (B&W) every frame<br>• Save masks as PNG | Detect every instance of a semantic category per frame.<br><br>• Complex semantic open-vocabulary — name any complex semantic instance category<br>• Semantic validation to reduce false positives<br>• Frame-by-frame (no temporal tracking)<br>• Binary masks (B&W) per instance |

## Installation

### Prerequisites

- CUDA-capable GPU
- Python >= 3.10, < 3.13
- [Ollama](https://ollama.com) installed and running on your system (used by generic semantic category segmentation)

### Install

```bash
pip install -e .
```

## Quick Start

Track an object across a folder of frames using a natural-language description:

```python
from eneas.segmentation import UniqueInstanceSegmenter

segmenter = UniqueInstanceSegmenter()

result = segmenter.segment(
    frames_path="./frames",
    text="the person in the red jacket",
)

# Binary masks (0 = background, 255 = object), one per frame, in memory
for frame_idx, mask in result.masks.items():
    print(frame_idx, mask.shape)
```

Or from the command line:

```bash
eneas unique_instance -i ./frames --text "the person in the red jacket" -o ./output
```

## Model Setup

**Automatic download**

Models download automatically from HuggingFace on first use — the main segmentation model (SeC-4B) and, for text-based segmentation, the grounding model.

```python
from eneas.segmentation import UniqueInstanceSegmenter

# Models download automatically on first use
segmenter = UniqueInstanceSegmenter()
```

**Custom model paths (optional)**

If you have pre-downloaded models or custom locations:

```python
segmenter = UniqueInstanceSegmenter(
    segmentation_model_path="/path/to/SeC-4B",       # main segmentation model
    grounding_model_path="/path/to/grounding-model"  # optional, for text-based segmentation
)
```

**Cache directory (optional)**

Models are cached using HuggingFace Hub's standard cache directory (`~/.cache/huggingface/`). Customize the location with `HF_HOME`:

```bash
export HF_HOME="/custom/cache/path"
```

## Usage

### CLI Usage

#### Unique Instance Segmentation

Track a specific object instance across frames:

```bash
# Point-based segmentation - annotate the first frame (default)
eneas unique_instance -i ./frames -p 400,300 -o ./output

# Specify which frame to annotate (recommended for better control)
eneas unique_instance -i ./frames -p 400,300 -f frame_0050.jpg -o ./output

# Multiple points for better accuracy
eneas unique_instance -i ./frames -p 400,300 -p 350,280 -f frame_0050.jpg

# Positive and negative points (refine segmentation)
eneas unique_instance -i ./frames -p 400,300 -p 350,280 -l 1 -l 0 -f frame_0050.jpg

# Text-based segmentation - use natural language instead of points
eneas unique_instance -i ./frames --text "the blonde woman with microphone"

# Text-based with specific frame and save masks
eneas unique_instance -i ./frames -t "red car" -f frame_0100.jpg --save-masks

# Save masks as PNG files (optional, always in memory)
eneas unique_instance -i ./frames -p 400,300 -f frame_0050.jpg --save-masks

# Use smaller encoder for faster processing
eneas unique_instance -i ./frames -p 400,300 -f frame_0050.jpg -s long-small

# Keep frames in GPU for faster processing (uses more VRAM)
eneas unique_instance -i ./frames -p 400,300 -f frame_0050.jpg --offload-frames-to-gpu

# See all options
eneas unique_instance --help
```

#### Generic Category Segmentation

Detect all instances of a semantic category across frames (no temporal tracking):

```bash
# Detect all instances of a semantic category
eneas generic_category -i ./frames --category "person" -o ./output

# Adjust confidence thresholds for precision/recall trade-off
eneas generic_category -i ./frames -c "chair" --accept-threshold 0.95 --reject-threshold 0.05

# Save debug visualizations to understand the detection pipeline
eneas generic_category -i ./frames -c "car" --save-debug

# Save masks as PNG files
eneas generic_category -i ./frames -c "bottle" --save-masks

# Enable verbose logging
eneas generic_category -i ./frames -c "person" -v

# See all options
eneas generic_category --help
```

**CLI features:**
- 🎯 Type-safe arguments with automatic validation
- 📝 Text-based grounding (open-vocabulary, zero-click annotation)
- 📊 Progress tracking and clear output
- 💾 Automatic model downloading
- ⚡ Multiple encoder sizes (long-large, long-small, long-tiny, small, tiny, etc.)
- 🔧 CUDA memory optimization for low-memory GPUs
- 💻 Flexible memory management (CPU/GPU offloading)

**Notes:**
- For unique instance: if you don't specify `-f/--annotation-frame`, the **first frame** in the directory is used by default.
- For generic category: adjust `--accept-threshold` (higher = more precision) and `--reject-threshold` (lower = more recall) based on your use case.

### Python API Usage

#### Unique Instance Segmentation

**Point-based segmentation:**

```python
from eneas.segmentation import UniqueInstanceSegmenter

# Initialize segmenter (auto-downloads model on first use)
segmenter = UniqueInstanceSegmenter()

# Segment a specific object instance - returns binary masks in memory
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=[(400, 300)],                    # Click coordinates on the object
    annotation_frame="frame_0050.jpg",      # Frame to annotate (omit to use first frame)
    output_dir="./outputs/"
)

# Access binary masks (always available in memory, no disk I/O by default)
print(f"✓ Processed {result.num_frames} frames")
for frame_idx, mask in result.masks.items():
    print(f"Frame {frame_idx}: mask shape {mask.shape}")  # (H, W) uint8 array

# Optionally save to disk
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=[(400, 300)],
    annotation_frame="frame_0050.jpg",
    save_masks=True  # Enable disk saving
)

# Now you can load from disk
import cv2
mask_0 = cv2.imread(result.mask_paths[0], cv2.IMREAD_GRAYSCALE)

# Check metadata
if result.initial_mask_path:
    print(f"✓ Initial mask visualization: {result.initial_mask_path}")
print(f"✓ Binary masks: {len(result.masks)} in memory")
if result.mask_paths:
    print(f"✓ Saved masks: {len(result.mask_paths)} PNG files")
```

**Text-based segmentation:**

```python
from eneas.segmentation import UniqueInstanceSegmenter

# Initialize segmenter
segmenter = UniqueInstanceSegmenter()

# Segment using text description
result = segmenter.segment(
    frames_path="./frames/",
    text="the blonde woman with microphone",
    annotation_frame="frame_0050.jpg",
    save_masks=True
)

# Access bounding box used for grounding
print(f"Detected bbox: {result.metadata['bbox']}")  # [x1, y1, x2, y2]
print(f"Text used: {result.metadata['text']}")

# Access binary masks
for frame_idx, mask in result.masks.items():
    print(f"Frame {frame_idx}: mask shape {mask.shape}")
```

**Note:** Text-based segmentation uses open-vocabulary grounding and downloads the grounding model automatically on first use.

#### Generic Category Segmentation

**Basic usage:**

```python
from eneas.segmentation import GenericCategorySegmenter

# Initialize segmenter (auto-downloads models on first use)
segmenter = GenericCategorySegmenter()

# Detect all instances of a category across frames
result = segmenter.segment(
    frames_path="/path/to/frames/",
    category="person",
    output_dir="./outputs/"
)

# Access binary masks - each frame can have multiple instances
print(f"✓ Processed {result.num_frames} frames")

# Masks are organized by frame index
for frame_idx, masks_list in result.masks.items():
    print(f"Frame {frame_idx}: {len(masks_list)} instances detected")
    for instance_idx, mask in enumerate(masks_list):
        print(f"  Instance {instance_idx}: mask shape {mask.shape}")  # (H, W) uint8 array
        # mask is binary: 0 (background) or 255 (object)

# Check detection metadata
total_detections = sum(len(dets) for dets in result.metadata['detections'].values())
print(f"Total detections across all frames: {total_detections}")
print(f"Validation used: {result.metadata['vlm_usage_percentage']:.1f}%")
```

**Advanced usage with thresholds:**

```python
from eneas.segmentation import GenericCategorySegmenter

# Initialize segmenter
segmenter = GenericCategorySegmenter()

# Fine-tune detection with custom thresholds
result = segmenter.segment(
    frames_path="/path/to/frames/",
    category="chair",
    accept_threshold=0.95,    # Higher = more precision (fewer false positives)
    reject_threshold=0.05,    # Lower = more recall (fewer false negatives)
    save_debug=True,          # Save debug visualizations
    save_masks=True,          # Save masks as PNG files
    output_dir="./outputs/"
)

# Access saved mask paths (when save_masks=True)
for frame_idx, mask_paths_list in result.mask_paths.items():
    print(f"Frame {frame_idx}: {len(mask_paths_list)} masks saved")
    for instance_idx, mask_path in enumerate(mask_paths_list):
        print(f"  Instance {instance_idx}: {mask_path}")
```

### Advanced Usage

```python
from eneas.segmentation import UniqueInstanceSegmenter

# Initialize segmenter
segmenter = UniqueInstanceSegmenter(
    sam_encoder="long-small",  # Faster encoder for development
)

# Segment with multiple points (positive and negative)
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=[(400, 300), (450, 350), (500, 200)],  # Multiple annotation points
    labels=[1, 1, 0],                              # 1=include, 0=exclude
    annotation_frame="frame_0100.jpg",
    output_dir="./outputs/",
    offload_frames_to_gpu=False,                   # Default: keep frames in CPU to save VRAM
    save_masks=True                                # Save masks as PNG files
)

# Access binary masks
import numpy as np
for frame_idx, mask in result.masks.items():
    # mask is numpy array (H, W) with values 0 (background) or 255 (object)
    num_object_pixels = np.sum(mask == 255)
    print(f"Frame {frame_idx}: {num_object_pixels} pixels in mask")
```

### Logging

```python
import logging
from eneas.segmentation import UniqueInstanceSegmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all segmentation operations will log progress
segmenter = UniqueInstanceSegmenter()
result = segmenter.segment(...)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_HOME` | HuggingFace cache directory (for all models) | `~/.cache/huggingface` |

Set custom cache directory:

```bash
export HF_HOME="/custom/cache/path"
```

## Examples

See the `examples/` directory for complete working examples:

**Unique Instance Segmentation:**
- `examples/unique_segmentation.py` - Basic unique instance tracking
- `examples/multi_point_unique_segmentation.py` - Multi-point refinement for complex objects
- `examples/advanced_unique_segmentation.py` - Advanced configuration and memory optimization
- `examples/grounding_unique_segmentation.py` - Text-based segmentation using natural language

**Generic Category Segmentation:**
- `examples/generic_segmentation.py` - Detect all instances of a category across frames

## Technical Details

Under the hood, ENEAS composes several models into a single open-vocabulary segmentation method.

**Unique instance segmentation**
- Built on [SeC](https://github.com/OpenIXCLab/SeC) (SeC-4B) for high-quality masks and bidirectional temporal propagation across the sequence.
- Text-based annotation uses [Florence-2](https://huggingface.co/microsoft/Florence-2-large) to ground the natural-language description into a region of the annotation frame (zero-click, open-vocabulary).

**Generic category segmentation** runs a multi-stage pipeline on each frame:
1. **Grounding** — [Florence-2](https://huggingface.co/microsoft/Florence-2-large) proposes candidate regions for the requested category.
2. **Semantic filtering** — [SigLIP](https://huggingface.co/google/siglip2-base-patch16-naflex) scores each region against the category using an ensemble of prompts, then accepts, rejects, or flags it as uncertain.
3. **Validation** — a vision-language model ([Qwen3-VL](https://ollama.com/library/qwen3-vl), served by [Ollama](https://ollama.com)) re-checks the uncertain regions to remove false positives.
4. **Segmentation** — SAM 2.1 turns the surviving regions into binary masks.

The validation model is configurable via `vlm_model=` (default `qwen3-vl:2b-instruct-q8_0`; use `qwen3-vl:4b-instruct-q8_0` for higher quality at the cost of more VRAM) and runs through your local Ollama server.

## Troubleshooting

### Model Download Issues

If automatic download fails, download the models manually and pass their paths:

```python
# SeC-4B: https://huggingface.co/OpenIXCLab/SeC-4B
segmenter = UniqueInstanceSegmenter(
    segmentation_model_path="/path/to/SeC-4B",
    grounding_model_path="/path/to/grounding-model"  # only needed for text-based segmentation (see Technical Details)
)
```

### Out of Memory (CUDA)

**Solution 1:** Reuse the same segmenter instance for multiple frame sequences to avoid reloading the model to GPU:

```python
segmenter = UniqueInstanceSegmenter()

# Process multiple frame sequences without reloading the model
for frames_path in frames_paths:
    result = segmenter.segment(frames_path=frames_path, points=[(400, 300)])
```

**Solution 2:** Ensure frames are offloaded to CPU (default behavior):

```python
# Python API - Frames stay in CPU by default (saves VRAM)
segmenter = UniqueInstanceSegmenter()
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=[(400, 300)],
    offload_frames_to_gpu=False  # This is the default (CPU)
)

# CLI - Don't use --offload-frames-to-gpu flag (frames in CPU is default)
eneas unique_instance -i ./frames -p 400,300
```

**Solution 3:** Use aggressive memory cleanup:

```python
segmenter = UniqueInstanceSegmenter(memory_cleanup_interval=5)  
```

```bash
eneas unique_instance -i ./frames -p 400,300 --memory-cleanup-interval 5
```

**Solution 4:** Use CUDA memory optimization:

```python
segmenter = UniqueInstanceSegmenter()
segmenter.optimize_cuda_memory()  # Enable expandable_segments
```

```bash
eneas unique_instance -i ./frames -p 400,300 --optimize-cuda-memory
```

## Contributing

This is an open-source project by **SperidLabs**. We welcome contributions!

If you'd like to contribute:
- 🐛 Report bugs via [GitHub Issues](https://github.com/speridlabs/eneas/issues)
- 💡 Suggest features via [GitHub Issues](https://github.com/speridlabs/eneas/issues)
- 🔧 Submit pull requests with improvements
- 📖 Improve documentation

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

ENEAS builds upon the SeC (Segment and Caption) model, which is also licensed under Apache 2.0.

## Citation

If you use ENEAS in your research, please cite:

```bibtex
@techreport{delpino2026eneas,
  title  = {ENEAS: Embedding-guided Neural Ensemble for Adaptive Segmentation},
  author = {del Pino, Javier and S\'anchez, Alejandro and Garabito, Chema},
  institution = {SperidLabs},
  year   = {2026},
  url    = {https://github.com/speridlabs/eneas}
}
```

## Acknowledgments

- Built on top of [SeC](https://github.com/OpenIXCLab/SeC) for video segmentation and temporal tracking
- Uses PyTorch, Transformers, and other open-source models and libraries (see [Technical Details](#technical-details))

## Support

- **Issues**: [GitHub Issues](https://github.com/speridlabs/eneas/issues)
- **Documentation**: [GitHub README](https://github.com/speridlabs/eneas#readme)
- **Company**: [SperidLabs](https://speridlabs.com)

---

**Made with ❤️ by [speridlabs.com](https://speridlabs.com)**
