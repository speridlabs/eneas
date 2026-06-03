"""
Generic Category Segmentation - Complete Example

Segment MULTIPLE instances of a category across frames (no temporal tracking).

This example shows:
1. How to segment all instances of a category (e.g., "chair", "person")
2. Configure detection thresholds for precision/recall tradeoff
3. Configure VLM model for validation
4. Access binary masks per frame and instance
5. Save masks as PNG images (optional)

Requirements:
- VLM server running: `ollama serve`
"""

import logging

import numpy as np
from eneas.segmentation import GenericCategorySegmenter

# Setup logging to see progress messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize segmenter (downloads models automatically on first use)
print("\n🚀 Initializing segmenter...")
segmenter = GenericCategorySegmenter(
    # vlm_model="qwen3-vl:2b-instruct-q8_0"  # Default: faster, less VRAM
    # vlm_model="qwen3-vl:4b-instruct-q8_0"  # Better quality, more VRAM
)

# Segment all instances of a category
print("\n🎯 Segmenting all chairs...")
result = segmenter.segment(
    frames_path="/path/to/your/frames/",
    category="chair",
    output_dir="./output_chairs/",
    accept_threshold=0.60,
    reject_threshold=0.10,
    save_masks=True,  # Save combined masks as PNG files
)

print("\n✅ Segmentation completed!")
print(f"   Processed {result.num_frames} frames")
print(f"   Output directory: {result.output_dir}")

# ============================================================================
# 1. ACCESS MASKS IN MEMORY (PER FRAME, PER INSTANCE)
# ============================================================================
print("\n📊 Accessing masks in memory...")

# Get masks for first frame
frame_idx = 0
frame_masks = result.masks[frame_idx]  # List of masks for this frame

print(f"Frame {frame_idx}:")
print(f"  - Number of instances detected: {len(frame_masks)}")

# Iterate through each instance in this frame
for instance_idx, mask in enumerate(frame_masks):
    print(f"\n  Instance {instance_idx + 1}:")
    print(f"    - Shape: {mask.shape}")  # (H, W)
    print(f"    - Data type: {mask.dtype}")  # uint8
    print(f"    - Values: 0 (background) and 255 (object)")
    print(f"    - Unique values: {np.unique(mask)}")


print("\n✨ Done!")
