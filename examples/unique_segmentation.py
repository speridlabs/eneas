"""
Unique Instance - Complete Example

Segment a SINGLE unique object instance across multiple frames
with temporal tracking.

This example shows:
1. How to segment a unique object instance across frame sequences
2. Access binary masks (numpy arrays with 0=background, 255=object)
3. Save masks as PNG images (optional)
4. Analyze and process masks
"""

import logging

import cv2
import numpy as np
from eneas.segmentation import UniqueInstanceSegmenter

# Setup logging to see progress messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize segmenter (downloads model automatically on first use)
print("\n🚀 Initializing segmenter...")
segmenter = UniqueInstanceSegmenter()

# Segment an object
print("\n🎯 Segmenting object...")
result = segmenter.segment(
    frames_path="/path/to/your/frames/",
    points=[(400, 300)],
    annotation_frame="frame_0050.jpg",  # Frame where you click (or None for first)
    output_dir="./output_masks/",
    save_masks=True,  # Save binary masks as PNG files
)

print("\n✅ Segmentation completed!")
print(f"   Processed {result.num_frames} frames")
print(f"   Output directory: {result.output_dir}")

# ============================================================================
# 1. ACCESS BINARY MASKS IN MEMORY
# ============================================================================
print("\n📊 Accessing binary masks in memory...")

# Get mask for first frame
frame_idx = 0
mask = result.masks[frame_idx]

print(f"Frame {frame_idx} mask:")
print(f"  - Shape: {mask.shape}")  # (H, W)
print(f"  - Data type: {mask.dtype}")  # uint8
print("  - Values: 0 (background) and 255 (object)")
print(f"  - Unique values: {np.unique(mask)}")

# ============================================================================
# 2. LOAD MASKS FROM DISK
# ============================================================================
print("\n💾 Loading masks from disk...")

# Masks are saved as PNG files
mask_path = result.mask_paths[0]
print(f"Loading: {mask_path}")

# Load as grayscale (0-255)
loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
print(f"Loaded mask shape: {loaded_mask.shape}")
print(f"Masks match: {np.array_equal(mask, loaded_mask)}")
