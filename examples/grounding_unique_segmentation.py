"""
Text-Based Grounding - Complete Example

Segment objects using natural language descriptions instead of manual point annotation.
This example uses Florence-2 for object grounding and SeC for temporal segmentation.

This example shows:
1. How to segment objects using text descriptions
2. Access binary masks and bounding box information
3. Save masks as PNG images
4. Analyze segmentation results
"""

import logging

import cv2
import numpy as np

from eneas.segmentation import UniqueInstanceSegmenter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

print("\n🚀 Initializing segmenter...")
segmenter = UniqueInstanceSegmenter()

print("\n🎯 Segmenting object using text...")
result = segmenter.segment(
    frames_path="/path/to/your/frames/",
    text="the blonde woman with microphone",
    annotation_frame="frame_0050.jpg",
    output_dir="./output_grounding/",
    save_masks=True,
)

print("\n✅ Segmentation completed!")
print(f"   Processed {result.num_frames} frames")
print(f"   Output directory: {result.output_dir}")

print("\n📊 Grounding information:")
print(f"   Text: '{result.metadata['text']}'")
print(f"   Detected bbox: {result.metadata['bbox']}")
print(f"   Mode: {result.metadata['mode']}")

print("\n🎨 Accessing binary masks in memory...")
frame_idx = 0
mask = result.masks[frame_idx]
print(f"Frame {frame_idx} mask:")
print(f"  - Shape: {mask.shape}")
print(f"  - Data type: {mask.dtype}")
print(f"  - Unique values: {np.unique(mask)}")

if result.mask_paths:
    print("\n💾 Loading masks from disk...")
    mask_path = result.mask_paths[0]
    print(f"Loading: {mask_path}")

    loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded mask shape: {loaded_mask.shape}")
    print(f"Masks match: {np.array_equal(mask, loaded_mask)}")

if result.initial_mask_path:
    print(f"\n🖼️  Initial mask visualization: {result.initial_mask_path}")
    print("   (Shows mask overlay + green bbox + text label)")

print("\n✨ Done!")
