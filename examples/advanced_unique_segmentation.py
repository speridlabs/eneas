"""
Advanced Configuration Example
==============================

This example shows how to use custom configuration and
advanced features of eneas, including:
- Setting environment variables
- Custom model paths
- SAM encoder selection
- CUDA optimizations
- GPU memory management
"""

import logging
import os

from eneas.segmentation import UniqueInstanceSegmenter

# ============================================================================
# OPTION 1: Configure via Environment Variables (before importing)
# ============================================================================
# You can set these BEFORE running the script or at the start of your code

# Set custom model cache directory (optional)
os.environ["HF_HOME"] = "/path/to/your/model/cache"


# Now when the model auto-downloads, it will go to this directory
# Default is: ~/.cache/eneas/

# ============================================================================
# OPTION 2: Configure via Constructor Parameters
# ============================================================================

# Configure logging to see detailed progress messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize segmenter with custom settings
segmenter = UniqueInstanceSegmenter(
    segmentation_model_path="/custom/path/to/SeC-4B",  # Or None for auto-download
    grounding_model_path="/custom/path/to/Florence-2",  # Or None for auto-download (only needed for text-based)
    sam_encoder="long-small",  # Faster encoder for development
    device="cuda:1",
    default_output_dir="./my_custom_outputs",
    memory_cleanup_interval=5,
)

print("✓ Segmenter initialized")
print(f"  Model cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
print(f"  Memory cleanup: every {segmenter.memory_cleanup_interval} frames")

# OPTIONAL: Optimize CUDA memory allocation
# Only needed if you have limited GPU memory or get Out-of-Memory errors
# This clears GPU cache and enables expandable memory segments (reduces fragmentation)
segmenter.optimize_cuda_memory()
print("✓ CUDA memory optimizations applied")

# Segment with advanced options
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=[(400, 300)],
    annotation_frame="frame_0100.jpg",
    output_dir="./outputs/advanced_example/",
    offload_frames_to_gpu=True,  # Keep frames in GPU for faster processing (Much more GPU memory needed)
    save_masks=True,  # Save binary masks to disk
)

print(f"\n✓ Segmentation complete: {result.num_frames} frames processed")
print(f"✓ Binary masks in memory: {len(result.masks)}")
print(f"✓ Masks saved to disk: {len(result.mask_paths)} PNG files")
