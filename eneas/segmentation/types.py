"""
Shared types for segmentation operations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    """Result of a segmentation operation.

    Attributes:
        masks: Dictionary mapping frame indices to binary masks (numpy arrays, 0=background, 255=foreground)
        num_frames: Number of frames successfully segmented
        output_dir: Directory where results were saved
        mask_paths: List of paths to saved mask images (if save_masks=True)
        metadata: Additional metadata about the segmentation
        initial_mask_path: Path to the initial mask visualization (None for generic segmentation)
    """

    masks: dict[int, np.ndarray]
    num_frames: int
    output_dir: str
    mask_paths: list[str]
    metadata: dict
    initial_mask_path: str | None = None
