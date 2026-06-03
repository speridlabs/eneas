"""
ENEAS - Embedding-guided Neural Ensemble for Adaptive Segmentation

Frame sequence segmentation library with temporal tracking and category detection.

Provides tools for:
- Unique instance segmentation with temporal tracking
- Generic category segmentation across frames
"""

__version__ = "0.1.0"

from .segmentation import (
    GenericCategorySegmenter,
    SegmentationResult,
    UniqueInstanceSegmenter,
)

__all__ = [
    "GenericCategorySegmenter",
    "SegmentationResult",
    "UniqueInstanceSegmenter",
]
