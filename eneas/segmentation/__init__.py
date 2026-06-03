"""
eneas Segmentation Module

Provides frame sequence segmentation tools for unique instance tracking
and generic category detection.
"""

from .generic_category import GenericCategorySegmenter
from .types import SegmentationResult
from .unique_instance import UniqueInstanceSegmenter

__all__ = [
    "UniqueInstanceSegmenter",
    "GenericCategorySegmenter",
    "SegmentationResult",
]
