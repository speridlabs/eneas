"""
Multi-Point Unique Instance

Refine unique instance segmentation using multiple annotation points
with positive and negative labels.

Positive points (label=1): Include these regions in the segmentation
Negative points (label=0): Exclude these regions from the segmentation

Use this when:
- Object has complex shape or texture
- Need to exclude shadows, reflections, or similar objects
- Single-point annotation is not accurate enough
"""

from eneas.segmentation import UniqueInstanceSegmenter

# Initialize segmenter
segmenter = UniqueInstanceSegmenter()

# Define multiple annotation points
points = [
    (400, 300),
    (450, 320),
    (420, 500),
    (380, 510),
]

# Define labels for each point
labels = [
    1,  # Point 1: positive (include)
    1,  # Point 2: positive (include)
    0,  # Point 3: negative (exclude)
    0,  # Point 4: negative (exclude)
]

# Segment with multiple points
result = segmenter.segment(
    frames_path="/path/to/frames/",
    points=points,
    labels=labels,
    annotation_frame="frame_0075.jpg",
    output_dir="./outputs/multi_point_example/",
)

# Print results
print("\n" + "=" * 60)
print("MULTI-POINT SEGMENTATION COMPLETE")
print("=" * 60)
print(f"✓ Used {len(points)} annotation points")
print(f"  - {sum(labels)} positive points (include regions)")
print(f"  - {len(labels) - sum(labels)} negative points (exclude regions)")
print(f"\n✓ Processed {result.num_frames} frames")
print(f"✓ Binary masks available: {len(result.masks)}")
