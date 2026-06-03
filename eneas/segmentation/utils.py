"""
Utility functions for segmentation tasks.

Includes NMS, IoU calculation, text conversion utilities, and image masking.
"""

import base64
import io
import logging
import re

import cv2
import inflect
import numpy as np
import spacy
from PIL import Image

logger = logging.getLogger(__name__)

# Load spacy model and inflect engine
_nlp = spacy.load("en_core_web_sm")
_inflect_engine = inflect.engine()

_CARRIER = "a photo of "
_CARRIER_LEN = len(_nlp(_CARRIER))
_NOUN_POS = ("NOUN", "PROPN")
_COORD_SPLIT = re.compile(r"(\s*/\s*|\s*&\s*|\s*,\s*|\s+and\s+|\s+or\s+)")
_EDGE_WS = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)


def _is_word_token(token) -> bool:
    return any(ch.isalpha() for ch in token.text)


def _base_np_tokens(span):
    """Tokens of the leading base noun phrase, cut before the first preposition."""
    tokens = list(span)
    end = len(tokens)
    for i, token in enumerate(tokens):
        if token.pos_ == "ADP":
            end = i
            break
    return tokens[:end] if end > 0 else tokens


def _primary_head(span):
    """Head of a simple phrase: the last word token of its base noun phrase."""
    base = _base_np_tokens(span)
    word_tokens = [t for t in base if _is_word_token(t)]
    return word_tokens[-1] if word_tokens else None


def _glued_group(span, token):
    """Contiguous no-whitespace orthographic word containing token."""
    doc = span.doc
    start = token.i
    while start - 1 >= span.start and doc[start - 1].whitespace_ == "":
        start -= 1
    end = token.i
    while end < span.end - 1 and doc[end].whitespace_ == "":
        end += 1
    return list(doc[start : end + 1])


def _resolve_inflection_token(span, head):
    """Pick the token to inflect inside head's orthographic word (hyphen compounds)."""
    group = _glued_group(span, head)
    word_tokens = [t for t in group if _is_word_token(t)]
    if len(word_tokens) <= 1:
        return head
    if any(t.pos_ == "ADP" for t in group):
        for t in group:
            if t.pos_ in _NOUN_POS:
                return t
        return word_tokens[0]
    for t in reversed(group):
        if t.pos_ in _NOUN_POS:
            return t
    return word_tokens[-1]


def _surface_plural(word: str) -> str:
    plural = _inflect_engine.plural(word)
    return plural if plural else word


def _surface_singular(word: str) -> str:
    singular = _inflect_engine.singular_noun(word)
    return singular if singular else word


def _looks_plural(word: str) -> bool:
    """Detect irregular plurals spaCy may mislabel, rejecting naive suffix-chops."""
    singular = _inflect_engine.singular_noun(word)
    if not singular:
        return False
    if _inflect_engine.plural(singular).lower() != word.lower():
        return False
    return singular.lower() != word.lower()[:-1]


def _is_plural(token) -> bool:
    if "Plur" in token.morph.get("Number"):
        return True
    return _looks_plural(token.text)


def _inflect_word(token, to_plural: bool) -> str:
    plural = _is_plural(token)
    if to_plural and not plural:
        return _surface_plural(token.text)
    if not to_plural and plural:
        return _surface_singular(token.text)
    return token.text


def _transform_simple(text: str, to_plural: bool) -> str:
    doc = _nlp(_CARRIER + text)
    span = doc[_CARRIER_LEN:]
    head = _primary_head(span)
    if head is None:
        return text
    target = _resolve_inflection_token(span, head)
    target_index = target.i if _is_word_token(target) else None

    pieces = []
    for token in span:
        if token.i == target_index:
            pieces.append(_inflect_word(token, to_plural) + token.whitespace_)
        else:
            pieces.append(token.text + token.whitespace_)
    return "".join(pieces)


def _transform(text: str, to_plural: bool) -> str:
    if not text or not text.strip():
        return text

    segments = _COORD_SPLIT.split(text)
    result = []
    for i, segment in enumerate(segments):
        if i % 2 == 1 or not segment.strip():
            result.append(segment)
            continue
        lead, core, trail = _EDGE_WS.match(segment).groups()
        result.append(lead + _transform_simple(core, to_plural) + trail)
    return "".join(result)


def smart_convert_to_singular(text: str) -> str:
    """Convert only the head noun(s) of the phrase to singular, preserving everything else.

    Detects the syntactic head via spaCy (parsing inside a carrier phrase for reliable
    tagging of bare inputs) and produces the surface form with inflect, splitting on
    coordinating separators and reconstructing original spacing/punctuation verbatim.

    Examples:
        "people" -> "person"
        "real people" -> "real person"
        "the blue chairs" -> "the blue chair"
        "chairs/stools" -> "chair/stool"
        "windows & doors" -> "window & door"

    Args:
        text: Text to convert (can contain multiple words and separators)

    Returns:
        Text with head noun(s) in singular
    """
    return _transform(text, to_plural=False)


def smart_convert_to_plural(text: str) -> str:
    """Convert only the head noun(s) of the phrase to plural, preserving everything else.

    Detects the syntactic head via spaCy (parsing inside a carrier phrase for reliable
    tagging of bare inputs) and produces the surface form with inflect, splitting on
    coordinating separators and reconstructing original spacing/punctuation verbatim.

    Examples:
        "person" -> "people"
        "real person" -> "real people"
        "the blue chair" -> "the blue chairs"
        "chair/stool" -> "chairs/stools"
        "window & door" -> "windows & doors"

    Args:
        text: Text to convert (can contain multiple words and separators)

    Returns:
        Text with head noun(s) in plural
    """
    return _transform(text, to_plural=True)


def calculate_iou(box1: list, box2: list) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_box_area(box: list) -> float:
    """Calculate area of a bounding box.

    Args:
        box: Bounding box [x1, y1, x2, y2]

    Returns:
        Area of the box
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def non_max_suppression(
    bboxes: list, labels: list, iou_threshold: float = 0.70
) -> tuple[list, list]:
    """Apply Non-Maximum Suppression without scores.

    When two boxes overlap (IoU > threshold), keep the larger box.

    Args:
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of labels
        iou_threshold: IoU threshold for considering boxes as duplicates (default: 0.70)

    Returns:
        Tuple of (filtered_bboxes, filtered_labels)
    """
    if len(bboxes) <= 1:
        return bboxes, labels

    # Calculate areas for all boxes
    areas = [calculate_box_area(box) for box in bboxes]

    # Sort by area (largest first)
    sorted_indices = sorted(range(len(bboxes)), key=lambda i: areas[i], reverse=True)

    keep_indices = []
    suppressed = set()

    for idx in sorted_indices:
        if idx in suppressed:
            continue

        keep_indices.append(idx)

        # Suppress smaller boxes that overlap with this one
        for other_idx in sorted_indices:
            if other_idx == idx or other_idx in suppressed:
                continue

            iou = calculate_iou(bboxes[idx], bboxes[other_idx])
            if iou > iou_threshold:
                suppressed.add(other_idx)
                logger.debug(
                    f"NMS: Suppressing box {other_idx} (area={areas[other_idx]:.1f}) "
                    f"due to overlap (IoU={iou:.3f}) with box {idx} (area={areas[idx]:.1f})"
                )

    # Return boxes in original order (keeping only non-suppressed ones)
    keep_indices_sorted = sorted(keep_indices)
    filtered_bboxes = [bboxes[i] for i in keep_indices_sorted]
    filtered_labels = [labels[i] for i in keep_indices_sorted]

    return filtered_bboxes, filtered_labels


def mask_overlapping_boxes(
    crop_image: Image.Image,
    current_bbox: list,
    all_bboxes: list,
    current_idx: int,
    crop_coords: tuple,
    max_mask_percentage: float = 60.0,
) -> Image.Image:
    """Mask overlapping regions from other boxes in the current crop.

    Paints black the regions of other boxes that overlap with the current box.
    If masking exceeds max_mask_percentage, returns original crop unmasked.

    Args:
        crop_image: PIL Image of the crop
        current_bbox: Bbox of current box [x1, y1, x2, y2] in original coordinates
        all_bboxes: List of all bboxes in original coordinates
        current_idx: Index of the current box
        crop_coords: Crop coordinates in original image (crop_x1, crop_y1, crop_x2, crop_y2)
        max_mask_percentage: Maximum allowed masking percentage (default: 60.0)

    Returns:
        PIL Image with overlapping regions masked (or unmasked if exceeds threshold)
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(c) for c in crop_coords]
    curr_x1, curr_y1, curr_x2, curr_y2 = [int(c) for c in current_bbox]

    crop_array = np.array(crop_image)
    total_pixels = crop_array.shape[0] * crop_array.shape[1]

    masked_crop_array = crop_array.copy()
    masked_pixels_count = 0

    for idx, other_bbox in enumerate(all_bboxes):
        if idx == current_idx:
            continue

        other_x1, other_y1, other_x2, other_y2 = [int(coord) for coord in other_bbox]

        # Check if there's overlap between current box and other box
        if not (
            other_x2 < curr_x1 or other_x1 > curr_x2 or other_y2 < curr_y1 or other_y1 > curr_y2
        ):
            # Calculate intersection region
            intersect_x1 = max(curr_x1, other_x1)
            intersect_y1 = max(curr_y1, other_y1)
            intersect_x2 = min(curr_x2, other_x2)
            intersect_y2 = min(curr_y2, other_y2)

            # Check if intersection falls within the crop
            if not (
                intersect_x2 < crop_x1
                or intersect_x1 > crop_x2
                or intersect_y2 < crop_y1
                or intersect_y1 > crop_y2
            ):
                # Convert to local crop coordinates
                mask_x1 = int(max(0, intersect_x1 - crop_x1))
                mask_y1 = int(max(0, intersect_y1 - crop_y1))
                mask_x2 = int(min(masked_crop_array.shape[1], intersect_x2 - crop_x1))
                mask_y2 = int(min(masked_crop_array.shape[0], intersect_y2 - crop_y1))

                region_pixels = (mask_y2 - mask_y1) * (mask_x2 - mask_x1)
                masked_pixels_count += region_pixels

                # Paint black the overlapping region
                masked_crop_array[mask_y1:mask_y2, mask_x1:mask_x2] = 0

    mask_percentage = (masked_pixels_count / total_pixels) * 100 if total_pixels > 0 else 0

    if mask_percentage > max_mask_percentage:
        logger.debug(
            f"Masking {mask_percentage:.1f}% > {max_mask_percentage}% - using unmasked crop"
        )
        return crop_image
    elif mask_percentage > 0:
        logger.debug(f"Masking {mask_percentage:.1f}% applied")
        return Image.fromarray(masked_crop_array)
    else:
        return crop_image


def expand_crop_to_minimum_size(
    crop_image: Image.Image, bbox: list, image_original: Image.Image, min_size: int = 32
) -> Image.Image:
    """Expand crop to minimum required size by taking pixels from original image.

    Required for VLM models that need minimum dimensions (e.g., Qwen3-VL requires 32x32).

    Args:
        crop_image: PIL Image of the current crop
        bbox: Original bbox [x1, y1, x2, y2]
        image_original: Full original PIL Image
        min_size: Minimum required size (default: 32)

    Returns:
        Expanded PIL Image that meets min_size × min_size
    """
    width, height = crop_image.size

    # If already meets minimum size, return unchanged
    if width >= min_size and height >= min_size:
        return crop_image

    x1, y1, x2, y2 = bbox
    img_width, img_height = image_original.size

    # Calculate expansion needed
    needed_width = max(0, min_size - width)
    needed_height = max(0, min_size - height)

    # Horizontal expansion (try symmetric, respect borders)
    expand_left = needed_width // 2
    expand_right = needed_width - expand_left

    if x1 - expand_left < 0:
        deficit = expand_left - x1
        expand_left = x1
        expand_right += deficit

    if x2 + expand_right > img_width:
        deficit = (x2 + expand_right) - img_width
        expand_right = img_width - x2
        expand_left += deficit
        if x1 - expand_left < 0:
            expand_left = x1

    # Vertical expansion
    expand_top = needed_height // 2
    expand_bottom = needed_height - expand_top

    if y1 - expand_top < 0:
        deficit = expand_top - y1
        expand_top = y1
        expand_bottom += deficit

    if y2 + expand_bottom > img_height:
        deficit = (y2 + expand_bottom) - img_height
        expand_bottom = img_height - y2
        expand_top += deficit
        if y1 - expand_top < 0:
            expand_top = y1

    # Calculate new coordinates
    new_x1 = max(0, x1 - expand_left)
    new_y1 = max(0, y1 - expand_top)
    new_x2 = min(img_width, x2 + expand_right)
    new_y2 = min(img_height, y2 + expand_bottom)

    # Extract expanded crop
    expanded_crop = image_original.crop((new_x1, new_y1, new_x2, new_y2))

    logger.debug(
        f"Crop expanded: {width}×{height} → {expanded_crop.size[0]}×{expanded_crop.size[1]}"
    )

    return expanded_crop


def image_to_base64_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI for VLM.

    Args:
        image: PIL Image to convert

    Returns:
        Data URI string (data:image/jpeg;base64,...)

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), color='red')
        >>> uri = image_to_base64_data_uri(img)
        >>> uri.startswith('data:image/jpeg;base64,')
        True
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=95)
    img_bytes = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def draw_bboxes(image: Image.Image, bboxes: list) -> Image.Image:
    """Draw red bounding boxes on image.

    Args:
        image: PIL Image
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]

    Returns:
        PIL Image with bboxes drawn (new copy)
    """
    # Convert PIL to cv2
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Draw red boxes
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
