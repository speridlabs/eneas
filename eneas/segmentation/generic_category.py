"""
GenericCategorySegmenter - Generic category segmentation for multiple instances.

Based on Florence-2 for object detection and grounding.
"""

import base64
import io
import logging
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model_manager import ModelManager
from .types import SegmentationResult
from .utils import (
    draw_bboxes,
    expand_crop_to_minimum_size,
    mask_overlapping_boxes,
    non_max_suppression,
    smart_convert_to_plural,
    smart_convert_to_singular,
)

logger = logging.getLogger(__name__)


class GenericCategorySegmenter:
    """
    Segmenter for generic categories (multiple instances per frame).

    Use cases:
    - "all chairs"
    - "all cars"
    - "all people"

    Multiple different instances, can vary frame by frame.
    No temporal tracking - each frame is processed independently.

    Returns binary masks (black & white) for each detected instance.

    Example:
        >>> from eneas.segmentation import GenericCategorySegmenter
        >>> segmenter = GenericCategorySegmenter()
        >>> result = segmenter.segment(
        ...     frames_path="/path/to/frames",
        ...     category="chair"
        ... )
        >>> print(f"Detected {result.num_frames} frames")
        >>> # Access masks for first frame
        >>> frame_0_masks = result.masks[0]  # List of masks for frame 0
    """

    SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        grounding_model_path: str | None = None,
        image_text_model_path: str | None = None,
        sam2_model_path: str | None = None,
        device: str | None = None,
        default_output_dir: str = "./outputs",
        vlm_model: str = "qwen3-vl:2b-instruct-q8_0",
    ):
        """
        Initialize the segmenter.

        Args:
            grounding_model_path: Path to Florence-2 model directory. If None, auto-downloads from HuggingFace
            image_text_model_path: Path to image-text model (SigLIP) directory. If None, auto-downloads from HuggingFace
            sam2_model_path: Path to SAM2 checkpoint file (.pt). If None, auto-downloads SAM2.1 large model
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects CUDA availability
            default_output_dir: Default directory for segmentation outputs
            vlm_model: Ollama model name for VLM validation. Default: "qwen3-vl:2b-instruct-q8_0"
                      Alternative: "qwen3-vl:4b-instruct-q8_0" (higher quality, more VRAM)

        Environment Variables:
            HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)

        Examples:
            >>> segmenter = GenericCategorySegmenter()
            >>> segmenter = GenericCategorySegmenter(device="cuda")
            >>> segmenter = GenericCategorySegmenter(grounding_model_path="/path/to/Florence-2")
            >>> segmenter = GenericCategorySegmenter(sam2_model_path="/path/to/sam2.1_hiera_large.pt")
            >>> # Use larger VLM for better quality
            >>> segmenter = GenericCategorySegmenter(vlm_model="qwen3-vl:4b-instruct-q8_0")
        """

        if grounding_model_path is not None:
            self.grounding_model_path = grounding_model_path
            self._auto_download_grounding_model = False
            logger.info(f"Using grounding model from: {grounding_model_path}")
        else:
            self.grounding_model_path = None
            self._auto_download_grounding_model = True
            logger.info("Grounding model will auto-download on first use")

        if image_text_model_path is not None:
            self.image_text_model_path = image_text_model_path
            self._auto_download_image_text_model = False
            logger.info(f"Using image-text model from: {image_text_model_path}")
        else:
            self.image_text_model_path = None
            self._auto_download_image_text_model = True
            logger.info("Image-text model will auto-download on first use")

        # Store VLM model name for Ollama
        self.vlm_model_name = vlm_model

        # Warn if using untested model
        supported_vlm_models = ["qwen3-vl:2b-instruct-q8_0", "qwen3-vl:4b-instruct-q8_0"]
        if vlm_model not in supported_vlm_models:
            logger.warning(
                f"VLM model '{vlm_model}' has not been tested. "
                f"Supported models: {', '.join(supported_vlm_models)}"
            )

        logger.info(f"VLM model (Ollama): {vlm_model}")

        if sam2_model_path is not None:
            self.sam2_model_path = sam2_model_path
            self._auto_download_sam2_model = False
            logger.info(f"Using SAM2 model from: {sam2_model_path}")
        else:
            self.sam2_model_path = None
            self._auto_download_sam2_model = True
            logger.info("SAM2 model will auto-download on first use")

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.default_output_dir = default_output_dir

        self.grounding_model = None
        self.grounding_processor = None
        self.image_text_model = None
        self.image_text_processor = None
        self.image_text_logit_bias = -10.0
        self.vlm_model = None

        self.sam2_predictor = None
        self.sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_vendor_path = os.path.join(os.path.dirname(__file__), "..", "vendor", "sam2")

        # Initialize model manager for auto-downloads
        self.model_manager = ModelManager()

        logger.info(f"GenericCategorySegmenter initialized with device: {self.device}")

    def _load_grounding_model(self):
        """Load Florence-2 grounding model lazily on first use.

        Raises:
            ImportError: If transformers cannot be imported
            RuntimeError: If auto-download fails
        """
        if self.grounding_model is not None:
            return

        grounding_model_id = "microsoft/Florence-2-large"

        if self._auto_download_grounding_model:
            logger.info(
                f"Auto-downloading grounding model ({grounding_model_id}) from HuggingFace..."
            )
            try:
                model_manager = ModelManager()
                downloaded_path = model_manager.download(grounding_model_id)
                self.grounding_model_path = str(downloaded_path)
                logger.info(f"Grounding model ready at: {downloaded_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Auto-download failed: {e}\n\n"
                    "You can manually download the model:\n"
                    f"  1. Visit: https://huggingface.co/{grounding_model_id}\n"
                    "  2. Download and extract\n"
                    "  3. Pass: GenericCategorySegmenter(grounding_model_path='/path/to/model')"
                ) from e

        logger.info(f"Loading grounding model from {self.grounding_model_path}...")

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.grounding_model = (
            AutoModelForCausalLM.from_pretrained(
                self.grounding_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

        self.grounding_processor = AutoProcessor.from_pretrained(
            self.grounding_model_path, trust_remote_code=True
        )

        logger.info("Grounding model loaded successfully")

    def _load_image_text_model(self):
        """Load SigLIP image-text model lazily on first use.

        Raises:
            ImportError: If transformers cannot be imported
            RuntimeError: If auto-download fails
        """
        if self.image_text_model is not None:
            return

        image_text_model_id = "google/siglip2-base-patch16-naflex"

        if self._auto_download_image_text_model:
            logger.info(
                f"Auto-downloading image-text model ({image_text_model_id}) from HuggingFace..."
            )
            try:
                model_manager = ModelManager()
                downloaded_path = model_manager.download(image_text_model_id)
                self.image_text_model_path = str(downloaded_path)
                logger.info(f"Image-text model ready at: {downloaded_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Auto-download failed: {e}\n\n"
                    "You can manually download the model:\n"
                    f"  1. Visit: https://huggingface.co/{image_text_model_id}\n"
                    "  2. Download and extract\n"
                    "  3. Pass: GenericCategorySegmenter(image_text_model_path='/path/to/model')"
                ) from e

        logger.info(f"Loading image-text model from {self.image_text_model_path}...")

        import torch.nn as nn
        from transformers import AutoModel, AutoProcessor

        if self.device == "cuda":
            self.image_text_model = AutoModel.from_pretrained(
                self.image_text_model_path, device_map="auto", torch_dtype=torch.float16
            ).eval()
        else:
            self.image_text_model = (
                AutoModel.from_pretrained(self.image_text_model_path).to(self.device).eval()
            )

        # Apply logit bias for probability calibration
        self.image_text_model.logit_bias = nn.Parameter(torch.tensor([self.image_text_logit_bias]))
        logger.info(f"Image-text logit bias applied: {self.image_text_logit_bias}")

        self.image_text_processor = AutoProcessor.from_pretrained(self.image_text_model_path)

        logger.info("Image-text model loaded successfully")

    def _load_vlm_model(self):
        """Verify Ollama VLM model is available.

        Raises:
            ImportError: If ollama cannot be imported
            RuntimeError: If Ollama server is not running or model not available
        """
        if self.vlm_model is not None:
            return

        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "ollama is required for VLM validation.\n"
                "Install it with: pip install ollama\n"
                "And ensure Ollama server is running: ollama serve"
            ) from e

        logger.info(f"Checking Ollama model: {self.vlm_model_name}")

        try:
            # Try to pull/verify model exists
            ollama.pull(self.vlm_model_name)
            logger.info(f"VLM model ready: {self.vlm_model_name}")
        except Exception as e:
            logger.warning(f"Could not pull model (server may be down or model unavailable): {e}")
            logger.info("Will attempt to use model anyway (may already be cached)")

        # Mark VLM model as loaded and ready for inference
        self.vlm_model = True

        logger.info("Ollama VLM ready")

    def _load_sam2_model(self):
        """Load SAM2.1 model lazily on first use.

        Raises:
            ImportError: If sam2 cannot be imported
            RuntimeError: If auto-download fails or model loading fails
        """
        if self.sam2_predictor is not None:
            return

        if self._auto_download_sam2_model:
            logger.info("Auto-downloading SAM2.1 checkpoint from direct URL...")
            try:
                sam2_url = (
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
                )
                checkpoint_path = self.model_manager.download_url(sam2_url, "sam2.1_hiera_large.pt")
                logger.info(f"SAM2 model ready at: {checkpoint_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Auto-download failed: {e}\n\n"
                    "You can manually download the model:\n"
                    f"  1. Visit: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt\n"
                    "  2. Save as sam2.1_hiera_large.pt\n"
                    "  3. Pass: GenericCategorySegmenter(sam2_model_path='/path/to/sam2.1_hiera_large.pt')"
                ) from e
        else:
            # User provided path to checkpoint file
            checkpoint_path = Path(self.sam2_model_path)
            if not checkpoint_path.exists():
                raise RuntimeError(f"SAM2 checkpoint not found: {checkpoint_path}")
            logger.info(f"Using SAM2 checkpoint from: {checkpoint_path}")

        # Config is always in vendor
        config_path = Path(self.sam2_vendor_path) / self.sam_model_cfg

        if not config_path.exists():
            raise RuntimeError(f"SAM2 config not found: {config_path}")

        # Load SAM2 model
        from eneas.vendor.sam2.build_sam import build_sam2
        from eneas.vendor.sam2.sam2_image_predictor import SAM2ImagePredictor

        # Build SAM2 model
        sam2_model = build_sam2(str(self.sam_model_cfg), str(checkpoint_path), device=self.device)

        # Create predictor
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        logger.info("SAM2 model loaded successfully")

    def _segment_bboxes_in_frame(
        self,
        frame_image: Image.Image,
        bboxes: list,
    ) -> list[np.ndarray]:
        """
        Segment multiple bounding boxes in a single frame using SAM2.1.

        Args:
            frame_image: PIL Image of the frame (RGB format)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            List of binary masks (H, W) with values 0 or 255 for each bbox
        """
        if len(bboxes) == 0:
            return []

        # Convert PIL to numpy array
        frame_image_np = np.array(frame_image)

        # Set image in predictor
        self.sam2_predictor.set_image(frame_image_np)

        # Convert bboxes to numpy array
        input_boxes = np.array(bboxes)

        # Predict masks
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Handle mask shape
        if len(masks.shape) == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        # Now masks is (num_boxes, H, W) bool

        # Convert to list of binary numpy masks (0 or 255)
        result_masks = []
        for mask in masks:
            # Fill small holes in mask (area <= 8 pixels)
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
            from eneas.vendor.SeC.inference.sam2.utils.misc import fill_holes_in_mask_scores
            mask_filled = fill_holes_in_mask_scores(mask_tensor, max_area=8)
            mask_filled_np = mask_filled.numpy()

            # Convert to binary (0 or 255)
            mask_binary = (mask_filled_np > 0).astype(np.uint8) * 255
            result_masks.append(mask_binary)

        return result_masks

    def _vlm_validate_single_crop(
        self,
        crop_image: Image.Image,
        target_text: str,
        num_predict: int = 2048,
        max_retries: int = 3,
    ) -> bool:
        """Validate a single crop using Ollama VLM with structured outputs.

        Args:
            crop_image: PIL Image of the crop (clean, no annotations)
            target_text: Target concept to validate (singular form, e.g., "person")
            num_predict: Maximum tokens for VLM response (default: 2048)
            max_retries: Maximum retry attempts if validation fails (default: 3)

        Returns:
            True if crop is validated as target, False otherwise
        """
        import ollama
        from pydantic import BaseModel

        # Define structured output schema
        class ValidationResult(BaseModel):
            reasoning: str
            is_target: bool

        # Convert image to base64
        img_byte_arr = io.BytesIO()
        crop_image.save(img_byte_arr, format="JPEG", quality=95)
        img_bytes = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Construct validation prompt
        prompt = f"""You are validating an object detection result.

TASK: Analyze the image and determine if it shows a **{target_text}**.

The image shows a cropped region from a larger scene. This region was detected by an AI system as possibly containing "{target_text}", but it may be a false positive.

CRITICAL THINKING QUESTIONS:
- What do you actually see in this image?
- Does it visually match the concept of "{target_text}"?
- Are you absolutely certain?
- Could this be a false positive (wrong detection)?

⚠️ IMPORTANT NOTES:
- The object may be partially visible or occluded (covered by other things) - this is still VALID if you can identify it
- Focus on what you SEE, not what the AI claimed to detect
- If detecting "person": ONLY real living humans count as TRUE. Statues, mannequins, dolls, paintings, photos, posters, or any artificial representations are FALSE.

Provide your response in JSON format with:
- "reasoning": Brief explanation of what you see and why it is/isn't {target_text}
- "is_target": true or false

Example responses:
{{"reasoning": "I see a real living person - natural skin texture, subtle movements or natural pose, wearing actual clothing.", "is_target": true}}
{{"reasoning": "This is clearly not a person - it's a wall with an electrical outlet and no human figure present.", "is_target": false}}
{{"reasoning": "This appears to be a statue or mannequin - rigid pose, uniform painted/plastic surface, artificial appearance, no signs of life.", "is_target": false}}
{{"reasoning": "I see a person in a photo/poster on the wall - this is a 2D image of a person, not an actual person in the scene.", "is_target": false}}"""

        for attempt in range(max_retries):
            try:
                logger.debug(f"VLM validation attempt {attempt + 1}/{max_retries}")

                messages = [{"role": "user", "content": prompt, "images": [img_base64]}]

                # Use structured outputs with Pydantic schema
                response = ollama.chat(
                    model=self.vlm_model_name,
                    messages=messages,
                    format=ValidationResult.model_json_schema(),
                    options={"temperature": 0.0, "num_predict": num_predict, "num_ctx": 8192},
                    keep_alive=-1,
                )

                # Parse and validate response using Pydantic
                result = ValidationResult.model_validate_json(response.message.content)

                logger.debug(f"VLM result: is_target={result.is_target}")
                logger.debug(f"VLM reasoning: {result.reasoning}")

                return result.is_target

            except Exception as e:
                logger.warning(f"VLM validation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Default: accept on failure to avoid blocking pipeline
                    logger.warning("VLM validation failed after all retries, defaulting to accept")
                    return True

        return True

    def _text_to_bbox(
        self,
        text: str,
        frame_image: Image.Image,
        accept_threshold: float = 0.90,
        reject_threshold: float = 0.10,
        save_debug: bool = False,
        output_dir: str = "",
        frame_name: str = "",
    ) -> tuple[list, list, bool]:
        """Detect and filter objects using multi-stage pipeline.

        Pipeline:
        1. Convert text to plural (once, for Florence)
        2. Detect with Florence-2 CAPTION_TO_PHRASE_GROUNDING
        3. Apply NMS to remove duplicates
        4. Convert text to singular (once, for SigLIP)
        5. Filter with image-text model semantic similarity
        6. VLM validation for uncertain boxes
        7. Return accepted + VLM-approved boxes

        Args:
            text: Text description of the object category
            frame_image: PIL Image of the frame
            accept_threshold: Threshold for accepting boxes automatically (default: 0.90)
            reject_threshold: Threshold for rejecting boxes automatically (default: 0.10)

        Returns:
            Tuple of (bboxes, labels, vlm_used) where vlm_used is True if VLM was called
        """
        # Stage 1: Convert to plural once (for Florence)
        text_plural = smart_convert_to_plural(text)
        logger.debug(f"Florence query: '{text}' → '{text_plural}'")

        # Stage 2: Florence-2 detection
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task_prompt + text_plural

        inputs = self.grounding_processor(text=prompt, images=frame_image, return_tensors="pt").to(
            self.device
        )
        inputs = {k: (v.to(self.grounding_model.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}

        generated_ids = self.grounding_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.grounding_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.grounding_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(frame_image.width, frame_image.height)
        )

        grounding_results = parsed_answer["<CAPTION_TO_PHRASE_GROUNDING>"]
        bboxes = grounding_results.get("bboxes", [])
        labels = grounding_results.get("labels", [])

        if not bboxes:
            logger.warning(f"No objects detected for text: '{text}'")
            return bboxes, labels, False

        logger.info(f"Florence detected {len(bboxes)} instances")

        # Save grounding debug (before NMS)
        if save_debug and len(bboxes) > 0:
            grounding_debug_dir = os.path.join(output_dir, "grounding_debug")
            grounding_img = draw_bboxes(frame_image.copy(), bboxes)
            grounding_path = os.path.join(grounding_debug_dir, f"{frame_name}.jpg")
            grounding_img.save(grounding_path, quality=95)
            logger.debug(f"Saved grounding debug: {grounding_path}")

        # Stage 3: Apply NMS
        if len(bboxes) > 1:
            original_count = len(bboxes)
            bboxes, labels = non_max_suppression(bboxes, labels, iou_threshold=0.70)
            removed_count = original_count - len(bboxes)
            if removed_count > 0:
                logger.info(f"NMS: Removed {removed_count} overlapping boxes")
                logger.info(f"After NMS: {len(bboxes)} instances")

        # Stage 4: Convert to singular once (for SigLIP)
        text_singular = smart_convert_to_singular(text)
        logger.debug(f"Image-text query: '{text}' → '{text_singular}'")

        # Stage 5: Image-text filtering
        if len(bboxes) > 0:
            accepted, rejected, uncertain, scores = self._image_text_filter_boxes(
                frame_image,
                bboxes,
                labels,
                text_singular,
                accept_threshold,
                reject_threshold,
                save_debug,
                output_dir,
                frame_name,
            )

            # Stage 6: VLM validation for uncertain boxes
            vlm_accepted = []
            vlm_used = False
            if len(uncertain) > 0 and self.vlm_model is not None:
                vlm_used = True
                logger.info(f"VLM validating {len(uncertain)} uncertain boxes...")

                for local_idx, global_idx in enumerate(uncertain):
                    bbox = bboxes[global_idx]
                    _label = labels[global_idx]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]

                    logger.debug(
                        f"VLM validating uncertain box {local_idx + 1}/{len(uncertain)} (global #{global_idx + 1})"
                    )

                    # Extract clean crop
                    crop = frame_image.crop((x1, y1, x2, y2))

                    # Mask overlapping regions
                    crop = mask_overlapping_boxes(crop, bbox, bboxes, global_idx, (x1, y1, x2, y2))

                    # Expand crop to minimum size (Qwen3-VL requires 32x32)
                    crop = expand_crop_to_minimum_size(crop, bbox, frame_image, min_size=32)

                    # Save VLM debug crop
                    if save_debug:
                        vlm_debug_dir = os.path.join(output_dir, "vlm_debug")
                        crop_path = os.path.join(
                            vlm_debug_dir, f"{frame_name}_vlm_crop_{global_idx + 1}.jpg"
                        )
                        crop.save(crop_path, quality=95)

                    # Validate with VLM
                    is_target = self._vlm_validate_single_crop(crop, text_singular)

                    if is_target:
                        logger.debug(f"VLM accepted box #{global_idx + 1}")
                        vlm_accepted.append(global_idx)
                    else:
                        logger.debug(f"VLM rejected box #{global_idx + 1}")

                logger.info(
                    f"VLM validation: {len(vlm_accepted)} accepted, {len(uncertain) - len(vlm_accepted)} rejected"
                )

            # Stage 7: Combine accepted + VLM-approved uncertain (rejected + VLM-rejected discarded)
            keep_indices = sorted(accepted + vlm_accepted)
            bboxes = [bboxes[i] for i in keep_indices]
            labels = [labels[i] for i in keep_indices]

            logger.info(f"Final result: {len(bboxes)} instances")

        return bboxes, labels, vlm_used

    def _image_text_filter_boxes(
        self,
        image: Image.Image,
        bboxes: list,
        labels: list,
        target_text: str,
        accept_threshold: float = 0.90,
        reject_threshold: float = 0.10,
        save_debug: bool = False,
        output_dir: str = "",
        frame_name: str = "",
    ) -> tuple[list, list, list, list]:
        """Filter bounding boxes using image-text model semantic similarity.

        Uses ensemble of prompts with MEAN strategy for robust filtering.

        Args:
            image: PIL Image (original, without boxes drawn)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            labels: List of labels from Florence
            target_text: Target concept (singular form, e.g., "person")
            accept_threshold: Threshold for accepting boxes automatically (default: 0.90)
            reject_threshold: Threshold for rejecting boxes automatically (default: 0.10)

        Returns:
            Tuple of:
            - accepted_indices: Indices of accepted boxes (score >= accept_threshold)
            - rejected_indices: Indices of rejected boxes (score < reject_threshold)
            - uncertain_indices: Indices of uncertain boxes (reject_threshold <= score < accept_threshold)
            - scores: List of similarity scores for each box
        """
        if len(bboxes) == 0:
            return [], [], [], []

        # Ensemble of prompt templates
        prompt_templates = [
            f"a photo of {target_text}",
            f"a photo of a {target_text}",
            f"This is a photo of {target_text}",
            f"This is a photo of a {target_text}",
            f"a cropped photo of {target_text}",
            f"a cropped photo of a {target_text}",
            f"an image of {target_text}",
            f"an image of a {target_text}",
            f"{target_text}",
            f"a {target_text}",
        ]

        # Remove duplicates maintaining order
        texts = []
        seen = set()
        for t in prompt_templates:
            if t not in seen:
                texts.append(t)
                seen.add(t)

        logger.info(f"Image-text filtering with {len(texts)} prompt variants (MEAN strategy)")
        logger.info(
            f"Accept threshold: >={accept_threshold}, Reject threshold: <{reject_threshold}"
        )

        # Step 1: Prepare all crops first
        all_crops = []
        for idx, (bbox, _label) in enumerate(zip(bboxes, labels, strict=True)):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Crop clean region
            crop = image.crop((x1, y1, x2, y2))

            # Mask overlapping regions
            crop = mask_overlapping_boxes(crop, bbox, bboxes, idx, (x1, y1, x2, y2))

            all_crops.append(crop)

            # Save image_text debug crops
            if save_debug:
                image_text_debug_dir = os.path.join(output_dir, "image_text_debug")
                crop_path = os.path.join(image_text_debug_dir, f"{frame_name}_crop_{idx + 1}.jpg")
                crop.save(crop_path, quality=95)

        # Step 2: Batch process all crops at once
        with torch.no_grad():
            inputs = self.image_text_processor(
                text=texts,
                images=all_crops,  # Process ALL crops in one batch
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(self.device)
            inputs = {k: (v.to(self.image_text_model.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}

            outputs = self.image_text_model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: [num_crops, num_prompts]
            probs = torch.sigmoid(logits_per_image)  # Shape: [num_crops, num_prompts]

        # Step 3: Process results for each crop individually
        scores = []
        accepted_indices = []
        rejected_indices = []
        uncertain_indices = []

        for idx, (_bbox, label) in enumerate(zip(bboxes, labels, strict=True)):
            # Extract scores for this specific crop
            crop_probs = probs[idx].cpu().numpy()  # Shape: [num_prompts]

            # MEAN strategy (average of all prompts)
            final_score = float(crop_probs.mean())

            # Stats for logging
            best_score = float(crop_probs.max())
            worst_score = float(crop_probs.min())
            best_prompt_idx = int(crop_probs.argmax())
            best_prompt = texts[best_prompt_idx]

            scores.append(final_score)

            # Classify according to thresholds
            if final_score >= accept_threshold:
                accepted_indices.append(idx)
                status = "ACCEPTED"
            elif final_score < reject_threshold:
                rejected_indices.append(idx)
                status = "REJECTED"
            else:
                uncertain_indices.append(idx)
                status = "UNCERTAIN"

            logger.debug(
                f"Box {idx + 1}: {label[:30]} | "
                f"MEAN={final_score:.4f} | "
                f"BEST='{best_prompt}'={best_score:.4f} | "
                f"WORST={worst_score:.4f} | "
                f"{status}"
            )

        logger.info(
            f"Image-text results: {len(accepted_indices)} accepted, "
            f"{len(rejected_indices)} rejected, {len(uncertain_indices)} uncertain"
        )

        return accepted_indices, rejected_indices, uncertain_indices, scores

    def segment(
        self,
        frames_path: str | list[str],
        category: str,
        output_dir: str | None = None,
        accept_threshold: float = 0.90,
        reject_threshold: float = 0.10,
        save_debug: bool = False,
        save_masks: bool = False,
    ) -> SegmentationResult:
        """
        Detect and segment instances of a category across multiple frames.

        Args:
            frames_path: Directory containing frames
            category: Category to detect (e.g., "chair", "person", "car")
            output_dir: Output directory for results
            accept_threshold: Image-text similarity threshold for auto-accepting boxes (default: 0.90)
            reject_threshold: Image-text similarity threshold for auto-rejecting boxes (default: 0.10)
            save_debug: Save debug visualizations (grounding_debug/, image_text_debug/, vlm_debug/, detections_debug/)
            save_masks: Save binary segmentation masks to disk (default: False)

        Returns:
            SegmentationResult with detection data and binary masks

        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If paths don't exist
            RuntimeError: If detection fails

        Examples:
            >>> segmenter = GenericCategorySegmenter()
            >>> result = segmenter.segment(
            ...     frames_path="./frames",
            ...     category="chair"
            ... )
            >>> # With masks
            >>> result = segmenter.segment(
            ...     frames_path="./frames",
            ...     category="person",
            ...     save_masks=True
            ... )
            >>> # Access masks: result.masks[frame_idx] returns list of masks
        """
        if output_dir is None:
            output_dir = self.default_output_dir

        # Validate frames_path
        if isinstance(frames_path, str):
            if not os.path.isdir(frames_path):
                raise FileNotFoundError(f"Frames directory not found: {frames_path}")
        else:
            raise NotImplementedError(
                "List of frame paths is not yet implemented. "
                "Please provide a directory path containing ordered frames."
            )

        # Load models
        self._load_grounding_model()
        self._load_image_text_model()
        self._load_vlm_model()

        # Load SAM2 model for segmentation
        self._load_sam2_model()

        # Start pure inference timer (after all model loading)
        logger.info("Models loaded. Starting pure inference timer...")
        inference_start_time = time.time()

        frames_dir = frames_path
        frame_names = self._get_frame_names(frames_dir)
        logger.info(f"Found {len(frame_names)} images")
        logger.info(f"Detecting category: '{category}'")

        # Create debug directories if needed
        if save_debug:
            grounding_debug_dir = os.path.join(output_dir, "grounding_debug")
            image_text_debug_dir = os.path.join(output_dir, "image_text_debug")
            vlm_debug_dir = os.path.join(output_dir, "vlm_debug")
            sam_debug_dir = os.path.join(output_dir, "sam_debug")
            detections_debug_dir = os.path.join(output_dir, "detections_debug")

            # Clean existing debug directories to avoid confusion with old files
            for debug_dir in [
                grounding_debug_dir,
                image_text_debug_dir,
                vlm_debug_dir,
                sam_debug_dir,
                detections_debug_dir,
            ]:
                if os.path.exists(debug_dir):
                    shutil.rmtree(debug_dir)
                    logger.info(f"Cleaned existing debug directory: {debug_dir}")

            os.makedirs(grounding_debug_dir, exist_ok=True)
            os.makedirs(image_text_debug_dir, exist_ok=True)
            os.makedirs(vlm_debug_dir, exist_ok=True)
            os.makedirs(sam_debug_dir, exist_ok=True)
            os.makedirs(detections_debug_dir, exist_ok=True)

            logger.info("Debug mode enabled - saving visualizations")

        # Process each frame independently
        all_detections = {}
        all_masks = {}
        vlm_usage_count = 0

        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = os.path.join(frames_dir, frame_name)
            frame_image = Image.open(frame_path).convert("RGB")

            logger.info(f"Processing frame {frame_idx + 1}/{len(frame_names)}: {frame_name}")

            # Get frame stem (without extension) for debug filenames
            frame_stem = Path(frame_name).stem

            # Detect and filter objects using full pipeline
            bboxes, labels, vlm_used = self._text_to_bbox(
                category,
                frame_image,
                accept_threshold,
                reject_threshold,
                save_debug,
                output_dir,
                frame_stem,
            )

            # Track VLM usage
            if vlm_used:
                vlm_usage_count += 1

            # Store detections for this frame
            frame_detections = []
            for bbox, label in zip(bboxes, labels, strict=True):
                frame_detections.append(
                    {
                        "bbox": bbox,
                        "label": label,
                    }
                )

            all_detections[frame_idx] = frame_detections

            # Segment bboxes using SAM2
            frame_masks = self._segment_bboxes_in_frame(frame_image, bboxes)
            all_masks[frame_idx] = frame_masks

            logger.info(f"  Segmented {len(frame_masks)} objects")

            # Save SAM segmentation debug (overlay masks on image)
            if save_debug and len(frame_masks) > 0:
                sam_debug_dir = os.path.join(output_dir, "sam_debug")

                # Convert PIL to numpy for overlay
                img_array = np.array(frame_image)

                # Create combined overlay with all masks
                overlay = img_array.copy()
                for mask in frame_masks:
                    overlay[mask > 0] = [0, 100, 255]  # Blue where mask is present

                # Blend original image with overlay (60% original, 40% overlay)
                blended = cv2.addWeighted(img_array, 0.6, overlay, 0.4, 0)

                # Convert back to PIL and save
                blended_img = Image.fromarray(blended)
                sam_path = os.path.join(sam_debug_dir, f"{frame_stem}.jpg")
                blended_img.save(sam_path, quality=95)

                logger.debug(f"Saved SAM debug visualization with {len(frame_masks)} masks")

            # Save detections debug (final result - always save, even if no detections)
            if save_debug:
                detections_debug_dir = os.path.join(output_dir, "detections_debug")
                if len(bboxes) > 0:
                    detections_img = draw_bboxes(frame_image.copy(), bboxes)
                else:
                    # No detections found - save original image without annotations
                    detections_img = frame_image.copy()
                detections_path = os.path.join(detections_debug_dir, f"{frame_stem}.jpg")
                detections_img.save(detections_path, quality=95)
                logger.debug(f"Saved detections debug: {detections_path}")

        # Calculate and log pure inference stats
        inference_end_time = time.time()
        pure_inference_time = inference_end_time - inference_start_time
        pure_fps = len(frame_names) / pure_inference_time if pure_inference_time > 0 else 0.0

        logger.info("Detection and segmentation completed successfully!")
        logger.info(f"Processed {len(frame_names)} frames")
        logger.info(f"==================================================")
        logger.info(f"Pure Inference Stats:")
        logger.info(f"  Total Time: {pure_inference_time:.4f}s")
        logger.info(f"  FPS: {pure_fps:.2f}")
        logger.info(
            f"  Latency per frame: {1 / pure_fps:.4f}s"
            if pure_fps > 0
            else "  Latency per frame: N/A"
        )
        logger.info(f"==================================================")

        # Calculate VLM usage percentage
        vlm_usage_percentage = (
            (vlm_usage_count / len(frame_names) * 100) if len(frame_names) > 0 else 0.0
        )

        # Save binary masks to disk if requested
        mask_paths = []
        if save_masks:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)

            logger.info("Saving binary masks...")
            for frame_idx in sorted(all_masks.keys()):
                frame_masks_list = all_masks[frame_idx]

                # Combine all masks for this frame using OR
                if len(frame_masks_list) > 0:
                    # Start with first mask
                    combined_mask = frame_masks_list[0].copy()
                    # OR with remaining masks
                    for mask in frame_masks_list[1:]:
                        combined_mask = combined_mask | mask
                else:
                    # No objects in this frame - create empty mask
                    # Get image dimensions from first frame
                    first_frame_path = os.path.join(frames_dir, frame_names[0])
                    first_frame = Image.open(first_frame_path).convert("RGB")
                    h, w = np.array(first_frame).shape[:2]
                    combined_mask = np.zeros((h, w), dtype=np.uint8)

                # Save as PNG (lossless, black & white)
                mask_filename = (
                    frame_names[frame_idx].replace(".jpg", ".png").replace(".jpeg", ".png")
                )
                mask_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_path, combined_mask)
                mask_paths.append(mask_path)

            logger.info(f"Masks saved to: {masks_dir}")

        result = SegmentationResult(
            masks=all_masks,
            num_frames=len(frame_names),
            output_dir=output_dir,
            mask_paths=mask_paths,
            metadata={
                "category": category,
                "detections": all_detections,
                "num_frames_total": len(frame_names),
                "accept_threshold": accept_threshold,
                "reject_threshold": reject_threshold,
                "vlm_usage_count": vlm_usage_count,
                "vlm_usage_percentage": vlm_usage_percentage,
            },
            initial_mask_path=None,
        )

        return result

    def _get_frame_names(self, frames_dir: str) -> list[str]:
        """Get sorted list of image files in directory.

        Args:
            frames_dir: Directory containing image frames

        Returns:
            Sorted list of frame filenames

        Raises:
            ValueError: If no valid image files found
        """
        frame_names = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith(self.SUPPORTED_IMAGE_FORMATS)]
        )

        if not frame_names:
            raise ValueError(
                f"No image files found in {frames_dir}. "
                f"Supported formats: {self.SUPPORTED_IMAGE_FORMATS}"
            )

        return frame_names
