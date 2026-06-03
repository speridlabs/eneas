"""
UniqueInstanceSegmenter - Unique instance segmentation with temporal tracking.

Based on SeC model for frame sequence object segmentation.
"""

import gc
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model_manager import ModelManager
from .types import SegmentationResult

logger = logging.getLogger(__name__)


class UniqueInstanceSegmenter:
    """
    Segmenter for unique instances with temporal tracking.

    Use cases:
    - "THAT specific statue"
    - "THAT red car"
    - "THAT particular person"

    A single instance that persists over time.
    Can disappear/reappear but remains THE SAME object.

    Returns binary masks (black & white) for each frame where:
    - Black (0) = Background
    - White (255) = Segmented object

    Example:
        >>> from eneas.segmentation import UniqueInstanceSegmenter
        >>> segmenter = UniqueInstanceSegmenter()  # Requires CUDA GPU
        >>> result = segmenter.segment(
        ...     frames_path="/path/to/frames",
        ...     points=[(100, 200), (150, 250)],
        ...     annotation_frame="frame_0050.jpg"
        ... )
        >>> print(f"Segmented {result.num_frames} frames")
        >>> # Access binary masks (always available in memory)
        >>> mask_frame_0 = result.masks[0]  # numpy array (H, W) with 0 and 255
        >>> # Optionally save to disk
        >>> result = segmenter.segment(..., save_masks=True)
        >>> mask_image = cv2.imread(result.mask_paths[0], cv2.IMREAD_GRAYSCALE)
    """

    SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png")
    DEFAULT_MEMORY_CLEANUP_INTERVAL = 10
    DEFAULT_PROGRESS_LOG_INTERVAL = 20

    # SAM encoder configurations
    SAM_ENCODERS = {
        # SAM 2.1 (Latest)
        "tiny": "sam2.1/sam2.1_hiera_t.yaml",
        "small": "sam2.1/sam2.1_hiera_s.yaml",
        "base": "sam2.1/sam2.1_hiera_b+.yaml",
        "large": "sam2.1/sam2.1_hiera_l.yaml",
        # LongSAM 2.1 (Default, better temporal consistency for frame sequences)
        "long-tiny": "longsam2.1/longsam2.1_hiera_t.yaml",
        "long-small": "longsam2.1/longsam2.1_hiera_s.yaml",
        "long-base": "longsam2.1/longsam2.1_hiera_b+.yaml",
        "long-large": "longsam2.1/longsam2.1_hiera_l.yaml",
        # SAM 2.0 (Legacy)
        "legacy-tiny": "sam2/sam2_hiera_t.yaml",
        "legacy-small": "sam2/sam2_hiera_s.yaml",
        "legacy-base": "sam2/sam2_hiera_b+.yaml",
        "legacy-large": "sam2/sam2_hiera_l.yaml",
    }

    def __init__(
        self,
        segmentation_model_path: str | None = None,
        grounding_model_path: str | None = None,
        sam_encoder: str = "long-large",
        device: str | None = None,
        default_output_dir: str = "./outputs",
        model_config_overrides: dict[str, str] | None = None,
        memory_cleanup_interval: int = 10,
    ):
        """
        Initialize the segmenter.

        Args:
            segmentation_model_path: Path to SeC model directory. If None, auto-downloads from HuggingFace
            grounding_model_path: Path to Florence-2 model directory. If None, auto-downloads when needed
            sam_encoder: SAM encoder variant. Options:
                - LongSAM 2.1 (best for temporal tracking): 'long-tiny', 'long-small', 'long-base', 'long-large' (default)
                - SAM 2.1: 'tiny', 'small', 'base', 'large'
                - SAM 2.0: 'legacy-tiny', 'legacy-small', 'legacy-base', 'legacy-large'
            device: Device to use ('cuda' recommended). If None, auto-detects CUDA availability
            default_output_dir: Default directory for segmentation outputs
            model_config_overrides: Additional Hydra config overrides for the segmentation model
            memory_cleanup_interval: Clean GPU memory every N frames (default: 10)

        Environment Variables:
            HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)

        Note:
            Requires CUDA GPU with bfloat16 support. CPU inference is not supported.

        Examples:
            >>> segmenter = UniqueInstanceSegmenter()
            >>> segmenter = UniqueInstanceSegmenter(sam_encoder="long-small")
            >>> segmenter = UniqueInstanceSegmenter(segmentation_model_path="/path/to/SeC-4B")
            >>> segmenter = UniqueInstanceSegmenter(device="cuda:1")
        """
        if sam_encoder not in self.SAM_ENCODERS:
            available = ", ".join(f"'{k}'" for k in self.SAM_ENCODERS.keys())
            raise ValueError(
                f"Invalid sam_encoder: '{sam_encoder}'. Available options: {available}"
            )

        self.sam_encoder = sam_encoder
        self.sam_config_path = self.SAM_ENCODERS[sam_encoder]
        logger.info(f"Using SAM encoder: {sam_encoder} ({self.sam_config_path})")

        if segmentation_model_path is not None:
            self.segmentation_model_path = segmentation_model_path
            self._auto_download_segmentation_model = False
            logger.info(f"Using segmentation model from: {segmentation_model_path}")
        else:
            self.segmentation_model_path = None
            self._auto_download_segmentation_model = True
            logger.info("Segmentation model will auto-download on first use")

        if grounding_model_path is not None:
            self.grounding_model_path = grounding_model_path
            self._auto_download_grounding_model = False
            logger.info(f"Using grounding model from: {grounding_model_path}")
        else:
            self.grounding_model_path = None
            self._auto_download_grounding_model = True

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                logger.warning(
                    "No CUDA device detected. SeC-4B requires CUDA GPU with bfloat16 support. "
                    "Inference will likely fail on CPU."
                )

        self.default_output_dir = default_output_dir
        self.memory_cleanup_interval = memory_cleanup_interval

        base_overrides = {
            "++model.non_overlap_masks": "false",
            "++model.grounding_encoder_config": self.sam_config_path,
        }
        if model_config_overrides:
            base_overrides.update(model_config_overrides)
        self.model_config_overrides = base_overrides

        self.segmentation_model = None
        self.segmentation_tokenizer = None
        self.grounding_model = None
        self.grounding_processor = None

        logger.info(f"UniqueInstanceSegmenter initialized with device: {self.device}")

    def optimize_cuda_memory(self) -> None:
        """
        Optimize CUDA memory allocation to reduce fragmentation.

        This method clears the CUDA cache and enables expandable memory segments,
        which helps prevent Out-of-Memory errors when processing long frame sequences or
        when GPU memory is limited. Only effective when using CUDA device.

        Call this method before segmentation if you experience memory issues.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("CUDA memory optimizations applied")

    def _validate_inputs(
        self,
        frames_path: str | list[str],
        points: list[tuple[int, int]],
        labels: list[int] | None,
    ) -> None:
        """
        Validate all input parameters.

        Args:
            frames_path: Path to frames directory or list of frame paths
            points: List of (x, y) coordinates
            labels: List of point labels (1 or 0)

        Raises:
            ValueError: If any input is invalid
            FileNotFoundError: If frames_path doesn't exist
        """
        # Validate frames_path
        if isinstance(frames_path, str):
            if not os.path.isdir(frames_path):
                raise FileNotFoundError(f"Frames directory not found: {frames_path}")
        else:
            raise NotImplementedError(
                "List of frame paths is not yet implemented. "
                "Please provide a directory path containing ordered frames."
            )

        # Validate points
        if not points:
            raise ValueError("At least one point must be provided")

        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in points):
            raise ValueError("Each point must be a tuple or list of two integers (x, y)")

        # Validate labels
        if labels is not None:
            if len(labels) != len(points):
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of points ({len(points)})"
                )
            if not all(label in (0, 1) for label in labels):
                raise ValueError("Labels must be 0 (negative) or 1 (positive)")

    def _load_segmentation_model(self):
        """Load SeC segmentation model lazily on first use.

        Raises:
            ImportError: If SeC modules cannot be imported
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If auto-download fails
        """
        if self.segmentation_model is not None:
            return

        if self._auto_download_segmentation_model:
            logger.info("Auto-downloading SeC-4B model from HuggingFace...")
            try:
                model_manager = ModelManager()
                downloaded_path = model_manager.download("OpenIXCLab/SeC-4B")
                self.segmentation_model_path = str(downloaded_path)
                logger.info(f"Model ready at: {downloaded_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Auto-download failed: {e}\n\n"
                    "You can manually download the model:\n"
                    "  1. Visit: https://huggingface.co/OpenIXCLab/SeC-4B\n"
                    "  2. Download and extract\n"
                    "  3. Pass: UniqueInstanceSegmenter(segmentation_model_path='/path/to/SeC-4B')"
                ) from e

        logger.info(f"Loading SeC model from {self.segmentation_model_path}...")

        model_path = Path(self.segmentation_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.segmentation_model_path}\n\n"
                "Options:\n"
                "  1. Auto-download: UniqueInstanceSegmenter()\n"
                "  2. Pass parameter: UniqueInstanceSegmenter(segmentation_model_path='/path/to/SeC-4B')\n"
                "  3. Manual download: https://huggingface.co/OpenIXCLab/SeC-4B"
            )

        try:
            from transformers import AutoTokenizer

            from eneas.vendor.SeC.inference.configuration_sec import SeCConfig
            from eneas.vendor.SeC.inference.modeling_sec import SeCModel
        except ImportError as e:
            raise ImportError(
                f"Failed to import SeC modules: {e}. "
                "This is an internal error with the vendored SeC code."
            ) from e

        if self.device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        config = SeCConfig.from_pretrained(str(model_path), trust_remote_code=True)

        hydra_overrides = [
            f"++model.{k.replace('++model.', '')}={v}"
            for k, v in self.model_config_overrides.items()
        ]
        config.hydra_overrides_extra = hydra_overrides

        if hasattr(config, "vision_config"):
            config.vision_config.use_flash_attn = False

        self.segmentation_model = (
            SeCModel.from_pretrained(
                str(model_path), config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(self.device)
        )

        self.segmentation_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")

    def _load_grounding_model(self):
        """Load grounding model lazily when needed for text-based segmentation.

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
                    "  3. Pass: UniqueInstanceSegmenter(grounding_model_path='/path/to/model')"
                ) from e

        logger.info(f"Loading grounding model from {self.grounding_model_path}...")

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.grounding_model = (
            AutoModelForCausalLM.from_pretrained(
                self.grounding_model_path, trust_remote_code=True, torch_dtype="auto"
            )
            .eval()
            .to(self.device)
        )

        self.grounding_processor = AutoProcessor.from_pretrained(
            self.grounding_model_path, trust_remote_code=True
        )

        logger.info("Grounding model loaded successfully")

    def _text_to_bbox(self, text: str, frame_image: Image) -> list[float]:
        """Use grounding model to detect object bounding box from text description.

        Args:
            text: Text description of the object
            frame_image: PIL Image of the frame

        Returns:
            Bounding box [x1, y1, x2, y2]

        Raises:
            ValueError: If no objects found for the text
        """
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task_prompt + text

        inputs = self.grounding_processor(text=prompt, images=frame_image, return_tensors="pt").to(
            self.device, torch.float16
        )

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

        bboxes = parsed_answer["<OPEN_VOCABULARY_DETECTION>"]["bboxes"]
        if not bboxes:
            raise ValueError(f"Grounding model could not detect any objects for text: '{text}'")

        bbox = bboxes[0]
        logger.info(f"Grounding model detected bbox: {bbox} for text: '{text}'")

        return bbox

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

    def _resolve_frame_index(
        self, annotation_frame: str | None, frame_names: list[str]
    ) -> tuple[int, str]:
        """Resolve annotation frame name to index and full name.

        Args:
            annotation_frame: Name of annotation frame (or None for first frame)
            frame_names: List of all available frame names

        Returns:
            Tuple of (frame_index, annotation_frame_name)

        Raises:
            ValueError: If annotation_frame is not found
        """
        if annotation_frame is None:
            return 0, frame_names[0]

        frame_basename = os.path.basename(annotation_frame)

        if frame_basename not in frame_names:
            raise ValueError(
                f"Annotation frame '{frame_basename}' not found in frames directory. "
                f"Available frames: {frame_names[:5]}... "
                f"(total: {len(frame_names)})"
            )

        return frame_names.index(frame_basename), frame_basename

    def _validate_points_in_bounds(
        self, points: list[tuple[int, int]], image_shape: tuple[int, int, int]
    ) -> None:
        """Validate that all points are within image bounds.

        Args:
            points: List of (x, y) coordinates
            image_shape: Image shape (height, width, channels)

        Raises:
            ValueError: If any point is out of bounds
        """
        height, width = image_shape[:2]

        for i, (x, y) in enumerate(points):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Point {i} at ({x}, {y}) is out of image bounds. Image size: {width}x{height}"
                )

    def segment(
        self,
        frames_path: str | list[str],
        points: list[tuple[int, int]] | None = None,
        annotation_frame: str | None = None,
        labels: list[int] | None = None,
        text: str | None = None,
        output_dir: str | None = None,
        offload_frames_to_gpu: bool = False,
        save_masks: bool = False,
        save_debug: bool = False,
    ) -> SegmentationResult:
        """
        Segment a unique instance across multiple frames.

        Args:
            frames_path: Directory containing ordered frames
            points: List of (x, y) coordinates (mutually exclusive with text)
            annotation_frame: Frame to annotate (or None for first frame)
            labels: Point labels 1=positive, 0=negative (only with points)
            text: Text description of object (mutually exclusive with points)
            output_dir: Output directory
            offload_frames_to_gpu: Keep frames in GPU (faster, more VRAM)
            save_masks: Save masks to disk
            save_debug: Save debug visualizations (sam_debug/)

        Returns:
            SegmentationResult with binary masks

        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If paths don't exist
            RuntimeError: If segmentation fails
        """
        if text is not None and points is not None:
            raise ValueError("'text' and 'points' are mutually exclusive")
        if text is None and points is None:
            raise ValueError("Either 'text' or 'points' must be provided")
        if text is not None and labels is not None:
            raise ValueError("'labels' cannot be used with 'text'")

        if output_dir is None:
            output_dir = self.default_output_dir

        if points is not None:
            self._validate_inputs(frames_path, points, labels)
        else:
            if isinstance(frames_path, str):
                if not os.path.isdir(frames_path):
                    raise FileNotFoundError(f"Frames directory not found: {frames_path}")

        frames_dir = frames_path
        frame_names = self._get_frame_names(frames_dir)
        logger.info(f"Found {len(frame_names)} images")

        frame_idx, annotation_frame = self._resolve_frame_index(annotation_frame, frame_names)

        initial_frame_path = os.path.join(frames_dir, annotation_frame)
        initial_frame = Image.open(initial_frame_path)
        initial_frame_np = np.array(initial_frame)

        logger.info(
            f"Annotation frame: {annotation_frame} ({initial_frame_np.shape[1]}x{initial_frame_np.shape[0]})"
        )

        if text is not None:
            logger.info(f"Using text-based grounding: '{text}'")
            self._load_grounding_model()

            bbox = self._text_to_bbox(text, initial_frame)
            bbox_array = np.array(bbox, dtype=np.float32)

            del self.grounding_model
            del self.grounding_processor
            self.grounding_model = None
            self.grounding_processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Grounding model unloaded from GPU")

        self._load_segmentation_model()

        # Start pure inference timer (after all model loading)
        logger.info("Models loaded. Starting pure inference timer...")
        inference_start_time = time.time()

        if text is not None:
            inference_state = self.segmentation_model.grounding_encoder.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=not offload_frames_to_gpu,
                offload_state_to_cpu=False,
            )

            logger.info("Processing bounding box...")
            ann_obj_id = 1
            _, out_obj_ids, out_mask_logits = (
                self.segmentation_model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    box=bbox_array,
                    points=None,
                    labels=None,
                    clear_old_points=True,
                )
            )

            init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            initial_mask_path = None
            if save_masks:
                os.makedirs(output_dir, exist_ok=True)
                initial_mask_path = os.path.join(output_dir, "initial_mask.jpg")
                self._save_initial_mask_with_bbox(
                    initial_frame_np, init_mask, bbox, text, frame_idx, initial_mask_path
                )
                logger.info(f"Initial mask saved: {initial_mask_path}")

            metadata = {
                "annotation_frame": annotation_frame,
                "text": text,
                "bbox": bbox,
                "mode": "text-based",
                "num_frames_total": len(frame_names),
                "offload_frames_to_gpu": offload_frames_to_gpu,
            }
        else:
            if labels is None:
                labels = [1] * len(points)

            logger.info(
                f"Using {len(points)} points in frame '{annotation_frame}' (index {frame_idx})"
            )

            for i, ((x, y), label) in enumerate(zip(points, labels, strict=True)):
                point_type = "POSITIVE" if label == 1 else "NEGATIVE"
                logger.debug(f"  Point {i + 1}: ({x}, {y}) - {point_type}")

            self._validate_points_in_bounds(points, initial_frame_np.shape)

            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, np.int32)

            inference_state = self.segmentation_model.grounding_encoder.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=not offload_frames_to_gpu,
                offload_state_to_cpu=False,
            )

            logger.info("Processing initial points...")
            ann_obj_id = 1
            _, out_obj_ids, out_mask_logits = (
                self.segmentation_model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    points=points_array,
                    labels=labels_array,
                )
            )

            init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            initial_mask_path = None
            if save_masks:
                os.makedirs(output_dir, exist_ok=True)
                initial_mask_path = os.path.join(output_dir, "initial_mask.jpg")
                self._save_initial_mask(
                    initial_frame_np, init_mask, points, labels, frame_idx, initial_mask_path
                )
                logger.info(f"Initial mask saved: {initial_mask_path}")

            metadata = {
                "annotation_frame": annotation_frame,
                "points": points,
                "labels": labels,
                "mode": "point-based",
                "num_frames_total": len(frame_names),
                "offload_frames_to_gpu": offload_frames_to_gpu,
            }

        # Propagate segmentation
        frame_segments = self._propagate_segmentation(
            inference_state, init_mask, frame_idx, len(frame_names)
        )

        logger.info(f"Propagation completed ({len(frame_segments)} frames)")

        # Convert frame_segments to binary masks dictionary
        binary_masks = {}
        for frame_idx, segments in frame_segments.items():
            # Get mask for object ID 1 (the single tracked instance)
            mask = segments[1]
            h, w = mask.shape[-2:]
            mask_binary = mask.reshape(h, w).astype(np.uint8) * 255  # 0 or 255
            binary_masks[frame_idx] = mask_binary

        # Save SAM segmentation debug (overlay masks on images)
        if save_debug:
            sam_debug_dir = os.path.join(output_dir, "sam_debug")
            os.makedirs(sam_debug_dir, exist_ok=True)

            logger.info("Saving SAM debug visualizations...")
            for frame_idx in sorted(binary_masks.keys()):
                mask = binary_masks[frame_idx]

                # Load original frame
                frame_path = os.path.join(frames_dir, frame_names[frame_idx])
                frame_img = cv2.imread(frame_path)

                # Create overlay with blue color for mask
                overlay = frame_img.copy()
                overlay[mask > 0] = [255, 100, 0]  # Blue in BGR

                # Blend original image with overlay (60% original, 40% overlay)
                blended = cv2.addWeighted(frame_img, 0.6, overlay, 0.4, 0)

                # Save
                debug_path = os.path.join(sam_debug_dir, frame_names[frame_idx])
                cv2.imwrite(debug_path, blended)

            logger.info(f"SAM debug visualizations saved to: {sam_debug_dir}")

        # Save binary masks to disk if requested
        mask_paths = []
        if save_masks:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)

            logger.info("Saving binary masks...")
            for frame_idx in sorted(binary_masks.keys()):
                mask = binary_masks[frame_idx]
                mask_filename = (
                    frame_names[frame_idx].replace(".jpg", ".png").replace(".jpeg", ".png")
                )
                mask_path = os.path.join(masks_dir, mask_filename)

                # Save as PNG (lossless, black & white)
                cv2.imwrite(mask_path, mask)
                mask_paths.append(mask_path)

            logger.info(f"Masks saved to: {masks_dir}")

        result = SegmentationResult(
            masks=binary_masks,
            num_frames=len(binary_masks),
            output_dir=output_dir,
            mask_paths=mask_paths,
            metadata=metadata,
            initial_mask_path=initial_mask_path,
        )

        # Calculate and log pure inference stats
        inference_end_time = time.time()
        pure_inference_time = inference_end_time - inference_start_time
        pure_fps = len(binary_masks) / pure_inference_time if pure_inference_time > 0 else 0.0

        logger.info("Segmentation completed successfully!")
        logger.info(f"Generated {len(binary_masks)} binary masks")
        logger.info("==================================================")
        logger.info("Pure Inference Stats:")
        logger.info(f"  Total Time: {pure_inference_time:.4f}s")
        logger.info(f"  FPS: {pure_fps:.2f}")
        logger.info(
            f"  Latency per frame: {1 / pure_fps:.4f}s"
            if pure_fps > 0
            else "  Latency per frame: N/A"
        )
        logger.info("==================================================")

        return result

    def _propagate_segmentation(
        self, inference_state, init_mask: np.ndarray, frame_idx: int, total_frames: int
    ) -> dict[int, dict[int, np.ndarray]]:
        """
        Propagate segmentation bidirectionally from initial frame.

        Args:
            inference_state: SeC inference state
            init_mask: Initial segmentation mask
            frame_idx: Index of initial frame
            total_frames: Total number of frames

        Returns:
            Dictionary mapping frame indices to segmentation masks
        """
        logger.info(f"Propagating segmentation across {total_frames} frames...")
        frame_segments = {}

        # Forward propagation
        logger.info(f"  Forward propagation from frame {frame_idx}...")
        frame_count = 0

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.segmentation_model.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            reverse=False,
            init_mask=init_mask,
            tokenizer=self.segmentation_tokenizer,
        ):
            frame_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            frame_count += 1

            # Periodic cleanup
            if frame_count % self.memory_cleanup_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Progress logging
            if frame_count % self.DEFAULT_PROGRESS_LOG_INTERVAL == 0:
                logger.info(f"    Processed {frame_count} frames...")

        logger.info(f"  Forward propagation completed ({frame_count} frames)")

        # Backward propagation
        if frame_idx > 0:
            logger.info(f"  Backward propagation from frame {frame_idx - 1}...")
            frame_count = 0

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.segmentation_model.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx - 1,
                reverse=True,
                init_mask=init_mask,
                tokenizer=self.segmentation_tokenizer,
            ):
                frame_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                frame_count += 1

                # Periodic cleanup
                if frame_count % self.memory_cleanup_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Progress logging
                if frame_count % self.DEFAULT_PROGRESS_LOG_INTERVAL == 0:
                    logger.info(f"    Processed {frame_count} frames...")

            logger.info(f"  Backward propagation completed ({frame_count} frames)")

        return frame_segments

    def _save_initial_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
        frame_idx: int,
        output_path: str,
    ) -> None:
        """Save visualization of initial mask with annotated points.

        Args:
            frame: Original frame image (RGB format)
            mask: Segmentation mask
            points: List of annotation points
            labels: Point labels (1=positive, 0=negative)
            frame_idx: Frame index
            output_path: Path to save visualization
        """
        # Convert RGB to BGR for OpenCV
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Create mask overlay (blue color)
        h, w = mask.shape[-2:]
        mask_binary = mask.reshape(h, w).astype(bool)
        overlay = vis_frame.copy()
        overlay[mask_binary] = [255, 144, 30]  # BGR: blue overlay

        # Blend original and overlay (60% original, 40% overlay)
        vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)

        # Draw points
        for (x, y), label in zip(points, labels, strict=True):
            if label == 1:
                # Green star for positive points
                color = (0, 255, 0)  # BGR
            else:
                # Red star for negative points
                color = (0, 0, 255)  # BGR

            # Draw star marker
            cv2.drawMarker(
                vis_frame,
                (int(x), int(y)),
                color,
                markerType=cv2.MARKER_STAR,
                markerSize=15,
                thickness=2,
            )

            # Add white border to marker for visibility
            cv2.drawMarker(
                vis_frame,
                (int(x), int(y)),
                (255, 255, 255),
                markerType=cv2.MARKER_STAR,
                markerSize=17,
                thickness=1,
            )

        # Add title text
        title = f"Initial Mask (Frame {frame_idx})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(title, font, font_scale, thickness)

        # Draw text background (semi-transparent black)
        cv2.rectangle(
            vis_frame,
            (5, 5),
            (15 + text_width, 15 + text_height + baseline),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            vis_frame,
            title,
            (10, 10 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        cv2.imwrite(output_path, vis_frame)

    def _save_initial_mask_with_bbox(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: list[float],
        text: str,
        frame_idx: int,
        output_path: str,
    ) -> None:
        """Save visualization of initial mask with bounding box (text mode).

        Args:
            frame: Original frame (RGB)
            mask: Segmentation mask
            bbox: Bounding box [x1, y1, x2, y2]
            text: Text description
            frame_idx: Frame index
            output_path: Save path
        """
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = mask.shape[-2:]
        mask_binary = mask.reshape(h, w).astype(bool)
        overlay = vis_frame.copy()
        overlay[mask_binary] = [255, 144, 30]
        vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

        label = f"'{text}'"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_y = max(y1 - 10, text_height + 5)

        cv2.rectangle(
            vis_frame,
            (x1, text_y - text_height - 5),
            (x1 + text_width + 5, text_y + baseline),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            vis_frame,
            label,
            (x1 + 2, text_y - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        title = f"Initial Mask (Frame {frame_idx}) - Text-based"
        (title_width, title_height), baseline = cv2.getTextSize(title, font, 1.0, 2)

        cv2.rectangle(
            vis_frame, (5, 5), (15 + title_width, 15 + title_height + baseline), (0, 0, 0), -1
        )

        cv2.putText(
            vis_frame, title, (10, 10 + title_height), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imwrite(output_path, vis_frame)
