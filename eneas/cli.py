"""
ENEAS CLI - Embedding-guided Neural Ensemble for Adaptive Segmentation

Command-line interface for frame sequence segmentation.
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import typer

from eneas.segmentation import GenericCategorySegmenter, UniqueInstanceSegmenter

app = typer.Typer(
    name="eneas",
    help="ENEAS - Embedding-guided Neural Ensemble for Adaptive Segmentation",
    add_completion=False,
)


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    # Also configure eneas loggers
    logging.getLogger("eneas").setLevel(level)


def parse_points(points_str: list[str]) -> list[tuple[int, int]]:
    """Parse point coordinates from CLI arguments."""
    result = []
    for i, point in enumerate(points_str):
        parts = point.split(",")
        if len(parts) != 2:
            typer.echo(f"Error: Point {i + 1} must be in format 'x,y', got '{point}'", err=True)
            raise typer.Exit(code=1)
        try:
            x, y = int(parts[0].strip()), int(parts[1].strip())
            result.append((x, y))
        except ValueError:
            typer.echo(
                f"Error: Point {i + 1} coordinates must be integers, got '{point}'", err=True
            )
            raise typer.Exit(code=1) from None
    return result


def parse_labels(labels_str: list[str] | None, num_points: int) -> list[int]:
    """Parse point labels from CLI arguments."""
    if not labels_str:
        # Default: all positive points
        return [1] * num_points

    if len(labels_str) != num_points:
        typer.echo(
            f"Error: Number of labels ({len(labels_str)}) must match number of points ({num_points})",
            err=True,
        )
        raise typer.Exit(code=1)

    result = []
    for i, label in enumerate(labels_str):
        try:
            val = int(label.strip())
            if val not in (0, 1):
                raise ValueError
            result.append(val)
        except ValueError:
            typer.echo(f"Error: Label {i + 1} must be 0 or 1, got '{label}'", err=True)
            raise typer.Exit(code=1) from None
    return result


def validate_paths(frames_path: Path, annotation_frame: str | None) -> None:
    """Validate input paths exist."""
    if not frames_path.exists():
        typer.echo(f"Error: Frames path does not exist: {frames_path}", err=True)
        raise typer.Exit(code=1)

    if not frames_path.is_dir():
        typer.echo(f"Error: Frames path is not a directory: {frames_path}", err=True)
        raise typer.Exit(code=1)

    # Check for image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [f for f in frames_path.iterdir() if f.suffix.lower() in image_extensions]
    if not image_files:
        typer.echo(f"Error: No image files found in: {frames_path}", err=True)
        raise typer.Exit(code=1)

    # Validate annotation frame if specified
    if annotation_frame:
        annotation_path = frames_path / annotation_frame
        if not annotation_path.exists():
            typer.echo(f"Error: Annotation frame not found: {annotation_path}", err=True)
            raise typer.Exit(code=1)


def print_banner():
    """Print welcome banner."""
    typer.echo("\n" + "=" * 70)
    typer.echo("  eneas - Frame Sequence Segmentation with Temporal Tracking")
    typer.echo("=" * 70 + "\n")


def print_config(config: dict):
    """Print configuration table."""
    typer.echo("Configuration:")
    typer.echo("-" * 70)
    for key, value in config.items():
        typer.echo(f"  {key:<30} {value}")
    typer.echo("-" * 70 + "\n")


def print_summary_unique_instance(result, elapsed_time: float) -> None:
    """Print unique instance segmentation results summary."""
    typer.echo("\n" + "=" * 70)
    typer.echo("  SEGMENTATION RESULTS")
    typer.echo("=" * 70)
    typer.echo(f"  Processed Frames:              {result.num_frames}")
    typer.echo(
        f"  Processing Time:               {elapsed_time:.2f}s ({result.num_frames / elapsed_time:.1f} fps)"
    )
    typer.echo(f"  Output Directory:              {result.output_dir}")

    if result.initial_mask_path:
        typer.echo(f"  Initial Mask Visualization:    {result.initial_mask_path}")

    if result.mask_paths:
        typer.echo(f"  Saved Mask Files:              {len(result.mask_paths)}")
        typer.echo(f"  First Mask File:               {result.mask_paths[0]}")

    # Metadata
    metadata = result.metadata
    typer.echo(f"\n  Annotation Frame:              {metadata['annotation_frame']}")
    typer.echo(f"  Segmentation Mode:             {metadata['mode']}")

    # Show mode-specific information
    if metadata["mode"] == "text-based":
        typer.echo(f"  Text Description:              {metadata['text']}")
        typer.echo(f"  Detected Bounding Box:         {metadata['bbox']}")
    else:  # point-based
        typer.echo(f"  Annotation Points:             {metadata['points']}")
        typer.echo(f"  Point Labels:                  {metadata['labels']}")

    typer.echo("=" * 70 + "\n")


def print_summary_generic_category(result, elapsed_time: float) -> None:
    """Print generic category detection results summary."""
    typer.echo("\n" + "=" * 70)
    typer.echo("  DETECTION RESULTS")
    typer.echo("=" * 70)
    typer.echo(f"  Processed Frames:              {result.num_frames}")
    typer.echo(
        f"  Processing Time:               {elapsed_time:.2f}s ({result.num_frames / elapsed_time:.1f} fps)"
    )
    typer.echo(f"  Output Directory:              {result.output_dir}")

    # Metadata
    metadata = result.metadata
    typer.echo(f"\n  Category:                      {metadata['category']}")
    typer.echo(f"  Accept Threshold:              {metadata['accept_threshold']}")
    typer.echo(f"  Reject Threshold:              {metadata['reject_threshold']}")

    # Count total detections
    total_detections = sum(len(dets) for dets in metadata["detections"].values())
    typer.echo(f"  Total Detections:              {total_detections}")

    # VLM usage statistics
    typer.echo(
        f"\n  VLM Validation Usage:          {metadata['vlm_usage_count']}/{metadata['num_frames_total']} frames ({metadata['vlm_usage_percentage']:.1f}%)"
    )

    typer.echo("=" * 70 + "\n")


@app.command(name="unique_instance")
def unique_instance(
    frames_path: Annotated[
        Path,
        typer.Option(
            "--frames-path",
            "-i",
            help="Directory containing frame sequence (images)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    points: Annotated[
        list[str] | None,
        typer.Option(
            "--points",
            "-p",
            help="Annotation points in format 'x,y'. Can specify multiple times. Example: -p 400,300 -p 350,280",
        ),
    ] = None,
    labels: Annotated[
        list[str] | None,
        typer.Option(
            "--labels",
            "-l",
            help="Point labels: 1 (positive/foreground) or 0 (negative/background). Must match number of points",
        ),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option(
            "--text",
            "-t",
            help="Text description of the object to segment (mutually exclusive with --points)",
        ),
    ] = None,
    annotation_frame: Annotated[
        str | None,
        typer.Option(
            "--annotation-frame",
            "-f",
            help="Frame filename to use for annotation. If not specified, uses first frame",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for results",
        ),
    ] = None,
    save_masks: Annotated[
        bool,
        typer.Option(
            "--save-masks",
            help="Save binary masks (including initial_mask.jpg visualization) as PNG files to disk",
        ),
    ] = False,
    offload_frames_to_gpu: Annotated[
        bool,
        typer.Option(
            "--offload-frames-to-gpu",
            help="Keep frames in GPU memory (faster but uses MUCH more VRAM). Default: False (CPU)",
        ),
    ] = False,
    sam_encoder: Annotated[
        str,
        typer.Option(
            "--sam-encoder",
            "-s",
            help="SAM encoder variant. LongSAM (long-*) best for temporal tracking. Options: long-large (default), long-small, long-tiny, small, tiny, etc.",
        ),
    ] = "long-large",
    memory_cleanup_interval: Annotated[
        int,
        typer.Option(
            "--memory-cleanup-interval",
            help="CUDA memory cleanup interval (frames). Lower = less memory, slower",
        ),
    ] = 10,
    optimize_cuda_memory: Annotated[
        bool,
        typer.Option(
            "--optimize-cuda-memory",
            help="Enable CUDA memory optimization. Useful for low-memory GPUs",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
    save_debug: Annotated[
        bool,
        typer.Option(
            "--save-debug",
            help="Save debug visualizations (sam_debug/)",
        ),
    ] = False,
):
    """
    Segment a unique object instance across frame sequences.

    NOTE: Requires a CUDA-capable GPU. CPU and MPS are not supported.
    """

    setup_logging(verbose)

    print_banner()

    try:
        if text is not None and points is not None:
            typer.echo("Error: --text and --points are mutually exclusive", err=True)
            raise typer.Exit(code=1)

        if text is None and points is None:
            typer.echo("Error: Either --text or --points must be provided", err=True)
            raise typer.Exit(code=1)

        if text is not None and labels is not None:
            typer.echo("Error: --labels cannot be used with --text", err=True)
            raise typer.Exit(code=1)

        validate_paths(frames_path, annotation_frame)

        if output_dir is None:
            output_dir = Path("./outputs")

        valid_encoders = [
            "tiny",
            "small",
            "base",
            "large",
            "long-tiny",
            "long-small",
            "long-base",
            "long-large",
            "legacy-tiny",
            "legacy-small",
            "legacy-base",
            "legacy-large",
        ]
        if sam_encoder not in valid_encoders:
            typer.echo(f"Error: Invalid sam_encoder '{sam_encoder}'", err=True)
            raise typer.Exit(code=1)

        config = {
            "Frames Path": str(frames_path),
            "Mode": "Text-based" if text else "Point-based",
            "Annotation Frame": annotation_frame or "[first frame]",
            "Output Directory": str(output_dir),
            "Save Masks to Disk": "Yes" if save_masks else "No",
            "Frames Location": "GPU (faster, more VRAM)"
            if offload_frames_to_gpu
            else "CPU (slower, less VRAM)",
            "SAM Encoder": sam_encoder,
            "Memory Cleanup Interval": str(memory_cleanup_interval),
            "CUDA Optimization": "Enabled" if optimize_cuda_memory else "Disabled",
        }

        if text:
            config["Text"] = text
        else:
            parsed_points = parse_points(points)
            parsed_labels = parse_labels(labels, len(parsed_points))
            config["Points"] = str(parsed_points)
            config["Labels"] = str(parsed_labels)

        print_config(config)

        typer.echo("Initializing segmenter (requires CUDA GPU)...")
        segmenter = UniqueInstanceSegmenter(
            sam_encoder=sam_encoder,
            memory_cleanup_interval=memory_cleanup_interval,
        )

        if optimize_cuda_memory:
            segmenter.optimize_cuda_memory()
            typer.echo("✓ CUDA memory optimization enabled")

        typer.echo("✓ Segmenter initialized\n")

        typer.echo("Processing frames...")
        start_time = time.time()

        if text:
            result = segmenter.segment(
                frames_path=str(frames_path),
                text=text,
                annotation_frame=annotation_frame,
                output_dir=str(output_dir),
                offload_frames_to_gpu=offload_frames_to_gpu,
                save_masks=save_masks,
                save_debug=save_debug,
            )
        else:
            result = segmenter.segment(
                frames_path=str(frames_path),
                points=parsed_points,
                labels=parsed_labels,
                annotation_frame=annotation_frame,
                output_dir=str(output_dir),
                offload_frames_to_gpu=offload_frames_to_gpu,
                save_masks=save_masks,
                save_debug=save_debug,
            )

        elapsed_time = time.time() - start_time

        typer.echo("✓ Segmentation complete!\n")

        print_summary_unique_instance(result, elapsed_time)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}\n", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command(name="generic_category")
def generic_category(
    frames_path: Annotated[
        Path,
        typer.Option(
            "--frames-path",
            "-i",
            help="Directory containing frame sequence (images)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    category: Annotated[
        str,
        typer.Option(
            "--category",
            "-c",
            help="Category to detect (e.g., 'person', 'chair', 'car')",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for results",
        ),
    ] = None,
    accept_threshold: Annotated[
        float,
        typer.Option(
            "--accept-threshold",
            help="Image-text similarity threshold for auto-accepting boxes (0.0-1.0)",
        ),
    ] = 0.90,
    reject_threshold: Annotated[
        float,
        typer.Option(
            "--reject-threshold",
            help="Image-text similarity threshold for auto-rejecting boxes (0.0-1.0)",
        ),
    ] = 0.10,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
    save_debug: Annotated[
        bool,
        typer.Option(
            "--save-debug",
            help="Save debug visualizations (grounding_debug/, image_text_debug/, vlm_debug/, sam_debug/, detections_debug/)",
        ),
    ] = False,
    save_masks: Annotated[
        bool,
        typer.Option(
            "--save-masks",
            help="Save binary segmentation masks as PNG files to disk",
        ),
    ] = False,
    vlm_model: Annotated[
        str,
        typer.Option(
            "--vlm-model",
            help="VLM model for validation. Options: 'qwen3-vl:2b-instruct-q8_0' (default, faster), 'qwen3-vl:4b-instruct-q8_0' (better quality)",
        ),
    ] = "qwen3-vl:2b-instruct-q8_0",
):
    """
    Detect and segment instances of a category across frame sequences.
    """

    setup_logging(verbose)

    print_banner()

    try:
        validate_paths(frames_path, annotation_frame=None)

        if output_dir is None:
            output_dir = Path("./outputs")

        # Validate thresholds
        if not 0.0 <= accept_threshold <= 1.0:
            typer.echo(
                f"Error: accept_threshold must be between 0.0 and 1.0, got {accept_threshold}",
                err=True,
            )
            raise typer.Exit(code=1)

        if not 0.0 <= reject_threshold <= 1.0:
            typer.echo(
                f"Error: reject_threshold must be between 0.0 and 1.0, got {reject_threshold}",
                err=True,
            )
            raise typer.Exit(code=1)

        if reject_threshold >= accept_threshold:
            typer.echo(
                f"Error: reject_threshold ({reject_threshold}) must be < accept_threshold ({accept_threshold})",
                err=True,
            )
            raise typer.Exit(code=1)

        config = {
            "Frames Path": str(frames_path),
            "Category": category,
            "Output Directory": str(output_dir),
            "Accept Threshold": f"{accept_threshold}",
            "Reject Threshold": f"{reject_threshold}",
            "Save Masks to Disk": "Yes" if save_masks else "No",
            "VLM Model": vlm_model,
        }

        print_config(config)

        typer.echo("Initializing detector (requires CUDA GPU)...")
        segmenter = GenericCategorySegmenter(vlm_model=vlm_model)

        typer.echo("✓ Detector initialized\n")

        typer.echo("Processing frames...")
        start_time = time.time()

        result = segmenter.segment(
            frames_path=str(frames_path),
            category=category,
            output_dir=str(output_dir),
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            save_debug=save_debug,
            save_masks=save_masks,
        )

        elapsed_time = time.time() - start_time

        typer.echo("✓ Segmentation complete!\n")

        print_summary_generic_category(result, elapsed_time)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}\n", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(code=1) from None


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
