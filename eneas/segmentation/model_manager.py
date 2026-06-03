"""
Model Manager - Simplified model download handling.

Handles downloading models from HuggingFace Hub and direct URLs.
Uses HuggingFace Hub's native caching system for all downloads.
"""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloads for eneas.

    Uses HuggingFace Hub's default cache (~/.cache/huggingface/hub/) for all models.
    Respects HF_HOME environment variable for custom cache locations.

    Examples:
        >>> manager = ModelManager()
        >>> # Download from HuggingFace Hub
        >>> model_path = manager.download("microsoft/Florence-2-large")
        >>> # Download from direct URL
        >>> sam2_path = manager.download_url(
        ...     "https://dl.fbaipublicfiles.com/.../sam2.1_hiera_large.pt",
        ...     "sam2.1_hiera_large.pt"
        ... )
    """

    def download(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub.

        Uses HuggingFace Hub's native caching and download resumption.
        The model is cached automatically and reused on subsequent calls.

        Args:
            model_id: HuggingFace model ID (e.g., 'microsoft/Florence-2-large')

        Returns:
            Path to model directory

        Raises:
            ImportError: If huggingface_hub is not installed
            RuntimeError: If download fails

        Examples:
            >>> manager = ModelManager()
            >>> path = manager.download("microsoft/Florence-2-large")
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for model downloads.\n"
                "Install with: pip install huggingface_hub"
            ) from e

        try:
            logger.info(f"Downloading {model_id} from HuggingFace Hub...")

            # Use HuggingFace's native caching
            # - Automatically uses ~/.cache/huggingface/hub/
            # - Respects HF_HOME environment variable
            # - Handles validation, resumable downloads, symlinks, etc.
            model_path = snapshot_download(repo_id=model_id)

            logger.info(f"Model ready at: {model_path}")
            return Path(model_path)

        except Exception as e:
            raise RuntimeError(
                f"Failed to download {model_id} from HuggingFace Hub: {e}\n\n"
                f"Manual download: https://huggingface.co/{model_id}"
            ) from e

    def download_url(self, url: str, filename: str) -> Path:
        """Download file from direct URL.

        Downloads to HuggingFace cache directory for consistency with other models.
        File is cached and reused on subsequent calls.

        Args:
            url: Direct download URL
            filename: Name to save file as

        Returns:
            Path to downloaded file

        Raises:
            RuntimeError: If download fails

        Examples:
            >>> manager = ModelManager()
            >>> path = manager.download_url(
            ...     "https://example.com/model.pt",
            ...     "model.pt"
            ... )
        """
        try:
            from huggingface_hub import HF_HOME
        except ImportError:
            # Fallback if huggingface_hub not available
            HF_HOME = None

        # Use HuggingFace cache directory for consistency
        cache_dir = Path(HF_HOME or Path.home() / ".cache" / "huggingface")
        file_path = cache_dir / "hub" / filename

        # Return cached file if exists
        if file_path.exists():
            logger.info(f"Using cached file: {file_path}")
            return file_path

        # Download file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {filename} from {url}...")

        try:
            urllib.request.urlretrieve(url, file_path)
            logger.info(f"Download complete: {file_path}")
            return file_path

        except Exception as e:
            raise RuntimeError(f"Failed to download from {url}: {e}") from e
