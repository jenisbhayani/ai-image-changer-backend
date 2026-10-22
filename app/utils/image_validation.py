import io
from typing import Tuple
from PIL import Image, ImageOps
from fastapi import HTTPException, status
from app.core.config import settings

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_upload(content_type: str | None, filename: str | None, data_length: int) -> None:
    if data_length == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty."
        )

    if data_length > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds the maximum allowed upload limit of {max_mb:.0f} MB."
        )

    # Check content type if available
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        # Also check file extension as fallback
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported image format: {content_type}. Supported formats are JPEG, PNG, and WebP."
                )


def load_and_normalize_image(image_bytes: bytes) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Safely decodes image bytes into a PIL Image, handles EXIF orientation,
    bounds maximum dimensions to prevent OOM, and returns (normalized_image, original_dimensions).
    """
    try:
        raw_image = Image.open(io.BytesIO(image_bytes))
        raw_image.verify()  # Fast check for file corruption
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded file is not a valid or readable image: {str(e)}"
        )

    # Re-open after verify() since verify() closes the handle
    image = Image.open(io.BytesIO(image_bytes))

    # Normalize EXIF orientation (e.g. photos taken vertically on smartphones)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    orig_width, orig_height = image.size

    # Check and bound maximum dimensions
    max_dim = settings.MAX_IMAGE_DIMENSION
    if orig_width > max_dim or orig_height > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    return image, (orig_width, orig_height)
