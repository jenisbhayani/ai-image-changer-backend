import base64
import io
import logging
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from app.services.upscale_service import (
    MAX_INPUT_DIMENSION_2X,
    MAX_INPUT_DIMENSION_4X,
    upscale_registry,
)
from app.utils.image_validation import load_and_normalize_image, validate_image_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/upscale/health")
async def upscale_health():
    """
    Returns the availability status of image super-resolution models and scales.
    """
    return upscale_registry.get_health_status()


@router.post("/upscale")
async def upscale_image(
    request: Request,
    image: UploadFile = File(...),
    scale: int = Query(2, ge=2, le=4, description="Upscale multiplier: 2 or 4"),
    mode: str = Query("photo", pattern="^(photo|illustration)$", description="Content type"),
    format: str = Query("base64", description="Output format: 'jpeg', 'png', or 'base64'"),
):
    """
    Upscales an uploaded image 2x or 4x with AI super-resolution or Lanczos fallback.
    Returns the upscaled image bytes or base64 JSON payload with metadata.
    """
    if scale not in (2, 4):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scale factor must be 2 or 4.",
        )

    logger.info(f"Upscale request received: scale={scale}x, mode={mode}, format={format}")

    try:
        contents = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded image data: {str(e)}",
        )

    # Validate image format and file size
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents),
    )

    # Decode, handle orientation, and bound dimensions according to scale factor
    max_input = MAX_INPUT_DIMENSION_4X if scale == 4 else MAX_INPUT_DIMENSION_2X
    pil_img, (orig_w, orig_h) = load_and_normalize_image(contents)

    # Enforce scale-specific input limit to prevent cloud timeout
    if pil_img.width > max_input or pil_img.height > max_input:
        pil_img.thumbnail((max_input, max_input), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        input_bytes = buf.getvalue()
    elif pil_img.size != (orig_w, orig_h):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        input_bytes = buf.getvalue()
    else:
        input_bytes = contents

    # Run upscaling with registry fallback
    output_bytes, model_used, processing_ms, is_basic = upscale_registry.process_upscale(
        input_bytes, scale=scale, mode=mode
    )

    # Extract final output dimensions
    try:
        out_pil = Image.open(io.BytesIO(output_bytes))
        out_w, out_h = out_pil.size
        out_format = out_pil.format or "JPEG"
    except Exception:
        out_w, out_h = orig_w * scale, orig_h * scale
        out_format = "JPEG"

    accept_header = request.headers.get("accept", "")
    if format == "base64" or "application/json" in accept_header:
        b64_data = base64.b64encode(output_bytes).decode("utf-8")
        return JSONResponse(
            content={
                "image_base64": b64_data,
                "model_used": model_used,
                "scale": scale,
                "mode": mode,
                "input_size": [orig_w, orig_h],
                "output_size": [out_w, out_h],
                "processing_ms": processing_ms,
                "is_basic_fallback": is_basic,
                "format": out_format,
            }
        )

    media_type = "image/png" if out_format.upper() == "PNG" else "image/jpeg"
    ext = "png" if out_format.upper() == "PNG" else "jpg"

    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="upscaled_{scale}x.{ext}"',
            "X-Model-Used": model_used,
            "X-Processing-Ms": str(processing_ms),
            "X-Scale-Factor": str(scale),
            "X-Is-Basic": str(is_basic),
            "X-Image-Width": str(out_w),
            "X-Image-Height": str(out_h),
            "Cache-Control": "no-cache",
        },
    )
