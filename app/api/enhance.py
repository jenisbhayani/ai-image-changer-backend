import base64
import io
import logging
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from app.services.enhance_service import enhance_registry
from app.utils.image_validation import load_and_normalize_image, validate_image_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/enhance/health")
async def enhance_health():
    """
    Returns the availability status of image enhancement models and modes.
    """
    return enhance_registry.get_health_status()


@router.post("/enhance")
async def enhance_image(
    request: Request,
    image: UploadFile = File(...),
    mode: str = Query("auto", pattern="^(auto|face|denoise)$", description="Enhancement mode"),
    strength: int = Query(50, ge=0, le=100, description="Enhancement strength from 0 to 100"),
    format: str = Query("base64", description="Output format: 'jpeg', 'png', or 'base64'"),
):
    """
    Enhances an uploaded image with AI restoration (CodeFormer) or deterministic fallback.
    Returns the enhanced image bytes or base64 JSON payload with metadata.
    """
    logger.info(f"Enhance request received: mode={mode}, strength={strength}, format={format}")

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

    # Decode, handle orientation, and bound dimensions
    pil_img, (orig_w, orig_h) = load_and_normalize_image(contents)

    # If image was downscaled to bound dimensions, re-encode for processing
    if pil_img.size != (orig_w, orig_h):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        input_bytes = buf.getvalue()
    else:
        input_bytes = contents

    # Run enhancement with registry fallback
    output_bytes, model_used, processing_ms = enhance_registry.process_enhancement(
        input_bytes, mode=mode, strength=strength
    )

    # Extract final output dimensions
    try:
        out_pil = Image.open(io.BytesIO(output_bytes))
        out_w, out_h = out_pil.size
        out_format = out_pil.format or "JPEG"
    except Exception:
        out_w, out_h = orig_w, orig_h
        out_format = "JPEG"

    accept_header = request.headers.get("accept", "")
    if format == "base64" or "application/json" in accept_header:
        b64_data = base64.b64encode(output_bytes).decode("utf-8")
        return JSONResponse(
            content={
                "image_base64": b64_data,
                "model_used": model_used,
                "mode": mode,
                "input_size": [orig_w, orig_h],
                "output_size": [out_w, out_h],
                "processing_ms": processing_ms,
                "format": out_format,
            }
        )

    media_type = "image/png" if out_format.upper() == "PNG" else "image/jpeg"
    ext = "png" if out_format.upper() == "PNG" else "jpg"

    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="enhanced.{ext}"',
            "X-Model-Used": model_used,
            "X-Processing-Ms": str(processing_ms),
            "X-Enhance-Mode": mode,
            "X-Image-Width": str(out_w),
            "X-Image-Height": str(out_h),
            "Cache-Control": "no-cache",
        },
    )
