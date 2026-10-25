import asyncio
import base64
import io
import os
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from app.services.hf_service import hf_service
from app.utils.image_validation import validate_image_upload

router = APIRouter()
_CONCURRENCY = int(os.environ.get("MAX_CONCURRENT_INFERENCE", "1"))
_SEM = asyncio.Semaphore(_CONCURRENCY)


@router.post("/remove-background")
async def remove_background(
    request: Request,
    image: UploadFile = File(...),
    format: str = Query("png", description="Output format: 'png' or 'base64'"),
    quality: Optional[str] = Query("balanced", description="Quality tier: 'fast', 'balanced', or 'high'"),
):
    """
    Studio-grade background removal endpoint.
    - Default primary engine: BiRefNet-general-lite (MIT licensed, commercial safe).
    - Preserves aspect ratio, fine hair strands, transparent regions, and multi-person subjects.
    - Eliminates dirty background artifacts and color fringes.
    """
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

    # Process via inference service
    tier = (quality or "balanced").strip().lower()
    async with _SEM:
        res = await run_in_threadpool(hf_service.remove_background, contents, tier)
    if isinstance(res, tuple):
        png_bytes, metadata = res
    else:
        png_bytes = res
        metadata = {}

    # Extract dimensions
    try:
        pil_img = Image.open(io.BytesIO(png_bytes))
        width, height = pil_img.size
    except Exception:
        width, height = 1024, 1024

    accept_header = request.headers.get("accept", "")
    if format == "base64" or "application/json" in accept_header:
        b64_data = base64.b64encode(png_bytes).decode("utf-8")
        payload = {
            "image_base64": b64_data,
            "width": width,
            "height": height,
            "format": "PNG",
            "model_used": metadata.get("model_used", "BiRefNet"),
            "quality_tier": metadata.get("quality_tier", tier),
            "confidence": metadata.get("confidence", 0.95),
            "warnings": metadata.get("warnings", []),
            "processing_ms": metadata.get("processing_ms", 0),
        }
        return JSONResponse(content=payload)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="background-removed.png"',
            "X-Image-Width": str(width),
            "X-Image-Height": str(height),
            "X-Model-Used": str(metadata.get("model_used", "BiRefNet")),
            "X-Quality-Tier": str(metadata.get("quality_tier", tier)),
            "Cache-Control": "no-cache",
        },
    )
