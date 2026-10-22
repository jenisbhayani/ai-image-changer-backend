import base64
import io
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from app.services.hf_service import hf_service
from app.utils.image_validation import validate_image_upload

router = APIRouter()


@router.post("/remove-background")
async def remove_background(
    request: Request,
    image: UploadFile = File(...),
    format: str = Query("png", description="Output format: 'png' or 'base64'")
):
    """
    Takes an uploaded image and passes it to the BiRefNet model on Hugging Face,
    returning transparent PNG bytes or a base64 JSON payload.
    """
    try:
        contents = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded image data: {str(e)}"
        )

    # Validate image format and file size
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents)
    )

    # Delegate to Hugging Face BiRefNet model
    png_bytes = hf_service.remove_background(contents)

    # Extract dimensions
    try:
        pil_img = Image.open(io.BytesIO(png_bytes))
        width, height = pil_img.size
    except Exception:
        width, height = 1024, 1024

    accept_header = request.headers.get("accept", "")
    if format == "base64" or "application/json" in accept_header:
        b64_data = base64.b64encode(png_bytes).decode("utf-8")
        return JSONResponse(content={
            "image_base64": b64_data,
            "width": width,
            "height": height,
            "format": "PNG"
        })

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="background-removed.png"',
            "X-Image-Width": str(width),
            "X-Image-Height": str(height),
            "Cache-Control": "no-cache",
        }
    )
