"""
FastAPI Edit and Passport Photo Studio Router (/api/v1/edit)
Endpoints:
1. POST /passport/analyze -> Biometric face analysis & crop recommendation
2. POST /background/replace -> AI cutout composite over solid/gradient/transparent
3. POST /render -> Full-resolution recipe rendering (crop, rotate, adjust, filter)
4. POST /compress -> Smart target file-size JPEG compression
5. GET /presets -> Server-driven country passport & visa specifications
"""

import base64
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse

from app.services.edit_service import image_edit_service
from app.services.passport_service import passport_analysis_service
from app.utils.image_validation import validate_image_upload

router = APIRouter(prefix="/edit")

# Server-driven country passport presets catalog
PASSPORT_PRESETS_DATA = [
    {
        "id": "india",
        "label": "India Passport / Visa",
        "region": "Asia",
        "width_mm": 35.0,
        "height_mm": 45.0,
        "dpi": 300,
        "width_px": 413,
        "height_px": 531,
        "head_ratio_range": [0.68, 0.80],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 250,
        "notes": "Verify with passportindia.gov.in. White background required, 70-80% face coverage.",
        "sheet_layouts": {"a4": {"cols": 4, "rows": 5, "total": 20}, "4x6": {"cols": 2, "rows": 3, "total": 6}},
    },
    {
        "id": "us",
        "label": "United States Passport / Visa",
        "region": "Americas",
        "width_mm": 50.8,
        "height_mm": 50.8,
        "dpi": 300,
        "width_px": 600,
        "height_px": 600,
        "head_ratio_range": [0.50, 0.69],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 240,
        "notes": "2x2 inches (51x51mm). Plain white or off-white background, head 1 to 1 3/8 inches.",
        "sheet_layouts": {"a4": {"cols": 3, "rows": 5, "total": 15}, "4x6": {"cols": 2, "rows": 2, "total": 4}},
    },
    {
        "id": "uk",
        "label": "United Kingdom Passport",
        "region": "Europe",
        "width_mm": 35.0,
        "height_mm": 45.0,
        "dpi": 300,
        "width_px": 413,
        "height_px": 531,
        "head_ratio_range": [0.64, 0.75],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#F0F0F0",
        "max_file_kb": 250,
        "notes": "Light grey or cream background required (not white). Head height 29mm to 34mm.",
        "sheet_layouts": {"a4": {"cols": 4, "rows": 5, "total": 20}, "4x6": {"cols": 2, "rows": 3, "total": 6}},
    },
    {
        "id": "eu",
        "label": "Schengen / European Union",
        "region": "Europe",
        "width_mm": 35.0,
        "height_mm": 45.0,
        "dpi": 300,
        "width_px": 413,
        "height_px": 531,
        "head_ratio_range": [0.70, 0.80],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 250,
        "notes": "35x45mm standard ICAO biometric. Light neutral background.",
        "sheet_layouts": {"a4": {"cols": 4, "rows": 5, "total": 20}, "4x6": {"cols": 2, "rows": 3, "total": 6}},
    },
    {
        "id": "canada",
        "label": "Canada Passport",
        "region": "Americas",
        "width_mm": 50.0,
        "height_mm": 70.0,
        "dpi": 300,
        "width_px": 591,
        "height_px": 827,
        "head_ratio_range": [0.62, 0.72],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 250,
        "notes": "50x70mm. Chin to crown must measure between 31mm and 36mm.",
        "sheet_layouts": {"a4": {"cols": 3, "rows": 3, "total": 9}, "4x6": {"cols": 2, "rows": 2, "total": 4}},
    },
    {
        "id": "australia",
        "label": "Australia Passport",
        "region": "Oceania",
        "width_mm": 35.0,
        "height_mm": 45.0,
        "dpi": 300,
        "width_px": 413,
        "height_px": 531,
        "head_ratio_range": [0.71, 0.80],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#F0F0F0",
        "max_file_kb": 250,
        "notes": "Plain, light grey or white background. Chin to crown 32mm to 36mm.",
        "sheet_layouts": {"a4": {"cols": 4, "rows": 5, "total": 20}, "4x6": {"cols": 2, "rows": 3, "total": 6}},
    },
    {
        "id": "china",
        "label": "China Passport / Visa",
        "region": "Asia",
        "width_mm": 33.0,
        "height_mm": 48.0,
        "dpi": 300,
        "width_px": 390,
        "height_px": 567,
        "head_ratio_range": [0.58, 0.71],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 240,
        "notes": "33x48mm. Pure white background, no white clothing.",
        "sheet_layouts": {"a4": {"cols": 4, "rows": 5, "total": 20}, "4x6": {"cols": 2, "rows": 3, "total": 6}},
    },
    {
        "id": "uae",
        "label": "UAE / Dubai Visa",
        "region": "Middle East",
        "width_mm": 40.0,
        "height_mm": 60.0,
        "dpi": 300,
        "width_px": 472,
        "height_px": 709,
        "head_ratio_range": [0.70, 0.80],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 300,
        "notes": "40x60mm. Clear white background with frontal gaze.",
        "sheet_layouts": {"a4": {"cols": 3, "rows": 4, "total": 12}, "4x6": {"cols": 2, "rows": 2, "total": 4}},
    },
    {
        "id": "pan_india",
        "label": "PAN Card (India)",
        "region": "Asia",
        "width_mm": 25.0,
        "height_mm": 35.0,
        "dpi": 300,
        "width_px": 295,
        "height_px": 413,
        "head_ratio_range": [0.70, 0.80],
        "eye_line_range": [0.56, 0.69],
        "default_bg_color": "#FFFFFF",
        "max_file_kb": 50,
        "notes": "Strict max 50 KB file size limit for NSDL/UTI portal upload.",
        "sheet_layouts": {"a4": {"cols": 6, "rows": 6, "total": 36}, "4x6": {"cols": 3, "rows": 4, "total": 12}},
    },
]


@router.get("/presets")
async def get_passport_presets():
    """Returns server-driven country passport and ID card presets."""
    return {
        "status": "success",
        "presets": PASSPORT_PRESETS_DATA,
        "count": len(PASSPORT_PRESETS_DATA),
    }


@router.post("/passport/analyze")
async def analyze_passport(
    image: UploadFile = File(...),
    preset_id: Optional[str] = Form(None),
    width_mm: Optional[float] = Form(None),
    height_mm: Optional[float] = Form(None),
):
    """
    Biometric face analysis and intelligent passport cropping calculation.
    """
    contents = await image.read()
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents),
    )

    # Calculate target ratio
    ratio = 35.0 / 45.0
    if preset_id:
        match = next((p for p in PASSPORT_PRESETS_DATA if p["id"] == preset_id.lower()), None)
        if match:
            ratio = match["width_mm"] / match["height_mm"]
    elif width_mm and height_mm and height_mm > 0:
        ratio = width_mm / height_mm

    result = passport_analysis_service.analyze_passport_photo(
        image_bytes=contents,
        target_ratio=ratio,
    )
    return {"status": "success", **result}


@router.post("/background/replace")
async def replace_background(
    request: Request,
    image: UploadFile = File(...),
    bg_type: str = Form("solid"),  # solid, gradient, transparent
    color: str = Form("#FFFFFF"),
    gradient_start: str = Form("#E0EAFC"),
    gradient_end: str = Form("#CFDEF3"),
    format: str = Query("base64", description="'base64' or 'png'"),
):
    """
    Removes background using primary AI model and composites over specified background.
    """
    contents = await image.read()
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents),
    )

    out_bytes, meta = image_edit_service.replace_background(
        image_bytes=contents,
        bg_type=bg_type,
        color=color,
        gradient_start=gradient_start,
        gradient_end=gradient_end,
    )

    if format == "base64" or "application/json" in request.headers.get("accept", ""):
        b64 = base64.b64encode(out_bytes).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "image_base64": b64,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "format": "PNG",
            **meta,
        })

    return Response(
        content=out_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="bg-replaced.png"'},
    )


@router.post("/render")
async def render_recipe_endpoint(
    request: Request,
    image: UploadFile = File(...),
    recipe_json: str = Form(...),
    format: str = Query("base64", description="'base64' or 'png'"),
):
    """
    Executes full-resolution render of an Edit Recipe (crop, rotate, adjustments, filters).
    """
    contents = await image.read()
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents),
    )

    try:
        recipe = json.loads(recipe_json)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid recipe JSON payload: {str(e)}",
        )

    out_bytes, meta = image_edit_service.render_recipe(
        image_bytes=contents,
        recipe=recipe,
    )

    if format == "base64" or "application/json" in request.headers.get("accept", ""):
        b64 = base64.b64encode(out_bytes).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "image_base64": b64,
            **meta,
        })

    return Response(
        content=out_bytes,
        media_type=meta.get("mime_type", "image/png"),
        headers={"Content-Disposition": 'inline; filename="rendered-edit.png"'},
    )


@router.post("/compress")
async def compress_image_endpoint(
    request: Request,
    image: UploadFile = File(...),
    target_kb: int = Form(...),
    min_quality: int = Form(30),
    format: str = Query("base64", description="'base64' or 'jpeg'"),
):
    """
    Binary-searches quality and scale to guarantee output size <= target_kb.
    """
    contents = await image.read()
    validate_image_upload(
        content_type=image.content_type,
        filename=image.filename,
        data_length=len(contents),
    )

    if target_kb < 10 or target_kb > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_kb must be between 10 KB and 10000 KB.",
        )

    out_bytes, meta = image_edit_service.compress_to_target_kb(
        image_bytes=contents,
        target_kb=target_kb,
        min_quality=min_quality,
    )

    if format == "base64" or "application/json" in request.headers.get("accept", ""):
        b64 = base64.b64encode(out_bytes).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "image_base64": b64,
            **meta,
        })

    return Response(
        content=out_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'inline; filename="compressed-{target_kb}kb.jpg"',
            "X-Target-KB": str(target_kb),
            "X-Final-KB": str(meta.get("final_kb")),
        },
    )
