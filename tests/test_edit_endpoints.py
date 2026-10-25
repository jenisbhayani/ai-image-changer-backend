"""
Tests for FastAPI edit & passport studio endpoints (/api/v1/edit).
"""

import base64
import io
import json
import os
import sys

# Ensure backend root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(TEST_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app

client = TestClient(app)


def test_get_passport_presets():
    """Verifies server-driven presets list endpoint."""
    response = client.get("/api/v1/edit/presets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "presets" in data
    assert len(data["presets"]) >= 8

    preset_ids = [p["id"] for p in data["presets"]]
    assert "india" in preset_ids
    assert "us" in preset_ids
    assert "eu" in preset_ids
    assert "uk" in preset_ids
    assert "canada" in preset_ids


def test_passport_analyze_endpoint():
    """Verifies face detection, biometric landmarks, suggested crop, and compliance checks."""
    portrait_path = os.path.join(BACKEND_ROOT, "eval", "images", "portrait_closeup", "portrait_closeup_sample.jpg")
    if not os.path.exists(portrait_path):
        pytest.skip("Portrait sample image not found")

    with open(portrait_path, "rb") as f:
        response = client.post(
            "/api/v1/edit/passport/analyze",
            files={"image": ("portrait.jpg", f, "image/jpeg")},
            data={"preset_id": "india"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["faces_count"] >= 1
    assert "suggested_crop" in data
    assert "checks" in data
    assert "roll_deg" in data

    crop = data["suggested_crop"]
    assert crop["width"] > 0
    assert crop["height"] > 0
    # Check Indian ratio ~ 35 / 45 = 0.777
    actual_ratio = crop["width"] / crop["height"]
    assert abs(actual_ratio - (35.0 / 45.0)) < 0.05

    check_ids = [c["id"] for c in data["checks"]]
    assert "single_face" in check_ids
    assert "head_tilt" in check_ids
    assert "head_size" in check_ids


def test_background_replace_endpoint():
    """Verifies AI cutout composition over custom solid/gradient backgrounds."""
    portrait_path = os.path.join(BACKEND_ROOT, "eval", "images", "portrait_closeup", "portrait_closeup_sample.jpg")
    if not os.path.exists(portrait_path):
        pytest.skip("Portrait sample image not found")

    # 1. Replace with pure white background (#FFFFFF)
    with open(portrait_path, "rb") as f:
        response = client.post(
            "/api/v1/edit/background/replace?format=base64",
            files={"image": ("portrait.jpg", f, "image/jpeg")},
            data={"bg_type": "solid", "color": "#FFFFFF"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "image_base64" in data
    assert data["format"] == "PNG"

    # Decode and verify dimensions
    raw_png = base64.b64decode(data["image_base64"])
    out_img = Image.open(io.BytesIO(raw_png))
    assert out_img.mode in ("RGB", "RGBA")

    # Corner pixel should be pure white background
    corner_pixel = out_img.getpixel((5, 5))
    assert corner_pixel[0] >= 250 and corner_pixel[1] >= 250 and corner_pixel[2] >= 250


def test_render_recipe_endpoint():
    """Verifies server rendering of edit recipes (crop, rotate, adjust, filter)."""
    # Create synthetic test image
    img = Image.new("RGB", (300, 300), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    recipe = {
        "rotate": 90,
        "flip": "horizontal",
        "crop": {"x": 20, "y": 20, "width": 200, "height": 200},
        "adjustments": {
            "brightness": 0.1,
            "contrast": 0.2,
            "saturation": 0.1,
            "sharpness": 0.5,
        },
        "filter_id": "vivid",
        "output": {"format": "JPEG", "quality": 90},
    }

    response = client.post(
        "/api/v1/edit/render?format=base64",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"recipe_json": json.dumps(recipe)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "image_base64" in data
    assert data["format"] == "JPEG"
    assert data["width"] == 200
    assert data["height"] == 200


def test_compress_to_target_kb_endpoint():
    """Verifies binary search compression guarantees file size <= target_kb."""
    img = Image.new("RGB", (1200, 1200), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    buf.seek(0)

    # Target 100 KB
    response = client.post(
        "/api/v1/edit/compress?format=base64",
        files={"image": ("large.jpg", buf, "image/jpeg")},
        data={"target_kb": 100},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_kb"] == 100
    assert data["final_kb"] <= 100.0

    # Target 250 KB
    buf.seek(0)
    response2 = client.post(
        "/api/v1/edit/compress?format=base64",
        files={"image": ("large.jpg", buf, "image/jpeg")},
        data={"target_kb": 250},
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["final_kb"] <= 250.0
