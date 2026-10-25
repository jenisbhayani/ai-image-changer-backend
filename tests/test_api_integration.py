"""
API Integration tests for POST /api/v1/remove-background.
Tests end-to-end HTTP request/response compatibility with real images and mobile app expectations.
"""

import base64
import io
import os
import sys

# Ensure backend root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(TEST_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app

client = TestClient(app)


def test_endpoint_base64_json_contract():
    """Verifies POST /api/v1/remove-background?format=base64 returns the exact JSON contract mobile expects."""
    # Create test image
    img = Image.new("RGB", (200, 200), (220, 180, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/remove-background?format=base64&quality=fast",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()

    # Mandatory mobile app fields
    assert "image_base64" in data
    assert "width" in data
    assert "height" in data
    assert data["format"] == "PNG"
    assert data["width"] == 200
    assert data["height"] == 200

    # New backward-compatible metadata fields
    assert "model_used" in data
    assert "quality_tier" in data
    assert "confidence" in data
    assert "processing_ms" in data

    # Verify base64 string decodes into a straight-alpha RGBA PNG
    raw_png = base64.b64decode(data["image_base64"])
    out_img = Image.open(io.BytesIO(raw_png))
    assert out_img.mode == "RGBA"
    assert out_img.size == (200, 200)


def test_endpoint_raw_png_binary():
    """Verifies POST /api/v1/remove-background?format=png returns raw image/png stream."""
    img = Image.new("RGB", (150, 150), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/v1/remove-background?format=png&quality=fast",
        files={"image": ("test.png", buf, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-Image-Width"] == "150"
    assert response.headers["X-Image-Height"] == "150"
    assert "X-Model-Used" in response.headers

    # Decodes correctly
    out_img = Image.open(io.BytesIO(response.content))
    assert out_img.mode == "RGBA"
    assert out_img.size == (150, 150)


def test_endpoint_with_poster_regression_image():
    """Tests the real RANG MORLA poster regression image on the API."""
    poster_path = os.path.join(BACKEND_ROOT, "eval", "images", "poster_graphic_text", "rang_morla.jpg")
    if not os.path.exists(poster_path):
        pytest.skip("Poster regression image not found")

    with open(poster_path, "rb") as f:
        response = client.post(
            "/api/v1/remove-background?format=base64&quality=balanced",
            files={"image": ("rang_morla.jpg", f, "image/jpeg")},
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "image_base64" in data
    raw_png = base64.b64decode(data["image_base64"])
    out_img = Image.open(io.BytesIO(raw_png))
    assert out_img.mode == "RGBA"
    # Ensure dimensions match original poster (918, 1024)
    assert out_img.size == (918, 1024)
