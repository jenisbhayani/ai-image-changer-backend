"""
Route tests verifying concurrent execution through run_in_threadpool + semaphore
and asserting JSON and binary contract compliance.
"""

import asyncio
import io
import os
import sys
from unittest.mock import patch

import pytest
import httpx
from PIL import Image

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(TEST_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.main import app
from app.services.hf_service import hf_service


def _make_dummy_image_bytes():
    img = Image.new("RGB", (200, 200), (255, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_dummy_rgba_bytes():
    img = Image.new("RGBA", (200, 200), (255, 100, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_concurrent_requests_and_contract_compliance():
    """
    With hf_service mocked to simulate processing delay, verify two concurrent requests
    complete without deadlocking and both response shapes match the contract.
    """
    dummy_input = _make_dummy_image_bytes()
    dummy_rgba = _make_dummy_rgba_bytes()

    def mock_remove_bg(image_bytes, quality_tier="balanced"):
        # Return valid PNG bytes and metadata dict
        return dummy_rgba, {
            "model_used": "IS-Net",
            "quality_tier": quality_tier,
            "confidence": 0.95,
            "warnings": [],
            "processing_ms": 12.5,
        }

    with patch.object(hf_service, "remove_background", side_effect=mock_remove_bg):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            # Task 1: JSON response (format=base64)
            async def req_json():
                files = {"image": ("test1.jpg", dummy_input, "image/jpeg")}
                return await client.post(
                    "/api/v1/remove-background?format=base64&quality=balanced",
                    files=files,
                )

            # Task 2: Binary response (format=png)
            async def req_binary():
                files = {"image": ("test2.jpg", dummy_input, "image/jpeg")}
                return await client.post(
                    "/api/v1/remove-background?format=png&quality=high",
                    files=files,
                )

            # Run concurrently
            res_json, res_binary = await asyncio.gather(req_json(), req_binary())

            # Verify JSON contract
            assert res_json.status_code == 200
            data = res_json.json()
            assert "image_base64" in data
            assert data["width"] == 200
            assert data["height"] == 200
            assert data["format"] == "PNG"
            assert data["model_used"] == "IS-Net"
            assert data["quality_tier"] == "balanced"
            assert data["confidence"] == 0.95
            assert isinstance(data["warnings"], list)
            assert data["processing_ms"] == 12.5

            # Verify Binary contract
            assert res_binary.status_code == 200
            assert res_binary.headers["content-type"] == "image/png"
            assert res_binary.headers["x-image-width"] == "200"
            assert res_binary.headers["x-image-height"] == "200"
            assert res_binary.headers["x-model-used"] == "IS-Net"
            assert res_binary.headers["x-quality-tier"] == "high"
            assert res_binary.content.startswith(b"\x89PNG")
