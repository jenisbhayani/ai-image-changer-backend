import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)


def create_test_image(width=100, height=100, color=(128, 64, 32)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_enhance_health():
    response = client.get("/api/v1/enhance/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "codeformer" in data["models"]
    assert "basic_pillow" in data["models"]


def test_enhance_validation_empty_file():
    response = client.post(
        "/api/v1/enhance",
        files={"image": ("test.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_enhance_validation_invalid_mode():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/enhance?mode=unsupported_mode",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_enhance_validation_invalid_strength():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/enhance?strength=150",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_enhance_pillow_fallback_base64():
    """
    Simulates CodeFormer failure to verify seamless fallback to Pillow Basic Enhancer.
    """
    img_bytes = create_test_image(80, 80)
    with patch(
        "app.services.enhance_service.EnhanceModelRegistry.enhance_with_codeformer",
        side_effect=RuntimeError("Simulated Cloud 429/503 rate limit"),
    ):
        response = client.post(
            "/api/v1/enhance?mode=auto&strength=60&format=base64",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert len(data["image_base64"]) > 0
        assert data["model_used"] == "Basic Enhancer"
        assert data["mode"] == "auto"
        assert data["input_size"] == [80, 80]
        assert data["output_size"] == [80, 80]
        assert "processing_ms" in data


def test_enhance_binary_response():
    """
    Verifies binary image response headers.
    """
    img_bytes = create_test_image(60, 60)
    with patch(
        "app.services.enhance_service.EnhanceModelRegistry.enhance_with_codeformer",
        side_effect=RuntimeError("Simulated timeout"),
    ):
        response = client.post(
            "/api/v1/enhance?mode=face&strength=40&format=jpeg",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            headers={"Accept": "image/jpeg"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-model-used"] == "Basic Enhancer"
        assert response.headers["x-enhance-mode"] == "face"
        assert len(response.content) > 0
