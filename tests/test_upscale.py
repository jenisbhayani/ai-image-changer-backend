import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)


def create_test_image(width=100, height=100, color=(60, 120, 180)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_upscale_health():
    response = client.get("/api/v1/upscale/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "scales" in data
    assert 2 in data["scales"]
    assert 4 in data["scales"]
    assert "models" in data


def test_upscale_validation_invalid_scale():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/upscale?scale=3",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "scale" in response.json()["detail"].lower()


def test_upscale_validation_invalid_mode():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/upscale?scale=2&mode=unsupported",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_upscale_fallback_2x():
    img_bytes = create_test_image(80, 80)
    with patch(
        "app.services.upscale_service.UpscaleModelRegistry.upscale_with_ai",
        side_effect=RuntimeError("Simulated cloud failure"),
    ):
        response = client.post(
            "/api/v1/upscale?scale=2&format=base64",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert data["scale"] == 2
        assert data["is_basic_fallback"] is True
        assert data["input_size"] == [80, 80]
        assert data["output_size"] == [160, 160]


def test_upscale_fallback_4x():
    img_bytes = create_test_image(60, 50)
    with patch(
        "app.services.upscale_service.UpscaleModelRegistry.upscale_with_ai",
        side_effect=RuntimeError("Simulated cloud timeout"),
    ):
        response = client.post(
            "/api/v1/upscale?scale=4&format=base64",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["scale"] == 4
        assert data["is_basic_fallback"] is True
        assert data["input_size"] == [60, 50]
        assert data["output_size"] == [240, 200]


def test_upscale_binary_headers():
    img_bytes = create_test_image(50, 50)
    with patch(
        "app.services.upscale_service.UpscaleModelRegistry.upscale_with_ai",
        side_effect=RuntimeError("Simulated failure"),
    ):
        response = client.post(
            "/api/v1/upscale?scale=2&format=jpeg",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            headers={"Accept": "image/jpeg"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-scale-factor"] == "2"
        assert "x-model-used" in response.headers
