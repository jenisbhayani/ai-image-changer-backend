from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_privacy_policy_html_endpoint():
    response = client.get("/privacy-policy")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Privacy Policy" in response.text
    assert "jdbhayani9099@gmail.com" in response.text


def test_terms_of_service_html_endpoint():
    response = client.get("/terms")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Terms of Service" in response.text


def test_app_config_endpoint():
    response = client.get("/api/v1/app-config")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["privacy_policy_url"] == "/privacy-policy"
    assert data["support_email"] == "jdbhayani9099@gmail.com"
