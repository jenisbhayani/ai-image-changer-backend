import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from app.api import background, edit, enhance, health, upscale
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware to allow requests from mobile app and development tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes (both /api/v1 and /api for maximum compatibility)
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(background.router, prefix=settings.API_V1_STR, tags=["Background Removal"])
app.include_router(edit.router, prefix=settings.API_V1_STR, tags=["Image Edit & Passport"])
app.include_router(edit.router, prefix="/api", tags=["Image Edit & Passport"])
app.include_router(enhance.router, prefix=settings.API_V1_STR, tags=["Enhance Image"])
app.include_router(enhance.router, prefix="/api", tags=["Enhance Image"])
app.include_router(upscale.router, prefix=settings.API_V1_STR, tags=["Upscale Image"])
app.include_router(upscale.router, prefix="/api", tags=["Upscale Image"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@app.get("/privacy-policy", response_class=HTMLResponse, tags=["Legal"])
def get_privacy_policy():
    """Serves the live, Google Play compliant HTML Privacy Policy."""
    path = os.path.join(STATIC_DIR, "privacy-policy.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Privacy Policy</h1><p>Contact: jdbhayani9099@gmail.com</p>")


@app.get("/terms", response_class=HTMLResponse, tags=["Legal"])
def get_terms_of_service():
    """Serves the live HTML Terms of Service."""
    path = os.path.join(STATIC_DIR, "terms-of-service.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Terms of Service</h1><p>Contact: jdbhayani9099@gmail.com</p>")


@app.get(f"{settings.API_V1_STR}/app-config", tags=["Remote Config"])
@app.get("/api/app-config", tags=["Remote Config"])
def get_app_config():
    """
    Returns server-driven dynamic app configuration.
    Allows updating policies, notices, and supported versions without APK updates.
    """
    return {
        "status": "success",
        "app_name": "AI Image Changer",
        "privacy_policy_url": "/privacy-policy",
        "terms_url": "/terms",
        "support_email": "jdbhayani9099@gmail.com",
        "min_version": "1.0.0",
        "latest_version": "1.0.0",
        "update_url": "https://play.google.com/store/apps/details?id=com.jdbhayani9.aiimagechanger",
        "maintenance_mode": False,
        "notice": None,
    }


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_id": settings.HF_MODEL_ID,
        "token_configured": bool(settings.HF_TOKEN.strip() and settings.HF_TOKEN != "hf_your_token_here"),
        "docs": "/docs",
        "privacy_policy": "/privacy-policy",
        "terms": "/terms",
        "app_config": f"{settings.API_V1_STR}/app-config",
        "health": f"{settings.API_V1_STR}/health",
        "remove_background": f"{settings.API_V1_STR}/remove-background",
        "enhance": f"{settings.API_V1_STR}/enhance",
        "enhance_health": f"{settings.API_V1_STR}/enhance/health",
        "upscale": f"{settings.API_V1_STR}/upscale",
        "upscale_health": f"{settings.API_V1_STR}/upscale/health",
    }

