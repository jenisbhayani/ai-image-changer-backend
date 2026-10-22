from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import background, health
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

# Register API v1 routes
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(background.router, prefix=settings.API_V1_STR, tags=["Background Removal"])


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_id": settings.HF_MODEL_ID,
        "token_configured": bool(settings.HF_TOKEN.strip() and settings.HF_TOKEN != "hf_your_token_here"),
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
        "remove_background": f"{settings.API_V1_STR}/remove-background",
    }
