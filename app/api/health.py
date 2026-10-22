from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health")
def get_health():
    """
    Check backend health and Hugging Face token configuration.
    """
    has_token = bool(settings.HF_TOKEN.strip() and settings.HF_TOKEN != "hf_your_token_here")
    return {
        "status": "ok",
        "backend": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "token_configured": has_token,
        "model_id": settings.HF_MODEL_ID,
        "inference_ready": True,
    }


@router.get("/model-info")
def get_model_info():
    """
    Information about the target Hugging Face cloud model.
    """
    return {
        "model_id": settings.HF_MODEL_ID,
        "api_provider": "Hugging Face Serverless Inference API",
        "token_configured": bool(settings.HF_TOKEN.strip() and settings.HF_TOKEN != "hf_your_token_here"),
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024),
    }
