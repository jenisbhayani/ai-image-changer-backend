import logging
import os
import tempfile
from fastapi import HTTPException, status
from gradio_client import Client, handle_file
from app.core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceInferenceService:
    def __init__(self):
        self._client = None

    def get_client(self) -> Client:
        token = settings.HF_TOKEN.strip()
        if not token or token == "hf_your_token_here":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hugging Face Access Token (HF_TOKEN) is not configured in backend/.env",
            )

        if self._client is None:
            try:
                logger.info("Initializing Hugging Face BiRefNet client...")
                self._client = Client("ZhengPeng7/BiRefNet_demo", token=token)
            except Exception as e:
                logger.error(f"Failed to connect to Hugging Face BiRefNet Space: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Could not connect to Hugging Face BiRefNet: {str(e)}",
                )
        return self._client

    def remove_background(self, image_bytes: bytes) -> bytes:
        client = self.get_client()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_input_path = temp_file.name

        try:
            logger.info("Running cloud BiRefNet inference on Hugging Face...")
            result = client.predict(
                images=handle_file(temp_input_path),
                resolution="1024x1024",
                weights_file="General",
                api_name="/image",
            )

            # result is a tuple/list: [original_image_path, result_transparent_png_path]
            output_file_path = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result[0]

            with open(output_file_path, "rb") as f:
                transparent_png_bytes = f.read()

            logger.info("Cloud BiRefNet inference completed successfully.")
            return transparent_png_bytes

        except Exception as e:
            logger.error(f"BiRefNet processing failed: {e}")
            # Reset client in case session expired
            self._client = None
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cloud BiRefNet processing failed: {str(e)}",
            )
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)


hf_service = HuggingFaceInferenceService()
