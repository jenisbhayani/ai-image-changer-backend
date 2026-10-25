import io
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, status
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from gradio_client import Client, handle_file

from app.core.config import settings
from app.services.model_registry import get_inference_service
from app.utils.cutout_refiner import cutout_refiner

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RMBG_MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "rmbg-1.4.onnx")


class HuggingFaceInferenceService:
    """
    Backwards-compatible inference facade routing to the high-accuracy
    ModelRegistry (BiRefNet-general-lite / IS-Net) with Gradio Cloud fallback.
    """

    def __init__(self, model_path: str = DEFAULT_RMBG_MODEL_PATH):
        self._model_path = model_path
        self._onnx_session: Optional[ort.InferenceSession] = None
        self._client: Optional[Client] = None

    def _get_onnx_session(self) -> Optional[ort.InferenceSession]:
        if self._onnx_session is None:
            if not os.path.exists(self._model_path):
                logger.warning(f"RMBG-1.4 model not found at {self._model_path}")
                return None
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = min(4, os.cpu_count() or 2)
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._onnx_session = ort.InferenceSession(
                    self._model_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                logger.error(f"Failed to load RMBG-1.4 ONNX session: {e}")
                self._onnx_session = None

        return self._onnx_session

    def remove_background_local(self, image_bytes: bytes) -> bytes:
        """
        Legacy RMBG-1.4 method retained for baseline regression testing.
        """
        session = self._get_onnx_session()
        if session is None:
            raise RuntimeError("Local RMBG-1.4 model session is unavailable")

        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        orig_w, orig_h = pil_img.size
        rgb_img = pil_img.convert("RGB")

        input_size = (1024, 1024)
        resized_img = rgb_img.resize(input_size, Image.Resampling.BILINEAR)

        img_arr = np.array(resized_img, dtype=np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 1.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        input_tensor = np.expand_dims(img_arr, axis=0)

        outputs = session.run(None, {"input": input_tensor})
        raw_mask = np.squeeze(outputs[0])

        refined_alpha = cutout_refiner.refine_alpha_mask(
            raw_mask,
            target_size=(orig_w, orig_h),
            rgb_image=rgb_img,
            subject_only=True,
            legacy_poster_mode=True,
        )

        clean_rgba = cutout_refiner.apply_cutout(rgb_img, refined_alpha, defringe_edges=True)
        out_buf = io.BytesIO()
        clean_rgba.save(out_buf, format="PNG", compress_level=6)
        return out_buf.getvalue()

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

    def remove_background_cloud(self, image_bytes: bytes) -> bytes:
        client = self.get_client()
        orig = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
            orig.save(tf, format="PNG")
            temp_input_path = tf.name
        try:
            result = client.predict(
                images=handle_file(temp_input_path), resolution="1024x1024",
                weights_file="General", api_name="/image")
            out_path = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result[0]
            alpha = np.array(Image.open(out_path).convert("RGBA").split()[-1], dtype=np.uint8)

            refined = cutout_refiner.refine_alpha_mask(
                alpha, target_size=orig.size, rgb_image=orig, subject_only=True)
            clean = cutout_refiner.apply_cutout(orig, refined, defringe_edges=True)
            buf = io.BytesIO()
            clean.save(buf, format="PNG", compress_level=6)   # optimize=True is very slow
            return buf.getvalue()
        except Exception as e:
            logger.error(f"BiRefNet processing failed: {e}")
            self._client = None
            raise HTTPException(status_code=500, detail=f"Cloud BiRefNet processing failed: {e}")
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    def remove_background(
        self,
        image_bytes: bytes,
        quality_tier: str = "balanced",
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Primary entrypoint: executes commercial-grade pipeline via ModelRegistry
        with automatic fallback to cloud if requested or on failure.
        """
        inference_service = get_inference_service()
        try:
            png_bytes, metadata = inference_service.remove_background(
                image_bytes=image_bytes,
                quality_tier=quality_tier,
            )
            return png_bytes, metadata
        except Exception as err:
            logger.warning(f"Local inference failed ({err}), attempting cloud fallback...")
            try:
                png_bytes = self.remove_background_cloud(image_bytes)
                return png_bytes, {
                    "model_used": "cloud-BiRefNet",
                    "quality_tier": quality_tier,
                    "confidence": 0.85,
                    "warnings": [f"Local inference error: {str(err)}"],
                    "processing_ms": 0,
                }
            except Exception:
                raise err


hf_service = HuggingFaceInferenceService()
