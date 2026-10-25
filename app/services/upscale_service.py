import io
import logging
import os
import tempfile
import time
from typing import Tuple
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps
from fastapi import HTTPException, status
from gradio_client import Client, handle_file
from app.core.config import settings

logger = logging.getLogger(__name__)

# Dimension limits to prevent free serverless timeouts and OOM
MAX_INPUT_DIMENSION_2X = 1536
MAX_INPUT_DIMENSION_4X = 1024
MAX_OUTPUT_DIMENSION = 4096

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "realesr-general-x4v3.onnx")


class UpscaleModelRegistry:
    def __init__(self, model_path: str = DEFAULT_ONNX_MODEL_PATH):
        self._model_path = model_path
        self._onnx_session: ort.InferenceSession | None = None
        self._codeformer_client: Client | None = None
        self._finegrain_client: Client | None = None

    def _get_onnx_session(self) -> ort.InferenceSession | None:
        if self._onnx_session is None:
            if not os.path.exists(self._model_path):
                logger.warning(f"Real-ESRGAN ONNX weights not found at {self._model_path}")
                return None
            try:
                logger.info(f"Loading Real-ESRGAN Compact ONNX model from {self._model_path}...")
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = min(4, os.cpu_count() or 2)
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._onnx_session = ort.InferenceSession(
                    self._model_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("Real-ESRGAN Compact ONNX model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                self._onnx_session = None

        return self._onnx_session

    def _get_codeformer_client(self) -> Client | None:
        token = settings.HF_TOKEN.strip()
        has_token = bool(token and token != "hf_your_token_here")
        if not has_token:
            return None

        if self._codeformer_client is None:
            try:
                logger.info("Connecting to CodeFormer Space for upscaling...")
                self._codeformer_client = Client("sczhou/CodeFormer", token=token)
            except Exception as e:
                logger.warning(f"Could not connect to CodeFormer Space: {e}")
                self._codeformer_client = None

        return self._codeformer_client

    def _get_finegrain_client(self) -> Client | None:
        token = settings.HF_TOKEN.strip()
        has_token = bool(token and token != "hf_your_token_here")
        if not has_token:
            return None

        if self._finegrain_client is None:
            try:
                logger.info("Connecting to FineGrain Super-Resolution Space...")
                self._finegrain_client = Client("FineGrain/finegrain-image-enhancer", token=token)
            except Exception as e:
                logger.warning(f"Could not connect to FineGrain Space: {e}")
                self._finegrain_client = None

        return self._finegrain_client

    def upscale_with_onnx(
        self, image_bytes: bytes, scale: int = 2, mode: str = "photo"
    ) -> Tuple[bytes, str]:
        """
        Executes local neural super-resolution via Real-ESRGAN Compact ONNX runtime on CPU:
        - 100% offline, zero cloud API calls, zero latency queue.
        - The model natively performs 4x neural reconstruction.
        - For 2x: Runs 4x inference and downsamples to 2x with Lanczos (super-sampled anti-aliasing).
        - For 4x: Returns full 4x neural super-resolution.
        """
        session = self._get_onnx_session()
        if session is None:
            raise RuntimeError("Real-ESRGAN ONNX session is not available")

        # Decode image using PIL with EXIF orientation correction
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        orig_w, orig_h = pil_img.size

        # Bound input if excessive to prevent CPU memory thrashing
        max_dim = MAX_INPUT_DIMENSION_4X if scale == 4 else MAX_INPUT_DIMENSION_2X
        if pil_img.width > max_dim or pil_img.height > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            orig_w, orig_h = pil_img.size

        # Normalize to float32 tensor [1, 3, H, W] in range [0.0, 1.0]
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # (C, H, W)
        input_tensor = np.expand_dims(img_np, axis=0)  # (1, 3, H, W)

        # Run ONNX inference
        outputs = session.run(None, {"input": input_tensor})
        out_tensor = outputs[0][0]  # Shape: (3, H*4, W*4)

        # Post-process: clamp [0.0, 1.0], transpose to (H*4, W*4, 3) and cast to uint8
        out_tensor = np.clip(out_tensor, 0.0, 1.0)
        out_np = (np.transpose(out_tensor, (1, 2, 0)) * 255.0).astype(np.uint8)
        out_pil = Image.fromarray(out_np, mode="RGB")

        if scale == 2:
            target_w = orig_w * 2
            target_h = orig_h * 2
            out_pil = out_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            model_name = "Real-ESRGAN AI (2x HD)"
        else:
            # scale == 4
            target_w = out_pil.width
            target_h = out_pil.height
            if target_w > MAX_OUTPUT_DIMENSION or target_h > MAX_OUTPUT_DIMENSION:
                ratio = min(MAX_OUTPUT_DIMENSION / target_w, MAX_OUTPUT_DIMENSION / target_h)
                out_pil = out_pil.resize(
                    (int(target_w * ratio), int(target_h * ratio)),
                    Image.Resampling.LANCZOS,
                )
            model_name = "Real-ESRGAN AI (4x Ultra HD)"

        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue(), model_name

    def upscale_with_ai(
        self, image_bytes: bytes, scale: int = 2, mode: str = "photo"
    ) -> Tuple[bytes, str]:
        """
        Attempts AI Super-Resolution:
        1. Local Real-ESRGAN Compact ONNX model (preferred, fast, 100% offline, $0 cost)
        2. Hugging Face Spaces fallback if token is configured
        """
        # Attempt 1: Local ONNX Model (no cloud dependency)
        try:
            return self.upscale_with_onnx(image_bytes, scale=scale, mode=mode)
        except Exception as onnx_err:
            logger.warning(f"Local ONNX upscale unavailable ({onnx_err}); checking cloud fallback...")

        # Attempt 2: Cloud AI if token is configured
        token = settings.HF_TOKEN.strip()
        has_token = bool(token and token != "hf_your_token_here")
        if not has_token:
            raise RuntimeError("Local ONNX model unavailable and no cloud token configured")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_input_path = temp_file.name

        try:
            client = self._get_codeformer_client()
            if client is None:
                raise RuntimeError("Cloud AI upscale client unavailable")

            logger.info(f"Running CodeFormer super-resolution (scale={scale}x, mode={mode})...")
            result = client.predict(
                image=handle_file(temp_input_path),
                face_align=True,
                background_enhance=True,
                face_upsample=True,
                upscale=2,
                codeformer_fidelity=0.6 if mode == "photo" else 0.8,
                api_name="/inference",
            )

            output_path = result[0] if isinstance(result, (list, tuple)) else result
            if isinstance(output_path, dict) and "path" in output_path:
                output_path = output_path["path"]

            with open(output_path, "rb") as f:
                output_bytes = f.read()

            if scale == 4:
                out_img = Image.open(io.BytesIO(output_bytes))
                out_img = out_img.resize(
                    (out_img.width * 2, out_img.height * 2), Image.Resampling.LANCZOS
                )
                out_img = out_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
                buf = io.BytesIO()
                out_img.save(buf, format="JPEG", quality=95, optimize=True)
                output_bytes = buf.getvalue()
                model_name = "CodeFormer AI (4x Ultra)"
            else:
                model_name = "CodeFormer AI (2x HD)"

            return output_bytes, model_name

        except Exception as e:
            logger.warning(f"Cloud AI upscale failed ({e}). Falling back to Pillow engine...")
            raise RuntimeError(f"Cloud AI failed: {str(e)}")

        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    def upscale_with_pillow(
        self, image_bytes: bytes, scale: int = 2
    ) -> Tuple[bytes, str]:
        """
        Deterministic, zero-latency server-side Lanczos super-sampling with unsharp masking.
        Acts as guaranteed fallback so user operations never fail completely.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            target_w = img.width * scale
            target_h = img.height * scale

            # Cap output to MAX_OUTPUT_DIMENSION
            if target_w > MAX_OUTPUT_DIMENSION or target_h > MAX_OUTPUT_DIMENSION:
                ratio = min(MAX_OUTPUT_DIMENSION / target_w, MAX_OUTPUT_DIMENSION / target_h)
                target_w = int(target_w * ratio)
                target_h = int(target_h * ratio)

            # High-fidelity Lanczos resampling
            upscaled = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            # High-frequency edge preservation unsharp mask
            radius = 1.6 if scale == 4 else 1.2
            percent = 140 if scale == 4 else 120
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))

            out = io.BytesIO()
            upscaled.save(out, format="JPEG", quality=95, optimize=True)
            return out.getvalue(), f"Basic Upscaler ({scale}x Lanczos)"

        except Exception as e:
            logger.error(f"Pillow fallback upscale failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image upscale engine error: {str(e)}",
            )

    def process_upscale(
        self, image_bytes: bytes, scale: int = 2, mode: str = "photo"
    ) -> Tuple[bytes, str, int, bool]:
        """
        Processes image upscaling with ordered fallback:
        1. AI Super-Resolution (Local ONNX Real-ESRGAN / Cloud Space)
        2. Deterministic Server-side Lanczos Super-Sampler
        Returns: (output_bytes, model_name, processing_ms, is_basic_fallback)
        """
        start_time = time.time()
        is_basic_fallback = False

        # Attempt 1: AI Super-Resolution
        try:
            output_bytes, model_name = self.upscale_with_ai(
                image_bytes, scale=scale, mode=mode
            )
        except Exception as ai_err:
            logger.warning(f"AI upscale unavailable or failed ({ai_err}). Falling back to Pillow...")
            # Attempt 2: Fallback to deterministic server-side Lanczos
            is_basic_fallback = True
            output_bytes, model_name = self.upscale_with_pillow(
                image_bytes, scale=scale
            )

        duration_ms = int((time.time() - start_time) * 1000)
        return output_bytes, model_name, duration_ms, is_basic_fallback

    def get_health_status(self) -> dict:
        token = settings.HF_TOKEN.strip()
        token_configured = bool(token and token != "hf_your_token_here")
        has_local_onnx = os.path.exists(self._model_path)
        return {
            "status": "healthy",
            "token_configured": token_configured,
            "scales": [2, 4],
            "models": {
                "local_realesrgan": {
                    "provider": "Local Real-ESRGAN Compact ONNX",
                    "available": has_local_onnx,
                    "scales": [2, 4],
                    "modes": ["photo", "illustration"],
                },
                "ai_super_res": {
                    "provider": "Local ONNX / Cloud AI",
                    "available": has_local_onnx or token_configured,
                    "scales": [2, 4],
                    "modes": ["photo", "illustration"],
                },
                "basic_lanczos": {
                    "provider": "Server-side Engine",
                    "available": True,
                    "scales": [2, 4],
                    "modes": ["photo", "illustration"],
                },
            },
            "default_scale": 2,
        }


upscale_registry = UpscaleModelRegistry()
