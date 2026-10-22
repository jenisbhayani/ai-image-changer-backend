import logging
from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from fastapi import HTTPException, status
from app.core.config import settings

logger = logging.getLogger(__name__)


class BiRefNetService:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.transform = None

    def initialize(self):
        """Loads BiRefNet weights on backend startup."""
        if self.is_loaded:
            return

        model_id = settings.BIREFNET_MODEL_ID
        logger.info(f"Loading BiRefNet model '{model_id}' on device: {self.device}...")

        try:
            from transformers import AutoModelForImageSegmentation

            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Input transformation
            inp_size = (settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE)
            self.transform = transforms.Compose([
                transforms.Resize(inp_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.is_loaded = True
            self.load_error = None
            logger.info(f"BiRefNet model '{model_id}' successfully loaded on {self.device}.")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed to load BiRefNet model '{model_id}': {e}", exc_info=True)

    def predict_mask(self, image: Image.Image) -> Image.Image:
        """
        Runs BiRefNet inference on a PIL Image and returns an 8-bit alpha mask (mode 'L')
        matching the exact width and height of the input image.
        """
        if not self.is_loaded or self.model is None or self.transform is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"BiRefNet inference model is currently unavailable on the server. ({self.load_error or 'Initializing'})"
            )

        try:
            # Prepare RGB input
            rgb_image = image.convert("RGB")
            orig_size = rgb_image.size  # (width, height)

            # Preprocess tensor
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()
                pred_tensor = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred_tensor)

            # Bilinear upsample mask back to original resolution
            alpha_mask = pred_pil.resize(orig_size, resample=Image.Resampling.BILINEAR)
            return alpha_mask

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The server ran out of memory processing this high-resolution image. Please choose a slightly smaller photo."
            )
        except Exception as e:
            logger.error(f"Inference error in BiRefNet: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cloud AI inference failed: {str(e)}"
            )


birefnet_service = BiRefNetService()
