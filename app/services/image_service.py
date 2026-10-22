import io
from typing import Tuple
from PIL import Image
from app.services.birefnet_service import birefnet_service


class ImageService:
    def process_background_removal(self, image: Image.Image) -> Tuple[bytes, int, int]:
        """
        Runs cloud BiRefNet segmentation on the image, applies the continuous alpha mask,
        and returns (png_bytes, width, height).
        """
        # Ensure image is in RGB mode before compositing alpha
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        # Predict smooth continuous alpha mask
        alpha_mask = birefnet_service.predict_mask(rgb_image)

        # Apply alpha mask directly to create transparent RGBA image
        rgba_image = rgb_image.copy()
        rgba_image.putalpha(alpha_mask)

        # Encode transparent PNG in-memory
        output_buffer = io.BytesIO()
        rgba_image.save(output_buffer, format="PNG", optimize=True)
        png_bytes = output_buffer.getvalue()

        return png_bytes, width, height


image_service = ImageService()
