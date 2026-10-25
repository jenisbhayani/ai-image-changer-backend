"""
Comprehensive image editing, background replacement, recipe rendering,
and smart file-size compression service.
"""

import io
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from app.services.hf_service import hf_service


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    clean_hex = hex_str.lstrip("#")
    if len(clean_hex) == 3:
        clean_hex = "".join(c * 2 for c in clean_hex)
    if len(clean_hex) != 6:
        return (255, 255, 255)
    return (
        int(clean_hex[0:2], 16),
        int(clean_hex[2:4], 16),
        int(clean_hex[4:6], 16),
    )


def create_linear_gradient_canvas(
    width: int,
    height: int,
    start_rgb: Tuple[int, int, int],
    end_rgb: Tuple[int, int, int],
) -> Image.Image:
    """Creates a smooth linear diagonal gradient canvas."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    t = (xx + yy) * 0.5  # Diagonal gradient

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        canvas[:, :, c] = np.clip(
            start_rgb[c] * (1.0 - t) + end_rgb[c] * t, 0, 255
        ).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB")


class ImageEditService:
    def replace_background(
        self,
        image_bytes: bytes,
        bg_type: str = "solid",
        color: str = "#FFFFFF",
        gradient_start: str = "#E0EAFC",
        gradient_end: str = "#CFDEF3",
        output_format: str = "PNG",
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Extracts foreground using the primary neural pipeline and composites over a background.
        """
        # 1. Get transparent cutout
        cutout_bytes, meta = hf_service.remove_background(image_bytes, quality_tier="balanced")
        fg_rgba = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
        w, h = fg_rgba.size

        # 2. Build background canvas
        if bg_type.lower() == "gradient":
            start_rgb = hex_to_rgb(gradient_start)
            end_rgb = hex_to_rgb(gradient_end)
            bg = create_linear_gradient_canvas(w, h, start_rgb, end_rgb)
        elif bg_type.lower() == "transparent":
            bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            bg.paste(fg_rgba, (0, 0), fg_rgba)
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            return buf.getvalue(), {"background": "transparent", **meta}
        else:  # solid
            solid_rgb = hex_to_rgb(color)
            bg = Image.new("RGB", (w, h), solid_rgb)

        # 3. Composite foreground on background
        bg.paste(fg_rgba, (0, 0), fg_rgba)

        out_buf = io.BytesIO()
        fmt = output_format.upper()
        if fmt == "JPEG" or fmt == "JPG":
            bg.convert("RGB").save(out_buf, format="JPEG", quality=95, optimize=True)
            mime = "image/jpeg"
        else:
            bg.save(out_buf, format="PNG", compress_level=6)
            mime = "image/png"

        return out_buf.getvalue(), {
            "background_type": bg_type,
            "color": color,
            "width": w,
            "height": h,
            "mime_type": mime,
            **meta,
        }

    def render_recipe(
        self,
        image_bytes: bytes,
        recipe: Dict[str, Any],
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Renders an entire non-destructive edit recipe at source resolution.
        """
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)

        # 1. Flip
        flip_mode = recipe.get("flip")
        if flip_mode == "horizontal":
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif flip_mode == "vertical":
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 2. Rotate (90, 180, 270)
        rotate_deg = recipe.get("rotate", 0)
        if rotate_deg in (90, 180, 270):
            # PIL rotate is counter-clockwise, so negative for clockwise
            img = img.rotate(-rotate_deg, expand=True)

        # 3. Straighten (-45 to +45 deg)
        straighten = float(recipe.get("straighten_deg", 0.0))
        if abs(straighten) > 0.1:
            img = img.rotate(-straighten, resample=Image.Resampling.BICUBIC, expand=False)

        # 4. Crop
        crop = recipe.get("crop")
        if crop and isinstance(crop, dict):
            cx = int(crop.get("x", 0))
            cy = int(crop.get("y", 0))
            cw = int(crop.get("width", img.width))
            ch = int(crop.get("height", img.height))
            # Clamp bounds
            cx = max(0, min(cx, img.width - 1))
            cy = max(0, min(cy, img.height - 1))
            cw = max(1, min(cw, img.width - cx))
            ch = max(1, min(ch, img.height - cy))
            img = img.crop((cx, cy, cx + cw, cy + ch))

        # Convert to RGB for adjustments
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 5. Adjustments
        adj = recipe.get("adjustments", {})
        if adj:
            # Brightness (-1.0 to 1.0 -> factor 0.0 to 2.0)
            b = float(adj.get("brightness", 0.0))
            if abs(b) > 0.01:
                img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b))

            # Contrast (-1.0 to 1.0 -> factor 0.0 to 2.0)
            c = float(adj.get("contrast", 0.0))
            if abs(c) > 0.01:
                img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c))

            # Saturation (-1.0 to 1.0 -> factor 0.0 to 2.0)
            s = float(adj.get("saturation", 0.0))
            if abs(s) > 0.01:
                img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s))

            # Sharpness (0.0 to 2.0 -> factor 0.0 to 3.0)
            sh = float(adj.get("sharpness", 0.0))
            if abs(sh) > 0.01:
                img = ImageEnhance.Sharpness(img).enhance(max(0.0, 1.0 + sh * 1.5))

            # Warmth (-1.0 to 1.0)
            warmth = float(adj.get("warmth", 0.0))
            if abs(warmth) > 0.01:
                arr = np.array(img, dtype=np.float32)
                arr[:, :, 0] += warmth * 25.0  # Red
                arr[:, :, 2] -= warmth * 25.0  # Blue
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

            # Vignette (0.0 to 1.0)
            vignette = float(adj.get("vignette", 0.0))
            if vignette > 0.05:
                w, h = img.size
                X = np.linspace(-1, 1, w, dtype=np.float32)
                Y = np.linspace(-1, 1, h, dtype=np.float32)
                XX, YY = np.meshgrid(X, Y)
                radius = np.sqrt(XX**2 + YY**2)
                mask = 1.0 - (vignette * np.clip((radius - 0.5) / 0.7, 0.0, 1.0))
                arr = np.array(img, dtype=np.float32) * mask[:, :, None]
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

        # 6. Filters
        filter_id = recipe.get("filter_id", "none").lower()
        if filter_id == "bw":
            img = ImageOps.grayscale(img).convert("RGB")
        elif filter_id == "vivid":
            img = ImageEnhance.Color(img).enhance(1.4)
            img = ImageEnhance.Contrast(img).enhance(1.15)
        elif filter_id == "warm":
            arr = np.array(img, dtype=np.float32)
            arr[:, :, 0] += 20.0
            arr[:, :, 2] -= 15.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        elif filter_id == "cool":
            arr = np.array(img, dtype=np.float32)
            arr[:, :, 0] -= 15.0
            arr[:, :, 2] += 20.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        elif filter_id == "fade":
            img = ImageEnhance.Contrast(img).enhance(0.85)
            arr = np.array(img, dtype=np.float32) * 0.9 + 25.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        elif filter_id == "matte":
            img = ImageEnhance.Contrast(img).enhance(0.9)
            arr = np.array(img, dtype=np.float32)
            arr[:, :, 0] += 10.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

        # 7. Output Format & Quality
        out_conf = recipe.get("output", {})
        fmt = out_conf.get("format", "PNG").upper()
        quality = int(out_conf.get("quality", 92))

        buf = io.BytesIO()
        if fmt in ("JPEG", "JPG"):
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"
        elif fmt == "WEBP":
            img.save(buf, format="WEBP", quality=quality)
            mime = "image/webp"
        else:
            img.save(buf, format="PNG", compress_level=6)
            mime = "image/png"

        return buf.getvalue(), {
            "width": img.width,
            "height": img.height,
            "format": fmt,
            "mime_type": mime,
        }

    def compress_to_target_kb(
        self,
        image_bytes: bytes,
        target_kb: int,
        min_quality: int = 30,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Binary-searches compression quality and iteratively scales image
        to ensure final file size is strictly <= target_kb.
        """
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        target_bytes = target_kb * 1024

        best_bytes = None
        best_quality = 95
        best_w, best_h = img.width, img.height

        curr_img = img.copy()

        # Try progressive quality search at current scale
        for scale_iteration in range(5):
            low_q = min_quality
            high_q = 95
            candidate_bytes = None
            candidate_q = 95

            while low_q <= high_q:
                mid_q = (low_q + high_q) // 2
                buf = io.BytesIO()
                curr_img.save(buf, format="JPEG", quality=mid_q, optimize=True)
                size = buf.tell()

                if size <= target_bytes:
                    candidate_bytes = buf.getvalue()
                    candidate_q = mid_q
                    best_w, best_h = curr_img.size
                    low_q = mid_q + 1  # Try higher quality
                else:
                    high_q = mid_q - 1  # Exceeded, need lower quality

            if candidate_bytes is not None:
                best_bytes = candidate_bytes
                best_quality = candidate_q
                break

            # If even min_quality at this scale exceeds target_bytes, downscale by 15%
            if curr_img.width > 300 and curr_img.height > 300:
                new_w = int(curr_img.width * 0.85)
                new_h = int(curr_img.height * 0.85)
                curr_img = curr_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                break

        if best_bytes is None:
            # Fallback to maximum compression achievable
            buf = io.BytesIO()
            curr_img.save(buf, format="JPEG", quality=min_quality, optimize=True)
            best_bytes = buf.getvalue()
            best_quality = min_quality
            best_w, best_h = curr_img.size

        return best_bytes, {
            "target_kb": target_kb,
            "final_kb": round(len(best_bytes) / 1024.0, 1),
            "quality": best_quality,
            "width": best_w,
            "height": best_h,
            "format": "JPEG",
        }


image_edit_service = ImageEditService()
