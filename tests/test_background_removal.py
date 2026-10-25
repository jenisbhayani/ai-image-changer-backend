"""
Comprehensive unit tests for background removal pipeline, refiner, and API compatibility.
"""

import base64
import io
import os
import sys

# Ensure backend root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(TEST_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.model_registry import letterbox_image, unletterbox_mask, get_inference_service
from app.utils.cutout_refiner import (
    cutout_refiner,
    fast_guided_filter,
    decontaminate_foreground_colors,
)


def test_preprocessing_various_formats():
    """Verifies preprocessing gracefully handles RGBA, Grayscale, CMYK, and non-square sizes."""
    # 1. Grayscale
    gray = Image.new("L", (320, 240), 128)
    lb_gray, pad_info = letterbox_image(gray.convert("RGB"))
    assert lb_gray.size == (1024, 1024)

    # 2. RGBA with transparent alpha
    rgba = Image.new("RGBA", (500, 300), (255, 0, 0, 128))
    lb_rgba, _ = letterbox_image(rgba.convert("RGB"))
    assert lb_rgba.size == (1024, 1024)

    # 3. CMYK
    cmyk = Image.new("CMYK", (200, 400), (100, 50, 0, 0))
    lb_cmyk, _ = letterbox_image(cmyk.convert("RGB"))
    assert lb_cmyk.size == (1024, 1024)

    # 4. Aspect unletterboxing
    mask_1024 = np.ones((1024, 1024), dtype=np.float32)
    unpadded = unletterbox_mask(mask_1024, pad_info, (320, 240))
    assert unpadded.shape == (240, 320)


def test_component_selection_preserves_group():
    """Verifies multi-person / group photos keep all subjects, unlike legacy refiner."""
    H, W = 800, 1000
    mask = np.zeros((H, W), dtype=np.float32)

    # Person 1 (left)
    cv2.circle(mask, (250, 400), 120, 0.95, -1)
    # Person 2 (right)
    cv2.circle(mask, (750, 400), 120, 0.95, -1)
    # Detached handbag / object
    cv2.rectangle(mask, (470, 490), (530, 570), 0.95, -1)
    # Tiny noise speck
    cv2.circle(mask, (50, 50), 2, 0.30, -1)

    dummy_rgb = np.ones((H, W, 3), dtype=np.uint8) * 128
    refined = cutout_refiner.refine_alpha_mask(
        mask, target_size=(W, H), rgb_image=dummy_rgb, subject_only=True
    )

    # Check both people and handbag are preserved
    assert refined[400, 250] > 200, "Person 1 was erroneously deleted"
    assert refined[400, 750] > 200, "Person 2 was erroneously deleted"
    assert refined[530, 500] > 200, "Handbag was erroneously deleted"
    # Noise speck should be suppressed
    assert refined[50, 50] < 50, "Noise speck was not cleaned"


def test_hole_fill_rule():
    """Verifies tiny low-confidence holes are filled while large openings are kept."""
    H, W = 600, 600
    mask = np.zeros((H, W), dtype=np.float32)
    # Solid subject
    cv2.circle(mask, (300, 300), 200, 0.95, -1)

    # Tiny speckle hole (e.g. dark button, radius 3) with low model confidence
    cv2.circle(mask, (300, 300), 3, 0.10, -1)
    # Real large opening (e.g. mug handle / gap between arm, radius 40)
    cv2.circle(mask, (300, 200), 40, 0.05, -1)

    dummy_rgb = np.ones((H, W, 3), dtype=np.uint8) * 150
    refined = cutout_refiner.refine_alpha_mask(
        mask, target_size=(W, H), rgb_image=dummy_rgb, subject_only=True
    )

    # Large opening should be preserved (transparent)
    assert refined[200, 300] == 0, "Large opening was erroneously filled"
    # Tiny dark button speckle should be filled
    assert refined[300, 300] > 100, "Small interior hole was not filled"


def test_color_decontamination():
    """Verifies background color bleed is removed from edges."""
    H, W = 100, 100
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Green background bleed on edge
    rgb[:, :, 1] = 200  # Bright green
    # Semi-transparent edge alpha
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[:, 50:] = 255  # Foreground
    alpha[:, 45:50] = 128  # Semi-transparent edge band

    decontaminated = decontaminate_foreground_colors(rgb, alpha)
    assert decontaminated.shape == (H, W, 3)
    # Pixels where alpha == 0 must have RGB set to 0
    assert np.all(decontaminated[:, :45] == 0), "Background RGB was not zeroed"


def test_quality_gate_checks():
    """Verifies quality gate detects degenerate masks."""
    dummy_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. Empty mask (<0.5%)
    empty_alpha = np.zeros((200, 200), dtype=np.uint8)
    passes, metrics, warnings = cutout_refiner.evaluate_quality_gate(empty_alpha, dummy_rgb)
    assert not passes
    assert any("Extremely small" in w for w in warnings)

    # 2. Solid mask (>98%)
    solid_alpha = np.full((200, 200), 255, dtype=np.uint8)
    passes, metrics, warnings = cutout_refiner.evaluate_quality_gate(solid_alpha, dummy_rgb)
    assert not passes
    assert any("Near whole-frame" in w for w in warnings)

    # 3. Normal subject (30% coverage)
    normal_alpha = np.zeros((200, 200), dtype=np.uint8)
    normal_alpha[50:150, 50:150] = 255
    passes, metrics, warnings = cutout_refiner.evaluate_quality_gate(normal_alpha, dummy_rgb)
    assert passes
    assert len(warnings) == 0


def test_api_response_schema_compatibility():
    """Verifies BackgroundRemovalService produces valid straight-alpha PNG and expected metadata."""
    service = get_inference_service()

    # Create dummy 100x100 test image bytes
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    png_bytes, metadata = service.remove_background(img_bytes, quality_tier="fast")
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

    # Ensure output is valid PNG with RGBA channels
    out_img = Image.open(io.BytesIO(png_bytes))
    assert out_img.mode == "RGBA"
    assert out_img.size == (100, 100)

    # Check metadata fields required for backward compatibility
    assert "model_used" in metadata
    assert "quality_tier" in metadata
    assert "confidence" in metadata
    assert "processing_ms" in metadata
