"""
Unit tests for the improved CutoutRefiner and color decontamination.
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(TEST_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import cv2
from app.utils.cutout_refiner import (
    cutout_refiner,
    fast_guided_filter,
    decontaminate_foreground_colors,
)


def test_two_large_blobs_group_kept():
    """Two large blobs of similar size (group) are both kept."""
    H, W = 600, 800
    m = np.zeros((H, W), dtype=np.float32)
    # Blob 1: radius 100
    cv2.circle(m, (250, 300), 100, 0.95, -1)
    # Blob 2: radius 90 (similar size, ~81% area of blob 1)
    cv2.circle(m, (550, 300), 90, 0.95, -1)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    assert refined[300, 250] > 200, "First group subject should be kept"
    assert refined[300, 550] > 200, "Second group subject should be kept"


def test_small_low_conf_speck_far_removed():
    """A small low-confidence speck far from the subject is removed."""
    H, W = 600, 800
    m = np.zeros((H, W), dtype=np.float32)
    # Main subject
    cv2.circle(m, (400, 300), 120, 0.95, -1)
    # Speck far away at corner (x=50, y=50) with low confidence
    cv2.circle(m, (50, 50), 4, 0.45, -1)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    assert refined[300, 400] > 200, "Main subject should be kept"
    assert refined[50, 50] == 0, "Distant low-confidence speck should be removed"


def test_small_high_conf_blob_near_subject_kept():
    """A small high-confidence blob near the subject (e.g. earring/string) is kept."""
    H, W = 600, 800
    m = np.zeros((H, W), dtype=np.float32)
    # Main subject centered at (400, 300), radius 100 (boundary is at x=500)
    cv2.circle(m, (400, 300), 100, 0.95, -1)
    # Small earring at x=515 (within 3% of long side = 24px from boundary)
    cv2.circle(m, (515, 300), 6, 0.90, -1)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    assert refined[300, 400] > 200, "Main subject should be kept"
    assert refined[300, 515] > 150, "Near high-confidence satellite should be kept"


def test_soft_edge_ramp_preserves_intermediate_alpha():
    """A soft edge ramp keeps intermediate alpha values (no hard cutoff): assert values between 30 and 225."""
    H, W = 400, 400
    m = np.zeros((H, W), dtype=np.float32)
    # Create smooth radial gradient from center
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - 200) ** 2 + (y - 200) ** 2)
    # Soft ramp between radius 80 and 120
    m = np.clip((120.0 - dist) / 40.0, 0.0, 1.0).astype(np.float32)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    semi_transparent = (refined >= 30) & (refined <= 225)
    assert np.sum(semi_transparent) > 100, "Soft edge ramp should preserve intermediate alpha values (30..225)"


def test_subject_touching_border_preserves_alpha():
    """A subject touching the image border keeps alpha > 0 on that border row."""
    H, W = 400, 400
    m = np.zeros((H, W), dtype=np.float32)
    # Subject touches top edge (y=0) and bottom edge (y=H-1)
    cv2.rectangle(m, (150, 0), (250, H - 1), 0.95, -1)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    assert refined[0, 200] > 0, "Top border touching subject must have alpha > 0"
    assert refined[H - 1, 200] > 0, "Bottom border touching subject must have alpha > 0"


def test_pinhole_filled_large_hole_kept():
    """A tiny enclosed low-confidence pinhole becomes opaque; a large enclosed hole (handle-like) stays transparent."""
    H, W = 600, 600
    m = np.zeros((H, W), dtype=np.float32)
    # Large solid subject
    cv2.circle(m, (300, 300), 200, 0.95, -1)

    # Tiny pinhole: area ~12 pixels (< 0.0001 * 360000 = 36 pixels) with low confidence
    cv2.circle(m, (300, 300), 2, 0.05, -1)
    # Large hole (handle-like): radius 40 (area ~5000 pixels)
    cv2.circle(m, (300, 180), 40, 0.0, -1)

    refined = cutout_refiner.refine_alpha_mask(m, target_size=(W, H), subject_only=True)
    # Pinhole filled
    assert refined[300, 300] > 200, "Tiny enclosed pinhole should be filled opaque"
    # Large opening kept transparent
    assert refined[180, 300] == 0, "Large handle hole should remain transparent"


def test_decontaminate_foreground_colors():
    """
    Pixels with alpha == 0 end up RGB (0,0,0);
    edge pixels near a red subject on a green background lose green tint (mean G drops vs the input).
    """
    H, W = 100, 100
    # Red subject on green background
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :50] = [0, 220, 0]    # Green background
    rgb[:, 50:] = [220, 0, 0]    # Red foreground

    # Simulate green background contamination on the edge pixels (columns 48 to 52)
    rgb[:, 48:52] = [120, 150, 0]

    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[:, 52:] = 255          # Solid foreground
    alpha[:, 48:52] = 160        # Semi-transparent edge

    decontaminated = decontaminate_foreground_colors(rgb, alpha)

    # 1. Pixels with alpha == 0 end up RGB (0,0,0)
    assert np.all(decontaminated[:, :48] == 0), "Pixels with alpha == 0 must have RGB (0,0,0)"

    # 2. Edge pixels near red subject on green background lose green tint (mean G drops vs the input)
    input_edge_green = np.mean(rgb[:, 48:52, 1])
    decontam_edge_green = np.mean(decontaminated[:, 48:52, 1])
    assert decontam_edge_green < input_edge_green, "Edge green tint should drop significantly after decontamination"


def test_empty_mask_handling():
    """Empty mask input returns an array of the right shape without crashing."""
    H, W = 200, 300
    empty_m = np.zeros((H, W), dtype=np.float32)

    refined = cutout_refiner.refine_alpha_mask(empty_m, target_size=(W, H), subject_only=True)
    assert refined.shape == (H, W)
    assert refined.dtype == np.uint8
    assert np.all(refined == 0)
