"""
Evaluation harness for AI Image Changer background removal pipelines.

Generates 5-panel contact sheets:
[Original | Alpha Mask | Checkerboard | Magenta (#FF00FF) | Dark Green (#003300)]
Computes quantitative benchmark metrics:
- Ground-truth metrics: IoU, MAE, Boundary F1, Trimap MAE
- Heuristic metrics: Uncertain band ratio, component count, edge halo score
- Resource metrics: Latency (p50, p95), Peak RAM
"""

import argparse
import io
import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Ensure backend root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def create_checkerboard(width: int, height: int, square_size: int = 16) -> Image.Image:
    """Creates a classic transparency checkerboard pattern."""
    img = Image.new("RGB", (width, height), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    dark_gray = (150, 150, 150)
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 1:
                draw.rectangle([x, y, x + square_size - 1, y + square_size - 1], fill=dark_gray)
    return img


def composite_cutout(cutout_rgba: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
    """Composites an RGBA cutout onto a solid RGB background."""
    bg = Image.new("RGB", cutout_rgba.size, bg_color)
    bg.paste(cutout_rgba, (0, 0), cutout_rgba)
    return bg


def create_contact_sheet(
    orig_rgb: Image.Image,
    cutout_rgba: Image.Image,
    max_panel_h: int = 400,
) -> Image.Image:
    """
    Builds a 5-panel contact sheet:
    [Original | Alpha Mask | Checkerboard | Saturated Magenta | Dark Green]
    """
    w, h = orig_rgb.size
    scale = max_panel_h / max(h, 1)
    panel_w = max(1, int(w * scale))
    panel_h = max(1, int(h * scale))

    # Resize images for contact sheet display
    thumb_orig = orig_rgb.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
    thumb_cutout = cutout_rgba.resize((panel_w, panel_h), Image.Resampling.BILINEAR)

    # Panel 1: Original
    p1 = thumb_orig

    # Panel 2: Alpha mask in grayscale
    alpha = thumb_cutout.getchannel("A") if "A" in thumb_cutout.getbands() else Image.new("L", (panel_w, panel_h), 255)
    p2 = Image.merge("RGB", (alpha, alpha, alpha))

    # Panel 3: Checkerboard
    checker = create_checkerboard(panel_w, panel_h, square_size=16)
    checker.paste(thumb_cutout, (0, 0), thumb_cutout)
    p3 = checker

    # Panel 4: Saturated Magenta (#FF00FF)
    p4 = composite_cutout(thumb_cutout, (255, 0, 255))

    # Panel 5: Dark Green (#003300)
    p5 = composite_cutout(thumb_cutout, (0, 51, 0))

    header_h = 32
    sheet_w = panel_w * 5
    sheet_h = panel_h + header_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)

    panels = [
        ("Original", p1),
        ("Alpha Mask", p2),
        ("Checkerboard", p3),
        ("Magenta (#FF00FF)", p4),
        ("Dark Green (#003300)", p5),
    ]

    for idx, (title, img) in enumerate(panels):
        x_offset = idx * panel_w
        sheet.paste(img, (x_offset, header_h))
        # Draw header dividing lines and title text
        draw.line([(x_offset, 0), (x_offset, sheet_h)], fill=(60, 60, 60), width=1)
        draw.text((x_offset + 8, 8), title, fill=(240, 240, 240))

    return sheet


def compute_mask_metrics(
    pred_alpha: np.ndarray,
    gt_alpha: Optional[np.ndarray],
    orig_rgb: np.ndarray,
) -> Dict[str, float]:
    """
    Computes both supervised (if GT exists) and unsupervised artifact metrics.
    """
    metrics: Dict[str, float] = {}
    pred_norm = pred_alpha.astype(np.float32) / 255.0

    # 1. Supervised metrics if ground truth is present
    if gt_alpha is not None:
        gt_norm = gt_alpha.astype(np.float32) / 255.0
        # Binarized at 0.5
        pred_bin = pred_norm > 0.5
        gt_bin = gt_norm > 0.5

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        metrics["iou"] = float(intersection / max(union, 1))
        metrics["mae"] = float(np.mean(np.abs(pred_norm - gt_norm)))

        # Boundary F-score with 3-pixel tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pred_edge = cv2.morphologyEx(pred_bin.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        gt_edge = cv2.morphologyEx(gt_bin.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

        gt_edge_dilated = cv2.dilate(gt_edge, kernel, iterations=2)
        pred_edge_dilated = cv2.dilate(pred_edge, kernel, iterations=2)

        precision = (pred_edge * gt_edge_dilated).sum() / max(pred_edge.sum(), 1)
        recall = (gt_edge * pred_edge_dilated).sum() / max(gt_edge.sum(), 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        metrics["boundary_f1"] = float(f1)

        # Trimap band error: error in the boundary region
        trimap_band = cv2.dilate(gt_bin.astype(np.uint8), kernel, iterations=5) - cv2.erode(
            gt_bin.astype(np.uint8), kernel, iterations=5
        )
        if trimap_band.sum() > 0:
            metrics["trimap_mae"] = float(np.sum(np.abs(pred_norm - gt_norm) * trimap_band) / trimap_band.sum())
        else:
            metrics["trimap_mae"] = metrics["mae"]

    # 2. Unsupervised / Heuristic artifact scores
    # A. Uncertain band ratio: fraction of mask in hazy 0.10 - 0.90 zone
    fg_area = float((pred_alpha > 25).sum())
    uncertain_px = float(((pred_alpha >= 25) & (pred_alpha <= 230)).sum())
    metrics["uncertain_ratio"] = float(uncertain_px / max(fg_area, 1.0))

    # B. Component fragmentation: number of disconnected islands (>100px)
    bin_mask = (pred_alpha > 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    significant_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 100)
    metrics["component_count"] = float(significant_components)

    # C. Edge halo measure: color discrepancy at the boundary
    # Sample 3px outer halo band vs 3px inner edge band
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_fg = cv2.dilate(bin_mask, kernel3, iterations=2)
    eroded_fg = cv2.erode(bin_mask, kernel3, iterations=2)
    outer_halo_band = (dilated_fg - bin_mask) > 0
    inner_edge_band = (bin_mask - eroded_fg) > 0

    if outer_halo_band.sum() > 10 and inner_edge_band.sum() > 10:
        # Measure mean color difference between foreground edge and removed background edge
        outer_rgb = orig_rgb[outer_halo_band].astype(np.float32)
        inner_rgb = orig_rgb[inner_edge_band].astype(np.float32)
        halo_metric = float(np.linalg.norm(np.mean(inner_rgb, axis=0) - np.mean(outer_rgb, axis=0)))
        metrics["halo_score"] = halo_metric
    else:
        metrics["halo_score"] = 0.0

    return metrics


def run_legacy_pipeline(img_bytes: bytes) -> Tuple[Image.Image, Image.Image]:
    """Runs the existing legacy pipeline (RMBG-1.4 + legacy cutout_refiner)."""
    from app.services.hf_service import HuggingFaceInferenceService

    service = HuggingFaceInferenceService()
    cutout_bytes = service.remove_background_local(img_bytes)
    cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    alpha = cutout.getchannel("A")
    return cutout, alpha


def run_pipeline_by_name(pipeline_name: str, img_bytes: bytes) -> Tuple[Image.Image, Image.Image]:
    """Dispatches to the requested pipeline."""
    if pipeline_name == "legacy":
        return run_legacy_pipeline(img_bytes)
    elif pipeline_name in ("fast", "balanced", "high"):
        from app.services.model_registry import get_inference_service

        service = get_inference_service()
        cutout_bytes, metadata = service.remove_background(img_bytes, quality_tier=pipeline_name)
        cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
        alpha = cutout.getchannel("A")
        return cutout, alpha
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


def main():
    parser = argparse.ArgumentParser(description="AI Image Changer Background Removal Evaluation Harness")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="legacy",
        choices=["legacy", "fast", "balanced", "high", "all"],
        help="Pipeline to evaluate",
    )
    parser.add_argument("--category", type=str, default="", help="Evaluate specific category only")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "images"),
        help="Path to evaluation images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "results"),
        help="Path to save evaluation artifacts",
    )
    args = parser.parse_args()

    pipelines_to_run = ["legacy"] if args.pipeline == "legacy" else (
        ["legacy", "fast", "balanced", "high"] if args.pipeline == "all" else [args.pipeline]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover images
    test_cases: List[Tuple[str, str, str, Optional[str]]] = []  # (category, stem, img_path, mask_path)
    for cat in sorted(os.listdir(args.eval_dir)):
        cat_dir = os.path.join(args.eval_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        if args.category and cat != args.category:
            continue
        for f in sorted(os.listdir(cat_dir)):
            if f.endswith(".mask.png") or not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            stem = os.path.splitext(f)[0]
            img_path = os.path.join(cat_dir, f)
            mask_path = os.path.join(cat_dir, f"{stem}.mask.png")
            test_cases.append((cat, stem, img_path, mask_path if os.path.exists(mask_path) else None))

    if not test_cases:
        print(f"No test images found in {args.eval_dir}")
        return

    print(f"\n=======================================================")
    print(f"  AI Image Changer - Evaluation Harness")
    print(f"  Test Images Found: {len(test_cases)}")
    print(f"  Pipelines to Run: {pipelines_to_run}")
    print(f"=======================================================\n")

    overall_results: Dict[str, Dict] = {}

    for pipe in pipelines_to_run:
        pipe_dir = os.path.join(args.output_dir, pipe)
        os.makedirs(pipe_dir, exist_ok=True)
        print(f"\n>>> Running Evaluation for Pipeline: [{pipe}]")

        pipe_records = []
        latencies = []
        ram_peaks = []

        for idx, (cat, stem, img_path, mask_path) in enumerate(test_cases, 1):
            print(f"  [{idx}/{len(test_cases)}] {cat} / {stem} ...", end=" ", flush=True)

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            orig_img = Image.open(io.BytesIO(img_bytes))
            orig_img = ImageOps.exif_transpose(orig_img).convert("RGB")
            orig_rgb = np.array(orig_img)

            gt_alpha = None
            if mask_path:
                gt_img = Image.open(mask_path).convert("L")
                gt_alpha = np.array(gt_img.resize(orig_img.size, Image.Resampling.NEAREST))

            # Benchmark inference & RAM
            tracemalloc.start()
            t0 = time.perf_counter()
            try:
                cutout_rgba, pred_alpha_img = run_pipeline_by_name(pipe, img_bytes)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                current_ram, peak_ram = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_ram_mb = peak_ram / (1024 * 1024)
            except Exception as e:
                tracemalloc.stop()
                print(f"FAILED: {e}")
                continue

            latencies.append(latency_ms)
            ram_peaks.append(peak_ram_mb)

            pred_alpha = np.array(pred_alpha_img)
            metrics = compute_mask_metrics(pred_alpha, gt_alpha, orig_rgb)
            metrics["latency_ms"] = latency_ms
            metrics["peak_ram_mb"] = peak_ram_mb
            metrics["category"] = cat
            metrics["stem"] = stem

            pipe_records.append(metrics)

            # Save cutout PNG
            cutout_path = os.path.join(pipe_dir, f"{cat}_{stem}_cutout.png")
            cutout_rgba.save(cutout_path, format="PNG")

            # Save contact sheet
            contact_sheet = create_contact_sheet(orig_img, cutout_rgba)
            contact_path = os.path.join(pipe_dir, f"{cat}_{stem}_contact.png")
            contact_sheet.save(contact_path, format="PNG")

            print(
                f"Done ({latency_ms:.0f}ms, RAM: {peak_ram_mb:.1f}MB, "
                f"Uncertain: {metrics.get('uncertain_ratio', 0):.2f}, "
                f"Components: {int(metrics.get('component_count', 0))})"
            )

        # Aggregate metrics
        latencies_sorted = sorted(latencies)
        p50_latency = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
        p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0
        mean_uncertain = np.mean([r["uncertain_ratio"] for r in pipe_records]) if pipe_records else 0
        mean_halo = np.mean([r["halo_score"] for r in pipe_records]) if pipe_records else 0
        mean_components = np.mean([r["component_count"] for r in pipe_records]) if pipe_records else 0
        peak_ram_max = max(ram_peaks) if ram_peaks else 0

        overall_results[pipe] = {
            "records": pipe_records,
            "latency_p50_ms": p50_latency,
            "latency_p95_ms": p95_latency,
            "mean_uncertain_ratio": float(mean_uncertain),
            "mean_halo_score": float(mean_halo),
            "mean_component_count": float(mean_components),
            "peak_ram_mb": float(peak_ram_max),
        }

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Pipeline':<12} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Peak RAM':<10} | {'Uncertain %':<12} | {'Halo Score':<12} | {'Components':<10}")
    print("-" * 95)
    for pipe, summary in overall_results.items():
        print(
            f"{pipe:<12} | "
            f"{summary['latency_p50_ms']:<10.1f} | "
            f"{summary['latency_p95_ms']:<10.1f} | "
            f"{summary['peak_ram_mb']:<7.1f} MB | "
            f"{summary['mean_uncertain_ratio'] * 100:<10.2f}% | "
            f"{summary['mean_halo_score']:<12.2f} | "
            f"{summary['mean_component_count']:<10.2f}"
        )
    print("=" * 95)

    # Save summary files
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    # Save Markdown report
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Background Removal Evaluation Summary\n\n")
        f.write("| Pipeline | p50 Latency | p95 Latency | Peak RAM | Uncertain % | Mean Halo Score | Mean Components |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for pipe, summary in overall_results.items():
            f.write(
                f"| **{pipe}** | {summary['latency_p50_ms']:.1f} ms | {summary['latency_p95_ms']:.1f} ms | "
                f"{summary['peak_ram_mb']:.1f} MB | {summary['mean_uncertain_ratio'] * 100:.2f}% | "
                f"{summary['mean_halo_score']:.2f} | {summary['mean_component_count']:.2f} |\n"
            )
        f.write("\n### Per-Category Breakdown\n\n")
        f.write("| Category | Pipeline | Latency (ms) | Uncertain % | Halo Score | Components |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for pipe, summary in overall_results.items():
            for rec in summary["records"]:
                f.write(
                    f"| {rec['category']} | {pipe} | {rec['latency_ms']:.1f} | "
                    f"{rec['uncertain_ratio']*100:.2f}% | {rec['halo_score']:.2f} | {rec['component_count']:.0f} |\n"
                )
    print(f"\nSaved benchmark results to:\n- {json_path}\n- {md_path}\n")


if __name__ == "__main__":
    main()
