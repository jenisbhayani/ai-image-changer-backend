"""
Verification script for bg removal refiner fixes.
Runs the 5 test images against the endpoint in both PNG and Base64 format,
generates color contact sheets, and benchmarks 12MP execution time.
"""

import asyncio
import base64
import io
import os
import sys
import time

from PIL import Image

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import httpx
from app.main import app
from scripts.preview_on_color import create_contact_sheet

IMAGES = [
    ("portrait_hair", os.path.join(BACKEND_ROOT, "eval", "images", "hair_fine", "hair_fine_sample.jpg")),
    ("group", os.path.join(BACKEND_ROOT, "eval", "images", "group", "group_sample.jpg")),
    ("product", os.path.join(BACKEND_ROOT, "eval", "images", "product_busy", "product_busy_sample.jpg")),
    ("animal", os.path.join(BACKEND_ROOT, "eval", "images", "pet_fur", "pet_fur_sample.jpg")),
    ("poster", os.path.join(BACKEND_ROOT, "eval", "images", "poster_graphic_text", "rang_morla.jpg")),
]

OUTPUT_DIR = os.path.join(BACKEND_ROOT, "eval", "results", "refiner_verification")


async def verify_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transport = httpx.ASGITransport(app=app)
    results_summary = []

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000", timeout=120.0) as client:
        print("=== 1. VERIFYING 5 REAL IMAGES (format=png and format=base64) ===")
        for name, img_path in IMAGES:
            print(f"\nProcessing {name}: {img_path}")
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            # 1. Format = png (binary)
            t0 = time.perf_counter()
            resp_png = await client.post(
                "/api/v1/remove-background?format=png&quality=balanced",
                files={"image": (os.path.basename(img_path), img_bytes, "image/jpeg")},
            )
            png_elapsed = (time.perf_counter() - t0) * 1000.0
            assert resp_png.status_code == 200, f"Failed PNG endpoint: {resp_png.text}"
            assert resp_png.headers["content-type"] == "image/png"
            cutout_png_path = os.path.join(OUTPUT_DIR, f"{name}_cutout.png")
            with open(cutout_png_path, "wb") as f:
                f.write(resp_png.content)

            # Generate contact sheet
            contact_path = os.path.join(OUTPUT_DIR, f"{name}_contact_sheet.png")
            create_contact_sheet(cutout_png_path, contact_path)

            # 2. Format = base64 (JSON)
            t0 = time.perf_counter()
            resp_b64 = await client.post(
                "/api/v1/remove-background?format=base64&quality=balanced",
                files={"image": (os.path.basename(img_path), img_bytes, "image/jpeg")},
            )
            b64_elapsed = (time.perf_counter() - t0) * 1000.0
            assert resp_b64.status_code == 200, f"Failed Base64 endpoint: {resp_b64.text}"
            b64_data = resp_b64.json()
            assert "image_base64" in b64_data
            assert b64_data["format"] == "PNG"

            # Check edge qualities
            rgba_check = Image.open(cutout_png_path).convert("RGBA")
            alpha_check = np.array(rgba_check.split()[-1], dtype=np.uint8)
            border_touch = (alpha_check[0, :] > 0).any() or (alpha_check[-1, :] > 0).any()

            results_summary.append({
                "name": name,
                "png_ms": round(png_elapsed, 1),
                "b64_ms": round(b64_elapsed, 1),
                "dimensions": f"{b64_data['width']}x{b64_data['height']}",
                "contact_sheet": contact_path,
                "border_touch": border_touch,
            })
            print(f"  -> PNG ({png_elapsed:.1f}ms), Base64 ({b64_elapsed:.1f}ms). Contact sheet: {contact_path}")

        print("\n=== 2. TIMING 12MP / 4K IMAGE (p50 of 3 runs) ===")
        large_path = os.path.join(BACKEND_ROOT, "eval", "images", "large_4k_photo", "large_4k_photo_sample.jpg")
        with open(large_path, "rb") as f:
            large_bytes = f.read()

        times = []
        for i in range(3):
            t0 = time.perf_counter()
            r = await client.post(
                "/api/v1/remove-background?format=png&quality=balanced",
                files={"image": ("large.jpg", large_bytes, "image/jpeg")},
            )
            elapsed = time.perf_counter() - t0
            assert r.status_code == 200
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")
        times.sort()
        p50 = times[1]
        print(f"  p50: {p50:.3f}s (Runs: {[round(t, 3) for t in times]})")

        print("\n=== 3. CONCURRENT NON-BLOCKING TEST ===")
        t0 = time.perf_counter()
        req1 = client.post(
            "/api/v1/remove-background?format=png&quality=balanced",
            files={"image": ("c1.jpg", img_bytes, "image/jpeg")},
        )
        req2 = client.post(
            "/api/v1/remove-background?format=base64&quality=balanced",
            files={"image": ("c2.jpg", img_bytes, "image/jpeg")},
        )
        r1, r2 = await asyncio.gather(req1, req2)
        total_parallel = time.perf_counter() - t0
        assert r1.status_code == 200 and r2.status_code == 200
        print(f"  Two parallel requests completed concurrently in {total_parallel:.3f}s")

    return results_summary, p50, total_parallel


if __name__ == "__main__":
    import numpy as np
    asyncio.run(verify_images())
