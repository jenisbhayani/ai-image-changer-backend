# Model Weights Catalog

This directory stores neural network weights for commercial background removal and cutout refinement.

## Model Inventory & Commercial License Audit

| Model File | Architecture | Parameters | License | Commercial Monetization Allowed? | Primary Use Case |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `isnet-general-use.onnx` | DIS (Dichotomous Image Segmentation) | ~44M | **MIT License** | **YES (Fully Approved)** | Primary engine for `balanced` & `fast` tiers and TTA `high` tier. Outstanding boundary detail on portraits, products, and graphics. |
| `rmbg-1.4.onnx` | BRIA RMBG-1.4 | ~44M | **bria-rmbg-1.4 (Non-Commercial)** | **NO (Requires BRIA Paid License)** | Disabled by default. Preserved exclusively behind the explicit `legacy` flag for comparative regression testing. |
| `realesr-general-x4v3.onnx` | Real-ESRGAN | ~17M | **BSD-3-Clause** | **YES** | Optional upscale service. |

## Upstream Sources & Checksums
- `isnet-general-use.onnx`:
  - Upstream Source: `https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx`
  - MD5: `fc16ebd8b0c10d971d3513d564d01e29`
  - File Size: ~178,648,008 bytes (~170.4 MB)
- `rmbg-1.4.onnx`:
  - Retained for legacy regression verification only.
