"""
Passport photo analysis service using OpenCV YuNet (MIT license).
Calculates precise biometric cropping according to official passport guidelines
and evaluates compliance checks (face count, tilt, head size, exposure, blur, uniformity).
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "face_detection_yunet_2023mar.onnx")


class PassportAnalysisService:
    def __init__(self, model_path: str = YUNET_MODEL_PATH):
        self._model_path = model_path
        self._detector: Optional[cv2.FaceDetectorYN] = None

    def _get_detector(self, width: int, height: int) -> Optional[cv2.FaceDetectorYN]:
        if not os.path.exists(self._model_path) or os.path.getsize(self._model_path) < 100_000:
            from app.services.model_registry import ensure_yunet_weights
            ensure_yunet_weights()
        if not os.path.exists(self._model_path):
            return None
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                model=self._model_path,
                config="",
                input_size=(width, height),
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=5000,
            )
        else:
            self._detector.setInputSize((width, height))
        return self._detector

    def analyze_passport_photo(
        self,
        image_bytes: bytes,
        target_ratio: float = 35.0 / 45.0,  # width / height (e.g. 0.777 for 35x45mm, 1.0 for US 2x2)
        head_min_pct: float = 0.68,  # head height ~68% - 80% of image height
        head_max_pct: float = 0.80,
        eye_y_min_pct: float = 0.56,  # eye line from bottom of photo
        eye_y_max_pct: float = 0.69,
    ) -> Dict[str, Any]:
        """
        Analyzes image for biometric passport standards and computes suggested crop coordinates.
        """
        # 1. Decode with EXIF orientation correction
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        w, h = pil_img.size
        rgb_arr = np.array(pil_img.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # Cap working size for fast face detection
        max_dim = 1600
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            work_w, work_h = int(w * scale), int(h * scale)
            work_bgr = cv2.resize(bgr_arr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work_w, work_h = w, h
            work_bgr = bgr_arr

        detector = self._get_detector(work_w, work_h)
        faces_detected = []
        if detector is not None:
            ret, faces = detector.detect(work_bgr)
            if ret and faces is not None:
                # Scale coordinates back to original image
                inv_scale = 1.0 / scale
                for f in faces:
                    box = [float(f[0] * inv_scale), float(f[1] * inv_scale), float(f[2] * inv_scale), float(f[3] * inv_scale)]
                    right_eye = [float(f[4] * inv_scale), float(f[5] * inv_scale)]
                    left_eye = [float(f[6] * inv_scale), float(f[7] * inv_scale)]
                    nose = [float(f[8] * inv_scale), float(f[9] * inv_scale)]
                    mouth_r = [float(f[10] * inv_scale), float(f[11] * inv_scale)]
                    mouth_l = [float(f[12] * inv_scale), float(f[13] * inv_scale)]
                    score = float(f[14])
                    faces_detected.append({
                        "box": box,
                        "right_eye": right_eye,
                        "left_eye": left_eye,
                        "nose": nose,
                        "mouth_r": mouth_r,
                        "mouth_l": mouth_l,
                        "score": score,
                    })

        # Sort faces by area descending (largest first)
        faces_detected.sort(key=lambda item: item["box"][2] * item["box"][3], reverse=True)
        faces_count = len(faces_detected)

        checks: List[Dict[str, Any]] = []

        # Check 1: Face detected
        if faces_count == 0:
            checks.append({
                "id": "face_found",
                "status": "fail",
                "message": "No face detected in photo. Please ensure clear frontal lighting and face the camera directly.",
            })
            return {
                "faces_count": 0,
                "face_box": None,
                "suggested_crop": self._fallback_crop(w, h, target_ratio),
                "checks": checks,
                "roll_deg": 0.0,
            }

        # Check 2: Single face check
        if faces_count > 1:
            checks.append({
                "id": "single_face",
                "status": "warn",
                "message": f"Multiple faces detected ({faces_count}). Only the primary subject should be in the photo.",
            })
        else:
            checks.append({
                "id": "single_face",
                "status": "pass",
                "message": "Single subject detected.",
            })

        primary_face = faces_detected[0]
        fb_x, fb_y, fb_w, fb_h = primary_face["box"]
        re_x, re_y = primary_face["right_eye"]
        le_x, le_y = primary_face["left_eye"]

        # Calculate Head Roll / Tilt Angle
        dx = le_x - re_x
        dy = le_y - re_y
        roll_deg = float(math.degrees(math.atan2(dy, dx))) if dx != 0 else 0.0

        if abs(roll_deg) > 5.0:
            checks.append({
                "id": "head_tilt",
                "status": "warn" if abs(roll_deg) < 12.0 else "fail",
                "message": f"Head is tilted ({abs(roll_deg):.1f}°). Passport authorities require a level head position.",
            })
        else:
            checks.append({
                "id": "head_tilt",
                "status": "pass",
                "message": "Head angle is level.",
            })

        # Estimate Chin and Head-Top:
        # Based on biometric facial anthropometry:
        # Chin is at ~ fb_y + fb_h
        # Head top (crown) is at ~ fb_y - 0.25 * fb_h (accounting for hair volume)
        chin_y = fb_y + fb_h
        head_top_y = max(0.0, fb_y - 0.25 * fb_h)
        eye_y = (re_y + le_y) * 0.5
        head_height = max(chin_y - head_top_y, fb_h * 1.25)

        # -------------------------------------------------------------
        # Biometric Crop Calculation
        # Target crop height H_crop such that head_height = ~75% of H_crop
        # Target eye line at ~60% from the bottom of H_crop (i.e. 40% from the top)
        # -------------------------------------------------------------
        target_head_pct = (head_min_pct + head_max_pct) * 0.5
        crop_h = head_height / target_head_pct
        crop_w = crop_h * target_ratio

        # Desired eye position from top: ~38% of crop height
        eye_target_offset_from_top = crop_h * 0.38
        crop_y = eye_y - eye_target_offset_from_top
        crop_x = (re_x + le_x) * 0.5 - (crop_w * 0.5)

        # Clamp and scale crop box inside image boundaries if possible
        if crop_w > w or crop_h > h:
            scale_down = min(w / crop_w, h / crop_h)
            crop_w *= scale_down
            crop_h *= scale_down

        crop_x = max(0.0, min(crop_x, w - crop_w))
        crop_y = max(0.0, min(crop_y, h - crop_h))

        suggested_crop = {
            "x": int(round(crop_x)),
            "y": int(round(crop_y)),
            "width": int(round(crop_w)),
            "height": int(round(crop_h)),
        }

        # Check 3: Head size within official range
        actual_head_pct = head_height / max(crop_h, 1.0)
        if actual_head_pct < head_min_pct - 0.05:
            checks.append({
                "id": "head_size",
                "status": "warn",
                "message": f"Subject appears too far from the camera ({int(actual_head_pct*100)}% of frame).",
            })
        elif actual_head_pct > head_max_pct + 0.05:
            checks.append({
                "id": "head_size",
                "status": "warn",
                "message": f"Subject is cropped too close ({int(actual_head_pct*100)}% of frame).",
            })
        else:
            checks.append({
                "id": "head_size",
                "status": "pass",
                "message": f"Head occupies {int(actual_head_pct*100)}% of photo height (meets 70-80% spec).",
            })

        # Check 4: Sharpness / Blur check (Laplacian variance on face ROI)
        x_int, y_int, w_int, h_int = int(max(0, fb_x)), int(max(0, fb_y)), int(min(w - fb_x, fb_w)), int(min(h - fb_y, fb_h))
        face_roi = rgb_arr[y_int : y_int + h_int, x_int : x_int + w_int]
        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
            if laplacian_var < 50.0:
                checks.append({
                    "id": "sharpness",
                    "status": "fail" if laplacian_var < 25.0 else "warn",
                    "message": "Photo is blurry. A crisp, sharp focus is required for biometric validation.",
                })
            else:
                checks.append({
                    "id": "sharpness",
                    "status": "pass",
                    "message": "Focus is sharp and clear.",
                })

            # Check 5: Brightness & Exposure
            mean_brightness = float(np.mean(gray_roi))
            if mean_brightness < 60.0:
                checks.append({
                    "id": "lighting",
                    "status": "fail",
                    "message": "Photo is underexposed or too dark. Retake with front lighting.",
                })
            elif mean_brightness > 220.0:
                checks.append({
                    "id": "lighting",
                    "status": "warn",
                    "message": "Face has highlights/glare that may wash out facial features.",
                })
            else:
                checks.append({
                    "id": "lighting",
                    "status": "pass",
                    "message": "Facial exposure and brightness are well balanced.",
                })

        # Check 6: Resolution adequacy (minimum 600px height for passport)
        if crop_h < 500:
            checks.append({
                "id": "resolution",
                "status": "warn",
                "message": f"Photo resolution ({int(crop_w)}×{int(crop_h)} px) is low. 600×600 px or higher recommended.",
            })
        else:
            checks.append({
                "id": "resolution",
                "status": "pass",
                "message": f"Resolution ({int(crop_w)}×{int(crop_h)} px) exceeds minimum requirements.",
            })

        return {
            "faces_count": faces_count,
            "face_box": {
                "x": int(fb_x),
                "y": int(fb_y),
                "width": int(fb_w),
                "height": int(fb_h),
            },
            "head_top_y": int(head_top_y),
            "chin_y": int(chin_y),
            "eye_y": int(eye_y),
            "roll_deg": round(roll_deg, 1),
            "suggested_crop": suggested_crop,
            "checks": checks,
        }

    def _fallback_crop(self, w: int, h: int, target_ratio: float) -> Dict[str, int]:
        current_ratio = w / h
        if current_ratio > target_ratio:
            crop_w = int(h * target_ratio)
            crop_h = h
        else:
            crop_w = w
            crop_h = int(w / target_ratio)
        return {
            "x": (w - crop_w) // 2,
            "y": (h - crop_h) // 2,
            "width": crop_w,
            "height": crop_h,
        }


import io
passport_analysis_service = PassportAnalysisService()
