"""Real-time quality assessment for camera capture in enrollment."""

import io
from typing import Any

import numpy as np
from PIL import Image


def assess_image_quality(image_data: bytes) -> dict[str, Any]:
    """
    Assess quality of a raw image (JPEG/PNG).
    
    Returns:
        {
            "quality_score": 0.0-1.0,
            "sharpness": float,
            "brightness": float,
            "contrast": float,
            "passed": bool,
            "feedback": [
                {"metric": "sharpness", "value": 78.5, "status": "good", "threshold": 55}
            ]
        }
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
        arr = np.array(img, dtype=np.float32)
        
        # 1. Sharpness (Laplacian variance)
        from scipy.ndimage import laplace
        laplacian = laplace(arr)
        sharpness = float(np.var(laplacian))
        sharpness_norm = min(100.0, sharpness / 10.0)  # Normalize to ~0-100
        
        # 2. Brightness (mean pixel value)
        brightness = float(np.mean(arr))
        
        # 3. Contrast (standard deviation)
        contrast = float(np.std(arr))
        
        # Thresholds
        SHARPNESS_THRESHOLD = 55.0
        BRIGHTNESS_THRESHOLD = (45.0, 220.0)
        CONTRAST_THRESHOLD = 15.0
        
        # Evaluate
        feedback = []
        passed = True
        
        # Sharpness check
        sharpness_pass = sharpness_norm >= SHARPNESS_THRESHOLD
        feedback.append({
            "metric": "sharpness",
            "value": round(sharpness_norm, 1),
            "threshold": SHARPNESS_THRESHOLD,
            "status": "good" if sharpness_pass else "poor",
            "message": "Image is sharp" if sharpness_pass else "Image is blurry"
        })
        passed = passed and sharpness_pass
        
        # Brightness check
        brightness_pass = (BRIGHTNESS_THRESHOLD[0] <= brightness <= BRIGHTNESS_THRESHOLD[1])
        brightness_status = "good" if brightness_pass else ("dark" if brightness < BRIGHTNESS_THRESHOLD[0] else "overexposed")
        feedback.append({
            "metric": "brightness",
            "value": round(brightness, 1),
            "threshold": f"{BRIGHTNESS_THRESHOLD[0]}-{BRIGHTNESS_THRESHOLD[1]}",
            "status": brightness_status,
            "message": "Lighting is optimal" if brightness_pass else f"Lighting is {brightness_status}"
        })
        passed = passed and brightness_pass
        
        # Contrast check (helps detect faces)
        contrast_pass = contrast >= CONTRAST_THRESHOLD
        feedback.append({
            "metric": "contrast",
            "value": round(contrast, 1),
            "threshold": CONTRAST_THRESHOLD,
            "status": "good" if contrast_pass else "low",
            "message": "Good contrast" if contrast_pass else "Low contrast - may affect face detection"
        })
        passed = passed and contrast_pass
        
        # Overall quality score (0-1)
        quality_score = (
            (min(1.0, sharpness_norm / SHARPNESS_THRESHOLD) * 0.4) +
            (1.0 if brightness_pass else 0.0) * 0.3 +
            (min(1.0, contrast / (CONTRAST_THRESHOLD * 3)) * 0.3)
        )
        
        return {
            "quality_score": round(quality_score, 2),
            "sharpness": round(sharpness_norm, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "passed": passed,
            "feedback": feedback,
        }
    except Exception as e:
        return {
            "quality_score": 0.0,
            "passed": False,
            "error": str(e),
            "feedback": []
        }


def quick_face_detection(image_data: bytes, recognizer) -> dict[str, Any]:
    """
    Quick face detection on frame (for camera preview).
    
    Args:
        image_data: Raw JPEG/PNG bytes
        recognizer: ArcFaceRecognizer instance
    
    Returns:
        {
            "faces_detected": 1,
            "largest_face": {
                "bbox": (x, y, w, h),
                "detector_score": 0.95,
                "face_ratio": 0.18,  # % of image
                "center": (cx, cy)
            }
        }
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_data))
        frame_array = np.array(img)
        
        # Detect faces using recognizer
        det_model = recognizer.det_model
        bboxes = det_model.detect(frame_array)
        
        if bboxes is None or len(bboxes) == 0:
            return {
                "faces_detected": 0,
                "message": "No face detected",
            }
        
        # Get largest face
        best_face = max(bboxes, key=lambda b: b[2] * b[3])  # Max by area
        bbox = best_face[:4]  # [x1, y1, x2, y2]
        score = float(best_face[4])
        
        # Calculate metrics
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1
        face_area = w * h
        frame_area = frame_array.shape[0] * frame_array.shape[1]
        face_ratio = (face_area / frame_area) if frame_area > 0 else 0.0
        
        # Center of face
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        return {
            "faces_detected": len(bboxes),
            "largest_face": {
                "bbox": (x1, y1, w, h),  # (x, y, width, height)
                "detector_score": round(float(score), 3),
                "face_ratio": round(face_ratio, 3),
                "center": (int(cx), int(cy)),
            }
        }
    except Exception as e:
        return {
            "faces_detected": 0,
            "error": str(e),
        }


def create_quality_router(settings: "Settings", store: "VectorStore", recognizer: "ArcFaceRecognizer") -> Any:
    """Create FastAPI router for quality assessment endpoints."""
    try:
        from fastapi import APIRouter, File, UploadFile
    except ImportError as exc:
        raise RuntimeError("Quality API needs FastAPI: pip install fastapi") from exc

    router = APIRouter(prefix="/api/v1", tags=["quality"])

    @router.post("/quality/assess")
    async def assess_frame_quality(file: UploadFile = File(...)) -> dict[str, Any]:
        """
        Assess quality of uploaded frame (from camera capture).
        
        Returns quality metrics for real-time feedback in enrollment UI.
        """
        image_data = await file.read()
        result = assess_image_quality(image_data)
        return result

    @router.post("/detection/quick")
    async def quick_detect_faces(file: UploadFile = File(...)) -> dict[str, Any]:
        """
        Quick face detection for camera preview (lightweight).
        
        Returns face bounding boxes and metrics for visualization.
        """
        image_data = await file.read()
        result = quick_face_detection(image_data, recognizer)
        return result

    return router
