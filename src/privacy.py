"""
Privacy Redaction Module for RoadWatch (Section 13: Ethics & Professional Conduct).

Automatically detects and blurs human faces (via OpenCV Haar Cascades) and
potential vehicle license plates (high-contrast rectangular character regions)
before images are stored or displayed in public reports.
"""

from typing import Tuple

import cv2
import numpy as np


def _blur_region(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Apply strong Gaussian blur to a rectangular region in-place."""
    img_h, img_w = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    kx = max(15, (w // 3) | 1)
    ky = max(15, (h // 3) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx, ky), 0)


def redact_privacy_regions(
    rgb_image: np.ndarray,
    blur_faces: bool = True,
    blur_plates: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Detect and blur human faces and vehicle license plates in an RGB road image.

    Args:
        rgb_image: Input RGB image (H, W, 3).
        blur_faces: Whether to detect and blur faces.
        blur_plates: Whether to detect and blur license-plate-like regions.

    Returns:
        (redacted_rgb, faces_blurred_count, plates_blurred_count)
    """
    output = rgb_image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    faces_count = 0
    plates_count = 0

    if blur_faces:
        try:
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
                )
                for (fx, fy, fw, fh) in faces:
                    _blur_region(output, int(fx), int(fy), int(fw), int(fh))
                    faces_count += 1
        except Exception:
            pass

    if blur_plates:
        try:
            # First try Russian/EU plate cascade if bundled with OpenCV
            plate_cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
            plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
            if not plate_cascade.empty():
                plates = plate_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 12)
                )
                for (px, py, pw, ph) in plates:
                    _blur_region(output, int(px), int(py), int(pw), int(ph))
                    plates_count += 1

            # Supplement with high-contrast morphological vertical-edge detector for Indian plates
            # (rectangular aspect ratio 2.0 to 5.5 with dense vertical character strokes)
            sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = float(w * h)
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / float(max(1, ch))
                area = float(cw * ch)
                # Plates typically appear in upper/middle bands and have aspect 2.4 - 5.2
                if 2.5 <= aspect <= 5.2 and (img_area * 0.0008) <= area <= (img_area * 0.035):
                    roi_gray = gray[cy : cy + ch, cx : cx + cw]
                    # High internal contrast (dark text on bright yellow/white plate)
                    if float(np.mean(roi_gray)) > 135.0 and float(np.std(roi_gray)) > 38.0:
                        _blur_region(output, cx, cy, cw, ch)
                        plates_count += 1
        except Exception:
            pass

    return output, faces_count, plates_count
