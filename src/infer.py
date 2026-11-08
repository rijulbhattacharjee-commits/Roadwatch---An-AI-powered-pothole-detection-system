"""
Model Loading and Inference Pipeline for RoadWatch.

Supports:
- Stage A: MobileNetV2 binary road classifier + Grad-CAM explainability
- Stage B: Ultralytics YOLOv8 object detector (`models/best.pt` or `yolov8n.pt`)
- Fallback morphological contour detector when custom YOLO weights are not yet trained,
  ensuring the application and tests work out-of-the-box offline.
- Stage C: Integration with `src.severity` for area ratio, severity, and priority scoring.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.gradcam import generate_gradcam_heatmap
from src.severity import LOW_THRESHOLD, HIGH_THRESHOLD, SeveritySummary, evaluate_severity


DEFAULT_YOLO_WEIGHTS = Path("models/best.pt")
DEFAULT_CLASSIFIER_WEIGHTS = Path("models/classifier_mobilenetv2.pt")


@dataclass
class InferenceResult:
    """Complete output from analyzing a single road image."""
    annotated_rgb: np.ndarray
    gradcam_overlay_rgb: np.ndarray
    gradcam_heatmap_rgb: np.ndarray
    classifier_pothole_prob: float
    classifier_label: str
    severity_summary: SeveritySummary
    inference_time_ms: float
    backend_used: str


def draw_detections(
    rgb_image: np.ndarray,
    summary: SeveritySummary,
) -> np.ndarray:
    """
    Draw color-coded bounding boxes, confidence scores, and severity badges on an RGB image.
    """
    canvas = rgb_image.copy()
    color_palette = {
        "Low": (46, 204, 113),      # Green
        "Medium": (243, 156, 18),   # Orange/Amber
        "High": (231, 76, 60),      # Red
        "None": (52, 152, 219),     # Blue
    }

    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(h, w) / 300)))
    font_scale = max(0.5, min(h, w) / 900.0)

    for det in summary.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = color_palette.get(det.severity, (231, 76, 60))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"#{det.id} Pothole {det.confidence:.2f} [{det.severity}]"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top_y = max(y1, th + baseline + 6)
        cv2.rectangle(
            canvas,
            (x1, top_y - th - baseline - 6),
            (x1 + tw + 8, top_y),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, top_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return canvas


def _heuristic_contour_detect(
    rgb_image: np.ndarray,
    conf_threshold: float = 0.35,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    OpenCV morphological dark-depression & rim-edge detector used as an offline
    fallback when `models/best.pt` has not yet been trained.
    """
    h, w = rgb_image.shape[:2]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    ksize = max(15, int(min(h, w) * 0.08) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    # Ignore top 20% sky/horizon region
    horizon_y = int(h * 0.20)
    blackhat[:horizon_y, :] = 0

    # Adaptive threshold based on strong dark anomalies
    mean_val, std_val = cv2.meanStdDev(blackhat)
    thresh_val = max(28.0, float(mean_val[0, 0] + 2.2 * std_val[0, 0]))
    _, binary = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(w * h)

    boxes: List[Tuple[float, float, float, float]] = []
    confs: List[float] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        # Filter noise (< 0.15% of image) and huge road shadows (> 45% of image)
        if box_area < img_area * 0.0015 or box_area > img_area * 0.45:
            continue
        aspect = bw / float(max(1, bh))
        if aspect < 0.2 or aspect > 6.0:
            continue

        roi = blackhat[y : y + bh, x : x + bw]
        intensity_score = min(1.0, float(np.mean(roi)) / 90.0)
        fill_ratio = float(cv2.contourArea(cnt)) / box_area
        conf = round(float(np.clip(0.30 + 0.45 * intensity_score + 0.20 * fill_ratio, 0.1, 0.96)), 3)

        if conf >= conf_threshold:
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
            confs.append(conf)

    # Sort highest confidence first
    if boxes:
        paired = sorted(zip(confs, boxes), key=lambda pair: pair[0], reverse=True)[:15]
        confs = [p[0] for p in paired]
        boxes = [p[1] for p in paired]

    return boxes, confs


class RoadWatchModelService:
    """
    Manages loading and running the Stage A Classifier and Stage B YOLO Detector.
    """

    def __init__(
        self,
        yolo_weights_path: Optional[ str | Path ] = None,
        classifier_weights_path: Optional[ str | Path ] = None,
    ):
        self.yolo_weights_path = Path(yolo_weights_path) if yolo_weights_path else DEFAULT_YOLO_WEIGHTS
        self.classifier_weights_path = (
            Path(classifier_weights_path) if classifier_weights_path else DEFAULT_CLASSIFIER_WEIGHTS
        )
        self.yolo_model = None
        self.classifier_model = None
        self.backend_name = "OpenCV Morphological Baseline (Offline Fallback)"
        self._load_models()

    def _load_models(self) -> None:
        """Attempt to load trained YOLOv8 and MobileNetV2 weights if present."""
        if self.yolo_weights_path.exists():
            try:
                from ultralytics import YOLO

                self.yolo_model = YOLO(str(self.yolo_weights_path))
                self.backend_name = f"Ultralytics YOLOv8 ({self.yolo_weights_path.name})"
            except Exception:
                self.yolo_model = None

        if self.classifier_weights_path.exists():
            try:
                import torch
                import torchvision.models as models

                model = models.mobilenet_v2(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(in_features, 2)
                state = torch.load(str(self.classifier_weights_path), map_location="cpu")
                model.load_state_dict(state)
                model.eval()
                self.classifier_model = model
            except Exception:
                self.classifier_model = None

    def predict(
        self,
        rgb_image: np.ndarray,
        conf_threshold: float = 0.35,
        low_thresh: float = LOW_THRESHOLD,
        high_thresh: float = HIGH_THRESHOLD,
    ) -> InferenceResult:
        """
        Run full two-stage RoadWatch inference on an RGB image.
        """
        t0 = time.perf_counter()
        h, w = rgb_image.shape[:2]

        # Stage B: Object Detection
        if self.yolo_model is not None:
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            result = self.yolo_model.predict(bgr, conf=conf_threshold, verbose=False)[0]
            raw_boxes = result.boxes.xyxy.cpu().numpy()
            raw_confs = result.boxes.conf.cpu().numpy()
            boxes = [tuple(map(float, b)) for b in raw_boxes]
            confs = [float(c) for c in raw_confs]
        else:
            boxes, confs = _heuristic_contour_detect(rgb_image, conf_threshold=conf_threshold)

        # Stage C: Severity Engine
        summary = evaluate_severity(
            boxes=boxes,
            confidences=confs,
            image_width=w,
            image_height=h,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
        )

        annotated_rgb = draw_detections(rgb_image, summary)

        # Stage A: Classifier + Grad-CAM Explainability
        gradcam_overlay, gradcam_heatmap, cls_prob = generate_gradcam_heatmap(
            rgb_image, model=self.classifier_model
        )
        # Align heuristic probability when using fallback and detections are found
        if self.classifier_model is None and summary.pothole_count > 0:
            cls_prob = round(max(cls_prob, summary.max_confidence), 4)
        cls_label = "Pothole Detected" if cls_prob >= 0.5 or summary.pothole_count > 0 else "Normal Road"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return InferenceResult(
            annotated_rgb=annotated_rgb,
            gradcam_overlay_rgb=gradcam_overlay,
            gradcam_heatmap_rgb=gradcam_heatmap,
            classifier_pothole_prob=cls_prob,
            classifier_label=cls_label,
            severity_summary=summary,
            inference_time_ms=elapsed_ms,
            backend_used=self.backend_name,
        )


_DEFAULT_SERVICE: Optional[RoadWatchModelService] = None


def get_model_service() -> RoadWatchModelService:
    """Singleton accessor for RoadWatchModelService."""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = RoadWatchModelService()
    return _DEFAULT_SERVICE


def analyse_image(
    rgb_image: np.ndarray,
    conf: float = 0.35,
    low_thresh: float = LOW_THRESHOLD,
    high_thresh: float = HIGH_THRESHOLD,
) -> Tuple[np.ndarray, int, float, str, List[float]]:
    """
    Convenience wrapper matching Section 10.3 of the Project Plan:
        `annotated, count, ratio, severity, confs = analyse_image(np.array(img), conf)`
    """
    service = get_model_service()
    res = service.predict(
        rgb_image,
        conf_threshold=conf,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    confs = [d.confidence for d in res.severity_summary.detections]
    return (
        res.annotated_rgb,
        res.severity_summary.pothole_count,
        res.severity_summary.total_area_ratio,
        res.severity_summary.overall_severity,
        confs,
    )
