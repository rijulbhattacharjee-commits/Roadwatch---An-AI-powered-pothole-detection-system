"""
Severity Engine for RoadWatch.

Calculates pothole bounding-box area ratios relative to the image area,
assigns per-pothole and overall road-segment severity ratings (Low / Medium / High),
and computes a 0-100 repair priority score.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# Default thresholds (configurable and tunable on validation set)
LOW_THRESHOLD: float = 0.02     # < 2% of image area -> Low
HIGH_THRESHOLD: float = 0.08    # >= 8% of image area -> High (2% to 8% -> Medium)


@dataclass
class PotholeDetection:
    """Represents a single detected pothole and its computed severity metrics."""
    id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    area_px: float
    area_ratio: float
    severity: str


@dataclass
class SeveritySummary:
    """Aggregated severity assessment for an entire road image or video frame."""
    image_width: int
    image_height: int
    pothole_count: int
    total_area_px: float
    total_area_ratio: float
    overall_severity: str
    priority_score: float
    max_confidence: float
    mean_confidence: float
    detections: List[PotholeDetection] = field(default_factory=list)


def classify_ratio(
    area_ratio: float,
    low_thresh: float = LOW_THRESHOLD,
    high_thresh: float = HIGH_THRESHOLD,
) -> str:
    """
    Classify an area ratio into 'None', 'Low', 'Medium', or 'High'.

    Args:
        area_ratio: Ratio of pothole area to total image area (0.0 to 1.0).
        low_thresh: Upper bound for Low severity (default 0.02 = 2%).
        high_thresh: Lower bound for High severity (default 0.08 = 8%).

    Returns:
        Severity string: 'None', 'Low', 'Medium', or 'High'.
    """
    if area_ratio <= 0.0:
        return "None"
    if area_ratio < low_thresh:
        return "Low"
    if area_ratio < high_thresh:
        return "Medium"
    return "High"


def compute_box_area(bbox: Sequence[float], img_w: int, img_h: int) -> float:
    """
    Compute the clipped pixel area of an (x1, y1, x2, y2) bounding box.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 coordinates (x1, y1, x2, y2), got {len(bbox)}")
    x1, y1, x2, y2 = bbox
    x1_c = max(0.0, min(float(img_w), float(x1)))
    y1_c = max(0.0, min(float(img_h), float(y1)))
    x2_c = max(0.0, min(float(img_w), float(x2)))
    y2_c = max(0.0, min(float(img_h), float(y2)))

    width = max(0.0, x2_c - x1_c)
    height = max(0.0, y2_c - y1_c)
    return width * height


def compute_priority_score(
    count: int,
    total_ratio: float,
    max_confidence: float,
    high_thresh: float = HIGH_THRESHOLD,
) -> float:
    """
    Compute a normalized 0-100 repair priority score for a road segment.

    Formula weights:
    - 55% from area ratio (saturates around 1.5x high_thresh = 12% of image area)
    - 25% from pothole count (saturates at 5 potholes)
    - 20% from maximum detection confidence
    """
    if count <= 0 or total_ratio <= 0.0:
        return 0.0

    saturation_ratio = max(high_thresh * 1.5, 0.01)
    area_component = min(1.0, total_ratio / saturation_ratio) * 55.0
    count_component = min(1.0, count / 5.0) * 25.0
    conf_component = max(0.0, min(1.0, max_confidence)) * 20.0

    score = area_component + count_component + conf_component
    return round(min(100.0, max(0.0, score)), 1)


def evaluate_severity(
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    image_width: int,
    image_height: int,
    low_thresh: float = LOW_THRESHOLD,
    high_thresh: float = HIGH_THRESHOLD,
) -> SeveritySummary:
    """
    Evaluate per-pothole and overall road severity given detected bounding boxes.

    Args:
        boxes: Sequence of (x1, y1, x2, y2) bounding boxes in pixels.
        confidences: Sequence of detection confidence scores (0.0 to 1.0).
        image_width: Width of the image in pixels (> 0).
        image_height: Height of the image in pixels (> 0).
        low_thresh: Threshold between Low and Medium severity.
        high_thresh: Threshold between Medium and High severity.

    Returns:
        SeveritySummary containing individual detections and overall metrics.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got ({image_width}, {image_height})"
        )
    if len(boxes) != len(confidences):
        raise ValueError(
            f"Length mismatch: {len(boxes)} boxes vs {len(confidences)} confidences"
        )

    image_area = float(image_width * image_height)
    detections: List[PotholeDetection] = []

    for idx, (box, conf) in enumerate(zip(boxes, confidences), start=1):
        area_px = compute_box_area(box, image_width, image_height)
        ratio = area_px / image_area
        # Individual pothole severity uses scaled thresholds (single pothole >= 4% is High)
        indiv_severity = classify_ratio(
            ratio, low_thresh=low_thresh * 0.5, high_thresh=high_thresh * 0.5
        )
        detections.append(
            PotholeDetection(
                id=idx,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=round(float(conf), 4),
                area_px=round(area_px, 2),
                area_ratio=round(ratio, 6),
                severity=indiv_severity,
            )
        )

    total_area_px = sum(d.area_px for d in detections)
    # Cap total ratio at 1.0 in case of overlapping bounding boxes
    total_area_ratio = min(1.0, total_area_px / image_area)
    count = len(detections)

    if count == 0:
        overall_severity = "None"
        max_conf = 0.0
        mean_conf = 0.0
    else:
        overall_severity = classify_ratio(
            total_area_ratio, low_thresh=low_thresh, high_thresh=high_thresh
        )
        max_conf = max(d.confidence for d in detections)
        mean_conf = sum(d.confidence for d in detections) / count

    priority = compute_priority_score(
        count=count,
        total_ratio=total_area_ratio,
        max_confidence=max_conf,
        high_thresh=high_thresh,
    )

    return SeveritySummary(
        image_width=image_width,
        image_height=image_height,
        pothole_count=count,
        total_area_px=round(total_area_px, 2),
        total_area_ratio=round(total_area_ratio, 6),
        overall_severity=overall_severity,
        priority_score=priority,
        max_confidence=round(max_conf, 4),
        mean_confidence=round(mean_conf, 4),
        detections=detections,
    )
