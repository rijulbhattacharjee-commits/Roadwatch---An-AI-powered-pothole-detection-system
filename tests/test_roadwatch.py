"""
Unit and functional tests for the RoadWatch pipeline:
- Severity Engine (`src/severity.py`)
- Preprocessing & Validation (`src/preprocess.py`)
- Grad-CAM & Inference (`src/gradcam.py`, `src/infer.py`)
- Report Generation (`src/report.py`)
"""

import io
import time

import numpy as np
import pytest
from PIL import Image

from src.gradcam import generate_gradcam_heatmap
from src.infer import analyse_image, get_model_service
from src.preprocess import (
    extract_gps_exif,
    load_and_preprocess_image,
    validate_file,
)
from src.report import (
    detections_to_dataframe,
    generate_csv_report,
    generate_pdf_report,
)
from src.severity import (
    classify_ratio,
    compute_box_area,
    compute_priority_score,
    evaluate_severity,
)


# ---------------------------------------------------------------------------
# 1. Severity Engine Tests
# ---------------------------------------------------------------------------

def test_classify_ratio_thresholds():
    assert classify_ratio(0.0) == "None"
    assert classify_ratio(0.005) == "Low"
    assert classify_ratio(0.0199) == "Low"
    assert classify_ratio(0.02) == "Medium"
    assert classify_ratio(0.05) == "Medium"
    assert classify_ratio(0.08) == "High"
    assert classify_ratio(0.25) == "High"


def test_compute_box_area_clipping():
    # Box completely inside 100x100 image -> 20 * 30 = 600
    assert compute_box_area([10, 10, 30, 40], 100, 100) == 600.0
    # Box extending outside image bounds gets clipped to [0, 100]
    assert compute_box_area([-10, -10, 50, 50], 100, 100) == 2500.0


def test_evaluate_severity_empty_and_populated():
    empty_summary = evaluate_severity([], [], 1000, 1000)
    assert empty_summary.pothole_count == 0
    assert empty_summary.total_area_ratio == 0.0
    assert empty_summary.overall_severity == "None"
    assert empty_summary.priority_score == 0.0

    # Two boxes on a 1000x1000 (1,000,000 px) image:
    # Box 1: 200x150 = 30,000 px (3.0%)
    # Box 2: 100x100 = 10,000 px (1.0%)
    # Total = 40,000 px (4.0% -> Medium)
    summary = evaluate_severity(
        boxes=[(100, 100, 300, 250), (500, 500, 600, 600)],
        confidences=[0.85, 0.65],
        image_width=1000,
        image_height=1000,
    )
    assert summary.pothole_count == 2
    assert pytest.approx(summary.total_area_ratio, rel=1e-4) == 0.04
    assert summary.overall_severity == "Medium"
    assert summary.priority_score > 0.0
    assert len(summary.detections) == 2


def test_evaluate_severity_invalid_inputs():
    with pytest.raises(ValueError):
        evaluate_severity([], [], 0, 500)
    with pytest.raises(ValueError):
        evaluate_severity([(0, 0, 10, 10)], [0.5, 0.6], 500, 500)


# ---------------------------------------------------------------------------
# 2. Preprocessing & File Validation Tests
# ---------------------------------------------------------------------------

def test_validate_file():
    ok, _ = validate_file("road_photo.jpg", file_size_bytes=1024 * 500)
    assert ok is True

    ok_ext, _ = validate_file("malware.exe", file_size_bytes=1024)
    assert ok_ext is False

    ok_empty, _ = validate_file("empty.png", file_size_bytes=0)
    assert ok_empty is False

    ok_huge, _ = validate_file("huge.jpg", file_size_bytes=100 * 1024 * 1024, max_size_mb=50.0)
    assert ok_huge is False


def test_load_and_preprocess_image_resizing():
    raw_arr = np.zeros((1600, 2400, 3), dtype=np.uint8)
    pil_img = Image.fromarray(raw_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")

    rgb, _ = load_and_preprocess_image(buf.getvalue(), max_dim=1200)
    assert rgb.shape == (800, 1200, 3)


def test_extract_gps_exif_none_when_absent():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    assert extract_gps_exif(pil_img) is None


# ---------------------------------------------------------------------------
# 3. Inference, Grad-CAM & Performance Tests
# ---------------------------------------------------------------------------

def test_gradcam_and_inference_performance():
    # Create synthetic 600x800 road image with a dark depression
    img = np.full((600, 800, 3), 120, dtype=np.uint8)
    img[320:420, 300:460] = 20  # Dark synthetic pothole

    t0 = time.perf_counter()
    annotated, count, ratio, severity, confs = analyse_image(img, conf=0.30)
    elapsed_sec = time.perf_counter() - t0

    assert annotated.shape == img.shape
    assert count >= 1
    assert ratio > 0.0
    assert severity in ("Low", "Medium", "High")
    assert len(confs) == count
    # Target metric from Section 2.4: <= 3 seconds per image on CPU
    assert elapsed_sec <= 3.0

    overlay, heatmap, prob = generate_gradcam_heatmap(img)
    assert overlay.shape == img.shape
    assert heatmap.shape == img.shape
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# 4. Report Generation Tests
# ---------------------------------------------------------------------------

def test_csv_and_pdf_report_generation():
    summary = evaluate_severity(
        boxes=[(50, 50, 200, 180)],
        confidences=[0.88],
        image_width=800,
        image_height=600,
    )
    df = detections_to_dataframe(summary)
    assert len(df) == 1

    csv_text = generate_csv_report(
        summary,
        source_name="test_road.jpg",
        gps_coords={"latitude": 12.9716, "longitude": 77.5946},
    )
    assert "test_road.jpg" in csv_text
    assert "12.9716" in csv_text

    try:
        pdf_bytes = generate_pdf_report(
            summary,
            source_name="test_road.jpg",
            classifier_prob=0.91,
            inference_time_ms=42.5,
            gps_coords={"latitude": 12.9716, "longitude": 77.5946},
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
    except ImportError:
        pytest.skip("fpdf2 not installed in current test environment")
