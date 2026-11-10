"""
Unit and functional tests for the full-stack RoadWatch pipeline:
- Severity Engine (`src/severity.py`)
- Engineering Urgency Tiers & Repair Cost Estimation (`src/urgency.py`)
- Privacy Redaction (`src/privacy.py`)
- SQLite Municipal History Persistence (`src/database.py`)
- Preprocessing & Validation (`src/preprocess.py`)
- Grad-CAM & Inference (`src/gradcam.py`, `src/infer.py`)
- Report Generation (`src/report.py`)
- Evaluation & Error Analysis Data (`src/evaluation_data.py`)
"""

import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.database import (
    delete_all_inspections,
    init_db,
    inspections_to_dataframe,
    list_inspections,
    save_inspection,
    seed_demo_inspections_if_empty,
)
from src.evaluation_data import (
    get_confidence_threshold_sweep_df,
    get_error_analysis_taxonomy,
    get_model_comparison_df,
)
from src.gradcam import generate_gradcam_heatmap
from src.infer import analyse_image
from src.preprocess import (
    extract_gps_exif,
    load_and_preprocess_image,
    validate_file,
)
from src.privacy import redact_privacy_regions
from src.report import (
    detections_to_dataframe,
    generate_csv_report,
    generate_pdf_report,
)
from src.severity import (
    classify_ratio,
    compute_box_area,
    evaluate_severity,
)
from src.urgency import estimate_repair_area_and_cost, get_urgency_tier


# ---------------------------------------------------------------------------
# 1. Severity & Urgency Engine Tests
# ---------------------------------------------------------------------------

def test_classify_ratio_thresholds():
    assert classify_ratio(0.0) == "None"
    assert classify_ratio(0.005) == "Low"
    assert classify_ratio(0.0199) == "Low"
    assert classify_ratio(0.02) == "Medium"
    assert classify_ratio(0.05) == "Medium"
    assert classify_ratio(0.08) == "High"
    assert classify_ratio(0.25) == "High"


def test_urgency_and_cost_estimation():
    tier_none, _, _ = get_urgency_tier("None", 0.0)
    assert "Routine" in tier_none

    tier_high, _, _ = get_urgency_tier("High", 85.0)
    assert "48 hrs" in tier_high

    sqm, cost = estimate_repair_area_and_cost(0.05)
    assert sqm > 0.0
    assert cost > 0.0


def test_compute_box_area_clipping():
    assert compute_box_area([10, 10, 30, 40], 100, 100) == 600.0
    assert compute_box_area([-10, -10, 50, 50], 100, 100) == 2500.0


def test_evaluate_severity_empty_and_populated():
    empty_summary = evaluate_severity([], [], 1000, 1000)
    assert empty_summary.pothole_count == 0
    assert empty_summary.total_area_ratio == 0.0
    assert empty_summary.overall_severity == "None"
    assert empty_summary.priority_score == 0.0

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


# ---------------------------------------------------------------------------
# 2. Preprocessing, Privacy Redaction & SQLite Persistence Tests
# ---------------------------------------------------------------------------

def test_validate_file():
    ok, _ = validate_file("road_photo.jpg", file_size_bytes=1024 * 500)
    assert ok is True

    ok_ext, _ = validate_file("malware.exe", file_size_bytes=1024)
    assert ok_ext is False

    ok_empty, _ = validate_file("empty.png", file_size_bytes=0)
    assert ok_empty is False


def test_privacy_redaction_preserves_shape():
    img = np.full((300, 500, 3), 120, dtype=np.uint8)
    redacted, faces, plates = redact_privacy_regions(img)
    assert redacted.shape == img.shape
    assert faces >= 0
    assert plates >= 0


def test_sqlite_database_lifecycle(tmp_path: Path):
    test_db = tmp_path / "test_roadwatch.db"
    init_db(test_db)
    seeded = seed_demo_inspections_if_empty(test_db)
    assert seeded == 5

    records = list_inspections(db_path=test_db)
    assert len(records) == 5

    df = inspections_to_dataframe(db_path=test_db)
    assert len(df) == 5

    delete_all_inspections(test_db)
    assert len(list_inspections(db_path=test_db)) == 0


# ---------------------------------------------------------------------------
# 3. Inference, Grad-CAM & Performance Tests
# ---------------------------------------------------------------------------

def test_gradcam_and_inference_performance():
    img = np.full((600, 800, 3), 120, dtype=np.uint8)
    img[320:420, 300:460] = 20

    t0 = time.perf_counter()
    annotated, count, ratio, severity, confs = analyse_image(img, conf=0.30)
    elapsed_sec = time.perf_counter() - t0

    assert annotated.shape == img.shape
    assert count >= 1
    assert ratio > 0.0
    assert severity in ("Low", "Medium", "High")
    assert len(confs) == count
    assert elapsed_sec <= 3.0

    overlay, heatmap, prob = generate_gradcam_heatmap(img)
    assert overlay.shape == img.shape
    assert heatmap.shape == img.shape
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# 4. Report Generation & Evaluation Data Tests
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
        road_name="MG Road",
        ward_name="Ward 11",
        gps_coords={"latitude": 12.9716, "longitude": 77.5946},
    )
    assert "test_road.jpg" in csv_text
    assert "MG Road" in csv_text

    dummy_img = np.full((300, 400, 3), 100, dtype=np.uint8)
    pdf_bytes = generate_pdf_report(
        summary,
        source_name="test_road.jpg",
        road_name="MG Road",
        ward_name="Ward 11",
        classifier_prob=0.91,
        inference_time_ms=42.5,
        gps_coords={"latitude": 12.9716, "longitude": 77.5946},
        annotated_rgb=dummy_img,
        gradcam_rgb=dummy_img,
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_evaluation_tables():
    assert len(get_model_comparison_df()) >= 4
    assert len(get_confidence_threshold_sweep_df()) >= 5
    assert len(get_error_analysis_taxonomy()) == 6
