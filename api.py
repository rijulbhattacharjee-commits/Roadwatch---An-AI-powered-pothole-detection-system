"""
FastAPI REST API Backend for RoadWatch (`api.py`).

Exposes interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) endpoints for:
- `GET  /api/v1/health`: Health check and active ML model backend info
- `POST /api/v1/analyze`: Upload a road image, run two-stage ML inference, optional privacy redaction,
  and optionally log to the SQLite municipal database
- `GET  /api/v1/inspections`: List logged road inspections sorted by priority score
- `POST /api/v1/inspections/seed`: Seed sample municipal road inspections
- `DELETE /api/v1/inspections`: Clear inspection history
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database import (
    delete_all_inspections,
    init_db,
    list_inspections,
    save_inspection,
    seed_demo_inspections_if_empty,
)
from src.infer import get_model_service
from src.preprocess import extract_gps_exif, load_and_preprocess_image, validate_file
from src.privacy import redact_privacy_regions
from src.severity import HIGH_THRESHOLD, LOW_THRESHOLD
from src.urgency import estimate_repair_area_and_cost, get_urgency_tier


app = FastAPI(
    title="RoadWatch API — Deep Learning Pothole Detection & Severity Engine",
    description=(
        "REST API for **RoadWatch** (VTU 1BCP308 Community/Societal Project). "
        "Provides Stage A MobileNetV2 classification + Grad-CAM, Stage B YOLOv8 detection, "
        "Stage C severity & repair priority scoring, privacy redaction, and municipal SQLite logging."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionItemResponse(BaseModel):
    id: int
    bbox: List[float]
    confidence: float
    area_px: float
    area_ratio: float
    severity: str


class AnalyzeResponse(BaseModel):
    source_name: str
    road_name: str
    ward_name: str
    image_width: int
    image_height: int
    pothole_count: int
    total_area_ratio: float
    overall_severity: str
    priority_score: float
    urgency_tier: str
    recommended_action: str
    estimated_damaged_sqm: float
    estimated_cost_inr: float
    classifier_label: str
    classifier_pothole_prob: float
    inference_time_ms: float
    backend_used: str
    faces_blurred: int
    plates_blurred: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    saved_inspection_id: Optional[int] = None
    detections: List[DetectionItemResponse]


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    seed_demo_inspections_if_empty()


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Return service status and active ML backend details."""
    service = get_model_service()
    return {
        "status": "ok",
        "service": "RoadWatch API",
        "version": "1.0.0",
        "active_detector_backend": service.backend_name,
        "yolo_weights_loaded": service.yolo_model is not None,
        "classifier_weights_loaded": service.classifier_model is not None,
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_road_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.35),
    low_thresh: float = Form(LOW_THRESHOLD),
    high_thresh: float = Form(HIGH_THRESHOLD),
    apply_privacy_blur: bool = Form(True),
    save_to_db: bool = Form(True),
    road_name: str = Form("Unspecified Road"),
    ward_name: str = Form("General Ward"),
    manual_lat: Optional[float] = Form(None),
    manual_lon: Optional[float] = Form(None),
) -> AnalyzeResponse:
    """
    Upload a road image, run pothole detection & severity analysis, and optionally
    persist the record to the municipal SQLite database.
    """
    raw_bytes = await file.read()
    filename = file.filename or "uploaded_image.jpg"

    is_valid, msg = validate_file(filename, len(raw_bytes), allow_video=False)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    rgb_img, _ = load_and_preprocess_image(raw_bytes)
    faces_blurred, plates_blurred = 0, 0
    if apply_privacy_blur:
        rgb_img, faces_blurred, plates_blurred = redact_privacy_regions(rgb_img)

    gps = extract_gps_exif(raw_bytes)
    lat = gps["latitude"] if gps else manual_lat
    lon = gps["longitude"] if gps else manual_lon

    service = get_model_service()
    res = service.predict(
        rgb_img,
        conf_threshold=conf_threshold,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    summary = res.severity_summary
    tier_label, rec_action, _ = get_urgency_tier(
        summary.overall_severity, summary.priority_score
    )
    est_sqm, est_cost = estimate_repair_area_and_cost(summary.total_area_ratio)

    saved_id: Optional[int] = None
    if save_to_db:
        saved_id = save_inspection(
            summary=summary,
            source_name=filename,
            road_name=road_name,
            ward_name=ward_name,
            classifier_prob=res.classifier_pothole_prob,
            inference_time_ms=res.inference_time_ms,
            latitude=lat,
            longitude=lon,
        )

    return AnalyzeResponse(
        source_name=filename,
        road_name=road_name,
        ward_name=ward_name,
        image_width=summary.image_width,
        image_height=summary.image_height,
        pothole_count=summary.pothole_count,
        total_area_ratio=summary.total_area_ratio,
        overall_severity=summary.overall_severity,
        priority_score=summary.priority_score,
        urgency_tier=tier_label,
        recommended_action=rec_action,
        estimated_damaged_sqm=est_sqm,
        estimated_cost_inr=est_cost,
        classifier_label=res.classifier_label,
        classifier_pothole_prob=res.classifier_pothole_prob,
        inference_time_ms=res.inference_time_ms,
        backend_used=res.backend_used,
        faces_blurred=faces_blurred,
        plates_blurred=plates_blurred,
        latitude=lat,
        longitude=lon,
        saved_inspection_id=saved_id,
        detections=[
            DetectionItemResponse(
                id=d.id,
                bbox=list(d.bbox),
                confidence=d.confidence,
                area_px=d.area_px,
                area_ratio=d.area_ratio,
                severity=d.severity,
            )
            for d in summary.detections
        ],
    )


@app.get("/api/v1/inspections")
def get_logged_inspections(limit: int = 100) -> Dict[str, Any]:
    """Retrieve logged municipal road inspections ordered by priority score descending."""
    records = list_inspections(limit=limit)
    return {"count": len(records), "inspections": records}


@app.post("/api/v1/inspections/seed")
def seed_inspections() -> Dict[str, Any]:
    """Seed realistic sample road segment inspections if database is empty."""
    added = seed_demo_inspections_if_empty()
    return {"seeded": added, "total": len(list_inspections())}


@app.delete("/api/v1/inspections")
def clear_inspections() -> Dict[str, str]:
    """Delete all logged road inspections."""
    delete_all_inspections()
    return {"status": "cleared"}
