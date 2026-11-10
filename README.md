# RoadWatch: Pothole Detection from Road Images using Deep Learning

> **Turn any road photo into an instant, explainable pothole report, with no special hardware and no manual inspection.**

**Course:** Community / Societal Project (1BCP308), VTU, Semester 3  
**SDGs Addressed:** SDG 9 (Industry, Innovation and Infrastructure), SDG 11 (Sustainable Cities and Communities), SDG 3 (Good Health and Well-being)  
**Programme Outcomes:** PO 5 (Modern Tool Usage), PO 6 (The Engineer and Society)

---

## Full-Stack Architecture & Key Features

**RoadWatch** combines a two-stage deep learning pipeline with a full-stack **Streamlit Web UI**, **FastAPI REST Backend**, and **SQLite Municipal Database**:

1. **Stage A — Baseline Classifier (`MobileNetV2` + `Grad-CAM`):** Image-level classification (`pothole` vs `normal`) with **Grad-CAM** heatmaps showing which surface regions triggered the prediction.
2. **Stage B — Object Detector (`YOLOv8`):** Bounding-box localisation of individual potholes with confidence scores (plus an OpenCV morphological depression/rim fallback when running offline without custom weights).
3. **Stage C — Severity, Priority & Municipal SLA Engine (`src/severity.py`, `src/urgency.py`):**
   - Computes $\text{Severity Ratio} = \frac{\sum \text{Box Area}}{\text{Image Area}}$
   - Assigns **Low** (`< 2%`), **Medium** (`2–8%`), or **High** (`> 8%`) ratings and a `0–100` **Repair Priority Score**.
   - Maps each road segment to an actionable **Municipal Maintenance SLA Tier** (`Immediate Action < 48 hrs`, `Scheduled Maintenance 1–2 Weeks`, `Preventive Seal & Patch`) and estimates damaged surface area ($\text{m}^2$) and patching cost (INR).
4. **Automatic Privacy Redaction (`src/privacy.py`):** Detects and blurs human faces and vehicle license plates before display or report export (Section 13 Ethics).
5. **Batch Ward Inspector & Video Frame Sampling:** Upload a batch of ward survey photos or a dashcam video clip to generate a ranked municipal work-order queue.
6. **City-Wide Inspection Map & SQLite Persistence (`src/database.py`):** Logs inspections with EXIF or pinned GPS coordinates onto an interactive multi-marker Folium map.
7. **Official Reports (`src/report.py`):** Exports both **CSV** work orders and formatted **PDF** inspection reports with embedded annotated and Grad-CAM images.
8. **REST API (`api.py`):** Interactive Swagger UI (`/docs`) exposing `/api/v1/health`, `/api/v1/analyze`, and `/api/v1/inspections`.

Full project specifications, syllabus mapping, and evaluation checklists are in [`docs/RoadWatch_Project_Plan.md`](docs/RoadWatch_Project_Plan.md).

---

## Repository Structure

```text
roadwatch/
├── data/
│   ├── raw/                  # Raw downloaded datasets (git-ignored)
│   ├── processed/            # Cleaned train / val / test splits in YOLO format
│   └── data.yaml             # Ultralytics dataset configuration
├── docs/
│   └── RoadWatch_Project_Plan.md
├── models/                   # Trained weights (classifier_mobilenetv2.pt & best.pt)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classifier_baseline.ipynb
│   └── 03_yolo_training.ipynb
├── src/
│   ├── __init__.py
│   ├── database.py           # SQLite persistence & municipal inspection history
│   ├── evaluation_data.py    # Ablation studies, PR sweep & error analysis taxonomy
│   ├── gradcam.py            # Grad-CAM heatmap generation for Stage A explainability
│   ├── infer.py              # Unified model loading & inference pipeline
│   ├── preprocess.py         # File validation, resizing, video sampling, EXIF GPS
│   ├── privacy.py            # Automatic face & license plate blurring (Section 13)
│   ├── report.py             # CSV, PDF (with embedded images), and Folium maps
│   ├── severity.py           # Area ratio, severity rating, and priority score
│   └── urgency.py            # Municipal SLA tier & repair cost estimation
├── tests/
│   └── test_roadwatch.py     # Unit and functional test suite (pytest)
├── api.py                    # FastAPI REST API backend (/docs)
├── app.py                    # Full-stack Streamlit web application
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Navigate to the repository
cd roadwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit & functional tests
pytest

# 4. Launch the Streamlit Web Application
streamlit run app.py

# 5. (Optional) Launch the FastAPI REST Backend & Swagger UI at http://localhost:8000/docs
uvicorn api:app --reload --port 8000
```

---

## Disclaimer

RoadWatch outputs are intended for **decision-support and academic evaluation**. Relative severity ratings and indicative repair costs depend on camera angle and distance; qualified civil engineers must verify road conditions before scheduling structural repairs.
