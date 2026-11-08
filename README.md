# RoadWatch: Pothole Detection from Road Images using Deep Learning

> **Turn any road photo into an instant, explainable pothole report, with no special hardware and no manual inspection.**

**Course:** Community / Societal Project (1BCP308), VTU, Semester 3  
**SDGs Addressed:** SDG 9 (Industry, Innovation and Infrastructure), SDG 11 (Sustainable Cities and Communities), SDG 3 (Good Health and Well-being)  
**Programme Outcomes:** PO 5 (Modern Tool Usage), PO 6 (The Engineer and Society)

---

## Overview

**RoadWatch** is a software-only web application that uses deep learning to detect potholes in road images and short videos. It combines:
1. **Stage A — Baseline Classifier (MobileNetV2):** Image-level classification (`pothole` vs `no pothole`) with **Grad-CAM** visual explanations.
2. **Stage B — Object Detector (YOLOv8):** Bounding-box localisation of individual potholes with confidence scores.
3. **Stage C — Severity & Reporting Engine:** Calculates relative affected road area, assigns a severity rating (**Low** `< 2%`, **Medium** `2–8%`, **High** `> 8%`) and a repair priority score, plots GPS EXIF coordinates on an interactive map if available, and exports downloadable **CSV** and **PDF** inspection reports.

Full project specifications, architecture diagrams, syllabus mapping, and evaluation checklists are in [`docs/RoadWatch_Project_Plan.md`](docs/RoadWatch_Project_Plan.md).

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
├── models/                   # Trained weights (classifier & YOLOv8)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classifier_baseline.ipynb
│   └── 03_yolo_training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Image/video validation, resizing, EXIF GPS extraction
│   ├── infer.py              # Model loading and inference pipeline
│   ├── severity.py           # Area ratio, severity rating, and priority score
│   ├── report.py             # CSV and PDF report generation
│   └── gradcam.py            # Grad-CAM heatmap generation for classifier explainability
├── tests/                    # Unit and functional tests (pytest)
├── app.py                    # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
pytest

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## Disclaimer

RoadWatch outputs are intended for **decision-support and academic research**. Relative severity ratings depend on camera angle and distance; qualified civil engineers must verify road conditions before scheduling structural repairs.
