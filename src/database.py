"""
SQLite Persistence Layer for RoadWatch (`src/database.py`).

Stores road inspection records, ward/road segment metadata, severity metrics,
repair cost estimates, and GPS coordinates so users can track city-wide potholes,
view an aggregated Folium heatmap/pin map, and export historical data.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.severity import SeveritySummary
from src.urgency import estimate_repair_area_and_cost, get_urgency_tier


DEFAULT_DB_PATH = Path("data/roadwatch_history.db")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the `inspections` table if it does not already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                road_name TEXT DEFAULT 'Unspecified Road',
                ward_name TEXT DEFAULT 'General Ward',
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                pothole_count INTEGER NOT NULL,
                total_area_ratio REAL NOT NULL,
                overall_severity TEXT NOT NULL,
                priority_score REAL NOT NULL,
                urgency_tier TEXT NOT NULL,
                estimated_sqm REAL NOT NULL,
                estimated_cost_inr REAL NOT NULL,
                classifier_prob REAL DEFAULT 0.0,
                inference_time_ms REAL DEFAULT 0.0,
                latitude REAL,
                longitude REAL,
                detections_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_inspection(
    summary: SeveritySummary,
    source_name: str,
    road_name: str = "MG Road / Ward Segment",
    ward_name: str = "Ward 12",
    classifier_prob: float = 0.0,
    inference_time_ms: float = 0.0,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """
    Save an inspection result to the SQLite database and return the inserted row ID.
    """
    init_db(db_path)
    tier_label, _, _ = get_urgency_tier(summary.overall_severity, summary.priority_score)
    est_sqm, est_cost = estimate_repair_area_and_cost(summary.total_area_ratio)

    det_list = [
        {
            "id": d.id,
            "bbox": list(d.bbox),
            "confidence": d.confidence,
            "area_px": d.area_px,
            "area_ratio": d.area_ratio,
            "severity": d.severity,
        }
        for d in summary.detections
    ]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO inspections (
                created_at, source_name, road_name, ward_name,
                image_width, image_height, pothole_count, total_area_ratio,
                overall_severity, priority_score, urgency_tier,
                estimated_sqm, estimated_cost_inr,
                classifier_prob, inference_time_ms,
                latitude, longitude, detections_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                source_name,
                road_name.strip() or "Unspecified Road",
                ward_name.strip() or "General Ward",
                summary.image_width,
                summary.image_height,
                summary.pothole_count,
                summary.total_area_ratio,
                summary.overall_severity,
                summary.priority_score,
                tier_label,
                est_sqm,
                est_cost,
                classifier_prob,
                inference_time_ms,
                latitude,
                longitude,
                json.dumps(det_list),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_inspections(
    limit: int = 200,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Return recent inspections ordered by priority_score DESC, created_at DESC.
    """
    init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM inspections
            ORDER BY priority_score DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def inspections_to_dataframe(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Return all logged inspections as a clean pandas DataFrame for the UI/CSV export."""
    records = list_inspections(db_path=db_path)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "detections_json" in df.columns:
        df = df.drop(columns=["detections_json"])
    return df


def delete_all_inspections(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Clear all logged inspections from the database."""
    init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM inspections")
        conn.commit()


def seed_demo_inspections_if_empty(db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Populate the SQLite database with 5 realistic Bengaluru municipal road segment
    inspections if the database is currently empty, so the City Map & History dashboard
    works immediately during live demos.
    """
    existing = list_inspections(limit=1, db_path=db_path)
    if existing:
        return 0

    from src.severity import evaluate_severity

    samples = [
        {
            "source_name": "outer_ring_road_bellandur.jpg",
            "road_name": "Outer Ring Road (Bellandur Flyover)",
            "ward_name": "Ward 150 - Bellandur",
            "boxes": [(120, 260, 410, 450), (490, 310, 680, 440), (210, 460, 350, 540)],
            "confs": [0.91, 0.84, 0.77],
            "lat": 12.9304,
            "lon": 77.6784,
        },
        {
            "source_name": "jc_road_junction.jpg",
            "road_name": "JC Road Main Junction",
            "ward_name": "Ward 118 - Sudhamanagar",
            "boxes": [(220, 300, 430, 430), (510, 350, 620, 440)],
            "confs": [0.86, 0.72],
            "lat": 12.9610,
            "lon": 77.5836,
        },
        {
            "source_name": "indiranagar_100ft_rd.jpg",
            "road_name": "100 Feet Road Indiranagar",
            "ward_name": "Ward 80 - Hoysala Nagar",
            "boxes": [(310, 360, 430, 430)],
            "confs": [0.68],
            "lat": 12.9784,
            "lon": 77.6408,
        },
        {
            "source_name": "hebbal_service_road.jpg",
            "road_name": "Hebbal Kempapura Service Road",
            "ward_name": "Ward 7 - Byatarayanapura",
            "boxes": [(90, 240, 420, 470), (460, 270, 740, 460)],
            "confs": [0.94, 0.89],
            "lat": 13.0358,
            "lon": 77.5970,
        },
        {
            "source_name": "jayanagar_4th_block.jpg",
            "road_name": "11th Main Road Jayanagar 4th Block",
            "ward_name": "Ward 168 - Pattabhiram Nagar",
            "boxes": [],
            "confs": [],
            "lat": 12.9279,
            "lon": 77.5831,
        },
    ]

    for item in samples:
        summary = evaluate_severity(
            boxes=item["boxes"],
            confidences=item["confs"],
            image_width=900,
            image_height=600,
        )
        save_inspection(
            summary=summary,
            source_name=item["source_name"],
            road_name=item["road_name"],
            ward_name=item["ward_name"],
            classifier_prob=0.92 if item["boxes"] else 0.08,
            inference_time_ms=38.4,
            latitude=item["lat"],
            longitude=item["lon"],
            db_path=db_path,
        )
    return len(samples)
