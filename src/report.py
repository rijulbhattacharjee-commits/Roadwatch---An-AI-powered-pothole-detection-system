"""
Report Generation & Map Visualization for RoadWatch (`src/report.py`).

Generates:
- Downloadable CSV table of detected potholes and summary statistics
- Downloadable formatted PDF inspection report (using fpdf2, with embedded annotated
  and Grad-CAM images, municipal urgency SLA tier, and indicative patching cost)
- Interactive Folium maps for single geotagged images and city-wide multi-inspection logs
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.severity import SeveritySummary
from src.urgency import estimate_repair_area_and_cost, get_urgency_tier


def detections_to_dataframe(summary: SeveritySummary) -> pd.DataFrame:
    """
    Convert a SeveritySummary's detections into a clean pandas DataFrame.
    """
    if not summary.detections:
        return pd.DataFrame(
            columns=[
                "Pothole ID",
                "X1 (px)",
                "Y1 (px)",
                "X2 (px)",
                "Y2 (px)",
                "Confidence",
                "Area (px²)",
                "Area Ratio (%)",
                "Severity",
            ]
        )

    rows = []
    for d in summary.detections:
        rows.append(
            {
                "Pothole ID": d.id,
                "X1 (px)": round(d.bbox[0], 1),
                "Y1 (px)": round(d.bbox[1], 1),
                "X2 (px)": round(d.bbox[2], 1),
                "Y2 (px)": round(d.bbox[3], 1),
                "Confidence": round(d.confidence, 3),
                "Area (px²)": round(d.area_px, 1),
                "Area Ratio (%)": round(d.area_ratio * 100.0, 2),
                "Severity": d.severity,
            }
        )
    return pd.DataFrame(rows)


def generate_csv_report(
    summary: SeveritySummary,
    source_name: str = "uploaded_image.jpg",
    road_name: str = "Unspecified Road",
    ward_name: str = "General Ward",
    gps_coords: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a CSV string containing metadata columns, urgency tier, estimated repair cost,
    and per-pothole detections.
    """
    tier_label, _, _ = get_urgency_tier(summary.overall_severity, summary.priority_score)
    est_sqm, est_cost = estimate_repair_area_and_cost(summary.total_area_ratio)

    df = detections_to_dataframe(summary)
    df["Source File"] = source_name
    df["Road Segment"] = road_name
    df["Ward / Zone"] = ward_name
    df["Overall Road Severity"] = summary.overall_severity
    df["Priority Score (0-100)"] = summary.priority_score
    df["Urgency Tier"] = tier_label
    df["Est. Damaged Area (m²)"] = est_sqm
    df["Est. Patch Cost (INR)"] = est_cost
    if gps_coords:
        df["Latitude"] = gps_coords.get("latitude")
        df["Longitude"] = gps_coords.get("longitude")

    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "Source File": source_name,
                    "Road Segment": road_name,
                    "Ward / Zone": ward_name,
                    "Pothole ID": 0,
                    "Confidence": 0.0,
                    "Area Ratio (%)": 0.0,
                    "Severity": "None",
                    "Overall Road Severity": summary.overall_severity,
                    "Priority Score (0-100)": summary.priority_score,
                    "Urgency Tier": tier_label,
                    "Est. Damaged Area (m²)": 0.0,
                    "Est. Patch Cost (INR)": 0.0,
                    "Latitude": gps_coords.get("latitude") if gps_coords else None,
                    "Longitude": gps_coords.get("longitude") if gps_coords else None,
                }
            ]
        )

    return df.to_csv(index=False)


def generate_pdf_report(
    summary: SeveritySummary,
    source_name: str = "uploaded_image.jpg",
    road_name: str = "Unspecified Road",
    ward_name: str = "General Ward",
    classifier_prob: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    gps_coords: Optional[Dict[str, float]] = None,
    annotated_rgb: Optional[np.ndarray] = None,
    gradcam_rgb: Optional[np.ndarray] = None,
) -> bytes:
    """
    Generate a structured municipal PDF inspection report using `fpdf2`.
    Embeds annotated detection and Grad-CAM images when provided, along with
    urgency SLA tier and indicative repair cost.
    """
    from fpdf import FPDF

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tier_label, recommended_action, _ = get_urgency_tier(
        summary.overall_severity, summary.priority_score
    )
    est_sqm, est_cost = estimate_repair_area_and_cost(summary.total_area_ratio)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Header Banner
    pdf.set_fill_color(28, 40, 51)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        12,
        "  RoadWatch - Municipal Pothole Inspection & Priority Report",
        ln=True,
        fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0,
        6,
        f"Generated: {timestamp}  |  Source: {source_name}  |  Road: {road_name} ({ward_name})",
        ln=True,
    )
    pdf.ln(3)

    # 1. Executive Summary & Engineering Assessment
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "1. Executive Summary & Repair Prioritization", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(
        95,
        6,
        f"Image Resolution: {summary.image_width} x {summary.image_height} px",
        border=0,
    )
    pdf.cell(
        95,
        6,
        f"Total Potholes Detected: {summary.pothole_count}",
        border=0,
        ln=True,
    )
    pdf.cell(
        95,
        6,
        f"Affected Road Area Ratio: {summary.total_area_ratio * 100:.2f}%",
        border=0,
    )
    pdf.cell(
        95,
        6,
        f"Overall Severity Rating: {summary.overall_severity}",
        border=0,
        ln=True,
    )
    pdf.cell(
        95,
        6,
        f"Repair Priority Score: {summary.priority_score:.1f} / 100.0",
        border=0,
    )
    pdf.cell(
        95,
        6,
        f"Urgency SLA Tier: {tier_label}",
        border=0,
        ln=True,
    )
    pdf.cell(
        95,
        6,
        f"Indicative Damaged Surface: ~{est_sqm:.2f} sq.m",
        border=0,
    )
    pdf.cell(
        95,
        6,
        f"Indicative Patching Cost: INR {est_cost:,.0f}",
        border=0,
        ln=True,
    )

    if classifier_prob is not None or inference_time_ms is not None:
        prob_str = f"{classifier_prob * 100:.1f}%" if classifier_prob is not None else "N/A"
        time_str = f"{inference_time_ms:.1f} ms" if inference_time_ms is not None else "N/A"
        pdf.cell(95, 6, f"Stage A Classifier Confidence: {prob_str}", border=0)
        pdf.cell(95, 6, f"Inference Latency: {time_str}", border=0, ln=True)

    if gps_coords:
        pdf.cell(
            0,
            6,
            f"Geotagged Coordinates: Lat {gps_coords['latitude']}, Lon {gps_coords['longitude']}",
            ln=True,
        )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Recommended Civil Action: {recommended_action}", ln=True)
    pdf.ln(3)

    # 2. Embedded Visual Evidence (Annotated Detector + Grad-CAM)
    temp_files: List[str] = []
    try:
        if annotated_rgb is not None:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, "2. Visual Evidence (YOLOv8 Detection & Grad-CAM Explainability)", ln=True)
            y_start = pdf.get_y()

            fd1, path1 = tempfile.mkstemp(suffix=".jpg")
            os.close(fd1)
            temp_files.append(path1)
            Image.fromarray(annotated_rgb).save(path1, format="JPEG", quality=85)

            if gradcam_rgb is not None:
                fd2, path2 = tempfile.mkstemp(suffix=".jpg")
                os.close(fd2)
                temp_files.append(path2)
                Image.fromarray(gradcam_rgb).save(path2, format="JPEG", quality=85)

                pdf.image(path1, x=12, y=y_start, w=90)
                pdf.image(path2, x=108, y=y_start, w=90)
                img_h_mm = 90.0 * (annotated_rgb.shape[0] / float(max(1, annotated_rgb.shape[1])))
                pdf.set_y(y_start + img_h_mm + 4)
            else:
                pdf.image(path1, x=12, y=y_start, w=115)
                img_h_mm = 115.0 * (annotated_rgb.shape[0] / float(max(1, annotated_rgb.shape[1])))
                pdf.set_y(y_start + img_h_mm + 4)
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except OSError:
                    pass

    # 3. Per-Pothole Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. Per-Pothole Bounding Box Breakdown", ln=True)

    if not summary.detections:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 7, "No potholes detected above the selected confidence threshold.", ln=True)
    else:
        pdf.set_font("Helvetica", "B", 9)
        col_widths = [16, 62, 32, 40, 36]
        headers = ["ID", "Bounding Box (x1,y1,x2,y2)", "Confidence", "Area Ratio (%)", "Severity"]
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 7, header, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for d in summary.detections:
            bbox_str = f"({int(d.bbox[0])}, {int(d.bbox[1])}, {int(d.bbox[2])}, {int(d.bbox[3])})"
            row = [
                str(d.id),
                bbox_str,
                f"{d.confidence:.2f}",
                f"{d.area_ratio * 100:.2f}%",
                d.severity,
            ]
            for width, val in zip(col_widths, row):
                pdf.cell(width, 6, val, border=1, align="C")
            pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        4.5,
        "Disclaimer: This report is generated by RoadWatch (VTU 1BCP308 Community/Societal Project) "
        "for decision support. Relative severity and indicative patching costs depend on camera angle "
        "and distance. Field verification by a qualified civil engineer must confirm repair scope.",
    )

    out = pdf.output()
    return bytes(out) if not isinstance(out, bytes) else out


def build_folium_map(
    gps_coords: Dict[str, float],
    summary: SeveritySummary,
    source_name: str = "Road Image",
    road_name: str = "Road Segment",
):
    """
    Create a Folium map centered on the extracted or pinned GPS coordinates with a
    color-coded severity marker.
    """
    import folium

    lat = gps_coords["latitude"]
    lon = gps_coords["longitude"]

    color_map = {
        "None": "green",
        "Low": "blue",
        "Medium": "orange",
        "High": "red",
    }
    marker_color = color_map.get(summary.overall_severity, "red")
    tier_label, _, _ = get_urgency_tier(summary.overall_severity, summary.priority_score)

    fmap = folium.Map(location=[lat, lon], zoom_start=15)
    popup_html = (
        f"<b>{road_name}</b> ({source_name})<br>"
        f"Potholes: <b>{summary.pothole_count}</b><br>"
        f"Severity: <b>{summary.overall_severity}</b> ({summary.total_area_ratio * 100:.2f}%)<br>"
        f"Priority Score: <b>{summary.priority_score:.1f}/100</b><br>"
        f"SLA: {tier_label}"
    )
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{road_name}: {summary.overall_severity} ({summary.pothole_count} potholes)",
        icon=folium.Icon(color=marker_color, icon="info-sign"),
    ).add_to(fmap)

    return fmap


def build_city_dashboard_map(inspections: List[Dict[str, Any]]):
    """
    Build a multi-marker municipal Folium map showing all geotagged inspections
    color-coded by severity and sized by priority score.
    """
    import folium

    geotagged = [
        r for r in inspections if r.get("latitude") is not None and r.get("longitude") is not None
    ]
    if not geotagged:
        center = [12.9716, 77.5946]
        zoom = 12
    else:
        avg_lat = sum(float(r["latitude"]) for r in geotagged) / len(geotagged)
        avg_lon = sum(float(r["longitude"]) for r in geotagged) / len(geotagged)
        center = [avg_lat, avg_lon]
        zoom = 12

    fmap = folium.Map(location=center, zoom_start=zoom)
    color_map = {
        "None": "green",
        "Low": "blue",
        "Medium": "orange",
        "High": "red",
    }

    for rec in geotagged:
        lat = float(rec["latitude"])
        lon = float(rec["longitude"])
        sev = str(rec.get("overall_severity", "Low"))
        color = color_map.get(sev, "blue")
        popup_html = (
            f"<b>#{rec.get('id')} — {rec.get('road_name')}</b><br>"
            f"Ward: {rec.get('ward_name')}<br>"
            f"Severity: <b>{sev}</b> | Potholes: {rec.get('pothole_count')}<br>"
            f"Area Ratio: {float(rec.get('total_area_ratio', 0.0)) * 100:.2f}%<br>"
            f"Priority Score: <b>{rec.get('priority_score')}/100</b><br>"
            f"Urgency: {rec.get('urgency_tier')}<br>"
            f"Est. Cost: INR {float(rec.get('estimated_cost_inr', 0)):,.0f}"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{rec.get('road_name')} [{sev} | Score: {rec.get('priority_score')}]",
            icon=folium.Icon(color=color, icon="road", prefix="fa"),
        ).add_to(fmap)

    return fmap
