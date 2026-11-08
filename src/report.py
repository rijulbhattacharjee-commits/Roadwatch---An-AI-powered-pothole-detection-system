"""
Report Generation & Map Visualization for RoadWatch.

Generates:
- Downloadable CSV table of detected potholes and summary statistics
- Downloadable formatted PDF inspection report (using fpdf2, with HTML fallback)
- Interactive Folium map when GPS EXIF coordinates are available
"""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.severity import SeveritySummary


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
    gps_coords: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a CSV string containing metadata header rows and per-pothole detections.
    """
    df = detections_to_dataframe(summary)
    df["Source File"] = source_name
    df["Overall Road Severity"] = summary.overall_severity
    df["Priority Score (0-100)"] = summary.priority_score
    if gps_coords:
        df["Latitude"] = gps_coords.get("latitude")
        df["Longitude"] = gps_coords.get("longitude")

    if df.empty:
        # Provide a single summary row when 0 potholes are detected
        df = pd.DataFrame(
            [
                {
                    "Source File": source_name,
                    "Pothole ID": 0,
                    "Confidence": 0.0,
                    "Area Ratio (%)": 0.0,
                    "Severity": "None",
                    "Overall Road Severity": summary.overall_severity,
                    "Priority Score (0-100)": summary.priority_score,
                    "Latitude": gps_coords.get("latitude") if gps_coords else None,
                    "Longitude": gps_coords.get("longitude") if gps_coords else None,
                }
            ]
        )

    return df.to_csv(index=False)


def generate_pdf_report(
    summary: SeveritySummary,
    source_name: str = "uploaded_image.jpg",
    classifier_prob: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    gps_coords: Optional[Dict[str, float]] = None,
) -> bytes:
    """
    Generate a structured PDF inspection report using `fpdf2`.
    Returns raw PDF bytes suitable for Streamlit's `st.download_button`.
    """
    from fpdf import FPDF

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title banner
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "RoadWatch - Automated Pothole Inspection Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {timestamp} | Source: {source_name}", ln=True)
    pdf.ln(4)

    # Summary Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)

    pdf.cell(
        0,
        7,
        f"Image Resolution: {summary.image_width} x {summary.image_height} px",
        ln=True,
    )
    pdf.cell(0, 7, f"Total Potholes Detected: {summary.pothole_count}", ln=True)
    pdf.cell(
        0,
        7,
        f"Total Affected Road Area Ratio: {summary.total_area_ratio * 100:.2f}%",
        ln=True,
    )
    pdf.cell(0, 7, f"Overall Road Severity Rating: {summary.overall_severity}", ln=True)
    pdf.cell(
        0,
        7,
        f"Repair Priority Score: {summary.priority_score:.1f} / 100.0",
        ln=True,
    )

    if classifier_prob is not None:
        pdf.cell(
            0,
            7,
            f"Stage A Classifier Pothole Probability: {classifier_prob * 100:.1f}%",
            ln=True,
        )
    if inference_time_ms is not None:
        pdf.cell(
            0, 7, f"Processing Time: {inference_time_ms:.1f} ms", ln=True
        )
    if gps_coords:
        pdf.cell(
            0,
            7,
            f"GPS Coordinates (EXIF): {gps_coords['latitude']}, {gps_coords['longitude']}",
            ln=True,
        )

    pdf.ln(5)

    # Per-Pothole Table
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Detected Pothole Breakdown", ln=True)

    if not summary.detections:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 8, "No potholes detected above the selected confidence threshold.", ln=True)
    else:
        pdf.set_font("Helvetica", "B", 10)
        col_widths = [18, 58, 30, 40, 35]
        headers = ["ID", "Bounding Box (x1,y1,x2,y2)", "Confidence", "Area Ratio (%)", "Severity"]
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 8, header, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
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
                pdf.cell(width, 7, val, border=1, align="C")
            pdf.ln()

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Disclaimer: This report is automatically generated by RoadWatch for decision support "
        "and academic evaluation. Relative severity depends on camera angle and distance. "
        "Field verification by a qualified civil engineer is recommended before scheduling repairs.",
    )

    out = pdf.output()
    return bytes(out) if not isinstance(out, bytes) else out


def build_folium_map(
    gps_coords: Dict[str, float],
    summary: SeveritySummary,
    source_name: str = "Road Image",
):
    """
    Create a Folium map centered on the extracted GPS EXIF coordinates with a
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

    fmap = folium.Map(location=[lat, lon], zoom_start=16)
    popup_html = (
        f"<b>{source_name}</b><br>"
        f"Potholes: {summary.pothole_count}<br>"
        f"Severity: <b>{summary.overall_severity}</b><br>"
        f"Area Ratio: {summary.total_area_ratio * 100:.2f}%<br>"
        f"Priority Score: {summary.priority_score:.1f}/100"
    )
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"RoadWatch: {summary.overall_severity} Severity ({summary.pothole_count} potholes)",
        icon=folium.Icon(color=marker_color, icon="info-sign"),
    ).add_to(fmap)

    return fmap
