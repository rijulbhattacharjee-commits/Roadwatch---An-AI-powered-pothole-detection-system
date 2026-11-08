"""
RoadWatch: Pothole Detection from Road Images using Deep Learning
Streamlit Web Application (app.py)
"""

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.infer import get_model_service
from src.preprocess import (
    extract_gps_exif,
    load_and_preprocess_image,
    sample_video_frames,
    validate_file,
)
from src.report import (
    build_folium_map,
    detections_to_dataframe,
    generate_csv_report,
    generate_pdf_report,
)
from src.severity import HIGH_THRESHOLD, LOW_THRESHOLD


st.set_page_config(
    page_title="RoadWatch | Pothole Detection & Severity Reporting",
    page_icon="🛣️",
    layout="wide",
)


def create_synthetic_demo_image() -> np.ndarray:
    """
    Create a realistic synthetic paved road image with simulated potholes
    so users can test the entire pipeline within 1 click without uploading a file.
    """
    h, w = 600, 900
    rng = np.random.default_rng(42)

    # Paved asphalt texture in lower 78%, sky in top 22%
    img = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.22)
    img[:horizon, :] = (185, 205, 225)  # Sky

    asphalt_base = np.full((h - horizon, w, 3), 115, dtype=np.int16)
    grain = rng.integers(-12, 13, size=(h - horizon, w, 1), dtype=np.int16)
    asphalt = np.clip(asphalt_base + grain, 70, 160).astype(np.uint8)
    img[horizon:, :] = asphalt

    # Road lane markings
    cv2.line(img, (80, h - 10), (340, horizon + 20), (220, 220, 220), 5)
    cv2.line(img, (w - 80, h - 10), (w - 340, horizon + 20), (220, 220, 220), 5)

    # Draw 3 realistic dark elliptical pothole depressions with rough rims
    potholes = [
        ((380, 390), (85, 42), 12),
        ((570, 470), (65, 34), -8),
        ((310, 260), (42, 22), 5),
    ]
    for center, axes, angle in potholes:
        # Outer rim crack
        cv2.ellipse(img, center, (axes[0] + 6, axes[1] + 5), angle, 0, 360, (68, 68, 70), -1)
        # Inner dark depression
        cv2.ellipse(img, center, axes, angle, 0, 360, (24, 24, 26), -1)
        # Inner texture highlight
        cv2.ellipse(
            img,
            (center[0] - 8, center[1] - 4),
            (int(axes[0] * 0.65), int(axes[1] * 0.6)),
            angle,
            0,
            360,
            (15, 15, 18),
            -1,
        )

    return img


def render_sidebar():
    st.sidebar.title("⚙️ Detection Settings")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Minimum detector confidence score to report a pothole.",
    )

    st.sidebar.subheader("📏 Severity Area Thresholds")
    low_pct = st.sidebar.slider(
        "Low → Medium Threshold (% of image area)",
        min_value=0.5,
        max_value=5.0,
        value=LOW_THRESHOLD * 100.0,
        step=0.5,
    )
    high_pct = st.sidebar.slider(
        "Medium → High Threshold (% of image area)",
        min_value=3.0,
        max_value=20.0,
        value=HIGH_THRESHOLD * 100.0,
        step=0.5,
    )
    show_gradcam = st.sidebar.checkbox("Show Stage A Grad-CAM Heatmap", value=True)

    service = get_model_service()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Active Detector Backend:**\n`{service.backend_name}`")
    st.sidebar.info(
        "**Note:** Severity is a relative indicator based on 2D bounding-box area ratio "
        "and is intended for repair prioritization support."
    )
    return conf_thresh, low_pct / 100.0, max(low_pct + 0.5, high_pct) / 100.0, show_gradcam


def render_image_results(
    rgb_img: np.ndarray,
    source_name: str,
    conf_thresh: float,
    low_thresh: float,
    high_thresh: float,
    show_gradcam: bool,
    gps_coords=None,
):
    service = get_model_service()
    res = service.predict(
        rgb_img,
        conf_threshold=conf_thresh,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    summary = res.severity_summary

    # Top-level metrics bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Potholes Detected", summary.pothole_count)
    m2.metric("Affected Area Ratio", f"{summary.total_area_ratio * 100:.2f}%")
    m3.metric("Overall Road Severity", summary.overall_severity)
    m4.metric("Repair Priority Score", f"{summary.priority_score:.1f} / 100")
    m5.metric("Inference Time", f"{res.inference_time_ms:.0f} ms")

    # Visual comparison columns
    if show_gradcam:
        c1, c2, c3 = st.columns(3)
        c1.subheader("1. Original Road Image")
        c1.image(rgb_img, use_container_width=True)

        c2.subheader("2. Stage B Detector & Severity")
        c2.image(
            res.annotated_rgb,
            caption=f"{summary.pothole_count} pothole(s) | Overall: {summary.overall_severity}",
            use_container_width=True,
        )

        c3.subheader("3. Stage A Grad-CAM Explainability")
        c3.image(
            res.gradcam_overlay_rgb,
            caption=f"{res.classifier_label} (Pothole Prob: {res.classifier_pothole_prob * 100:.1f}%)",
            use_container_width=True,
        )
    else:
        c1, c2 = st.columns(2)
        c1.subheader("Original Road Image")
        c1.image(rgb_img, use_container_width=True)
        c2.subheader("Annotated Pothole Detection")
        c2.image(
            res.annotated_rgb,
            caption=f"{summary.pothole_count} pothole(s) | Overall: {summary.overall_severity}",
            use_container_width=True,
        )

    st.markdown("---")

    # Per-pothole breakdown table & downloadable reports
    col_table, col_dl = st.columns([2, 1])

    with col_table:
        st.subheader("📋 Per-Pothole Breakdown")
        df = detections_to_dataframe(summary)
        if df.empty:
            st.success("No potholes detected above the selected confidence threshold.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with col_dl:
        st.subheader("📥 Download Inspection Report")
        csv_data = generate_csv_report(
            summary=summary, source_name=source_name, gps_coords=gps_coords
        )
        st.download_button(
            label="📄 Download CSV Report",
            data=csv_data,
            file_name=f"roadwatch_report_{Path(source_name).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        try:
            pdf_bytes = generate_pdf_report(
                summary=summary,
                source_name=source_name,
                classifier_prob=res.classifier_pothole_prob,
                inference_time_ms=res.inference_time_ms,
                gps_coords=gps_coords,
            )
            st.download_button(
                label="📑 Download PDF Inspection Report",
                data=pdf_bytes,
                file_name=f"roadwatch_report_{Path(source_name).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF generation unavailable: {exc}")

    # Optional GPS EXIF Map View
    if gps_coords:
        st.markdown("---")
        st.subheader("🗺️ Geotagged Pothole Location (EXIF GPS)")
        st.write(
            f"**Latitude:** `{gps_coords['latitude']}` | **Longitude:** `{gps_coords['longitude']}`"
        )
        try:
            from streamlit_folium import st_folium

            fmap = build_folium_map(gps_coords, summary, source_name=source_name)
            st_folium(fmap, width=800, height=380)
        except Exception:
            map_df = pd.DataFrame(
                [{"lat": gps_coords["latitude"], "lon": gps_coords["longitude"]}]
            )
            st.map(map_df)


def main():
    st.title("🛣️ RoadWatch: Deep Learning Pothole Detection & Severity Reporting")
    st.markdown(
        "**Turn any road photo or short video into an instant, explainable pothole report — "
        "with bounding-box localization, severity rating, and downloadable CSV/PDF reports.**"
    )

    conf_thresh, low_thresh, high_thresh, show_gradcam = render_sidebar()

    tab_img, tab_vid, tab_about = st.tabs(
        ["📷 Road Image Analysis", "🎥 Road Video Analysis", "ℹ️ About & Methodology"]
    )

    with tab_img:
        col_up, col_demo = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload a road surface image (JPG, JPEG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
            )
        with col_demo:
            st.write("")
            st.write("")
            run_demo = st.button("🚀 Try Sample Road Image", use_container_width=True)

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            valid, msg = validate_file(uploaded_file.name, len(raw_bytes), allow_video=False)
            if not valid:
                st.error(msg)
            else:
                rgb_img, _ = load_and_preprocess_image(raw_bytes)
                gps = extract_gps_exif(raw_bytes)
                render_image_results(
                    rgb_img=rgb_img,
                    source_name=uploaded_file.name,
                    conf_thresh=conf_thresh,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                    show_gradcam=show_gradcam,
                    gps_coords=gps,
                )
        elif run_demo:
            demo_img = create_synthetic_demo_image()
            render_image_results(
                rgb_img=demo_img,
                source_name="sample_road_demo.jpg",
                conf_thresh=conf_thresh,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                show_gradcam=show_gradcam,
                gps_coords={"latitude": 12.9716, "longitude": 77.5946},
            )
        else:
            st.info(
                "👆 Upload a road photo above or click **Try Sample Road Image** to run an instant inspection."
            )

    with tab_vid:
        vid_file = st.file_uploader(
            "Upload a short road video clip (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov", "mkv"],
        )
        every_n = st.slider("Sample 1 frame every N frames", 5, 60, 15, step=5)
        if vid_file is not None:
            raw_vid = vid_file.getvalue()
            valid, msg = validate_file(vid_file.name, len(raw_vid), allow_video=True)
            if not valid:
                st.error(msg)
            else:
                with st.spinner("Sampling and analyzing video frames..."):
                    frames = sample_video_frames(raw_vid, every_n_frames=every_n, max_frames=15)
                if not frames:
                    st.warning("Could not extract frames from the uploaded video.")
                else:
                    service = get_model_service()
                    rows = []
                    worst_frame_res = None
                    worst_ratio = -1.0

                    for f_idx, ts_sec, rgb_frame in frames:
                        res = service.predict(
                            rgb_frame,
                            conf_threshold=conf_thresh,
                            low_thresh=low_thresh,
                            high_thresh=high_thresh,
                        )
                        s = res.severity_summary
                        rows.append(
                            {
                                "Frame #": f_idx,
                                "Timestamp (s)": ts_sec,
                                "Potholes": s.pothole_count,
                                "Area Ratio (%)": round(s.total_area_ratio * 100.0, 2),
                                "Severity": s.overall_severity,
                                "Priority Score": s.priority_score,
                            }
                        )
                        if s.total_area_ratio > worst_ratio:
                            worst_ratio = s.total_area_ratio
                            worst_frame_res = (f_idx, ts_sec, res)

                    st.subheader(f"📊 Video Summary ({len(frames)} Sampled Frames)")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    if worst_frame_res is not None:
                        f_idx, ts_sec, best_res = worst_frame_res
                        st.subheader(
                            f"🚨 Peak Severity Frame (Frame #{f_idx} @ {ts_sec}s — "
                            f"{best_res.severity_summary.overall_severity})"
                        )
                        st.image(best_res.annotated_rgb, use_container_width=True)

    with tab_about:
        st.markdown(
            """
            ### How RoadWatch Works
            1. **Stage A — Baseline Classifier (`MobileNetV2` + `Grad-CAM`):**
               Classifies whether an image contains road damage and produces a Grad-CAM heatmap highlighting salient regions.
            2. **Stage B — Object Detector (`YOLOv8`):**
               Localizes each individual pothole with bounding boxes and confidence scores.
            3. **Stage C — Severity & Priority Engine:**
               - Computes $\\text{Severity Ratio} = \\frac{\\sum \\text{Box Area}}{\\text{Image Area}}$
               - **Low:** $< 2\\%$ | **Medium:** $2\\% - 8\\%$ | **High:** $> 8\\%$
               - Generates downloadable **CSV** and **PDF** reports for municipal bodies, civil engineers, and citizen groups.
            """
        )


if __name__ == "__main__":
    main()
