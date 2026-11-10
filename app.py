"""
RoadWatch: Pothole Detection from Road Images using Deep Learning
Full-Stack Streamlit Web Application (`app.py`)
Course: Community / Societal Project (1BCP308), VTU, Semester 3
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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
from src.infer import get_model_service
from src.preprocess import (
    extract_gps_exif,
    load_and_preprocess_image,
    sample_video_frames,
    validate_file,
)
from src.privacy import redact_privacy_regions
from src.report import (
    build_city_dashboard_map,
    build_folium_map,
    detections_to_dataframe,
    generate_csv_report,
    generate_pdf_report,
)
from src.severity import HIGH_THRESHOLD, LOW_THRESHOLD
from src.urgency import estimate_repair_area_and_cost, get_urgency_tier


st.set_page_config(
    page_title="RoadWatch | Pothole Detection & Municipal Prioritization",
    page_icon="🛣️",
    layout="wide",
)

# Initialize SQLite database and seed realistic sample records if empty
init_db()
seed_demo_inspections_if_empty()


def create_synthetic_demo_image(variant: int = 0) -> np.ndarray:
    """
    Create a realistic synthetic paved road image with simulated potholes
    so users can test the entire pipeline within 1 click without uploading a file.
    """
    h, w = 600, 900
    rng = np.random.default_rng(42 + variant)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.22)
    img[:horizon, :] = (185, 205, 225)

    asphalt_base = np.full((h - horizon, w, 3), 115, dtype=np.int16)
    grain = rng.integers(-12, 13, size=(h - horizon, w, 1), dtype=np.int16)
    asphalt = np.clip(asphalt_base + grain, 70, 160).astype(np.uint8)
    img[horizon:, :] = asphalt

    # Road lane markings
    cv2.line(img, (80, h - 10), (340, horizon + 20), (220, 220, 220), 5)
    cv2.line(img, (w - 80, h - 10), (w - 340, horizon + 20), (220, 220, 220), 5)

    potholes = [
        ((380, 390), (85, 42), 12),
        ((570, 470), (65, 34), -8),
        ((310, 260), (42, 22), 5),
    ]
    for center, axes, angle in potholes:
        cv2.ellipse(img, center, (axes[0] + 6, axes[1] + 5), angle, 0, 360, (68, 68, 70), -1)
        cv2.ellipse(img, center, axes, angle, 0, 360, (24, 24, 26), -1)
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
    st.sidebar.title("⚙️ RoadWatch Controls")
    conf_thresh = st.sidebar.slider(
        "Detector Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Minimum confidence score to report a pothole bounding box.",
    )

    st.sidebar.subheader("📏 Severity Area Thresholds")
    low_pct = st.sidebar.slider(
        "Low → Medium (% of image area)",
        min_value=0.5,
        max_value=5.0,
        value=LOW_THRESHOLD * 100.0,
        step=0.5,
    )
    high_pct = st.sidebar.slider(
        "Medium → High (% of image area)",
        min_value=3.0,
        max_value=20.0,
        value=HIGH_THRESHOLD * 100.0,
        step=0.5,
    )

    st.sidebar.subheader("🔒 Privacy & Explainability")
    show_gradcam = st.sidebar.checkbox("Show Stage A Grad-CAM Heatmap", value=True)
    privacy_blur = st.sidebar.checkbox(
        "Auto-Blur Faces & License Plates (Ethics Sec. 13)",
        value=True,
        help="Automatically redacts human faces and vehicle registration plates before display and report export.",
    )
    auto_save_db = st.sidebar.checkbox(
        "Log Inspection to City SQLite Database",
        value=True,
        help="Automatically saves the inspection to the City Map & History database.",
    )

    service = get_model_service()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Active Detector Backend:**\n`{service.backend_name}`")
    st.sidebar.caption("**REST API Docs:** `http://localhost:8000/docs` (`uvicorn api:app`)")
    return (
        conf_thresh,
        low_pct / 100.0,
        max(low_pct + 0.5, high_pct) / 100.0,
        show_gradcam,
        privacy_blur,
        auto_save_db,
    )


def render_single_inspection(
    rgb_img: np.ndarray,
    source_name: str,
    road_name: str,
    ward_name: str,
    conf_thresh: float,
    low_thresh: float,
    high_thresh: float,
    show_gradcam: bool,
    privacy_blur: bool,
    auto_save_db: bool,
    gps_coords: Optional[Dict[str, float]] = None,
):
    faces_blurred, plates_blurred = 0, 0
    if privacy_blur:
        rgb_img, faces_blurred, plates_blurred = redact_privacy_regions(rgb_img)

    service = get_model_service()
    res = service.predict(
        rgb_img,
        conf_threshold=conf_thresh,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    summary = res.severity_summary
    tier_label, rec_action, badge_hex = get_urgency_tier(
        summary.overall_severity, summary.priority_score
    )
    est_sqm, est_cost = estimate_repair_area_and_cost(summary.total_area_ratio)

    if auto_save_db:
        lat = gps_coords.get("latitude") if gps_coords else None
        lon = gps_coords.get("longitude") if gps_coords else None
        save_inspection(
            summary=summary,
            source_name=source_name,
            road_name=road_name,
            ward_name=ward_name,
            classifier_prob=res.classifier_pothole_prob,
            inference_time_ms=res.inference_time_ms,
            latitude=lat,
            longitude=lon,
        )

    # Top KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Potholes Detected", summary.pothole_count)
    k2.metric("Affected Area Ratio", f"{summary.total_area_ratio * 100:.2f}%")
    k3.metric("Overall Severity", summary.overall_severity)
    k4.metric("Priority Score", f"{summary.priority_score:.1f} / 100")
    k5.metric("Est. Patch Cost", f"₹{est_cost:,.0f}", f"~{est_sqm:.2f} m²")
    k6.metric("CPU Inference Time", f"{res.inference_time_ms:.0f} ms")

    # Actionable Engineering Urgency Banner
    st.markdown(
        f"""
        <div style="padding:12px 16px; border-radius:8px; border-left:6px solid {badge_hex}; background-color:rgba(128,128,128,0.08); margin-bottom:14px;">
            <b>🛠️ Municipal Maintenance SLA Tier:</b> <span style="color:{badge_hex}; font-weight:700;">{tier_label}</span><br>
            <b>Recommended Civil Action:</b> {rec_action}
            {f'<br><small>🔒 Privacy Redaction Active: {faces_blurred} face(s) and {plates_blurred} plate region(s) blurred.</small>' if privacy_blur else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Panels
    if show_gradcam:
        c1, c2, c3 = st.columns(3)
        c1.subheader("1. Preprocessed Road Image")
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
            caption=f"{res.classifier_label} (Prob: {res.classifier_pothole_prob * 100:.1f}%)",
            use_container_width=True,
        )
    else:
        c1, c2 = st.columns(2)
        c1.subheader("Preprocessed Road Image")
        c1.image(rgb_img, use_container_width=True)
        c2.subheader("Annotated Pothole Detection")
        c2.image(
            res.annotated_rgb,
            caption=f"{summary.pothole_count} pothole(s) | Overall: {summary.overall_severity}",
            use_container_width=True,
        )

    st.markdown("---")
    col_table, col_dl = st.columns([2, 1])

    with col_table:
        st.subheader("📋 Per-Pothole Bounding Box Breakdown")
        df = detections_to_dataframe(summary)
        if df.empty:
            st.success("No potholes detected above the selected confidence threshold.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with col_dl:
        st.subheader("📥 Export Official Report")
        csv_data = generate_csv_report(
            summary=summary,
            source_name=source_name,
            road_name=road_name,
            ward_name=ward_name,
            gps_coords=gps_coords,
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
                road_name=road_name,
                ward_name=ward_name,
                classifier_prob=res.classifier_pothole_prob,
                inference_time_ms=res.inference_time_ms,
                gps_coords=gps_coords,
                annotated_rgb=res.annotated_rgb,
                gradcam_rgb=res.gradcam_overlay_rgb if show_gradcam else None,
            )
            st.download_button(
                label="📑 Download Municipal PDF Report (with Images)",
                data=pdf_bytes,
                file_name=f"roadwatch_report_{Path(source_name).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF generation unavailable: {exc}")

    if gps_coords:
        st.markdown("---")
        st.subheader("🗺️ Geotagged Inspection Location")
        st.caption(
            f"**Road:** {road_name} ({ward_name}) | **Latitude:** `{gps_coords['latitude']}` | **Longitude:** `{gps_coords['longitude']}`"
        )
        try:
            from streamlit_folium import st_folium

            fmap = build_folium_map(
                gps_coords, summary, source_name=source_name, road_name=road_name
            )
            st_folium(fmap, width=850, height=360, returned_objects=[])
        except Exception:
            st.map(pd.DataFrame([{"lat": gps_coords["latitude"], "lon": gps_coords["longitude"]}]))


def main():
    st.title("🛣️ RoadWatch: Deep Learning Pothole Detection & Municipal Prioritization")
    st.markdown(
        "**Turn any road photo or video into an instant, explainable engineering report — "
        "combining Stage A MobileNetV2 + Grad-CAM, Stage B YOLOv8 localization, and Stage C Severity & Priority Scoring.**"
    )

    (
        conf_thresh,
        low_thresh,
        high_thresh,
        show_gradcam,
        privacy_blur,
        auto_save_db,
    ) = render_sidebar()

    tabs = st.tabs(
        [
            "📷 Single Road Inspection",
            "🗂️ Batch Ward Inspector",
            "🎥 Video Frame Analysis",
            "🗺️ City Map & History DB",
            "📊 Model Evaluation & Error Analysis",
        ]
    )

    # -----------------------------------------------------------------------
    # TAB 1: Single Road Image Inspection
    # -----------------------------------------------------------------------
    with tabs[0]:
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns([2, 2, 1.5, 1.5])
        road_name = meta_c1.text_input("Road Segment Name", value="MG Road Eastbound")
        ward_name = meta_c2.text_input("Municipal Ward / Zone", value="Ward 111 - Shantala Nagar")
        manual_lat = meta_c3.number_input(
            "Fallback Latitude (if no EXIF)", value=12.9716, format="%.4f"
        )
        manual_lon = meta_c4.number_input(
            "Fallback Longitude (if no EXIF)", value=77.5946, format="%.4f"
        )

        col_up, col_demo = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload a road surface image (JPG, JPEG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
            )
        with col_demo:
            st.write("")
            st.write("")
            run_demo = st.button("🚀 1-Click Sample Road Demo", use_container_width=True)

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            valid, msg = validate_file(uploaded_file.name, len(raw_bytes), allow_video=False)
            if not valid:
                st.error(msg)
            else:
                rgb_img, _ = load_and_preprocess_image(raw_bytes)
                exif_gps = extract_gps_exif(raw_bytes)
                gps = exif_gps if exif_gps else {"latitude": float(manual_lat), "longitude": float(manual_lon)}
                render_single_inspection(
                    rgb_img=rgb_img,
                    source_name=uploaded_file.name,
                    road_name=road_name,
                    ward_name=ward_name,
                    conf_thresh=conf_thresh,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                    show_gradcam=show_gradcam,
                    privacy_blur=privacy_blur,
                    auto_save_db=auto_save_db,
                    gps_coords=gps,
                )
        elif run_demo:
            demo_img = create_synthetic_demo_image()
            render_single_inspection(
                rgb_img=demo_img,
                source_name="sample_road_demo.jpg",
                road_name=road_name,
                ward_name=ward_name,
                conf_thresh=conf_thresh,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                show_gradcam=show_gradcam,
                privacy_blur=privacy_blur,
                auto_save_db=auto_save_db,
                gps_coords={"latitude": float(manual_lat), "longitude": float(manual_lon)},
            )
        else:
            st.info(
                "👆 Upload a road photo above or click **1-Click Sample Road Demo** to get an annotated inspection report within 3 clicks."
            )

    # -----------------------------------------------------------------------
    # TAB 2: Batch Road Segment / Ward Inspector
    # -----------------------------------------------------------------------
    with tabs[1]:
        st.subheader("🗂️ Multi-Image Batch Road Survey & Repair Queue Ranking")
        st.markdown(
            "Upload multiple road photos from a ward survey to automatically rank every segment by **Repair Priority Score (0–100)** and estimate total patching budget."
        )
        batch_ward = st.text_input("Survey Ward / Corridor Name", value="Ward 150 - Outer Ring Road Corridor")
        batch_files = st.file_uploader(
            "Select multiple road images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        if batch_files:
            service = get_model_service()
            batch_rows = []
            for idx, bf in enumerate(batch_files):
                raw_b = bf.getvalue()
                ok, _ = validate_file(bf.name, len(raw_b), allow_video=False)
                if not ok:
                    continue
                rgb_b, _ = load_and_preprocess_image(raw_b)
                if privacy_blur:
                    rgb_b, _, _ = redact_privacy_regions(rgb_b)
                res_b = service.predict(
                    rgb_b,
                    conf_threshold=conf_thresh,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                )
                s_b = res_b.severity_summary
                tier_b, _, _ = get_urgency_tier(s_b.overall_severity, s_b.priority_score)
                sqm_b, cost_b = estimate_repair_area_and_cost(s_b.total_area_ratio)
                gps_b = extract_gps_exif(raw_b)

                if auto_save_db:
                    save_inspection(
                        summary=s_b,
                        source_name=bf.name,
                        road_name=f"Segment #{idx + 1} ({bf.name})",
                        ward_name=batch_ward,
                        classifier_prob=res_b.classifier_pothole_prob,
                        inference_time_ms=res_b.inference_time_ms,
                        latitude=gps_b["latitude"] if gps_b else None,
                        longitude=gps_b["longitude"] if gps_b else None,
                    )

                batch_rows.append(
                    {
                        "File Name": bf.name,
                        "Potholes": s_b.pothole_count,
                        "Affected Area (%)": round(s_b.total_area_ratio * 100.0, 2),
                        "Severity": s_b.overall_severity,
                        "Priority Score": s_b.priority_score,
                        "Urgency SLA": tier_b,
                        "Est. Area (m²)": sqm_b,
                        "Est. Cost (INR)": cost_b,
                        "Latency (ms)": res_b.inference_time_ms,
                    }
                )

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows).sort_values(
                    by="Priority Score", ascending=False
                )
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Images Processed", len(batch_df))
                b2.metric("Total Potholes Found", int(batch_df["Potholes"].sum()))
                b3.metric(
                    "High-Severity Segments",
                    int((batch_df["Severity"] == "High").sum()),
                )
                b4.metric(
                    "Total Est. Patching Budget",
                    f"₹{batch_df['Est. Cost (INR)'].sum():,.0f}",
                )

                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Ranked Batch Survey CSV",
                    data=batch_df.to_csv(index=False),
                    file_name="roadwatch_batch_ward_survey.csv",
                    mime="text/csv",
                )

    # -----------------------------------------------------------------------
    # TAB 3: Video Frame Sampling Analysis
    # -----------------------------------------------------------------------
    with tabs[2]:
        st.subheader("🎥 Dashcam / Phone Road Video Frame Sampling")
        vid_file = st.file_uploader(
            "Upload a short road video clip (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
        )
        every_n = st.slider("Sample 1 frame every N video frames", 5, 60, 15, step=5)
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
                        if privacy_blur:
                            rgb_frame, _, _ = redact_privacy_regions(rgb_frame)
                        res = service.predict(
                            rgb_frame,
                            conf_threshold=conf_thresh,
                            low_thresh=low_thresh,
                            high_thresh=high_thresh,
                        )
                        s = res.severity_summary
                        tier_v, _, _ = get_urgency_tier(s.overall_severity, s.priority_score)
                        rows.append(
                            {
                                "Frame #": f_idx,
                                "Timestamp (s)": ts_sec,
                                "Potholes": s.pothole_count,
                                "Area Ratio (%)": round(s.total_area_ratio * 100.0, 2),
                                "Severity": s.overall_severity,
                                "Priority Score": s.priority_score,
                                "Urgency Tier": tier_v,
                            }
                        )
                        if s.total_area_ratio > worst_ratio:
                            worst_ratio = s.total_area_ratio
                            worst_frame_res = (f_idx, ts_sec, res)

                    st.subheader(f"📊 Video Timeline Summary ({len(frames)} Sampled Frames)")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    if worst_frame_res is not None:
                        f_idx, ts_sec, best_res = worst_frame_res
                        st.subheader(
                            f"🚨 Peak Severity Frame (Frame #{f_idx} @ {ts_sec}s — "
                            f"{best_res.severity_summary.overall_severity})"
                        )
                        st.image(best_res.annotated_rgb, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 4: City-Wide Inspection Map & SQLite History Log
    # -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("🗺️ Municipal Pothole Prioritization Map & SQLite History")
        records = list_inspections(limit=200)
        hist_df = inspections_to_dataframe()

        top_c1, top_c2, top_c3, top_c4 = st.columns(4)
        top_c1.metric("Logged Road Inspections", len(records))
        total_potholes_logged = sum(int(r["pothole_count"]) for r in records)
        top_c2.metric("Total Potholes Mapped", total_potholes_logged)
        high_sev_count = sum(1 for r in records if r["overall_severity"] == "High")
        top_c3.metric("High-Urgency Road Segments (<48h)", high_sev_count)
        total_budget = sum(float(r["estimated_cost_inr"]) for r in records)
        top_c4.metric("Cumulative Est. Repair Budget", f"₹{total_budget:,.0f}")

        try:
            from streamlit_folium import st_folium

            city_map = build_city_dashboard_map(records)
            st_folium(city_map, width=1050, height=430, returned_objects=[])
        except Exception:
            geo_pts = [
                {"lat": float(r["latitude"]), "lon": float(r["longitude"])}
                for r in records
                if r.get("latitude") is not None and r.get("longitude") is not None
            ]
            if geo_pts:
                st.map(pd.DataFrame(geo_pts))

        st.markdown("#### 📋 Prioritized Municipal Work-Order Queue (Sorted by Priority Score)")
        if not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            btn_c1, btn_c2 = st.columns([1, 1])
            btn_c1.download_button(
                "📥 Export Full Municipal Database (CSV)",
                data=hist_df.to_csv(index=False),
                file_name="roadwatch_municipal_history.csv",
                mime="text/csv",
            )
            if btn_c2.button("🗑️ Reset & Re-Seed Demo Database"):
                delete_all_inspections()
                seed_demo_inspections_if_empty()
                st.rerun()

    # -----------------------------------------------------------------------
    # TAB 5: Model Evaluation, Experiments & Error Analysis (Section 8)
    # -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("📊 Section 8: ML Evaluation, Ablation Studies & Error Analysis")
        st.markdown(
            "This dashboard summarizes the four required experiments from **Section 8.5** and the **Error Analysis Taxonomy (Section 8.4)** for Review 3 and Exhibition viva."
        )

        st.markdown("#### 1. Classifier vs. Detector & Augmentation Ablation (Experiments 1, 2 & 3)")
        comp_df = get_model_comparison_df()
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("#### 2. Effect of Confidence Threshold on Precision, Recall & F1 (Experiment 4)")
        sweep_df = get_confidence_threshold_sweep_df()
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            st.line_chart(
                sweep_df.set_index("Confidence Threshold")[["Precision", "Recall", "F1-Score"]]
            )
        with sc2:
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)
            st.caption(
                "**Why 0.35 is default:** Recall (`0.81`) is prioritized over precision (`0.80`) because missing a dangerous pothole is more harmful than a false alarm (Section 8.3)."
            )

        st.markdown("#### 3. Error Analysis Taxonomy & Engineering Mitigations (Section 8.4)")
        err_df = pd.DataFrame(get_error_analysis_taxonomy())
        st.dataframe(err_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
