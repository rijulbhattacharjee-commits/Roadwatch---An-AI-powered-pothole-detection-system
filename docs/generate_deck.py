"""
Script to generate the comprehensive RoadWatch presentation (.pptx)
using python-pptx with clean typography, styled cards, and executive layout.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def build_presentation(output_path: str):
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Palette
    C_NAVY = RGBColor(26, 37, 48)       # #1A2530
    C_SLATE = RGBColor(44, 62, 80)      # #2C3E50
    C_ACCENT = RGBColor(231, 76, 60)    # #E74C3C (RoadWatch Red)
    C_ORANGE = RGBColor(243, 156, 18)   # #F39C12 (Warning Amber)
    C_GREEN = RGBColor(39, 174, 96)     # #27AE60 (Safety Green)
    C_BLUE = RGBColor(41, 128, 185)     # #2980B9 (Tech Blue)
    C_LIGHT = RGBColor(245, 247, 250)   # #F5F7FA
    C_WHITE = RGBColor(255, 255, 255)
    C_MUTED = RGBColor(127, 140, 141)   # #7F8C8D
    C_TEXT = RGBColor(40, 40, 40)

    def add_header(slide, title_text: str, category_text: str = "ROADWATCH | VTU COMMUNITY PROJECT 1BCP308"):
        # Top accent bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = C_ACCENT
        bar.line.fill.background()

        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.5), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.65))
        tf = t_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = C_NAVY

    def add_card(slide, left, top, width, height, bg_color=C_LIGHT, border_color=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.fill.background()

    # Red accent stripe
    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = C_ACCENT
    stripe.line.fill.background()

    t_box1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11), Inches(4.5))
    tf1 = t_box1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "VTU SEMESTER 3 | COMMUNITY / SOCIETAL PROJECT (1BCP308)"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = C_ORANGE
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    p_main.text = "RoadWatch: Pothole Detection & Municipal Severity Prioritization"
    p_main.font.size = Pt(32)
    p_main.font.bold = True
    p_main.font.color.rgb = C_WHITE
    p_main.space_after = Pt(10)

    p_sub = tf1.add_paragraph()
    p_sub.text = "An explainable deep learning full-stack system transforming ordinary road images into instant, geotagged civil repair intelligence."
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(189, 195, 199)
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "SDGs Addressed: SDG 9 (Infrastructure), SDG 11 (Sustainable Cities), SDG 3 (Good Health & Road Safety)\nProgramme Outcomes: PO 5 (Modern Tool Usage), PO 6 (The Engineer and Society)"
    p_meta.font.size = Pt(12)
    p_meta.font.color.rgb = C_LIGHT

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Problem: Why Traditional Road Monitoring Fails", "1. BACKGROUND & PROBLEM STATEMENT")

    cards_data_s2 = [
        ("The Societal Challenge", "• Potholes form as monsoon rainwater seeps into asphalt cracks, breaking the binder layer under heavy traffic.\n• Leads to severe vehicle suspension damage, fatal two-wheeler accidents, traffic gridlock, and inflated municipal budgets.\n• High-priority concern across Indian urban and rural arterial road networks.", C_ACCENT),
        ("Current Inspection Bottlenecks", "• Slow: Defects discovered only after citizen outrage or accidents occur.\n• Unscalable: Municipal bodies lack personnel to manually inspect hundreds of kilometers regularly.\n• Subjective: Different road inspectors assign inconsistent severity ratings, causing arbitrary budget allocation.", C_ORANGE),
        ("The RoadWatch Solution", "• Low-Cost & Software-Only: No expensive LiDAR, drone fleets, or accelerometer rigs required.\n• Instant & Objective: Upload ordinary smartphone or dashcam road photos to obtain automated bounding boxes.\n• Actionable Municipal SLA: Delivers standardized severity ratios, priority scores (0–100), and repair cost estimates.", C_GREEN)
    ]
    for i, (title, text, color) in enumerate(cards_data_s2):
        x = Inches(0.8 + i * 4.0)
        add_card(s2, x, Inches(1.6), Inches(3.7), Inches(5.2), border_color=color)
        tb = s2.shapes.add_textbox(x + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(15)
        pt.font.bold = True
        pt.font.color.rgb = color
        pt.space_after = Pt(12)
        pb = tf.add_paragraph()
        pb.text = text
        pb.font.size = Pt(11)
        pb.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 3: Objectives & Target Metrics
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Project Objectives & Benchmark Targets", "2. SPECIFICATIONS & TARGET METRICS")

    # Left: Objectives
    add_card(s3, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.2))
    tb_obj = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(6.4), Inches(4.8))
    tf_obj = tb_obj.text_frame
    tf_obj.word_wrap = True
    p = tf_obj.paragraphs[0]
    p.text = "Key Project Objectives (Plan Sec. 2.3)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(10)
    p2 = tf_obj.add_paragraph()
    p2.text = (
        "1. Literature & Gap Survey: Analyze manual, sensor, and deep-learning approaches.\n\n"
        "2. Labelled Dataset Pipeline: Clean, split (no video frame leakage), and augment public datasets.\n\n"
        "3. Two-Stage ML Architecture: Baseline MobileNetV2 classifier + YOLOv8 object detector.\n\n"
        "4. Quantitative Severity Engine: Compute bounding-box area ratios and 0-100 priority scores.\n\n"
        "5. Full-Stack Web Application: Build Streamlit interactive UI + FastAPI REST API + SQLite history.\n\n"
        "6. Societal & Ethical Compliance: Automated privacy redaction (faces/plates) and downloadable PDF/CSV reports."
    )
    p2.font.size = Pt(11.5)
    p2.font.color.rgb = C_TEXT

    # Right: Target Metrics Card
    add_card(s3, Inches(7.9), Inches(1.6), Inches(4.6), Inches(5.2), border_color=C_BLUE)
    tb_met = s3.shapes.add_textbox(Inches(8.1), Inches(1.8), Inches(4.2), Inches(4.8))
    tf_met = tb_met.text_frame
    tf_met.word_wrap = True
    pm = tf_met.paragraphs[0]
    pm.text = "Target Engineering Metrics (Plan Sec. 2.4)"
    pm.font.size = Pt(16)
    pm.font.bold = True
    pm.font.color.rgb = C_BLUE
    pm.space_after = Pt(12)
    pm2 = tf_met.add_paragraph()
    pm2.text = (
        "• Classifier Recall (Pothole Class):\n  ≥ 90% (Missing a pothole is more dangerous than a false alarm)\n\n"
        "• Classifier Accuracy:\n  ≥ 90% on held-out test split\n\n"
        "• Detector mAP@0.5:\n  ≥ 0.60 (realistic for challenging public road datasets)\n\n"
        "• CPU Inference Latency:\n  ≤ 3.0 seconds per road image\n\n"
        "• Application Usability:\n  Any user gets an actionable report within ≤ 3 clicks"
    )
    pm2.font.size = Pt(12)
    pm2.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 4: Full-Stack System Architecture
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Multi-Tier Full-Stack System Design", "3. ARCHITECTURE & WORKFLOW")

    layers = [
        ("Presentation Layer", "Streamlit 5-Tab Web UI", "Single image, multi-image batch ward survey, video frame sampling, city map, and evaluation dashboard.", C_BLUE),
        ("REST API Layer", "FastAPI (`api.py`)", "Swagger / OpenAPI endpoints (`/api/v1/analyze`, `/health`, `/inspections`) for third-party municipal platform integration.", C_NAVY),
        ("Intelligence Layer", "Two-Stage ML Pipeline", "Stage A: MobileNetV2 + Grad-CAM explainability.\nStage B: Ultralytics YOLOv8 detector + OpenCV morphological depression fallback.", C_ACCENT),
        ("Engineering Engine", "Severity, SLA & Privacy", "Stage C area ratio formula, 0-100 priority score, municipal repair SLA tier, cost estimation (INR), and face/plate blurring.", C_ORANGE),
        ("Persistence & Export", "SQLite & Reporting", "SQLite inspection history database, interactive Folium city map, and downloadable CSV and PDF reports with embedded images.", C_GREEN),
    ]
    for i, (layer_name, tech, desc, col) in enumerate(layers):
        y = Inches(1.5 + i * 1.1)
        add_card(s4, Inches(0.8), y, Inches(11.7), Inches(0.95), border_color=col)
        tb = s4.shapes.add_textbox(Inches(1.0), y + Inches(0.08), Inches(11.3), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = f"{layer_name}: {tech}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 5: What We Have Built (Current Accomplishments)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Current Progress: Full-Stack Implementation Complete", "4. WHAT HAS BEEN DONE")

    built_items = [
        ("1. Git Repository & Structure", "• Connected to GitHub remote repo.\n• Clean modular architecture adhering to Section 7.5.\n• Automated .gitignore & directory scaffolding.", C_BLUE),
        ("2. Severity & Urgency Engine", "• Relative area ratio formula (Low <2%, Med 2-8%, High >8%).\n• Priority score (0-100) combining area, count & confidence.\n• Municipal SLA tier (<48h, 1-2 wks) & INR cost estimator.", C_ACCENT),
        ("3. Explainability & Privacy", "• Grad-CAM visual heatmaps for decision transparency.\n• Automated Haar cascade face & plate redaction (Section 13).\n• Offline OpenCV depression detector fallback.", C_ORANGE),
        ("4. Full-Stack App & API", "• 5-tab Streamlit web application with 1-click synthetic demo.\n• FastAPI REST service with Swagger UI at /docs.\n• SQLite history persistence & interactive Folium map.", C_GREEN),
    ]
    for i, (title, bullets, col) in enumerate(built_items):
        col_idx = i % 2
        row_idx = i // 2
        x = Inches(0.8 + col_idx * 5.9)
        y = Inches(1.5 + row_idx * 2.7)
        add_card(s5, x, y, Inches(5.6), Inches(2.4), border_color=col)
        tb = s5.shapes.add_textbox(x + Inches(0.2), y + Inches(0.15), Inches(5.2), Inches(2.1))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(14)
        pt.font.bold = True
        pt.font.color.rgb = col
        pt.space_after = Pt(6)
        pb = tf.add_paragraph()
        pb.text = bullets
        pb.font.size = Pt(10.5)
        pb.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 6: Streamlit UI & Interactive Features
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Interactive Application Features (Streamlit UI)", "5. USER INTERFACE & CAPABILITIES")

    tabs_info = [
        ("Tab 1: Single Road Inspection", "Upload image or click 1-Click Sample Demo; view side-by-side Original, YOLOv8 Detection, and Grad-CAM Heatmap; download CSV and PDF report."),
        ("Tab 2: Batch Ward Survey", "Upload dozens of ward road photos simultaneously; automatically ranks road segments into a prioritized repair queue with cumulative patching cost."),
        ("Tab 3: Video Frame Analysis", "Upload dashcam or mobile phone MP4/AVI clips; samples 1 frame every N frames and automatically flags the peak-severity frame."),
        ("Tab 4: City Map & SQLite History", "Geotagged multi-marker Folium city map color-coded by severity, integrated with SQLite database for tracking municipal work orders over time."),
        ("Tab 5: Evaluation & Error Analysis", "Interactive tables for model comparison, augmentation ablation, confidence threshold sweep, and failure-case error taxonomy."),
    ]
    for i, (tab_title, desc) in enumerate(tabs_info):
        y = Inches(1.5 + i * 1.1)
        add_card(s6, Inches(0.8), y, Inches(11.7), Inches(0.95))
        tb = s6.shapes.add_textbox(Inches(1.0), y + Inches(0.08), Inches(11.3), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = tab_title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pb = tf.add_paragraph()
        pb.text = desc
        pb.font.size = Pt(10.5)
        pb.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 7: Severity Scoring & Municipal Repair Economics
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Mathematical Formulation: Severity, Priority & Cost", "6. CORE ALGORITHMS")

    # Left: Severity & Priority
    add_card(s7, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb_s7_1 = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_s7_1 = tb_s7_1.text_frame
    tf_s7_1.word_wrap = True
    p = tf_s7_1.paragraphs[0]
    p.text = "Severity & Priority Scoring (Plan Sec. 9.3)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(8)
    p2 = tf_s7_1.add_paragraph()
    p2.text = (
        "1. Area Ratio Formula:\n"
        "   Ratio = ∑ (Pothole Bounding Box Area) / (Image Area)\n\n"
        "2. Severity Rating Classification:\n"
        "   • Low Severity:    Ratio < 2.0%\n"
        "   • Medium Severity: 2.0% ≤ Ratio < 8.0%\n"
        "   • High Severity:   Ratio ≥ 8.0%\n\n"
        "3. Priority Score (0 to 100):\n"
        "   Score = 55% × (Area Ratio Saturation)\n"
        "         + 25% × (Pothole Count / 5)\n"
        "         + 20% × (Max Detection Confidence)\n"
        "   Normalizes road segments into a clear ranking for civic work orders."
    )
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT

    # Right: Municipal SLA & Cost
    add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_ORANGE)
    tb_s7_2 = s7.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_s7_2 = tb_s7_2.text_frame
    tf_s7_2.word_wrap = True
    pm = tf_s7_2.paragraphs[0]
    pm.text = "Municipal SLA Tiers & Cost Estimation"
    pm.font.size = Pt(15)
    pm.font.bold = True
    pm.font.color.rgb = C_ORANGE
    pm.space_after = Pt(8)
    pm2 = tf_s7_2.add_paragraph()
    pm2.text = (
        "1. Maintenance SLA Tiers:\n"
        "   • Immediate Action (< 48 hrs): High severity or Score ≥ 65.\n"
        "     Cordon off road; emergency hot-mix asphalt patching.\n"
        "   • Scheduled Maintenance (1–2 Weeks): Medium severity.\n"
        "     Standard ward queue semi-permanent patching.\n"
        "   • Preventive Seal & Patch (Monthly): Low severity.\n"
        "     Crack sealing before monsoon water ingress.\n\n"
        "2. Indicative Repair Budget Calculation:\n"
        "   • Damaged Area (m²) = Total Ratio × Assumed View Area (18 m²) × 0.72 packing factor.\n"
        "   • Patching Cost = Damaged Area × ₹1,450 / m² standard bituminous rate."
    )
    pm2.font.size = Pt(11)
    pm2.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 8: Explainability & Ethical Privacy Redaction
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Trust & Ethics: Grad-CAM Saliency and Privacy Blurring", "7. EXPLAINABILITY & ETHICS")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_BLUE)
    tb_xai = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_xai = tb_xai.text_frame
    tf_xai.word_wrap = True
    p = tf_xai.paragraphs[0]
    p.text = "Stage A Grad-CAM Explainability"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BLUE
    p.space_after = Pt(8)
    p2 = tf_xai.add_paragraph()
    p2.text = (
        "• The Black-Box Problem: Deep networks make predictions without intuitive justifications, creating resistance among civil engineers.\n\n"
        "• Grad-CAM Visual Heatmaps: Gradient-weighted Class Activation Mapping calculates gradients at MobileNetV2's final convolutional layer (`features[-1]`).\n\n"
        "• Visual Verification: Overlays a jet heatmap showing whether the network genuinely focused on sunken pothole edges rather than roadside artifacts.\n\n"
        "• Offline Morphological Fallback: Saliency map based on dark depression black-hat transforms and Sobel gradients."
    )
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT

    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_GREEN)
    tb_eth = s8.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_eth = tb_eth.text_frame
    tf_eth.word_wrap = True
    pm = tf_eth.paragraphs[0]
    pm.text = "Ethical Conduct & Privacy (Plan Sec. 13)"
    pm.font.size = Pt(15)
    pm.font.bold = True
    pm.font.color.rgb = C_GREEN
    pm.space_after = Pt(8)
    pm2 = tf_eth.add_paragraph()
    pm2.text = (
        "• Pedestrian & Driver Privacy: Public road photos frequently capture bystander faces and vehicle registration plates.\n\n"
        "• Automated Dual Redaction (`src/privacy.py`):\n"
        "   1. Face Detection: Fast Haar cascade scans and applies heavy Gaussian blur to facial regions.\n"
        "   2. Plate Detection: Plate cascade + high-contrast rectangular character-stroke morphology blurs number plates.\n\n"
        "• Compliance & Neutrality: The system is academic and non-partisan, evaluating road conditions objectively without civic blame."
    )
    pm2.font.size = Pt(11)
    pm2.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 9: Error Analysis Taxonomy (Required for Viva)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Error Analysis & Failure Modes (Plan Section 8.4)", "8. ERROR ANALYSIS & MITIGATION")

    errors = [
        ("Tree & Building Shadows", "False Positive (FP)", "Sharp dark illumination edges on sunny asphalt mimic pothole rims.", "HSV color jittering during training & texture roughness filters."),
        ("Manhole Covers & Grates", "False Positive (FP)", "Circular/rectangular dark metal covers have high contrast boundaries.", "Hard-negative mining in dataset; train explicit utility class."),
        ("Wet Puddles & Specular Glare", "FP / Inaccurate Box", "Standing water reflects sky/sun, hiding internal crater texture.", "Post-rain dataset augmentation; daylight clear capture guidance."),
        ("Fresh Bituminous Patches", "False Positive (FP)", "Dark asphalt repairs contrast strongly with weathered grey road.", "Grad-CAM check: patched tarmac lacks sunken depression shadow."),
        ("Distant / Horizon Potholes", "False Negative (FN)", "Potholes < 16x16 px lose texture features at 640x640 resolution.", "Capture road images within 3–8 meters or use SAHI tiled inference."),
        ("Vehicle Motion Blur", "False Negative (FN)", "High-speed camera shutter blur smears pothole rim edges.", "Motion blur data augmentation; lower confidence threshold to 0.25."),
    ]
    for i, (cat, etype, cause, mit) in enumerate(errors):
        col_idx = i % 2
        row_idx = i // 2
        x = Inches(0.8 + col_idx * 5.9)
        y = Inches(1.5 + row_idx * 1.8)
        bcol = C_ACCENT if "FP" in etype else C_ORANGE
        add_card(s9, x, y, Inches(5.6), Inches(1.65), border_color=bcol)
        tb = s9.shapes.add_textbox(x + Inches(0.15), y + Inches(0.1), Inches(5.3), Inches(1.45))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = f"{cat} [{etype}]"
        pt.font.size = Pt(12)
        pt.font.bold = True
        pt.font.color.rgb = bcol
        pb = tf.add_paragraph()
        pb.text = f"Cause: {cause}\nMitigation: {mit}"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 10: Roadmap & Next Steps (What We Are Planning Ahead)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Roadmap: What We Are Planning to Do Ahead", "9. PROJECT ROADMAP")

    phases = [
        ("Phase 1: Environment & Pipeline", "Nov 10 – Nov 13", "• Complete Python 3.11/3.12 environment setup with PyTorch & Ultralytics.\n• Download Kaggle & Roboflow datasets into data/raw/.\n• Run 01_eda.ipynb to verify zero cross-split leakage.", C_BLUE),
        ("Phase 2: Model Training & Tuning", "Nov 14 – Nov 22", "• Train MobileNetV2 head & fine-tune on GPU (Colab/Kaggle).\n• Train YOLOv8n detector on data.yaml for 60 epochs.\n• Tune confidence threshold on validation set (target mAP@0.5 ≥ 0.60).", C_ACCENT),
        ("Phase 3: Civil Faculty Review", "Nov 23 – Nov 27", "• Present severity formula & repair SLA to civil engineering guide.\n• Calibrate cost per m² based on local PWD/BBMP schedule of rates.\n• Finalize Review 2 presentation & prototype demo.", C_ORANGE),
        ("Phase 4: Cloud Deploy & Exhibition", "Nov 28 – Dec 9", "• Deploy live demo on Hugging Face Spaces / Streamlit Community Cloud.\n• Generate final documentation report & viva slide deck.\n• Prepare 2-minute backup demo video & exhibition poster.", C_GREEN),
    ]
    for i, (title, dates, desc, col) in enumerate(phases):
        x = Inches(0.8 + i * 3.0)
        add_card(s10, x, Inches(1.6), Inches(2.8), Inches(5.2), border_color=col)
        tb = s10.shapes.add_textbox(x + Inches(0.15), Inches(1.8), Inches(2.5), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = col
        p_d = tf.add_paragraph()
        p_d.text = dates
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_MUTED
        p_d.space_after = Pt(8)
        pb = tf.add_paragraph()
        pb.text = desc
        pb.font.size = Pt(10)
        pb.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 11: Societal Impact & Alignment to Evaluation Rubric
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Societal Impact, Syllabus Mapping & Evaluation Checklist", "10. EVALUATION & IMPACT")

    # Left: SDGs & Stakeholders
    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    tb_imp = s11.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_imp = tb_imp.text_frame
    tf_imp.word_wrap = True
    p = tf_imp.paragraphs[0]
    p.text = "Societal Impact & Beneficiaries"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(8)
    p2 = tf_imp.add_paragraph()
    p2.text = (
        "• UN Sustainable Development Goals:\n"
        "  - SDG 9 (Infrastructure): Resilient road asset management.\n"
        "  - SDG 11 (Sustainable Cities): Safer urban transit & lower commute friction.\n"
        "  - SDG 3 (Health & Well-being): Preventing two-wheeler road fatalities.\n\n"
        "• Direct Stakeholders:\n"
        "  - Municipal bodies (BBMP / PWD / Smart Cities): Objective prioritization.\n"
        "  - Commuters & Two-Wheeler Riders: Safer road transit.\n"
        "  - Resident Welfare Associations (RWAs): Verifiable evidence for civic complaints."
    )
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT

    # Right: VTU Rubric Mapping
    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), border_color=C_GREEN)
    tb_rub = s11.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_rub = tb_rub.text_frame
    tf_rub.word_wrap = True
    pm = tf_rub.paragraphs[0]
    pm.text = "VTU Course Evaluation Mapping (1BCP308)"
    pm.font.size = Pt(15)
    pm.font.bold = True
    pm.font.color.rgb = C_GREEN
    pm.space_after = Pt(8)
    pm2 = tf_rub.add_paragraph()
    pm2.text = (
        "• Review 1 (15 Marks): Problem identification, MoRTH statistics, literature review, teamwork roles.\n\n"
        "• Review 2 (15 Marks): Working full-stack prototype, baseline metrics, Grad-CAM explainability, challenges log.\n\n"
        "• Review 3 (20 Marks): Completed technical testing on held-out test split, error analysis, comprehensive report & viva.\n\n"
        "• Exhibition Criteria: High societal relevance, dual ML model, automated privacy redaction, and instant usability."
    )
    pm2.font.size = Pt(11)
    pm2.font.color.rgb = C_TEXT

    # =========================================================================
    # SLIDE 12: Conclusion & Q&A
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = C_NAVY
    bg12.line.fill.background()

    # Red accent stripe
    stripe12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5))
    stripe12.fill.solid()
    stripe12.fill.fore_color.rgb = C_ACCENT
    stripe12.line.fill.background()

    tb12 = s12.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11), Inches(4.5))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p_end_badge = tf12.paragraphs[0]
    p_end_badge.text = "ROADWATCH | COMMUNITY / SOCIETAL PROJECT (1BCP308)"
    p_end_badge.font.size = Pt(12)
    p_end_badge.font.bold = True
    p_end_badge.font.color.rgb = C_ORANGE
    p_end_badge.space_after = Pt(14)

    p_end_title = tf12.add_paragraph()
    p_end_title.text = "Thank You! Questions & Discussion"
    p_end_title.font.size = Pt(36)
    p_end_title.font.bold = True
    p_end_title.font.color.rgb = C_WHITE
    p_end_title.space_after = Pt(14)

    p_end_body = tf12.add_paragraph()
    p_end_body.text = (
        "• Live Prototype: Full-stack Streamlit App & FastAPI REST service ready for demonstration.\n"
        "• GitHub Repository: https://github.com/rijulbhattacharjee-commits/Roadwatch---An-AI-powered-pothole-detection-system\n"
        "• Open for feedback on severity thresholds, cost estimation parameters, and municipal integration."
    )
    p_end_body.font.size = Pt(14)
    p_end_body.font.color.rgb = RGBColor(220, 225, 230)

    # Save presentation
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Presentation successfully saved to: {output_path}")


if __name__ == "__main__":
    out = "roadwatch/docs/RoadWatch_Comprehensive_Presentation.pptx"
    build_presentation(out)
