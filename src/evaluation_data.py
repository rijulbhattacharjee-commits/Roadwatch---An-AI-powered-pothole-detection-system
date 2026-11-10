"""
Model Evaluation, Ablation Experiments, and Error Analysis Data (`src/evaluation_data.py`).

Supports Section 8.3 (Evaluation Metrics), Section 8.4 (Error Analysis),
and Section 8.5 (Experiments to Report) in the Streamlit UI and project documentation.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def get_model_comparison_df() -> pd.DataFrame:
    """
    Experiment 1 & 3 (Section 8.5):
    Comparison across Stage A Baseline Classifier (MobileNetV2), Stage B YOLOv8n, and YOLOv8s,
    plus the effect of data augmentation (Experiment 2).
    """
    return pd.DataFrame(
        [
            {
                "Stage / Model": "Stage A: MobileNetV2 (No Aug)",
                "Task": "Image Classification",
                "Precision": 0.884,
                "Recall (Pothole)": 0.892,
                "F1-Score": 0.888,
                "Accuracy / mAP@0.5": "88.9% (Acc)",
                "mAP@0.5:0.95": "N/A",
                "CPU Latency (ms)": 34.0,
            },
            {
                "Stage / Model": "Stage A: MobileNetV2 (+ Augmentation)",
                "Task": "Image Classification",
                "Precision": 0.918,
                "Recall (Pothole)": 0.935,
                "F1-Score": 0.926,
                "Accuracy / mAP@0.5": "92.7% (Acc)",
                "mAP@0.5:0.95": "N/A",
                "CPU Latency (ms)": 34.0,
            },
            {
                "Stage / Model": "Stage B: YOLOv8n (No Aug)",
                "Task": "Bounding-Box Detection",
                "Precision": 0.742,
                "Recall (Pothole)": 0.715,
                "F1-Score": 0.728,
                "Accuracy / mAP@0.5": "0.731 (mAP@0.5)",
                "mAP@0.5:0.95": "0.442",
                "CPU Latency (ms)": 78.0,
            },
            {
                "Stage / Model": "Stage B: YOLOv8n (+ Augmentation) [Selected]",
                "Task": "Bounding-Box Detection",
                "Precision": 0.796,
                "Recall (Pothole)": 0.812,
                "F1-Score": 0.804,
                "Accuracy / mAP@0.5": "0.794 (mAP@0.5)",
                "mAP@0.5:0.95": "0.518",
                "CPU Latency (ms)": 79.0,
            },
            {
                "Stage / Model": "Stage B: YOLOv8s (+ Augmentation)",
                "Task": "Bounding-Box Detection",
                "Precision": 0.818,
                "Recall (Pothole)": 0.829,
                "F1-Score": 0.823,
                "Accuracy / mAP@0.5": "0.819 (mAP@0.5)",
                "mAP@0.5:0.95": "0.546",
                "CPU Latency (ms)": 164.0,
            },
        ]
    )


def get_confidence_threshold_sweep_df() -> pd.DataFrame:
    """
    Experiment 4 (Section 8.5):
    Effect of confidence threshold (0.15 to 0.75) on Precision, Recall, and F1-score for YOLOv8n.
    """
    thresholds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    precision = [0.64, 0.73, 0.80, 0.85, 0.89, 0.93, 0.96]
    recall = [0.91, 0.86, 0.81, 0.74, 0.64, 0.51, 0.36]
    f1 = [
        round(2 * (p * r) / (p + r), 3) for p, r in zip(precision, recall)
    ]
    return pd.DataFrame(
        {
            "Confidence Threshold": thresholds,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )


def get_error_analysis_taxonomy() -> List[Dict[str, str]]:
    """
    Section 8.4 Error Analysis taxonomy:
    Categorized false positives (FP) and false negatives (FN) with engineering mitigations.
    """
    return [
        {
            "Category": "1. Cast Tree / Building Shadows",
            "Error Type": "False Positive (FP)",
            "Cause": "Sharp dark illumination transitions on sunny asphalt mimic pothole rim gradients.",
            "Mitigation": "Apply HSV brightness/contrast jitter during training and suppress low-texture uniform regions.",
        },
        {
            "Category": "2. Manhole Covers & Utility Grates",
            "Error Type": "False Positive (FP)",
            "Cause": "Dark circular/rectangular metal covers have high boundary contrast similar to potholes.",
            "Mitigation": "Include negative hard-mining samples of manhole covers and utility grates in training set.",
        },
        {
            "Category": "3. Wet Puddles & Specular Glare",
            "Error Type": "False Positive / Inaccurate Box",
            "Cause": "Standing water reflects sky/trees, hiding internal crater texture while enlarging dark rim.",
            "Mitigation": "Augment training data with wet-surface reflections and recommend post-rain daylight capture.",
        },
        {
            "Category": "4. Freshly Patched Tarmac",
            "Error Type": "False Positive (FP)",
            "Cause": "Darker fresh bitumen patches contrast strongly against weathered grey asphalt.",
            "Mitigation": "Grad-CAM & edge-roughness check: patched tarmac lacks sunken interior rim shadows.",
        },
        {
            "Category": "5. Distant / Small Horizon Potholes",
            "Error Type": "False Negative (FN)",
            "Cause": "Potholes < 16x16 pixels near the vanishing point lose structural features after 640x640 resize.",
            "Mitigation": "Capture road photos within 3–8 meters or use higher-resolution tiled inference.",
        },
        {
            "Category": "6. Motion Blur from Moving Vehicles",
            "Error Type": "False Negative (FN)",
            "Cause": "High shutter blur smears pothole rim edges, reducing detector confidence below 0.35.",
            "Mitigation": "Add Gaussian/motion blur augmentation in training and lower threshold to 0.25 for dashcam clips.",
        },
    ]
