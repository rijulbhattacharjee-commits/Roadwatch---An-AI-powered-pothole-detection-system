"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for RoadWatch.

Provides visual explanations (heatmaps) showing which road regions influenced
the Stage A MobileNetV2 classifier's prediction. Includes a lightweight
OpenCV texture/dark-depression saliency fallback when PyTorch classifier weights
are not yet trained or loaded.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def generate_gradcam_heatmap(
    rgb_image: np.ndarray,
    model=None,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a Grad-CAM heatmap and blended overlay for an RGB road image.

    If a PyTorch MobileNetV2 `model` is provided, computes true Grad-CAM gradients
    on the final convolutional block (`model.features[-1]`).
    Otherwise, computes a deterministic structural/dark-depression saliency map
    as an explainable visual baseline until classifier weights are trained.

    Args:
        rgb_image: Input image in RGB format (H, W, 3), uint8.
        model: Optional PyTorch MobileNetV2 model.
        target_class: Optional target class index (defaults to predicted or pothole class 1).
        alpha: Heatmap overlay opacity (0.0 to 1.0).

    Returns:
        (overlay_rgb, heatmap_rgb, pothole_probability)
    """
    h, w = rgb_image.shape[:2]

    if model is not None:
        try:
            import torch
            import torch.nn.functional as F

            model.eval()
            device = next(model.parameters()).device

            # Standard ImageNet preprocessing for MobileNetV2 (224x224)
            resized = cv2.resize(rgb_image, (224, 224)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (resized - mean) / std
            tensor = (
                torch.from_numpy(norm.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
                .requires_grad_(True)
            )

            activations = []
            gradients = []

            target_layer = model.features[-1]
            handle_fwd = target_layer.register_forward_hook(
                lambda m, i, o: activations.append(o)
            )
            handle_bwd = target_layer.register_full_backward_hook(
                lambda m, gi, go: gradients.append(go[0])
            )

            try:
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pothole_prob = float(probs[0, 1].item()) if probs.shape[1] > 1 else float(probs[0, 0].item())

                cls_idx = target_class if target_class is not None else (1 if logits.shape[1] > 1 else 0)
                model.zero_grad()
                logits[0, cls_idx].backward()

                grads = gradients[0].detach().cpu().numpy()[0]      # (C, H', W')
                acts = activations[0].detach().cpu().numpy()[0]       # (C, H', W')
                weights = np.mean(grads, axis=(1, 2))                 # (C,)

                cam = np.zeros(acts.shape[1:], dtype=np.float32)
                for c_idx, weight in enumerate(weights):
                    cam += weight * acts[c_idx]

                cam = np.maximum(cam, 0)
                if cam.max() > 0:
                    cam = cam / cam.max()

                cam_resized = cv2.resize(cam, (w, h))
                heatmap_uint8 = np.uint8(255 * cam_resized)
                heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                overlay_rgb = cv2.addWeighted(
                    rgb_image, 1.0 - alpha, heatmap_rgb, alpha, 0
                )
                return overlay_rgb, heatmap_rgb, round(pothole_prob, 4)
            finally:
                handle_fwd.remove()
                handle_bwd.remove()
        except Exception:
            # Fall back cleanly if PyTorch or hook fails
            pass

    # Fallback: Multi-scale road surface anomaly / depression saliency map
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Black-hat morphological transform highlights dark depressions against brighter road tarmac
    kernel_size = max(15, int(min(h, w) * 0.08) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    # Combine with local gradient edges (rough pothole rims)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Focus on lower 75% of the image (typical road surface region)
    spatial_prior = np.linspace(0.25, 1.0, h, dtype=np.float32)[:, None]
    saliency = (0.65 * blackhat + 0.35 * grad_mag) * spatial_prior
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=max(3.0, min(h, w) * 0.02))

    max_val = float(saliency.max())
    if max_val > 1e-5:
        saliency_norm = saliency / max_val
    else:
        saliency_norm = np.zeros_like(saliency)

    # Heuristic probability estimate from anomaly contrast
    anomaly_energy = float(np.percentile(blackhat, 98)) / 255.0
    pothole_prob = float(np.clip(anomaly_energy * 2.2, 0.05, 0.98))

    heatmap_uint8 = np.uint8(255 * saliency_norm)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_rgb, alpha, 0)

    return overlay_rgb, heatmap_rgb, round(pothole_prob, 4)
