"""
Engineering Urgency Tiers, Repair Action Recommendations, and Cost Estimation
extend Stage C of RoadWatch.
"""

from typing import Tuple


def get_urgency_tier(severity: str, priority_score: float) -> Tuple[str, str, str]:
    """
    Map overall road severity and priority score (0-100) to an actionable municipal
    maintenance SLA tier, recommended action, and UI badge color.

    Returns:
        (tier_label, recommended_action, badge_color_hex)
    """
    if severity == "None" or priority_score <= 0.0:
        return (
            "Routine Monitoring",
            "No immediate pothole intervention required; include in standard seasonal survey.",
            "#2ecc71",
        )
    if severity == "High" or priority_score >= 65.0:
        return (
            "Immediate Action (< 48 hrs)",
            "Cordoning / warning signage recommended; schedule hot-mix asphalt patching within 48 hours.",
            "#e74c3c",
        )
    if severity == "Medium" or priority_score >= 30.0:
        return (
            "Scheduled Maintenance (1–2 Weeks)",
            "Prioritize in ward maintenance queue; cold-mix or semi-permanent patch repair within 14 days.",
            "#f39c12",
        )
    return (
        "Preventive Seal & Patch (Monthly Queue)",
        "Crack sealing or surface dressing to prevent monsoon water ingress and expansion.",
        "#3498db",
    )


def estimate_repair_area_and_cost(
    total_area_ratio: float,
    assumed_road_view_sqm: float = 18.0,
    cost_per_sqm_inr: float = 1450.0,
) -> Tuple[float, float]:
    """
    Provide an indicative (decision-support) surface repair area (m²) and estimated
    patching material/labor cost in INR based on assumed camera road footprint.

    Args:
        total_area_ratio: Summed pothole bounding-box ratio (0.0 to 1.0).
        assumed_road_view_sqm: Approximate visible road surface area in frame (default 18 m²).
        cost_per_sqm_inr: Approximate municipal bituminous patching rate per m² in INR.

    Returns:
        (estimated_damaged_sqm, estimated_cost_inr)
    """
    if total_area_ratio <= 0.0:
        return 0.0, 0.0
    # Apply a 0.72 packing factor because elliptical potholes occupy ~72% of their bounding box
    est_sqm = max(0.05, total_area_ratio * assumed_road_view_sqm * 0.72)
    est_cost = est_sqm * cost_per_sqm_inr
    return round(est_sqm, 2), round(est_cost, 0)
