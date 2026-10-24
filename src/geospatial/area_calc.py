"""Geodesic and projected pixel area calculations.

Converts raster pixel counts into real-world geographic surface areas (hectares and km²)
accounting for Coordinate Reference Systems (UTM planar vs WGS84 geodesic).
"""

from typing import Any

import pandas as pd
from pyproj import Geod
from rasterio.crs import CRS

from src.change_detection.classifier import CHANGE_CLASSES


def get_pixel_area_sqm(profile: dict[str, Any]) -> float:
    """Calculate the ground area of a single pixel in square meters.

    For projected CRS (e.g. UTM), computes |dx * dy|.
    For geographic CRS (e.g. EPSG:4326), computes geodesic area at the raster centroid.
    """
    transform = profile.get("transform")
    crs = profile.get("crs")

    if transform is None:
        return 100.0  # Default 10m x 10m Sentinel-2 pixel

    dx = abs(transform[0])
    dy = abs(transform[4])

    is_geographic = False
    if crs is not None:
        crs_obj = CRS.from_user_input(crs)
        is_geographic = crs_obj.is_geographic

    if is_geographic:
        # Geographic coordinates in degrees: use WGS84 ellipsoid geodesic polygon
        width = profile.get("width", 100)
        height = profile.get("height", 100)
        center_x = transform[2] + (width / 2.0) * transform[0]
        center_y = transform[5] + (height / 2.0) * transform[4]

        geod = Geod(ellps="WGS84")
        lons = [center_x, center_x + dx, center_x + dx, center_x, center_x]
        lats = [center_y, center_y, center_y - dy, center_y - dy, center_y]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area)

    # Projected coordinates in meters (UTM standard)
    return float(dx * dy)


def compute_change_area_statistics(
    pixel_counts: dict[int, int],
    profile: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute tabular area statistics and summary metrics from classified pixel counts.

    Args:
        pixel_counts: Dictionary mapping class ID (0 to 5) to pixel count.
        profile: Rasterio profile containing CRS and transform metadata.

    Returns:
        Tuple of:
            - stats_df: Pandas DataFrame with Class, Area (ha), Area (km²), Share (%).
            - summary_metrics: High-level KPIs (Total Area, Valid Area, Net Change ha).
    """
    pixel_area_sqm = get_pixel_area_sqm(profile)

    # Compute exact unrounded area values directly from pixel counts
    n_severe = pixel_counts.get(1, 0)
    n_mod_loss = pixel_counts.get(2, 0)
    n_stable = pixel_counts.get(3, 0)
    n_mod_gain = pixel_counts.get(4, 0)
    n_sig_gain = pixel_counts.get(5, 0)

    severe_loss_ha = (n_severe * pixel_area_sqm) / 10000.0
    moderate_loss_ha = (n_mod_loss * pixel_area_sqm) / 10000.0
    loss_ha = ((n_severe + n_mod_loss) * pixel_area_sqm) / 10000.0
    gain_ha = ((n_mod_gain + n_sig_gain) * pixel_area_sqm) / 10000.0
    stable_ha = (n_stable * pixel_area_sqm) / 10000.0
    net_change_ha = gain_ha - loss_ha

    # Total valid pixels (classes 1 to 5, excluding 0)
    valid_pixels = sum(pixel_counts.get(cid, 0) for cid in range(1, 6))
    total_pixels = sum(pixel_counts.values())

    total_valid_ha = (valid_pixels * pixel_area_sqm) / 10000.0
    total_study_ha = (total_pixels * pixel_area_sqm) / 10000.0

    records = []
    for cid in [1, 2, 3, 4, 5, 0]:
        count = pixel_counts.get(cid, 0)
        area_ha = (count * pixel_area_sqm) / 10000.0
        area_km2 = area_ha / 100.0

        share_valid = (count / valid_pixels * 100.0) if (cid != 0 and valid_pixels > 0) else 0.0
        share_scene = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0

        meta = CHANGE_CLASSES.get(cid, {})
        records.append(
            {
                "Class ID": cid,
                "Category": meta.get("name", f"Class {cid}"),
                "Pixel Count": count,
                "Area (ha)": round(area_ha, 2),
                "Area (km²)": round(area_km2, 4),
                "Share (%)": round(share_valid if cid != 0 else share_scene, 2),
                "Share of Valid (%)": round(share_valid, 2),
                "Share of Scene (%)": round(share_scene, 2),
                "Color": meta.get("color", "#CCCCCC"),
            }
        )

    stats_df = pd.DataFrame(records)

    summary_metrics = {
        "pixel_area_sqm": pixel_area_sqm,
        "total_study_ha": round(total_study_ha, 2),
        "total_valid_ha": round(total_valid_ha, 2),
        "valid_coverage_pct": (valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "severe_loss_ha": round(severe_loss_ha, 2),
        "moderate_loss_ha": round(moderate_loss_ha, 2),
        "total_loss_ha": round(loss_ha, 2),
        "total_gain_ha": round(gain_ha, 2),
        "stable_ha": round(stable_ha, 2),
        "net_change_ha": round(net_change_ha, 2),
    }

    return stats_df, summary_metrics
