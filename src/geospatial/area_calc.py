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

    records = []
    # Total valid pixels (classes 1 to 5, excluding 0)
    valid_pixels = sum(pixel_counts.get(cid, 0) for cid in range(1, 6))
    total_pixels = sum(pixel_counts.values())

    for cid in [1, 2, 3, 4, 5, 0]:
        count = pixel_counts.get(cid, 0)
        area_ha = (count * pixel_area_sqm) / 10000.0
        area_km2 = area_ha / 100.0

        if cid != 0 and valid_pixels > 0:
            share_pct = (count / valid_pixels) * 100.0
        elif cid == 0 and total_pixels > 0:
            share_pct = (count / total_pixels) * 100.0
        else:
            share_pct = 0.0

        meta = CHANGE_CLASSES.get(cid, {})
        records.append(
            {
                "Class ID": cid,
                "Category": meta.get("name", f"Class {cid}"),
                "Pixel Count": count,
                "Area (ha)": round(area_ha, 2),
                "Area (km²)": round(area_km2, 4),
                "Share (%)": round(share_pct, 2),
                "Color": meta.get("color", "#CCCCCC"),
            }
        )

    stats_df = pd.DataFrame(records)

    # Calculate net canopy change: (Gains) - (Losses)
    loss_ha = stats_df[stats_df["Class ID"].isin([1, 2])]["Area (ha)"].sum()
    gain_ha = stats_df[stats_df["Class ID"].isin([4, 5])]["Area (ha)"].sum()
    net_change_ha = gain_ha - loss_ha

    severe_loss_ha = stats_df[stats_df["Class ID"] == 1]["Area (ha)"].sum()
    moderate_loss_ha = stats_df[stats_df["Class ID"] == 2]["Area (ha)"].sum()
    stable_ha = stats_df[stats_df["Class ID"] == 3]["Area (ha)"].sum()

    total_valid_ha = (valid_pixels * pixel_area_sqm) / 10000.0
    total_study_ha = (total_pixels * pixel_area_sqm) / 10000.0

    summary_metrics = {
        "pixel_area_sqm": pixel_area_sqm,
        "total_study_ha": total_study_ha,
        "total_valid_ha": total_valid_ha,
        "valid_coverage_pct": (valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "severe_loss_ha": severe_loss_ha,
        "moderate_loss_ha": moderate_loss_ha,
        "total_loss_ha": loss_ha,
        "total_gain_ha": gain_ha,
        "stable_ha": stable_ha,
        "net_change_ha": net_change_ha,
    }

    return stats_df, summary_metrics
