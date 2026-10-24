"""Change detection thresholding and ecological categorization.

Categorizes differential vegetation indices (dNDVI) into standard ecological classes:
- Severe Loss (potential deforestation, clear-cut, wildfire)
- Moderate Loss (canopy thinning, drought, seasonal dieback)
- Relatively Stable (mature canopy, seasonal equilibrium)
- Moderate Gain (canopy regrowth, post-disturbance recovery)
- Significant Gain (vigorous greening, reforestation)
"""

from typing import Any

import numpy as np

# Change classes metadata definition
CHANGE_CLASSES: dict[int, dict[str, Any]] = {
    1: {
        "name": "Severe Loss",
        "description": "Substantial canopy removal (dNDVI < -0.25). Corresponds to clear-cutting, fire scar, or intensive clearing.",
        "color": "#D90429",  # Crimson
        "rgba": (217, 4, 41, 230),
    },
    2: {
        "name": "Moderate Loss",
        "description": "Noticeable vegetation thinning (-0.25 <= dNDVI < -0.10). Corresponds to selective logging, drought, or seasonal dieback.",
        "color": "#F77F00",  # Amber
        "rgba": (247, 127, 0, 220),
    },
    3: {
        "name": "Relatively Stable",
        "description": "Vegetation index remains within normal baseline fluctuation (-0.10 <= dNDVI <= +0.10).",
        "color": "#ADB5BD",  # Neutral Slate Gray
        "rgba": (173, 181, 189, 140),
    },
    4: {
        "name": "Moderate Gain",
        "description": "Vegetation index increase (+0.10 < dNDVI <= +0.25). Corresponds to crop growth, pioneer regrowth, or post-disturbance recovery.",
        "color": "#52B788",  # Medium Green
        "rgba": (82, 183, 136, 220),
    },
    5: {
        "name": "Significant Gain",
        "description": "Vigorous canopy expansion (dNDVI > +0.25). Corresponds to intensive reforestation, bamboo expansion, or agricultural green-up.",
        "color": "#1B4332",  # Deep Forest Green
        "rgba": (27, 67, 50, 240),
    },
    0: {
        "name": "Masked / Nodata",
        "description": "Cloud, shadow, permanent water, or missing observation pixels.",
        "color": "#000000",
        "rgba": (0, 0, 0, 0),
    },
}


def classify_dndvi(
    dndvi_array: np.ndarray,
    severe_loss_threshold: float = -0.25,
    moderate_loss_threshold: float = -0.10,
    moderate_gain_threshold: float = 0.10,
    significant_gain_threshold: float = 0.25,
) -> tuple[np.ndarray, dict[int, int]]:
    """Classify continuous dNDVI values into discrete ecological change classes.

    Args:
        dndvi_array: 2D array of dNDVI values (NaN for invalid/nodata).
        severe_loss_threshold: Threshold below which change is classified as Severe Loss.
        moderate_loss_threshold: Upper threshold for Moderate Loss.
        moderate_gain_threshold: Lower threshold for Moderate Gain.
        significant_gain_threshold: Lower threshold for Significant Gain.

    Returns:
        Tuple of:
            - class_raster: 2D uint8 array with class IDs (0 to 5).
            - pixel_counts: Dictionary mapping class ID to integer pixel count.
    """
    if severe_loss_threshold >= moderate_loss_threshold:
        raise ValueError("severe_loss_threshold must be strictly less than moderate_loss_threshold")
    if moderate_gain_threshold >= significant_gain_threshold:
        raise ValueError("moderate_gain_threshold must be strictly less than significant_gain_threshold")

    dndvi = np.asarray(dndvi_array, dtype=np.float32)
    classes = np.zeros(dndvi.shape, dtype=np.uint8)

    valid = np.isfinite(dndvi)

    # Class 1: Severe Loss
    classes[valid & (dndvi < severe_loss_threshold)] = 1

    # Class 2: Moderate Loss
    classes[valid & (dndvi >= severe_loss_threshold) & (dndvi < moderate_loss_threshold)] = 2

    # Class 3: Relatively Stable
    classes[valid & (dndvi >= moderate_loss_threshold) & (dndvi <= moderate_gain_threshold)] = 3

    # Class 4: Moderate Gain
    classes[valid & (dndvi > moderate_gain_threshold) & (dndvi <= significant_gain_threshold)] = 4

    # Class 5: Significant Gain
    classes[valid & (dndvi > significant_gain_threshold)] = 5

    # Compute pixel counts
    pixel_counts = {cid: int((classes == cid).sum()) for cid in CHANGE_CLASSES}

    return classes, pixel_counts


def create_change_rgba(class_raster: np.ndarray) -> np.ndarray:
    """Render an RGBA uint8 image (H, W, 4) from a class raster for visual mapping."""
    h, w = class_raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for cid, meta in CHANGE_CLASSES.items():
        mask = class_raster == cid
        rgba[mask] = meta["rgba"]

    return rgba
