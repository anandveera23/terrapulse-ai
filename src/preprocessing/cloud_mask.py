"""Cloud, shadow, and atmospheric anomaly masking.

Supports Sentinel-2 SCL (Scene Classification Layer) standard quality masks
as well as a multispectral thresholding fallback when SCL is unavailable.
"""


import numpy as np

# Sentinel-2 SCL Classes (ESA Level-2A specification)
SCL_CLASSES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOW",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW_OR_ICE",
}


def apply_scl_mask(
    scl_array: np.ndarray,
    mask_clouds: bool = True,
    mask_shadows: bool = True,
    mask_snow: bool = True,
    mask_unclassified: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate a boolean mask from Sentinel-2 SCL (Scene Classification Layer).

    Args:
        scl_array: 2D array of integer SCL class values (0 to 11).
        mask_clouds: Mask medium/high probability clouds and thin cirrus (classes 8, 9, 10).
        mask_shadows: Mask cloud shadows and cast shadows (classes 2, 3).
        mask_snow: Mask snow/ice (class 11).
        mask_unclassified: Mask unclassified pixels (class 7).

    Returns:
        Tuple of:
            - valid_mask: 2D boolean array where True indicates clear, valid pixels.
            - quality_stats: Dictionary containing cloud %, shadow %, and clear % metrics.
    """
    scl = np.asarray(scl_array, dtype=np.int32)
    total_pixels = scl.size

    # Invalid classes to exclude
    invalid_classes = {0, 1}  # Always exclude nodata and saturated/defective
    if mask_clouds:
        invalid_classes.update({8, 9, 10})
    if mask_shadows:
        invalid_classes.update({2, 3})
    if mask_snow:
        invalid_classes.add(11)
    if mask_unclassified:
        invalid_classes.add(7)

    # Boolean valid mask
    valid_mask = ~np.isin(scl, list(invalid_classes))

    # Compute quality metrics
    cloud_pixels = np.isin(scl, [8, 9, 10]).sum()
    shadow_pixels = np.isin(scl, [2, 3]).sum()
    nodata_pixels = (scl == 0).sum()
    valid_pixels = valid_mask.sum()

    stats = {
        "cloud_pct": float(cloud_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "shadow_pct": float(shadow_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "nodata_pct": float(nodata_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "clear_valid_pct": float(valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "total_pixels": int(total_pixels),
        "valid_pixels": int(valid_pixels),
    }

    return valid_mask, stats


def detect_clouds_fallback(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    blue_thresh: float = 0.22,
) -> np.ndarray:
    """Multispectral heuristic cloud and shadow detector when SCL is unavailable.

    Clouds exhibit high visible reflectance (especially Blue) and relatively flat
    spectral response, distinct from vegetated surfaces.

    Args:
        blue: Blue surface reflectance [0.0 - 1.0].
        red: Red surface reflectance [0.0 - 1.0].
        nir: NIR surface reflectance [0.0 - 1.0].
        blue_thresh: Reflectance threshold in Blue band for bright cloud detection.

    Returns:
        valid_mask: 2D boolean array where True indicates clear pixels.
    """
    blue_f = np.asarray(blue, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)
    nir_f = np.asarray(nir, dtype=np.float32)

    # Bright clouds: High blue reflectance and high visible brightness
    is_cloud = (blue_f > blue_thresh) & (red_f > 0.18)

    # Dense cloud shadows: extremely low reflectance across all bands
    is_shadow = (nir_f < 0.05) & (red_f < 0.04) & (blue_f < 0.06)

    # Valid is neither cloud, shadow, nor NaN
    is_valid = (~is_cloud) & (~is_shadow) & np.isfinite(blue_f) & np.isfinite(red_f) & np.isfinite(nir_f)
    return is_valid
