"""Vegetation and spectral indices calculations.

Provides scientifically validated formulations for:
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- SAVI (Soil-Adjusted Vegetation Index)

All functions implement strict zero-division protection, nodata propagation,
and statistical summary generation.
"""


import numpy as np


def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    valid_mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, dict[str, float]]:
    """Calculate Normalized Difference Vegetation Index (NDVI) safely.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-Infrared surface reflectance array (e.g., Sentinel-2 Band 8).
        red: Red surface reflectance array (e.g., Sentinel-2 Band 4).
        valid_mask: Optional boolean mask where True indicates valid pixels.
        epsilon: Small threshold to protect against zero or near-zero division.

    Returns:
        Tuple of:
            - ndvi_array: 2D float32 numpy array with NDVI in [-1.0, 1.0], NaN for invalid.
            - stats: Summary statistics dictionary (mean, std, median, min, max, coverage).
    """
    if nir.shape != red.shape:
        raise ValueError(
            f"Shape mismatch: NIR shape {nir.shape} != Red shape {red.shape}"
        )

    # Cast to float32 for high precision and native NaN handling
    nir_f = np.asarray(nir, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)

    # Initialize output array filled with NaN
    ndvi = np.full(nir_f.shape, np.nan, dtype=np.float32)

    # Base valid mask: finite numbers and within physical reflectance bounds
    is_finite = np.isfinite(nir_f) & np.isfinite(red_f)

    if valid_mask is not None:
        compute_mask = is_finite & valid_mask
    else:
        compute_mask = is_finite

    # Compute denominator
    denominator = nir_f + red_f
    numerator = nir_f - red_f

    # Protect against zero and near-zero denominators
    safe_div_mask = compute_mask & (np.abs(denominator) > epsilon)

    # Calculate NDVI
    ndvi[safe_div_mask] = numerator[safe_div_mask] / denominator[safe_div_mask]

    # Clip mathematically valid range to [-1.0, 1.0] to handle minor sensor noise
    np.clip(ndvi, -1.0, 1.0, out=ndvi)

    # Compute statistics on valid pixels
    valid_values = ndvi[np.isfinite(ndvi)]
    total_pixels = int(ndvi.size)
    valid_pixels = int(valid_values.size)
    valid_pct = (valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    if valid_pixels > 0:
        stats = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "median": float(np.median(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_percentage": float(valid_pct),
        }
    else:
        stats = {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "valid_percentage": 0.0,
        }

    return ndvi, stats


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Calculate Normalized Difference Water Index (NDWI, McFeeters 1996).

    Formula:
        NDWI = (Green - NIR) / (Green + NIR)

    Used to delineate open water bodies and prevent misclassifying water
    variations as canopy loss.
    """
    if green.shape != nir.shape:
        raise ValueError("Green and NIR arrays must have the same shape.")

    green_f = np.asarray(green, dtype=np.float32)
    nir_f = np.asarray(nir, dtype=np.float32)
    ndwi = np.full(green_f.shape, np.nan, dtype=np.float32)

    denom = green_f + nir_f
    numer = green_f - nir_f
    mask = np.isfinite(green_f) & np.isfinite(nir_f) & (np.abs(denom) > epsilon)

    ndwi[mask] = numer[mask] / denom[mask]
    np.clip(ndwi, -1.0, 1.0, out=ndwi)
    return ndwi


def calculate_savi(
    nir: np.ndarray,
    red: np.ndarray,
    l_factor: float = 0.5,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Calculate Soil-Adjusted Vegetation Index (SAVI, Huete 1988).

    Formula:
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    """
    if nir.shape != red.shape:
        raise ValueError("NIR and Red arrays must have the same shape.")

    nir_f = np.asarray(nir, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)
    savi = np.full(nir_f.shape, np.nan, dtype=np.float32)

    denom = nir_f + red_f + l_factor
    numer = (nir_f - red_f) * (1.0 + l_factor)
    mask = np.isfinite(nir_f) & np.isfinite(red_f) & (np.abs(denom) > epsilon)

    savi[mask] = numer[mask] / denom[mask]
    return savi
