"""Differential vegetation index (dNDVI) calculation.

Computes multi-temporal pixel-wise difference rasters and statistical distribution
metrics across valid co-registered observations.
"""


import numpy as np


def compute_dndvi(
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
    valid_mask_t1: np.ndarray | None = None,
    valid_mask_t2: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute difference raster: dNDVI = NDVI_t2 - NDVI_t1.

    A negative dNDVI signifies vegetation index decline.
    A positive dNDVI signifies vegetation index recovery or greening.

    Args:
        ndvi_t1: Baseline NDVI array (Observation 1).
        ndvi_t2: Follow-up NDVI array (Observation 2).
        valid_mask_t1: Boolean mask of clear/valid pixels for T1.
        valid_mask_t2: Boolean mask of clear/valid pixels for T2.
        water_mask: Boolean mask where True indicates permanent water (excluded from loss).

    Returns:
        Tuple of:
            - dndvi: 2D float32 array with dNDVI values, NaN for invalid/masked pixels.
            - stats: Dictionary of statistical summary metrics.
    """
    if ndvi_t1.shape != ndvi_t2.shape:
        raise ValueError(
            f"Shape mismatch: T1 shape {ndvi_t1.shape} != T2 shape {ndvi_t2.shape}"
        )

    t1_f = np.asarray(ndvi_t1, dtype=np.float32)
    t2_f = np.asarray(ndvi_t2, dtype=np.float32)

    # Initialize output array with NaN
    dndvi = np.full(t1_f.shape, np.nan, dtype=np.float32)

    # Valid mask is joint intersection of both observations being finite
    common_valid = np.isfinite(t1_f) & np.isfinite(t2_f)

    if valid_mask_t1 is not None:
        common_valid &= valid_mask_t1
    if valid_mask_t2 is not None:
        common_valid &= valid_mask_t2
    if water_mask is not None:
        common_valid &= (~water_mask)

    # Compute dNDVI on valid pixels only
    dndvi[common_valid] = t2_f[common_valid] - t1_f[common_valid]

    # Clip mathematically possible range to [-2.0, 2.0]
    np.clip(dndvi, -2.0, 2.0, out=dndvi)

    valid_vals = dndvi[np.isfinite(dndvi)]
    total_pixels = int(dndvi.size)
    valid_count = int(valid_vals.size)

    if valid_count > 0:
        decreases = (valid_vals < -0.10).sum()
        increases = (valid_vals > 0.10).sum()
        stable = valid_count - (decreases + increases)

        stats = {
            "mean_dndvi": float(np.mean(valid_vals)),
            "std_dndvi": float(np.std(valid_vals)),
            "median_dndvi": float(np.median(valid_vals)),
            "min_dndvi": float(np.min(valid_vals)),
            "max_dndvi": float(np.max(valid_vals)),
            "valid_pixels": valid_count,
            "total_pixels": total_pixels,
            "decline_pixel_pct": float(decreases / valid_count * 100.0),
            "increase_pixel_pct": float(increases / valid_count * 100.0),
            "stable_pixel_pct": float(stable / valid_count * 100.0),
        }
    else:
        stats = {
            "mean_dndvi": float("nan"),
            "std_dndvi": float("nan"),
            "median_dndvi": float("nan"),
            "min_dndvi": float("nan"),
            "max_dndvi": float("nan"),
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "decline_pixel_pct": 0.0,
            "increase_pixel_pct": 0.0,
            "stable_pixel_pct": 0.0,
        }

    return dndvi, stats
