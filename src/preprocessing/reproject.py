"""Geospatial raster reprojection, grid alignment, and spatial intersection.

Ensures observation pairs share identical Coordinate Reference Systems (CRS),
spatial extents, pixel resolutions, and affine grid dimensions.
"""

from typing import Any

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject


def align_and_match_rasters(
    array1: np.ndarray,
    profile1: dict[str, Any],
    array2: np.ndarray,
    profile2: dict[str, Any],
    resampling_method: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align array2 to match the exact CRS, extent, shape, and transform of array1.

    Args:
        array1: Reference 2D or 3D numpy array (Observation 1).
        profile1: Rasterio profile dictionary for array1 (CRS, transform, width, height).
        array2: Target 2D or 3D numpy array to be warped/aligned (Observation 2).
        profile2: Rasterio profile dictionary for array2.
        resampling_method: Interpolation mode (bilinear for continuous, nearest for discrete).

    Returns:
        Tuple of:
            - aligned_array1: Reference array matching target specs.
            - aligned_array2: Warped array matching array1 shape and spatial reference.
            - unified_profile: Target rasterio profile.
    """
    src_crs = profile2.get("crs")
    dst_crs = profile1.get("crs")

    src_transform = profile2.get("transform")
    dst_transform = profile1.get("transform")

    dst_height = int(profile1.get("height", array1.shape[-2]))
    dst_width = int(profile1.get("width", array1.shape[-1]))

    # Quick check: if shapes, crs, and transforms already match, return directly
    if (
        src_crs == dst_crs
        and src_transform == dst_transform
        and array1.shape == array2.shape
    ):
        return array1, array2, profile1

    # Allocate destination buffer for array2
    if array2.ndim == 2:
        dst_array2 = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=array2.astype(np.float32),
            destination=dst_array2,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_method,
        )
    elif array2.ndim == 3:
        num_bands = array2.shape[0]
        dst_array2 = np.full((num_bands, dst_height, dst_width), np.nan, dtype=np.float32)
        for b in range(num_bands):
            reproject(
                source=array2[b].astype(np.float32),
                destination=dst_array2[b],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling_method,
            )
    else:
        raise ValueError(f"Unsupported array dimension: {array2.ndim}")

    unified_profile = profile1.copy()
    unified_profile.update(
        {
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "crs": dst_crs,
        }
    )

    return array1, dst_array2, unified_profile


def check_spatial_compatibility(profile1: dict[str, Any], profile2: dict[str, Any]) -> bool:
    """Check whether two raster profiles share compatible geospatial metadata."""
    return not (not profile1.get("crs") or not profile2.get("crs"))
