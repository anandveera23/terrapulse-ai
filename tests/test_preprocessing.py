"""Unit tests for preprocessing, cloud masking, and reprojection."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.preprocessing.cloud_mask import apply_scl_mask, detect_clouds_fallback
from src.preprocessing.reproject import align_and_match_rasters


def test_scl_cloud_and_shadow_masking():
    """Verify SCL masking correctly filters clouds, shadows, and nodata."""
    # SCL values:
    # 0 = Nodata, 4 = Vegetation, 8 = Medium Cloud, 9 = High Cloud, 3 = Cloud Shadow, 6 = Water
    scl = np.array([[0, 4], [8, 9], [3, 6]], dtype=np.int32)

    valid_mask, stats = apply_scl_mask(scl, mask_clouds=True, mask_shadows=True)

    # Valid should only be vegetation (4) and water (6)
    assert not valid_mask[0, 0]  # nodata
    assert valid_mask[0, 1]      # vegetation
    assert not valid_mask[1, 0]  # cloud med
    assert not valid_mask[1, 1]  # cloud high
    assert not valid_mask[2, 0]  # shadow
    assert valid_mask[2, 1]      # water

    assert stats["total_pixels"] == 6
    assert stats["valid_pixels"] == 2
    assert stats["cloud_pct"] == pytest.approx(33.333, rel=1e-2)
    assert stats["shadow_pct"] == pytest.approx(16.666, rel=1e-2)


def test_detect_clouds_fallback():
    """Verify fallback spectral threshold detector masks bright clouds."""
    # Pixel 0: Clear vegetation (Blue=0.03, Red=0.04, NIR=0.50)
    # Pixel 1: Bright cloud (Blue=0.35, Red=0.32, NIR=0.40)
    blue = np.array([[0.03, 0.35]], dtype=np.float32)
    red = np.array([[0.04, 0.32]], dtype=np.float32)
    nir = np.array([[0.50, 0.40]], dtype=np.float32)

    mask = detect_clouds_fallback(blue, red, nir)
    assert mask[0, 0] == True
    assert mask[0, 1] == False


def test_raster_alignment_matching_profiles():
    """Verify that rasters with matching grids pass through without reprojection distortion."""
    arr1 = np.ones((50, 50), dtype=np.float32)
    arr2 = np.ones((50, 50), dtype=np.float32) * 2.0
    transform = from_origin(100.0, 100.0, 10.0, 10.0)

    profile = {
        "crs": "EPSG:32720",
        "transform": transform,
        "width": 50,
        "height": 50,
    }

    out1, out2, _out_prof = align_and_match_rasters(arr1, profile, arr2, profile)
    assert out1.shape == (50, 50)
    assert out2.shape == (50, 50)
    assert np.all(out1 == 1.0)
    assert np.all(out2 == 2.0)
