"""Tests for edge cases, corrupted inputs, and pipeline failure modes."""

import io

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.change_detection.differencing import compute_dndvi
from src.data.sentinel_loader import load_sentinel_geotiff
from src.indices.vegetation import calculate_ndvi


def test_invalid_upload_single_band_error():
    """Verify that a single-band raster raises ValueError for missing required bands."""
    buffer = io.BytesIO()
    transform = from_origin(100.0, 100.0, 10.0, 10.0)

    # Write single-band GeoTIFF
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:32720",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((1, 20, 20), dtype=np.uint8))

    buffer.seek(0)
    with pytest.raises(ValueError, match="at least 2 bands"):
        load_sentinel_geotiff(buffer)


def test_corrupted_file_bytes_error():
    """Verify that corrupted random byte streams raise rasterio.RasterioIOError."""
    corrupted_buffer = io.BytesIO(b"NotARealGeoTIFFHeaderData1234567890")
    with pytest.raises(rasterio.RasterioIOError):
        load_sentinel_geotiff(corrupted_buffer)


def test_all_nodata_raster_handling():
    """Verify that an all-NaN or all-nodata raster produces safe zeroed stats without crashing."""
    all_nan_nir = np.full((30, 30), np.nan, dtype=np.float32)
    all_nan_red = np.full((30, 30), np.nan, dtype=np.float32)

    ndvi, stats = calculate_ndvi(all_nan_nir, all_nan_red)

    assert np.all(np.isnan(ndvi))
    assert stats["valid_pixels"] == 0
    assert stats["valid_percentage"] == 0.0
    assert np.isnan(stats["mean"])


def test_differencing_shape_mismatch_error():
    """Verify compute_dndvi raises ValueError on non-matching array shapes."""
    arr1 = np.ones((10, 10), dtype=np.float32)
    arr2 = np.ones((10, 12), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dndvi(arr1, arr2)
