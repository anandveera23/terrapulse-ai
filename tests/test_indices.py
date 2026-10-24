"""Unit tests for vegetation and spectral indices (src/indices/vegetation.py)."""

import numpy as np
import pytest

from src.indices.vegetation import calculate_ndvi, calculate_ndwi, calculate_savi


def test_ndvi_known_synthetic_values():
    """Verify NDVI calculations on synthetic arrays with mathematically known values."""
    # Pixel 0: Dense forest (NIR=0.50, Red=0.05) -> (0.45 / 0.55) = 0.81818...
    # Pixel 1: Bare soil (NIR=0.20, Red=0.20) -> 0.0
    # Pixel 2: Water body (NIR=0.02, Red=0.04) -> (-0.02 / 0.06) = -0.33333...
    nir = np.array([[0.50, 0.20], [0.02, 0.60]], dtype=np.float32)
    red = np.array([[0.05, 0.20], [0.04, 0.10]], dtype=np.float32)

    ndvi, stats = calculate_ndvi(nir, red)

    assert pytest.approx(ndvi[0, 0], rel=1e-4) == (0.50 - 0.05) / (0.50 + 0.05)
    assert pytest.approx(ndvi[0, 1], abs=1e-5) == 0.0
    assert pytest.approx(ndvi[1, 0], rel=1e-4) == (0.02 - 0.04) / (0.02 + 0.04)
    assert pytest.approx(ndvi[1, 1], rel=1e-4) == (0.60 - 0.10) / (0.60 + 0.10)

    assert stats["valid_pixels"] == 4
    assert stats["total_pixels"] == 4
    assert stats["valid_percentage"] == 100.0


def test_ndvi_zero_denominator_protection():
    """Verify zero and near-zero denominators yield NaN without raising runtime errors."""
    nir = np.array([[0.0, 1e-9]], dtype=np.float32)
    red = np.array([[0.0, -1e-9]], dtype=np.float32)

    ndvi, stats = calculate_ndvi(nir, red, epsilon=1e-6)

    # Both pixels should safely be NaN
    assert np.isnan(ndvi[0, 0])
    assert np.isnan(ndvi[0, 1])
    assert stats["valid_pixels"] == 0


def test_ndvi_nan_and_nodata_preservation():
    """Verify NaN values in input bands propagate safely to NaN in output."""
    nir = np.array([[np.nan, 0.5], [0.4, 0.3]], dtype=np.float32)
    red = np.array([[0.1, np.nan], [0.1, 0.1]], dtype=np.float32)

    ndvi, stats = calculate_ndvi(nir, red)

    assert np.isnan(ndvi[0, 0])
    assert np.isnan(ndvi[0, 1])
    assert not np.isnan(ndvi[1, 0])
    assert not np.isnan(ndvi[1, 1])
    assert stats["valid_pixels"] == 2
    assert stats["total_pixels"] == 4
    assert stats["valid_percentage"] == 50.0


def test_ndvi_clipping_bounds():
    """Verify output NDVI values are strictly bounded within [-1.0, 1.0]."""
    nir = np.array([[1.2, -0.2]], dtype=np.float32)
    red = np.array([[-0.1, 0.5]], dtype=np.float32)

    ndvi, _ = calculate_ndvi(nir, red)
    valid_vals = ndvi[np.isfinite(ndvi)]
    assert np.all(valid_vals >= -1.0)
    assert np.all(valid_vals <= 1.0)


def test_ndvi_shape_mismatch():
    """Verify ValueError is raised if NIR and Red shapes differ."""
    nir = np.zeros((10, 10), dtype=np.float32)
    red = np.zeros((10, 12), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_ndvi(nir, red)


def test_ndwi_and_savi_basic():
    """Verify NDWI and SAVI calculations."""
    green = np.array([[0.10]], dtype=np.float32)
    nir = np.array([[0.05]], dtype=np.float32)
    ndwi = calculate_ndwi(green, nir)
    assert pytest.approx(ndwi[0, 0], rel=1e-4) == (0.10 - 0.05) / (0.10 + 0.05)

    red = np.array([[0.05]], dtype=np.float32)
    savi = calculate_savi(nir, red, l_factor=0.5)
    assert savi.shape == (1, 1)
    assert not np.isnan(savi[0, 0])
