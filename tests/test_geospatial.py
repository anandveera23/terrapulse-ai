"""Unit tests for geospatial area calculations."""

from rasterio.transform import from_origin

from src.geospatial.area_calc import compute_change_area_statistics, get_pixel_area_sqm


def test_pixel_area_utm_projected():
    """Verify pixel area calculation in UTM planar space."""
    transform = from_origin(300000.0, 8840000.0, 10.0, 10.0)
    profile = {
        "crs": "EPSG:32720",
        "transform": transform,
        "width": 100,
        "height": 100,
    }

    area_sqm = get_pixel_area_sqm(profile)
    assert area_sqm == 100.0  # 10m x 10m = 100 m²


def test_change_area_aggregation():
    """Verify conversion of pixel counts to hectares, km², and net balance."""
    transform = from_origin(100.0, 100.0, 10.0, 10.0)
    profile = {
        "crs": "EPSG:32720",
        "transform": transform,
        "width": 100,
        "height": 100,
    }

    # 100 pixels @ 100m² = 10,000 m² = 1 hectare
    pixel_counts = {
        1: 500,   # Severe Loss: 50,000 m² = 5.0 ha
        2: 200,   # Moderate Loss: 20,000 m² = 2.0 ha
        3: 1000,  # Stable: 100,000 m² = 10.0 ha
        4: 100,   # Moderate Gain: 10,000 m² = 1.0 ha
        5: 50,    # Sig Gain: 5,000 m² = 0.5 ha
        0: 150,   # Masked
    }

    _df, summary = compute_change_area_statistics(pixel_counts, profile)

    assert summary["pixel_area_sqm"] == 100.0
    assert summary["severe_loss_ha"] == 5.0
    assert summary["moderate_loss_ha"] == 2.0
    assert summary["total_loss_ha"] == 7.0
    assert summary["total_gain_ha"] == 1.5
    assert summary["net_change_ha"] == -5.5  # 1.5 - 7.0 = -5.5 ha
    assert summary["stable_ha"] == 10.0

    # Total valid pixels = 500+200+1000+100+50 = 1850 pixels = 18.5 ha
    assert summary["total_valid_ha"] == 18.5
