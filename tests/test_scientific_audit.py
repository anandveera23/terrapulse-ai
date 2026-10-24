"""Scientific correctness and data integrity audit test suite.

Validates exact end-to-end calculations, common valid-pixel masks,
area quantifications, and nodata handling using synthetic rasters
with strictly known expected outputs.
"""

import io

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.change_detection.classifier import classify_dndvi
from src.change_detection.differencing import compute_dndvi
from src.data.sample_loader import load_sample_pair
from src.data.sentinel_loader import load_sentinel_geotiff
from src.geospatial.area_calc import compute_change_area_statistics
from src.indices.vegetation import calculate_ndvi, calculate_ndwi
from src.preprocessing.cloud_mask import apply_scl_mask
from src.preprocessing.reproject import align_and_match_rasters


def test_synthetic_raster_end_to_end_audit():
    """Verify complete analytical pipeline on a 4x4 synthetic raster with known transitions.

    Layout (16 pixels, 10m x 10m = 100 m² per pixel):
    - Pixels 0-1: Severe Loss (T1 NDVI=0.80, T2 NDVI=0.20 -> dNDVI=-0.60)
    - Pixels 2-3: Moderate Loss (T1 NDVI=0.80, T2 NDVI=0.65 -> dNDVI=-0.15)
    - Pixels 4-7: Relatively Stable (T1 NDVI=0.80, T2 NDVI=0.80 -> dNDVI=0.00)
    - Pixels 8-9: Moderate Gain (T1 NDVI=0.50, T2 NDVI=0.65 -> dNDVI=+0.15)
    - Pixels 10-11: Significant Gain (T1 NDVI=0.30, T2 NDVI=0.70 -> dNDVI=+0.40)
    - Pixels 12-13: Permanent Water (T1 & T2 Green=0.10, NIR=0.02 -> NDWI=+0.667)
    - Pixels 14-15: Cloud in T2 (SCL=9 in T2)
    """
    # 4x4 grid (16 pixels)
    height, width = 4, 4
    transform = from_origin(1000.0, 2000.0, 10.0, 10.0)
    profile = {
        "crs": "EPSG:32720",
        "transform": transform,
        "width": width,
        "height": height,
    }

    # Synthesize T1 & T2 NIR and Red bands to achieve desired NDVIs
    # NDVI = (NIR - Red) / (NIR + Red)
    # If NIR + Red = 1.0, then NIR = (1 + NDVI) / 2, Red = (1 - NDVI) / 2
    target_ndvi_t1 = np.array([
        [0.80, 0.80, 0.80, 0.80],
        [0.80, 0.80, 0.80, 0.80],
        [0.50, 0.50, 0.30, 0.30],
        [-0.60, -0.60, 0.80, 0.80],
    ], dtype=np.float32)

    target_ndvi_t2 = np.array([
        [0.20, 0.20, 0.65, 0.65],
        [0.80, 0.80, 0.80, 0.80],
        [0.65, 0.65, 0.70, 0.70],
        [-0.60, -0.60, 0.80, 0.80],
    ], dtype=np.float32)

    t1_nir = (1.0 + target_ndvi_t1) / 2.0
    t1_red = (1.0 - target_ndvi_t1) / 2.0
    t1_green = np.full((4, 4), 0.08, dtype=np.float32)
    t1_green[3, :2] = 0.12  # Water green
    t1_nir[3, :2] = 0.02    # Water low NIR
    t1_red[3, :2] = 0.04    # Water red

    t2_nir = (1.0 + target_ndvi_t2) / 2.0
    t2_red = (1.0 - target_ndvi_t2) / 2.0
    t2_green = np.full((4, 4), 0.08, dtype=np.float32)
    t2_green[3, :2] = 0.12  # Water green
    t2_nir[3, :2] = 0.02    # Water low NIR
    t2_red[3, :2] = 0.04    # Water red

    # SCL masks: 4 = vegetation, 6 = water, 9 = high cloud
    scl_t1 = np.full((4, 4), 4, dtype=np.int32)
    scl_t1[3, :2] = 6  # Water

    scl_t2 = np.full((4, 4), 4, dtype=np.int32)
    scl_t2[3, :2] = 6  # Water
    scl_t2[3, 2:] = 9  # High probability cloud

    # 1. Quality masks
    mask_t1, q_t1 = apply_scl_mask(scl_t1)
    mask_t2, q_t2 = apply_scl_mask(scl_t2)

    # In T1: 16 total, 0 clouds, 0 shadows -> 16 valid pixels (100%)
    assert q_t1["total_pixels"] == 16
    assert q_t1["valid_pixels"] == 16
    assert q_t1["clear_valid_pct"] == 100.0

    # In T2: 16 total, 2 clouds -> 14 valid pixels (87.5%)
    assert q_t2["total_pixels"] == 16
    assert q_t2["valid_pixels"] == 14
    assert q_t2["cloud_pct"] == pytest.approx(12.5, rel=1e-3)
    assert q_t2["clear_valid_pct"] == pytest.approx(87.5, rel=1e-3)

    # 2. Water exclusion mask via NDWI
    ndwi_t1 = calculate_ndwi(t1_green, t1_nir)
    ndwi_t2 = calculate_ndwi(t2_green, t2_nir)
    water_mask = (ndwi_t1 > 0.05) | (ndwi_t2 > 0.05)
    assert water_mask[3, 0] == True
    assert water_mask[3, 1] == True
    assert water_mask[0, 0] == False

    # 3. NDVI calculation
    ndvi_1, _ = calculate_ndvi(t1_nir, t1_red, valid_mask=mask_t1)
    ndvi_2, _ = calculate_ndvi(t2_nir, t2_red, valid_mask=mask_t2)

    # 4. Multi-temporal Differencing with Water & Cloud Exclusion
    dndvi, dndvi_stats = compute_dndvi(
        ndvi_1,
        ndvi_2,
        valid_mask_t1=mask_t1,
        valid_mask_t2=mask_t2,
        water_mask=water_mask,
    )

    # Common valid pixels:
    # 16 total - 2 (water) - 2 (clouds in T2) = 12 valid pixels!
    assert dndvi_stats["valid_pixels"] == 12
    assert dndvi_stats["total_pixels"] == 16

    # 5. Classification
    _classes, counts = classify_dndvi(dndvi)
    assert counts[1] == 2  # Severe Loss: pixels (0,0) and (0,1)
    assert counts[2] == 2  # Moderate Loss: pixels (0,2) and (0,3)
    assert counts[3] == 4  # Stable: row 1 (pixels 4-7)
    assert counts[4] == 2  # Moderate Gain: pixels (2,0) and (2,1)
    assert counts[5] == 2  # Significant Gain: pixels (2,2) and (2,3)
    assert counts[0] == 4  # Masked: 2 water + 2 clouds = 4

    # 6. Area Calculation
    df, summary = compute_change_area_statistics(counts, profile)

    # 10m x 10m pixel = 100 m² = 0.01 ha
    assert summary["pixel_area_sqm"] == 100.0
    assert summary["severe_loss_ha"] == 0.02
    assert summary["moderate_loss_ha"] == 0.02
    assert summary["total_loss_ha"] == 0.04
    assert summary["total_gain_ha"] == 0.04
    assert summary["net_change_ha"] == 0.00  # 0.04 gain - 0.04 loss = 0.00
    assert summary["stable_ha"] == 0.04

    # Coverage: 12 valid out of 16 = 75.0%
    assert summary["valid_coverage_pct"] == 75.0
    assert summary["total_valid_ha"] == 0.12
    assert summary["total_study_ha"] == 0.16

    # Verify that shares of valid pixels sum to ~100.0% within standard 2-decimal rounding tolerance (+-0.02%)
    valid_shares_sum = df[df["Class ID"] != 0]["Share of Valid (%)"].sum()
    assert pytest.approx(valid_shares_sum, abs=0.05) == 100.0


def test_nodata_sentinel_geotiff_handling():
    """Verify that GeoTIFFs containing custom nodata values (-9999) convert safely to NaN."""
    buffer = io.BytesIO()
    transform = from_origin(100.0, 100.0, 10.0, 10.0)

    # Create 5-band raster with nodata = -9999
    # Band 1-4: NIR, Red, etc., with pixel (0,0) having nodata=-9999
    bands = np.full((5, 10, 10), 1000, dtype=np.int16)
    bands[:, 0, 0] = -9999

    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=5,
        dtype=rasterio.int16,
        crs="EPSG:32720",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(bands)

    buffer.seek(0)
    scene = load_sentinel_geotiff(buffer)

    # Pixel (0,0) should be NaN, not -9999 or -0.9999
    assert np.isnan(scene.blue[0, 0])
    assert np.isnan(scene.green[0, 0])
    assert np.isnan(scene.red[0, 0])
    assert np.isnan(scene.nir[0, 0])

    # Other pixels should be valid reflectance (1000 / 10000 = 0.10)
    assert pytest.approx(scene.nir[1, 1], abs=1e-4) == 0.10


def test_rondonia_sample_exact_numerical_consistency():
    """Verify exact numerical consistency of the Rondônia sample benchmark."""
    scene1, scene2, _meta = load_sample_pair("Rondônia, Brazil (Deforestation Front)")

    arr1 = np.stack([scene1.blue, scene1.green, scene1.red, scene1.nir])
    arr2 = np.stack([scene2.blue, scene2.green, scene2.red, scene2.nir])
    aligned1, aligned2, unified_profile = align_and_match_rasters(
        arr1, scene1.profile, arr2, scene2.profile
    )

    assert unified_profile["crs"] == "EPSG:32720"
    assert unified_profile["width"] == 180
    assert unified_profile["height"] == 180

    mask1, _ = apply_scl_mask(scene1.scl)
    mask2, _ = apply_scl_mask(scene2.scl)
    ndwi1 = calculate_ndwi(aligned1[1], aligned1[3])
    ndwi2 = calculate_ndwi(aligned2[1], aligned2[3])
    water_mask = (ndwi1 > 0.05) | (ndwi2 > 0.05)

    ndvi1, stats1 = calculate_ndvi(aligned1[3], aligned1[2], valid_mask=mask1)
    ndvi2, stats2 = calculate_ndvi(aligned2[3], aligned2[2], valid_mask=mask2)

    # T1 Mean NDVI in rainforest range (~0.77 - 0.78)
    assert pytest.approx(stats1["mean"], abs=0.02) == 0.774
    assert stats1["valid_percentage"] == 100.0

    # T2 Mean NDVI drop due to clearing (~0.49 - 0.51)
    assert pytest.approx(stats2["mean"], abs=0.02) == 0.498
    assert pytest.approx(stats2["valid_percentage"], abs=0.5) == 99.2

    dndvi, _dndvi_stats = compute_dndvi(
        ndvi1, ndvi2, valid_mask_t1=mask1, valid_mask_t2=mask2, water_mask=water_mask
    )
    _classes, counts = classify_dndvi(dndvi)
    _stats_df, summary = compute_change_area_statistics(counts, unified_profile)

    # 180 x 180 @ 10m = 324.0 ha total scene
    assert summary["total_study_ha"] == 324.0

    # 30,883 valid joint canopy pixels = 308.83 ha (displays as 308.8 ha)
    assert pytest.approx(summary["total_valid_ha"], abs=0.1) == 308.8
    # 95.3% coverage
    assert pytest.approx(summary["valid_coverage_pct"], abs=0.1) == 95.3

    # Losses and net balance
    assert summary["total_loss_ha"] > 90.0
    assert summary["total_gain_ha"] == 0.0
    # Net balance is negative
    assert summary["net_change_ha"] == -summary["total_loss_ha"]
    assert pytest.approx(summary["net_change_ha"], abs=0.5) == -98.5


def test_mau_forest_sample_exact_numerical_consistency():
    """Verify exact numerical consistency and gain inclusion for the Mau Forest benchmark."""
    scene1, scene2, _meta = load_sample_pair("Mau Forest Complex, Kenya (Canopy Loss & Recovery)")

    arr1 = np.stack([scene1.blue, scene1.green, scene1.red, scene1.nir])
    arr2 = np.stack([scene2.blue, scene2.green, scene2.red, scene2.nir])
    aligned1, aligned2, unified_profile = align_and_match_rasters(
        arr1, scene1.profile, arr2, scene2.profile
    )

    assert unified_profile["crs"] == "EPSG:32736"
    assert unified_profile["width"] == 160
    assert unified_profile["height"] == 160

    mask1, _ = apply_scl_mask(scene1.scl)
    mask2, _ = apply_scl_mask(scene2.scl)
    ndwi1 = calculate_ndwi(aligned1[1], aligned1[3])
    ndwi2 = calculate_ndwi(aligned2[1], aligned2[3])
    water_mask = (ndwi1 > 0.05) | (ndwi2 > 0.05)

    ndvi1, _stats1 = calculate_ndvi(aligned1[3], aligned1[2], valid_mask=mask1)
    ndvi2, _stats2 = calculate_ndvi(aligned2[3], aligned2[2], valid_mask=mask2)

    dndvi, _ = compute_dndvi(
        ndvi1, ndvi2, valid_mask_t1=mask1, valid_mask_t2=mask2, water_mask=water_mask
    )
    _classes, counts = classify_dndvi(dndvi)
    _stats_df, summary = compute_change_area_statistics(counts, unified_profile)

    # Total area: 160 x 160 @ 10m = 256.0 ha
    assert summary["total_study_ha"] == 256.0
    assert summary["total_valid_ha"] == 256.0
    assert summary["valid_coverage_pct"] == 100.0

    # Mau Forest exhibits both canopy loss and regrowth/reforestation
    assert summary["total_loss_ha"] > 0.0
    assert summary["total_gain_ha"] > 0.0

    # Net balance mathematically equals gain minus loss
    expected_net = round(summary["total_gain_ha"] - summary["total_loss_ha"], 2)
    assert summary["net_change_ha"] == expected_net

