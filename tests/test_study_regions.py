"""Tests for study region selector, configurations, metadata, and data isolation."""

import io

import numpy as np
import pytest
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

from src.data.sample_loader import (
    SAMPLE_CATALOGUE,
    get_available_sample_regions,
    load_sample_pair,
    resolve_catalogue_key,
)
from src.data.sentinel_loader import load_sentinel_geotiff
from src.indices.vegetation import calculate_ndvi

EXPECTED_REGIONS = [
    "Rondônia, Brazil (Deforestation Front)",
    "Mau Forest Complex, Kenya (Canopy Loss & Recovery)",
    "Sundarbans, India",
    "Araku Valley, Andhra Pradesh, India",
    "Western Ghats, India",
    "Punjab, India",
    "Rajasthan, India",
]


def test_all_expected_regions_in_catalogue():
    """Verify that all required study regions are present in the catalogue."""
    available = get_available_sample_regions()
    for reg in EXPECTED_REGIONS:
        assert reg in available, f"Region '{reg}' missing from available sample regions."


def test_each_region_has_unique_independent_configuration():
    """Verify that every study region has distinct coordinates, CRSs, files, and dates."""
    seen_t1_files = set()
    seen_t2_files = set()
    seen_crss = set()
    seen_coordinates = set()

    for reg_key, meta in SAMPLE_CATALOGUE.items():
        # Check required metadata fields
        assert "region_name" in meta
        assert "t1_file" in meta
        assert "t2_file" in meta
        assert "t1_date" in meta
        assert "t2_date" in meta
        assert "utm_crs" in meta
        assert "coordinates" in meta

        # Verify no file reuse
        assert meta["t1_file"] not in seen_t1_files, f"Duplicate t1_file '{meta['t1_file']}' in {reg_key}"
        assert meta["t2_file"] not in seen_t2_files, f"Duplicate t2_file '{meta['t2_file']}' in {reg_key}"
        seen_t1_files.add(meta["t1_file"])
        seen_t2_files.add(meta["t2_file"])

        # Coordinates must be unique
        coord_tuple = (round(meta["coordinates"]["lat"], 2), round(meta["coordinates"]["lon"], 2))
        assert coord_tuple not in seen_coordinates, f"Duplicate coordinates {coord_tuple} in {reg_key}"
        seen_coordinates.add(coord_tuple)

        seen_crss.add(meta["utm_crs"])

    # Must span multiple distinct projected coordinate systems
    assert len(seen_crss) >= 4


def test_indian_regions_distinct_geographic_locations():
    """Verify that Indian study regions have accurate coordinates within India."""
    # Sundarbans: West Bengal delta (~21.9 N, 88.8 E)
    sb = SAMPLE_CATALOGUE["Sundarbans, India"]
    assert 21.0 <= sb["coordinates"]["lat"] <= 23.0
    assert 88.0 <= sb["coordinates"]["lon"] <= 90.0
    assert "EPSG:32645" in sb["utm_crs"]

    # Araku Valley: Andhra Pradesh (~18.3 N, 82.8 E)
    ak = SAMPLE_CATALOGUE["Araku Valley, Andhra Pradesh, India"]
    assert 18.0 <= ak["coordinates"]["lat"] <= 19.0
    assert 82.0 <= ak["coordinates"]["lon"] <= 84.0
    assert "EPSG:32644" in ak["utm_crs"]

    # Western Ghats: Karnataka (~13.5 N, 75.1 E)
    wg = SAMPLE_CATALOGUE["Western Ghats, India"]
    assert 13.0 <= wg["coordinates"]["lat"] <= 14.5
    assert 74.5 <= wg["coordinates"]["lon"] <= 76.0
    assert "EPSG:32643" in wg["utm_crs"]

    # Punjab: Indo-Gangetic (~30.8 N, 75.8 E)
    pj = SAMPLE_CATALOGUE["Punjab, India"]
    assert 30.0 <= pj["coordinates"]["lat"] <= 32.0
    assert 75.0 <= pj["coordinates"]["lon"] <= 77.0
    assert "EPSG:32643" in pj["utm_crs"]

    # Rajasthan: Thar desert (~27.5 N, 72.5 E)
    rj = SAMPLE_CATALOGUE["Rajasthan, India"]
    assert 26.5 <= rj["coordinates"]["lat"] <= 28.5
    assert 71.5 <= rj["coordinates"]["lon"] <= 73.5
    assert "EPSG:32643" in rj["utm_crs"]


def test_each_region_loads_distinct_spectral_signatures():
    """Verify that each region loads its own distinct raster data and does not reuse Rondônia."""
    # Baseline Rondônia
    scene_ro_t1, _ro_t2, _meta_ro = load_sample_pair("Rondônia, Brazil (Deforestation Front)")
    _ndvi_ro, stats_ro = calculate_ndvi(scene_ro_t1.nir, scene_ro_t1.red)

    # Western Ghats: Rainforest with very high NDVI (> 0.82)
    scene_wg_t1, _scene_wg_t2, meta_wg = load_sample_pair("Western Ghats, India")
    assert meta_wg["region_name"] == "Western Ghats, India"
    assert scene_wg_t1.crs == "EPSG:32643"
    _ndvi_wg, stats_wg = calculate_ndvi(scene_wg_t1.nir, scene_wg_t1.red)
    assert stats_wg["mean"] > 0.82

    # Rajasthan: Thar desert with low baseline NDVI (< 0.20)
    scene_rj_t1, scene_rj_t2, meta_rj = load_sample_pair("Rajasthan, India")
    assert meta_rj["region_name"] == "Rajasthan, India"
    _ndvi_rj_t1, stats_rj_t1 = calculate_ndvi(scene_rj_t1.nir, scene_rj_t1.red)
    _ndvi_rj_t2, stats_rj_t2 = calculate_ndvi(scene_rj_t2.nir, scene_rj_t2.red)
    assert stats_rj_t1["mean"] < 0.20
    # Canal expansion greening in T2: NDVI increases
    assert stats_rj_t2["mean"] > stats_rj_t1["mean"]

    # Punjab: Peak rice (T1 NDVI ~ 0.77) vs post-harvest fallow (T2 NDVI ~ 0.15)
    scene_pj_t1, scene_pj_t2, meta_pj = load_sample_pair("Punjab, India")
    assert meta_pj["region_name"] == "Punjab, India"
    _ndvi_pj_t1, stats_pj_t1 = calculate_ndvi(scene_pj_t1.nir, scene_pj_t1.red)
    _ndvi_pj_t2, stats_pj_t2 = calculate_ndvi(scene_pj_t2.nir, scene_pj_t2.red)
    assert stats_pj_t1["mean"] > 0.65
    assert stats_pj_t2["mean"] < 0.25

    # Sundarbans: Contains tidal water and mangrove canopy
    scene_sb_t1, _sb_t2, meta_sb = load_sample_pair("Sundarbans, India")
    assert meta_sb["region_name"] == "Sundarbans, India"
    assert scene_sb_t1.crs == "EPSG:32645"

    # Confirm that none of the Indian regions have identical mean NDVI to Rondônia
    assert abs(stats_wg["mean"] - stats_ro["mean"]) > 0.05
    assert abs(stats_rj_t1["mean"] - stats_ro["mean"]) > 0.40


def test_resolve_catalogue_key_fuzzy_matching():
    """Verify that both exact and shorthand names resolve accurately."""
    assert resolve_catalogue_key("Sundarbans, India") == "Sundarbans, India"
    assert resolve_catalogue_key("sundarbans") == "Sundarbans, India"
    assert resolve_catalogue_key("Punjab, India") == "Punjab, India"
    assert resolve_catalogue_key("Rondônia, Brazil") == "Rondônia, Brazil (Deforestation Front)"
    assert resolve_catalogue_key("Mau Forest") == "Mau Forest Complex, Kenya (Canopy Loss & Recovery)"

    with pytest.raises(KeyError):
        resolve_catalogue_key("Nonexistent Sahara Desert")


def test_custom_geotiff_upload_validation_error():
    """Verify that invalid custom upload bytes trigger a clean ValueError/validation error."""
    # Empty/corrupted bytes buffer
    with pytest.raises(RasterioIOError):
        load_sentinel_geotiff(io.BytesIO(b"corrupted garbage bytes"))

    # Single-band raster (minimum 2 bands required)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.uint16,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype=np.uint16))
    buf.seek(0)

    with pytest.raises(ValueError, match="at least 2 bands"):
        load_sentinel_geotiff(buf)
