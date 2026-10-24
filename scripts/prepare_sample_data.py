"""Generate high-fidelity, georeferenced Sentinel-2 sample datasets.

Produces 5-band GeoTIFFs (B02, B03, B04, B08, SCL) for:
1. Rondônia, Brazil (Deforestation Front - 2020 vs 2023)
2. Mau Forest Complex, Kenya (Degradation & Regrowth - 2019 vs 2024)
3. Sundarbans, India (Mangrove Delta & Coastal Dynamics - 2021 vs 2024)
4. Araku Valley, Andhra Pradesh, India (Eastern Ghats Agroforestry - 2021 vs 2024)
5. Western Ghats, India (Evergreen Rainforest & Montane Canopy - 2020 vs 2024)
6. Punjab, India (Indo-Gangetic Agricultural Phenology - 2021 Sep vs Nov)
7. Rajasthan, India (Thar Arid Scrub & Canal Command Greening - 2020 vs 2023)
"""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_geotiff_5band(
    filepath: Path,
    height: int,
    width: int,
    crs: str,
    transform: Any,
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    scl: np.ndarray,
) -> None:
    """Save a standardized 5-band Sentinel-2 Level-2A GeoTIFF."""
    b2 = (np.clip(blue, 0.0001, 1.0) * 10000).astype(np.uint16)
    b3 = (np.clip(green, 0.0001, 1.0) * 10000).astype(np.uint16)
    b4 = (np.clip(red, 0.0001, 1.0) * 10000).astype(np.uint16)
    b8 = (np.clip(nir, 0.0001, 1.0) * 10000).astype(np.uint16)
    scl_u16 = scl.astype(np.uint16)

    stacked = np.stack([b2, b3, b4, b8, scl_u16])
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=5,
        dtype=rasterio.uint16,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(stacked)
        dst.set_band_description(1, "B02 - Blue")
        dst.set_band_description(2, "B03 - Green")
        dst.set_band_description(3, "B04 - Red")
        dst.set_band_description(4, "B08 - NIR")
        dst.set_band_description(5, "SCL - Scene Classification")
    print(f"Saved: {filepath}")


def create_rondonia_pair(output_dir: Path, force: bool = False) -> None:
    """Generate Rondônia sample pair with genuine fishbone clearing pattern."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and (output_dir / "rondonia_2020_t1.tif").exists() and (output_dir / "rondonia_2023_t2.tif").exists():
        print(f"Rondônia sample rasters already exist in {output_dir}")
        return

    height, width = 180, 180  # 180 x 180 pixels @ 10m = 1.8km x 1.8km = 324 ha
    res = 10.0  # meters

    west = 300000.0
    north = 8840000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32720"

    np.random.seed(42)

    # --- BASELINE T1 (July 2020: Mostly Intact Rainforest) ---
    t1_nir = np.random.normal(0.48, 0.03, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.045, 0.008, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.075, 0.01, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.03, 0.005, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)  # 4 = Vegetation

    # River
    river_x = np.linspace(20, 45, height).astype(int) + (np.sin(np.linspace(0, 6, height)) * 8).astype(int)
    for y in range(height):
        rx = river_x[y]
        for dx in range(-2, 3):
            if 0 <= rx + dx < width:
                t1_nir[y, rx + dx] = 0.02
                t1_red[y, rx + dx] = 0.03
                t1_green[y, rx + dx] = 0.05
                t1_blue[y, rx + dx] = 0.07
                t1_scl[y, rx + dx] = 6  # 6 = Water

    # North-South Road at x = 90
    for y in range(height):
        for dx in range(-1, 2):
            t1_nir[y, 90 + dx] = 0.16
            t1_red[y, 90 + dx] = 0.18
            t1_green[y, 90 + dx] = 0.15
            t1_blue[y, 90 + dx] = 0.12
            t1_scl[y, 90 + dx] = 5  # 5 = Not vegetated

    # A few early small agricultural clearings in 2020
    for cy, cx, r in [(40, 110, 8), (130, 70, 10)]:
        y_grid, x_grid = np.ogrid[:height, :width]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        cleared = dist <= r
        t1_nir[cleared] = 0.22
        t1_red[cleared] = 0.16
        t1_green[cleared] = 0.14
        t1_blue[cleared] = 0.11
        t1_scl[cleared] = 5

    # Clip reflectance [0.0, 1.0]
    for b in [t1_blue, t1_green, t1_red, t1_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    # --- FOLLOW-UP T2 (July 2023: Heavy Fishbone Deforestation) ---
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Perpendicular secondary logging roads branch off main road (Fishbone pattern)
    branch_rows = [30, 60, 95, 125, 155]
    for by in branch_rows:
        # East branches
        t2_nir[by - 1:by + 2, 90:165] = 0.15
        t2_red[by - 1:by + 2, 90:165] = 0.18
        t2_green[by - 1:by + 2, 90:165] = 0.14
        t2_blue[by - 1:by + 2, 90:165] = 0.12
        t2_scl[by - 1:by + 2, 90:165] = 5

        # West branches
        t2_nir[by - 1:by + 2, 25:90] = 0.15
        t2_red[by - 1:by + 2, 25:90] = 0.18
        t2_green[by - 1:by + 2, 25:90] = 0.14
        t2_blue[by - 1:by + 2, 25:90] = 0.12
        t2_scl[by - 1:by + 2, 25:90] = 5

        # Large cleared parcel blocks along secondary roads
        for px, py, pw, ph in [(100, by - 12, 45, 25), (35, by - 14, 40, 28)]:
            if 0 <= py < height - ph and 0 <= px < width - pw:
                t2_nir[py:py + ph, px:px + pw] = np.random.normal(0.16, 0.02, (ph, pw))
                t2_red[py:py + ph, px:px + pw] = np.random.normal(0.18, 0.02, (ph, pw))
                t2_green[py:py + ph, px:px + pw] = np.random.normal(0.13, 0.015, (ph, pw))
                t2_blue[py:py + ph, px:px + pw] = np.random.normal(0.10, 0.015, (ph, pw))
                t2_scl[py:py + ph, px:px + pw] = 5

    # River intact in T2
    for y in range(height):
        rx = river_x[y]
        for dx in range(-2, 3):
            if 0 <= rx + dx < width:
                t2_nir[y, rx + dx] = 0.02
                t2_red[y, rx + dx] = 0.03
                t2_green[y, rx + dx] = 0.05
                t2_blue[y, rx + dx] = 0.07
                t2_scl[y, rx + dx] = 6

    # Test cloud and shadow
    cy_cloud, cx_cloud = 15, 160
    y_g, x_g = np.ogrid[:height, :width]
    cloud_dist = np.sqrt((x_g - cx_cloud)**2 + (y_g - cy_cloud)**2)
    cloud_mask = cloud_dist <= 7
    t2_blue[cloud_mask] = 0.42
    t2_green[cloud_mask] = 0.40
    t2_red[cloud_mask] = 0.38
    t2_nir[cloud_mask] = 0.44
    t2_scl[cloud_mask] = 9

    shadow_mask = np.sqrt((x_g - (cx_cloud - 8))**2 + (y_g - (cy_cloud + 6))**2) <= 6
    t2_nir[shadow_mask] = 0.04
    t2_red[shadow_mask] = 0.03
    t2_green[shadow_mask] = 0.03
    t2_blue[shadow_mask] = 0.04
    t2_scl[shadow_mask] = 3

    for b in [t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "rondonia_2020_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "rondonia_2023_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_mau_forest_pair(output_dir: Path, force: bool = False) -> None:
    """Generate Mau Forest sample pair with degradation and regrowth zones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and (output_dir / "mau_forest_2019_t1.tif").exists() and (output_dir / "mau_forest_2024_t2.tif").exists():
        print(f"Mau Forest sample rasters already exist in {output_dir}")
        return

    height, width = 160, 160
    res = 10.0
    west = 750000.0
    north = 9940000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32736"

    np.random.seed(101)

    t1_nir = np.random.normal(0.44, 0.04, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.05, 0.01, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.08, 0.012, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.035, 0.007, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Degraded eastern section
    t1_nir[:, 110:] = np.random.normal(0.28, 0.03, (height, 50))
    t1_red[:, 110:] = np.random.normal(0.12, 0.02, (height, 50))
    t1_scl[:, 110:] = 5

    # T2
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Encroachment loss
    t2_nir[20:80, 20:65] = np.random.normal(0.19, 0.02, (60, 45))
    t2_red[20:80, 20:65] = np.random.normal(0.17, 0.02, (60, 45))
    t2_scl[20:80, 20:65] = 5

    # Reforestation gain
    t2_nir[70:140, 115:150] = np.random.normal(0.42, 0.03, (70, 35))
    t2_red[70:140, 115:150] = np.random.normal(0.06, 0.01, (70, 35))
    t2_scl[70:140, 115:150] = 4

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "mau_forest_2019_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "mau_forest_2024_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_sundarbans_pair(output_dir: Path) -> None:
    """Generate Sundarbans mangrove delta pair with tidal creeks, erosion, and colonization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180
    res = 10.0
    # West Bengal UTM Zone 45N (EPSG:32645)
    west = 685000.0
    north = 2426000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32645"

    np.random.seed(301)

    # T1 (2021): Dense intertidal mangrove canopy
    t1_nir = np.random.normal(0.42, 0.03, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.045, 0.008, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.075, 0.01, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.032, 0.005, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Wide tidal estuaries and branching mangrove channels
    y_coords, x_coords = np.ogrid[:height, :width]
    main_channel = np.abs(x_coords - (55 + np.sin(y_coords / 22.0) * 16)) <= 7
    branch_channel = (y_coords >= 70) & (np.abs((x_coords - 55) - (y_coords - 70) * 0.7) <= 5)
    water_mask = main_channel | branch_channel

    t1_nir[water_mask] = 0.015
    t1_red[water_mask] = 0.035
    t1_green[water_mask] = 0.065
    t1_blue[water_mask] = 0.085
    t1_scl[water_mask] = 6  # Water

    # Intertidal mudflat in southeast quadrant (NDVI ~ 0.20)
    mudflat_mask = (y_coords >= 120) & (x_coords >= 120) & ((x_coords + y_coords) % 5 <= 3)
    t1_nir[mudflat_mask] = 0.17
    t1_red[mudflat_mask] = 0.13
    t1_green[mudflat_mask] = 0.11
    t1_blue[mudflat_mask] = 0.08
    t1_scl[mudflat_mask] = 5  # Bare soil / mudflat

    # T2 (2024):
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Cyclone erosion & tidal inundation dieback along west bank of main estuary (Loss)
    erosion_mask = (y_coords >= 30) & (y_coords <= 100) & (x_coords >= 32) & (x_coords <= 52) & (~water_mask)
    t2_nir[erosion_mask] = np.random.normal(0.16, 0.02, t2_nir[erosion_mask].shape)
    t2_red[erosion_mask] = np.random.normal(0.14, 0.015, t2_red[erosion_mask].shape)
    t2_green[erosion_mask] = 0.09
    t2_blue[erosion_mask] = 0.07
    t2_scl[erosion_mask] = 5

    # Pioneer Avicennia mangrove colonization on the accreted mudflat (Gain)
    regrowth_mask = (y_coords >= 130) & (y_coords <= 165) & (x_coords >= 130) & (x_coords <= 165) & mudflat_mask
    t2_nir[regrowth_mask] = np.random.normal(0.39, 0.025, t2_nir[regrowth_mask].shape)
    t2_red[regrowth_mask] = np.random.normal(0.055, 0.008, t2_red[regrowth_mask].shape)
    t2_green[regrowth_mask] = 0.08
    t2_blue[regrowth_mask] = 0.035
    t2_scl[regrowth_mask] = 4

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "sundarbans_2021_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "sundarbans_2024_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_araku_valley_pair(output_dir: Path) -> None:
    """Generate Araku Valley pair with shade coffee agroforestry and shifting hill cultivation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180
    res = 10.0
    # Andhra Pradesh UTM Zone 44N (EPSG:32644)
    west = 695000.0
    north = 2026000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32644"

    np.random.seed(402)

    # T1 (2021): Undulating montane plateau with shade coffee and semi-evergreen canopy
    t1_nir = np.random.normal(0.46, 0.035, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.05, 0.009, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.08, 0.01, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.035, 0.006, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Stream in valley depression
    stream_x = (np.linspace(130, 40, height) + np.cos(np.linspace(0, 5, height)) * 10).astype(int)
    for y in range(height):
        sx = stream_x[y]
        for dx in range(-1, 2):
            if 0 <= sx + dx < width:
                t1_nir[y, sx + dx] = 0.02
                t1_red[y, sx + dx] = 0.035
                t1_green[y, sx + dx] = 0.055
                t1_blue[y, sx + dx] = 0.075
                t1_scl[y, sx + dx] = 6

    # Podu shifting agriculture terrace (bare soil / dry fallow)
    t1_nir[25:65, 25:65] = np.random.normal(0.22, 0.02, (40, 40))
    t1_red[25:65, 25:65] = np.random.normal(0.16, 0.015, (40, 40))
    t1_scl[25:65, 25:65] = 5

    # T2 (2024):
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Hill slope clearing in eastern sector (Loss)
    t2_nir[100:150, 110:160] = np.random.normal(0.18, 0.02, (50, 50))
    t2_red[100:150, 110:160] = np.random.normal(0.17, 0.015, (50, 50))
    t2_scl[100:150, 110:160] = 5

    # Silver oak shade tree agroforestry expansion on old fallow terrace (Gain)
    t2_nir[30:60, 30:60] = np.random.normal(0.44, 0.03, (30, 30))
    t2_red[30:60, 30:60] = np.random.normal(0.055, 0.008, (30, 30))
    t2_scl[30:60, 30:60] = 4

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "araku_valley_2021_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "araku_valley_2024_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_western_ghats_pair(output_dir: Path) -> None:
    """Generate Western Ghats wet tropical evergreen rainforest pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180
    res = 10.0
    # Karnataka UTM Zone 43N (EPSG:32643)
    west = 510000.0
    north = 1494000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32643"

    np.random.seed(503)

    # T1 (2020): Pristine tropical wet evergreen canopy (very high NIR, very low Red)
    t1_nir = np.random.normal(0.54, 0.03, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.032, 0.006, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.07, 0.009, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.025, 0.004, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Riparian stream
    for y in range(height):
        sx = int(60 + np.sin(y / 15.0) * 8)
        for dx in range(-1, 2):
            if 0 <= sx + dx < width:
                t1_nir[y, sx + dx] = 0.015
                t1_red[y, sx + dx] = 0.03
                t1_green[y, sx + dx] = 0.05
                t1_blue[y, sx + dx] = 0.07
                t1_scl[y, sx + dx] = 6

    # T2 (2024):
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Localized linear access clearing along eastern edge (Loss)
    t2_nir[50:130, 150:165] = np.random.normal(0.18, 0.02, (80, 15))
    t2_red[50:130, 150:165] = np.random.normal(0.17, 0.015, (80, 15))
    t2_scl[50:130, 150:165] = 5

    # Secondary bamboo gap regeneration in west (Gain)
    t2_nir[130:160, 20:45] = np.random.normal(0.56, 0.025, (30, 25))
    t2_red[130:160, 20:45] = np.random.normal(0.030, 0.005, (30, 25))
    t2_scl[130:160, 20:45] = 4

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "western_ghats_2020_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "western_ghats_2024_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_punjab_pair(output_dir: Path) -> None:
    """Generate Punjab agricultural basin pair (Peak Kharif Rice vs Post-Harvest Fallow)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180
    res = 10.0
    # Punjab UTM Zone 43N (EPSG:32643)
    west = 574000.0
    north = 3416000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32643"

    np.random.seed(604)

    # T1 (Sept 2021): Peak Kharif paddy season (intense green agricultural parcels)
    t1_nir = np.random.normal(0.44, 0.03, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.05, 0.008, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.09, 0.01, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.04, 0.006, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Irrigation canal network running N-S at x=80 and E-W at y=90
    t1_nir[:, 78:82] = 0.02
    t1_red[:, 78:82] = 0.035
    t1_green[:, 78:82] = 0.06
    t1_blue[:, 78:82] = 0.08
    t1_scl[:, 78:82] = 6

    t1_nir[88:92, :] = 0.02
    t1_red[88:92, :] = 0.035
    t1_green[88:92, :] = 0.06
    t1_blue[88:92, :] = 0.08
    t1_scl[88:92, :] = 6

    # T2 (Nov 2021): Post-harvest fallow, bare soil, and crop residue clearing
    t2_nir = np.random.normal(0.19, 0.02, (height, width)).astype(np.float32)
    t2_red = np.random.normal(0.21, 0.02, (height, width)).astype(np.float32)
    t2_green = np.random.normal(0.15, 0.015, (height, width)).astype(np.float32)
    t2_blue = np.random.normal(0.11, 0.012, (height, width)).astype(np.float32)
    t2_scl = np.full((height, width), 5, dtype=np.int32)  # Bare soil

    # Keep canals intact in T2
    t2_nir[:, 78:82] = 0.02
    t2_red[:, 78:82] = 0.035
    t2_green[:, 78:82] = 0.06
    t2_blue[:, 78:82] = 0.08
    t2_scl[:, 78:82] = 6

    t2_nir[88:92, :] = 0.02
    t2_red[88:92, :] = 0.035
    t2_green[88:92, :] = 0.06
    t2_blue[88:92, :] = 0.08
    t2_scl[88:92, :] = 6

    # Retain a small vegetative shelterbelt/orchard in southwest (Stable)
    t2_nir[130:170, 20:60] = t1_nir[130:170, 20:60]
    t2_red[130:170, 20:60] = t1_red[130:170, 20:60]
    t2_green[130:170, 20:60] = t1_green[130:170, 20:60]
    t2_blue[130:170, 20:60] = t1_blue[130:170, 20:60]
    t2_scl[130:170, 20:60] = 4

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "punjab_2021_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "punjab_2021_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_rajasthan_pair(output_dir: Path) -> None:
    """Generate Rajasthan Thar desert scrub and canal irrigation greening pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180
    res = 10.0
    # Rajasthan UTM Zone 43N (EPSG:32643)
    west = 252000.0
    north = 3045000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32643"

    np.random.seed(705)

    # T1 (Oct 2020): Dominant bare desert sand & sparse arid scrub
    t1_red = np.random.normal(0.28, 0.025, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.22, 0.02, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.16, 0.015, (height, width)).astype(np.float32)
    t1_nir = np.random.normal(0.30, 0.025, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 5, dtype=np.int32)  # Bare soil/desert

    # Indira Gandhi Feeder Canal passing N-S at x = 40
    t1_nir[:, 38:43] = 0.02
    t1_red[:, 38:43] = 0.04
    t1_green[:, 38:43] = 0.06
    t1_blue[:, 38:43] = 0.08
    t1_scl[:, 38:43] = 6

    # T2 (Oct 2023): Canal command expansion creating green agricultural crop parcels (Gain)
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Canal irrigation expansion in eastern quadrant
    for px, py, pw, ph in [(80, 20, 45, 50), (120, 85, 45, 50), (75, 110, 40, 45)]:
        t2_nir[py:py + ph, px:px + pw] = np.random.normal(0.44, 0.03, (ph, pw))
        t2_red[py:py + ph, px:px + pw] = np.random.normal(0.065, 0.01, (ph, pw))
        t2_green[py:py + ph, px:px + pw] = np.random.normal(0.10, 0.012, (ph, pw))
        t2_blue[py:py + ph, px:px + pw] = np.random.normal(0.04, 0.006, (ph, pw))
        t2_scl[py:py + ph, px:px + pw] = 4  # Vegetated crop

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    write_geotiff_5band(output_dir / "rajasthan_2020_t1.tif", height, width, crs, transform, t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    write_geotiff_5band(output_dir / "rajasthan_2023_t2.tif", height, width, crs, transform, t2_blue, t2_green, t2_red, t2_nir, t2_scl)


if __name__ == "__main__":
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    create_rondonia_pair(sample_dir)
    create_mau_forest_pair(sample_dir)
    create_sundarbans_pair(sample_dir)
    create_araku_valley_pair(sample_dir)
    create_western_ghats_pair(sample_dir)
    create_punjab_pair(sample_dir)
    create_rajasthan_pair(sample_dir)
    print("All sample datasets successfully generated!")
