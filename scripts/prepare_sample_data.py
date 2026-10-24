"""Generate high-fidelity, georeferenced Sentinel-2 sample datasets.

Produces 5-band GeoTIFFs (B02, B03, B04, B08, SCL) for:
1. Rondônia, Brazil (Deforestation Front - 2020 vs 2023)
2. Mau Forest, Kenya (Degradation & Regrowth - 2019 vs 2024)
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_rondonia_pair(output_dir: Path):
    """Generate Rondônia sample pair with genuine fishbone clearing pattern."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 180, 180  # 180 x 180 pixels @ 10m = 1.8km x 1.8km = 324 ha
    res = 10.0  # meters

    # Rondônia UTM Zone 20S coordinates (approx -10.45 lat, -62.8 long)
    # EPSG:32720
    west = 300000.0
    north = 8840000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32720"

    np.random.seed(42)

    # --- BASELINE T1 (July 2020: Mostly Intact Rainforest) ---
    # High NIR (~0.45-0.55), Low Red (~0.03-0.06), Low Blue (~0.02-0.04), Moderate Green (~0.06-0.09)
    # NDVI ~ 0.78 - 0.86
    t1_nir = np.random.normal(0.48, 0.03, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.045, 0.008, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.075, 0.01, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.03, 0.005, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)  # 4 = Vegetation

    # Add a meandering river / stream
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

    # Add a main pioneer road running North-South at x = 90
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

    # --- T2 (July 2023: Dramatic Fishbone Deforestation) ---
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
                # Clear-cut bare soil / degraded pasture: Low NIR (~0.14-0.18), High Red (~0.16-0.22) -> NDVI ~ 0.0-0.15
                t2_nir[py:py + ph, px:px + pw] = np.random.normal(0.16, 0.02, (ph, pw))
                t2_red[py:py + ph, px:px + pw] = np.random.normal(0.18, 0.02, (ph, pw))
                t2_green[py:py + ph, px:px + pw] = np.random.normal(0.13, 0.015, (ph, pw))
                t2_blue[py:py + ph, px:px + pw] = np.random.normal(0.10, 0.015, (ph, pw))
                t2_scl[py:py + ph, px:px + pw] = 5

    # Keep river intact in T2
    for y in range(height):
        rx = river_x[y]
        for dx in range(-2, 3):
            if 0 <= rx + dx < width:
                t2_nir[y, rx + dx] = 0.02
                t2_red[y, rx + dx] = 0.03
                t2_green[y, rx + dx] = 0.05
                t2_blue[y, rx + dx] = 0.07
                t2_scl[y, rx + dx] = 6

    # Add a small isolated cloud and shadow in corner of T2 to test cloud masking
    cy_cloud, cx_cloud = 15, 160
    y_g, x_g = np.ogrid[:height, :width]
    cloud_dist = np.sqrt((x_g - cx_cloud)**2 + (y_g - cy_cloud)**2)
    cloud_mask = cloud_dist <= 7
    t2_blue[cloud_mask] = 0.42
    t2_green[cloud_mask] = 0.40
    t2_red[cloud_mask] = 0.38
    t2_nir[cloud_mask] = 0.44
    t2_scl[cloud_mask] = 9  # 9 = High probability cloud

    # Cloud shadow
    shadow_mask = np.sqrt((x_g - (cx_cloud - 8))**2 + (y_g - (cy_cloud + 6))**2) <= 6
    t2_nir[shadow_mask] = 0.04
    t2_red[shadow_mask] = 0.03
    t2_green[shadow_mask] = 0.03
    t2_blue[shadow_mask] = 0.04
    t2_scl[shadow_mask] = 3  # 3 = Cloud shadow

    for b in [t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    # Convert to standard Sentinel-2 DN integer scaling (DN = reflectance * 10000)
    def save_scene(filename, blue, green, red, nir, scl):
        b2 = (blue * 10000).astype(np.uint16)
        b3 = (green * 10000).astype(np.uint16)
        b4 = (red * 10000).astype(np.uint16)
        b8 = (nir * 10000).astype(np.uint16)
        scl_u16 = scl.astype(np.uint16)

        stacked = np.stack([b2, b3, b4, b8, scl_u16])
        filepath = output_dir / filename
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

    save_scene("rondonia_2020_t1.tif", t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    save_scene("rondonia_2023_t2.tif", t2_blue, t2_green, t2_red, t2_nir, t2_scl)


def create_mau_forest_pair(output_dir: Path):
    """Generate Mau Forest sample pair with degradation and regrowth zones."""
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 160, 160
    res = 10.0
    # Kenya UTM Zone 36S (EPSG:32736)
    west = 750000.0
    north = 9940000.0
    transform = from_origin(west, north, res, res)
    crs = "EPSG:32736"

    np.random.seed(101)

    # T1 (2019): Highland Montane Forest
    t1_nir = np.random.normal(0.44, 0.04, (height, width)).astype(np.float32)
    t1_red = np.random.normal(0.05, 0.01, (height, width)).astype(np.float32)
    t1_green = np.random.normal(0.08, 0.012, (height, width)).astype(np.float32)
    t1_blue = np.random.normal(0.035, 0.007, (height, width)).astype(np.float32)
    t1_scl = np.full((height, width), 4, dtype=np.int32)

    # Some previously degraded shrubland in eastern section (NDVI ~ 0.40)
    t1_nir[:, 110:] = np.random.normal(0.28, 0.03, (height, 50))
    t1_red[:, 110:] = np.random.normal(0.12, 0.02, (height, 50))
    t1_scl[:, 110:] = 5

    # T2 (2024):
    # Zone A: Illegal encroachment & canopy thinning in western quadrant
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    t2_blue = t1_blue.copy()
    t2_scl = t1_scl.copy()

    # Encroachment loss (NDVI drops from ~0.8 to ~0.2)
    t2_nir[20:80, 20:65] = np.random.normal(0.19, 0.02, (60, 45))
    t2_red[20:80, 20:65] = np.random.normal(0.17, 0.02, (60, 45))
    t2_scl[20:80, 20:65] = 5

    # Zone B: Community reforestation & bamboo recovery in eastern sector (NDVI increases from ~0.4 to ~0.72)
    t2_nir[70:140, 115:150] = np.random.normal(0.42, 0.03, (70, 35))
    t2_red[70:140, 115:150] = np.random.normal(0.06, 0.01, (70, 35))
    t2_scl[70:140, 115:150] = 4  # Recovered to vegetation

    for b in [t1_blue, t1_green, t1_red, t1_nir, t2_blue, t2_green, t2_red, t2_nir]:
        np.clip(b, 0.001, 0.999, out=b)

    def save_scene(filename, blue, green, red, nir, scl):
        b2 = (blue * 10000).astype(np.uint16)
        b3 = (green * 10000).astype(np.uint16)
        b4 = (red * 10000).astype(np.uint16)
        b8 = (nir * 10000).astype(np.uint16)
        scl_u16 = scl.astype(np.uint16)
        stacked = np.stack([b2, b3, b4, b8, scl_u16])
        filepath = output_dir / filename
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

    save_scene("mau_forest_2019_t1.tif", t1_blue, t1_green, t1_red, t1_nir, t1_scl)
    save_scene("mau_forest_2024_t2.tif", t2_blue, t2_green, t2_red, t2_nir, t2_scl)


if __name__ == "__main__":
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    create_rondonia_pair(sample_dir)
    create_mau_forest_pair(sample_dir)
    print("All sample datasets successfully generated!")
