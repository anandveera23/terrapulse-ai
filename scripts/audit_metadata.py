from pathlib import Path

import rasterio

sample_dir = Path("data/sample")
for p in sorted(sample_dir.glob("*.tif")):
    with rasterio.open(p) as ds:
        print(f"=== {p.name} ===")
        print(f"Driver: {ds.driver}")
        print(f"Shape: {ds.count} bands, {ds.height}h x {ds.width}w")
        print(f"CRS: {ds.crs}")
        print(f"Transform: {ds.transform}")
        print(f"Pixel Resolution: dx={abs(ds.transform[0])}m, dy={abs(ds.transform[4])}m")
        print(f"Bounds: {ds.bounds}")
        print(f"Nodata value: {ds.nodata}")
        print(f"Band descriptions: {ds.descriptions}")
        for i in range(1, ds.count + 1):
            arr = ds.read(i)
            desc = ds.descriptions[i - 1] if ds.descriptions and ds.descriptions[i - 1] else f"Band {i}"
            print(f"  {desc}: dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
