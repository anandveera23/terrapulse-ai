import sys

sys.path.insert(0, ".")
import numpy as np

from src.change_detection.classifier import classify_dndvi
from src.change_detection.differencing import compute_dndvi
from src.data.sample_loader import load_sample_pair
from src.geospatial.area_calc import compute_change_area_statistics
from src.indices.vegetation import calculate_ndvi, calculate_ndwi
from src.preprocessing.cloud_mask import apply_scl_mask
from src.preprocessing.reproject import align_and_match_rasters

for reg in ['Rondônia, Brazil (Deforestation Front)', 'Mau Forest Complex, Kenya (Canopy Loss & Recovery)']:
    scene1, scene2, meta = load_sample_pair(reg)
    arr1 = np.stack([scene1.blue, scene1.green, scene1.red, scene1.nir])
    arr2 = np.stack([scene2.blue, scene2.green, scene2.red, scene2.nir])
    aligned1, aligned2, unified_profile = align_and_match_rasters(arr1, scene1.profile, arr2, scene2.profile)
    
    mask1, qual1 = apply_scl_mask(scene1.scl)
    mask2, qual2 = apply_scl_mask(scene2.scl)
    ndwi1 = calculate_ndwi(aligned1[1], aligned1[3])
    ndwi2 = calculate_ndwi(aligned2[1], aligned2[3])
    water_mask = (ndwi1 > 0.05) | (ndwi2 > 0.05)
    
    ndvi1, stats1 = calculate_ndvi(aligned1[3], aligned1[2], valid_mask=mask1)
    ndvi2, stats2 = calculate_ndvi(aligned2[3], aligned2[2], valid_mask=mask2)
    
    dndvi, dndvi_stats = compute_dndvi(ndvi1, ndvi2, mask1, mask2, water_mask=water_mask)
    classes, counts = classify_dndvi(dndvi)
    stats_df, summary = compute_change_area_statistics(counts, unified_profile)
    
    print(f"=== REGION: {reg} ===")
    print(f"Raster dimensions: {unified_profile['height']}h x {unified_profile['width']}w = {unified_profile['height']*unified_profile['width']} total pixels")
    print(f"CRS: {unified_profile['crs']}")
    print(f"Pixel size: dx={abs(unified_profile['transform'][0])}m, dy={abs(unified_profile['transform'][4])}m -> {abs(unified_profile['transform'][0])*abs(unified_profile['transform'][4])} m^2/pixel")
    from pyproj import Transformer
    transformer = Transformer.from_crs(unified_profile['crs'], 'EPSG:4326', always_xy=True)
    min_x = unified_profile['transform'][2]
    max_y = unified_profile['transform'][5]
    max_x = min_x + unified_profile['width'] * unified_profile['transform'][0]
    min_y = max_y + unified_profile['height'] * unified_profile['transform'][4]
    lon_min, lat_max = transformer.transform(min_x, max_y)
    lon_max, lat_min = transformer.transform(max_x, min_y)
    print(f"WGS84 Lat/Lon Extent: [{lat_min:.6f}, {lon_min:.6f}] to [{lat_max:.6f}, {lon_max:.6f}]")
    print(f"Center Lat/Lon: [{(lat_min+lat_max)/2:.6f}, {(lon_min+lon_max)/2:.6f}]")
    print(f"Total scene area: {unified_profile['height']*unified_profile['width']*abs(unified_profile['transform'][0])*abs(unified_profile['transform'][4])/10000.0} ha")
    print(f"T1 Mean NDVI: {stats1['mean']:.4f}, Valid %: {stats1['valid_percentage']:.2f}% ({stats1['valid_pixels']} pixels)")
    print(f"T2 Mean NDVI: {stats2['mean']:.4f}, Valid %: {stats2['valid_percentage']:.2f}% ({stats2['valid_pixels']} pixels)")
    print(f"dNDVI Mean: {dndvi_stats['mean_dndvi']:.4f}, Valid count: {dndvi_stats['valid_pixels']}")
    print(f"Water masked pixels: {np.sum(water_mask)} ({np.sum(water_mask)*0.01:.2f} ha)")
    print(f"Mask1 valid: {np.sum(mask1)}, Mask2 valid: {np.sum(mask2)}")
    print(f"Summary metrics: {summary}")
    print("Classes Table:")
    for idx, row in stats_df.iterrows():
        print(f"  Class {row['Class ID']} ({row['Category']}): count={row['Pixel Count']}, area_ha={row['Area (ha)']:.2f}, share_valid={row['Share of Valid (%)']:.2f}%, share_scene={row['Share of Scene (%)']:.2f}%")
    print()
