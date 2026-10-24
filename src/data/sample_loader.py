"""Sample dataset catalogue and loader for TerraPulse AI.

Provides out-of-the-box, verified Sentinel-2 observation pairs:
1. Rondônia Deforestation Front (Brazil)
2. Mau Forest Canopy Dynamics (Kenya)
"""

from pathlib import Path

from src.data.sentinel_loader import SentinelScene, load_sentinel_geotiff

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

SAMPLE_CATALOGUE: dict[str, dict[str, str]] = {
    "Rondônia, Brazil (Deforestation Front)": {
        "description": "Amazonian tropical rainforest showing intensive fishbone clearing along pioneer road corridors between 2020 and 2023.",
        "region_name": "Rondônia, Brazil",
        "t1_file": "rondonia_2020_t1.tif",
        "t1_date": "2020-07-18",
        "t2_file": "rondonia_2023_t2.tif",
        "t2_date": "2023-07-22",
        "primary_threat": "Agricultural expansion & logging corridors",
        "utm_crs": "EPSG:32720 (UTM Zone 20S)",
    },
    "Mau Forest Complex, Kenya (Canopy Loss & Recovery)": {
        "description": "Critical East African water tower exhibiting localized canopy thinning alongside community reforestation and bamboo recovery zones.",
        "region_name": "Mau Forest, Kenya",
        "t1_file": "mau_forest_2019_t1.tif",
        "t1_date": "2019-01-14",
        "t2_file": "mau_forest_2024_t2.tif",
        "t2_date": "2024-01-20",
        "primary_threat": "Encroachment & selective logging vs Reforestation",
        "utm_crs": "EPSG:32736 (UTM Zone 36S)",
    },
}


def get_available_sample_regions() -> list[str]:
    """Return a list of available curated sample region names."""
    return list(SAMPLE_CATALOGUE.keys())


def load_sample_pair(region_key: str) -> tuple[SentinelScene, SentinelScene, dict[str, str]]:
    """Load the paired Sentinel-2 scenes for the specified sample region.

    Args:
        region_key: Name of region from get_available_sample_regions().

    Returns:
        Tuple of (scene_t1, scene_t2, metadata_dict).
    """
    if region_key not in SAMPLE_CATALOGUE:
        raise KeyError(f"Region '{region_key}' not found in sample catalogue.")

    info = SAMPLE_CATALOGUE[region_key]
    t1_path = SAMPLE_DIR / info["t1_file"]
    t2_path = SAMPLE_DIR / info["t2_file"]

    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError(
            f"Sample GeoTIFFs missing in {SAMPLE_DIR}. Run scripts/prepare_sample_data.py."
        )

    scene_t1 = load_sentinel_geotiff(
        str(t1_path),
        acquisition_date=info["t1_date"],
        region_name=info["region_name"],
    )
    scene_t2 = load_sentinel_geotiff(
        str(t2_path),
        acquisition_date=info["t2_date"],
        region_name=info["region_name"],
    )

    return scene_t1, scene_t2, info
