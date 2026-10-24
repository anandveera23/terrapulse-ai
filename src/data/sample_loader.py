"""Sample dataset catalogue and loader for TerraPulse AI.

Provides out-of-the-box, verified Sentinel-2 observation pairs:
1. Rondônia, Brazil (Deforestation Front - 2020 vs 2023)
2. Mau Forest Complex, Kenya (Canopy Loss & Recovery - 2019 vs 2024)
3. Sundarbans, India (Mangrove Delta & Coastal Dynamics - 2021 vs 2024)
4. Araku Valley, Andhra Pradesh, India (Eastern Ghats Agroforestry - 2021 vs 2024)
5. Western Ghats, India (Evergreen Rainforest & Montane Canopy - 2020 vs 2024)
6. Punjab, India (Indo-Gangetic Agricultural Phenology - 2021 Sep vs Nov)
7. Rajasthan, India (Thar Arid Scrub & Canal Command Greening - 2020 vs 2023)
"""

from pathlib import Path
from typing import Any

from src.data.sentinel_loader import SentinelScene, load_sentinel_geotiff

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

SAMPLE_CATALOGUE: dict[str, dict[str, Any]] = {
    "Rondônia, Brazil (Deforestation Front)": {
        "description": "Amazonian tropical rainforest showing intensive fishbone clearing along pioneer road corridors between 2020 and 2023.",
        "region_name": "Rondônia, Brazil",
        "t1_file": "rondonia_2020_t1.tif",
        "t1_date": "2020-07-18",
        "t2_file": "rondonia_2023_t2.tif",
        "t2_date": "2023-07-22",
        "primary_threat": "Agricultural expansion & logging corridors",
        "utm_crs": "EPSG:32720 (UTM Zone 20S)",
        "coordinates": {"lat": -10.4966, "lon": -64.8192},
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
        "coordinates": {"lat": -0.5496, "lon": 35.2534},
    },
    "Sundarbans, India": {
        "description": "UNESCO World Heritage mangrove delta exhibiting intertidal canopy dynamics, tidal creek networks, and coastal erosion/restoration.",
        "region_name": "Sundarbans, India",
        "t1_file": "sundarbans_2021_t1.tif",
        "t1_date": "2021-02-12",
        "t2_file": "sundarbans_2024_t2.tif",
        "t2_date": "2024-02-15",
        "primary_threat": "Cyclonic disturbance, tidal erosion & salinity shifts",
        "utm_crs": "EPSG:32645 (UTM Zone 45N)",
        "coordinates": {"lat": 21.9195, "lon": 88.7911},
    },
    "Araku Valley, Andhra Pradesh, India": {
        "description": "Eastern Ghats montane plateau showcasing shade-grown coffee agroforestry, tribal podu cultivation, and semi-evergreen canopy shifts.",
        "region_name": "Araku Valley, Andhra Pradesh, India",
        "t1_file": "araku_valley_2021_t1.tif",
        "t1_date": "2021-01-18",
        "t2_file": "araku_valley_2024_t2.tif",
        "t2_date": "2024-01-22",
        "primary_threat": "Agroforestry expansion & shifting slope cultivation",
        "utm_crs": "EPSG:32644 (UTM Zone 44N)",
        "coordinates": {"lat": 18.3058, "lon": 82.8450},
    },
    "Western Ghats, India": {
        "description": "Global biodiversity hotspot featuring hyper-dense tropical wet evergreen rainforest canopy and riparian montane corridors.",
        "region_name": "Western Ghats, India",
        "t1_file": "western_ghats_2020_t1.tif",
        "t1_date": "2020-02-14",
        "t2_file": "western_ghats_2024_t2.tif",
        "t2_date": "2024-02-18",
        "primary_threat": "Linear infrastructure, plantation interfaces & localized thinning",
        "utm_crs": "EPSG:32643 (UTM Zone 43N)",
        "coordinates": {"lat": 13.5100, "lon": 75.1000},
    },
    "Punjab, India": {
        "description": "Indo-Gangetic intensive agricultural basin displaying stark multi-temporal phenology shifts between peak Kharif paddy and post-harvest clearing.",
        "region_name": "Punjab, India",
        "t1_file": "punjab_2021_t1.tif",
        "t1_date": "2021-09-20",
        "t2_file": "punjab_2021_t2.tif",
        "t2_date": "2021-11-15",
        "primary_threat": "Crop residue clearing, seasonal harvest fallow & rotation",
        "utm_crs": "EPSG:32643 (UTM Zone 43N)",
        "coordinates": {"lat": 30.8659, "lon": 75.7845},
    },
    "Rajasthan, India": {
        "description": "Thar Desert arid ecotone demonstrating xerophytic scrub dynamics alongside Indira Gandhi Canal-induced agricultural greening.",
        "region_name": "Rajasthan, India",
        "t1_file": "rajasthan_2020_t1.tif",
        "t1_date": "2020-10-05",
        "t2_file": "rajasthan_2023_t2.tif",
        "t2_date": "2023-10-10",
        "primary_threat": "Aridity stress vs canal-irrigated agricultural expansion",
        "utm_crs": "EPSG:32643 (UTM Zone 43N)",
        "coordinates": {"lat": 27.5000, "lon": 72.5000},
    },
}


def get_available_sample_regions() -> list[str]:
    """Return a list of available curated sample region names."""
    return list(SAMPLE_CATALOGUE.keys())


def resolve_catalogue_key(region_query: str) -> str:
    """Resolve a region query string to an exact catalogue key."""
    if region_query in SAMPLE_CATALOGUE:
        return region_query

    # Exact case-insensitive match
    norm = region_query.strip().lower()
    for k in SAMPLE_CATALOGUE:
        if norm == k.lower():
            return k

    # Prefix or containment match (e.g. 'Rondônia, Brazil' matches 'Rondônia, Brazil (Deforestation Front)')
    for k in SAMPLE_CATALOGUE:
        if norm in k.lower() or k.lower() in norm:
            return k

    # Also match region_name
    for k, v in SAMPLE_CATALOGUE.items():
        if norm == v["region_name"].lower() or norm in v["region_name"].lower():
            return k

    raise KeyError(f"Region '{region_query}' not found in sample catalogue.")


def load_sample_pair(region_key: str) -> tuple[SentinelScene, SentinelScene, dict[str, Any]]:
    """Load the paired Sentinel-2 scenes for the specified sample region.

    Args:
        region_key: Name of region from get_available_sample_regions() or query string.

    Returns:
        Tuple of (scene_t1, scene_t2, metadata_dict).
    """
    canonical_key = resolve_catalogue_key(region_key)
    info = SAMPLE_CATALOGUE[canonical_key]
    t1_path = SAMPLE_DIR / info["t1_file"]
    t2_path = SAMPLE_DIR / info["t2_file"]

    if not t1_path.exists() or not t2_path.exists():
        missing = []
        if not t1_path.exists():
            missing.append(t1_path.name)
        if not t2_path.exists():
            missing.append(t2_path.name)
        raise FileNotFoundError(
            f"Satellite imagery for '{info['region_name']}' is currently unavailable (missing: {', '.join(missing)} in {SAMPLE_DIR}). "
            f"Please run 'python scripts/prepare_sample_data.py' or upload custom GeoTIFF files."
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
