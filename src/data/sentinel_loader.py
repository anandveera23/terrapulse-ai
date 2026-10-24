"""Sentinel-2 multispectral raster loader and validator.

Supports Level-2A surface reflectance GeoTIFF ingestion from file paths,
byte buffers (Streamlit file upload), and in-memory arrays.
"""

import io
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile


class SentinelScene:
    """Represents a validated multispectral satellite observation."""

    def __init__(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        scl: np.ndarray | None,
        profile: dict[str, Any],
        acquisition_date: str = "Unknown Date",
        region_name: str = "Custom Region",
    ):
        self.blue = blue
        self.green = green
        self.red = red
        self.nir = nir
        self.scl = scl
        self.profile = profile
        self.acquisition_date = acquisition_date
        self.region_name = region_name

        self.crs = profile.get("crs")
        self.transform = profile.get("transform")
        self.height, self.width = blue.shape

    def get_rgb_composite(self, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
        """Generate a contrast-stretched True Color (RGB: B04, B03, B02) composite (uint8)."""
        r = self.red
        g = self.green
        b = self.blue

        rgb = np.stack([r, g, b], axis=-1)
        rgb_stretched = np.zeros_like(rgb, dtype=np.uint8)

        for i in range(3):
            band = rgb[:, :, i]
            valid = band[np.isfinite(band) & (band > 0)]
            if valid.size > 0:
                low = np.percentile(valid, p_min)
                high = np.percentile(valid, p_max)
                if high > low:
                    stretched = np.clip((band - low) / (high - low), 0.0, 1.0) * 255.0
                    rgb_stretched[:, :, i] = stretched.astype(np.uint8)
        return rgb_stretched

    def get_false_color_composite(self, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
        """Generate False Color Infrared (CIR: B08 NIR, B04 Red, B03 Green) composite (uint8).

        Dense healthy vegetation appears bright red/crimson in CIR.
        """
        nir = self.nir
        r = self.red
        g = self.green

        cir = np.stack([nir, r, g], axis=-1)
        cir_stretched = np.zeros_like(cir, dtype=np.uint8)

        for i in range(3):
            band = cir[:, :, i]
            valid = band[np.isfinite(band) & (band > 0)]
            if valid.size > 0:
                low = np.percentile(valid, p_min)
                high = np.percentile(valid, p_max)
                if high > low:
                    stretched = np.clip((band - low) / (high - low), 0.0, 1.0) * 255.0
                    cir_stretched[:, :, i] = stretched.astype(np.uint8)
        return cir_stretched


def load_sentinel_geotiff(
    source: str | io.BytesIO | bytes,
    acquisition_date: str = "Unknown Date",
    region_name: str = "Custom Region",
) -> SentinelScene:
    """Load and validate a Sentinel-2 GeoTIFF from disk or an uploaded memory buffer.

    Expected band layouts:
    - 4 bands: [B02 Blue, B03 Green, B04 Red, B08 NIR]
    - 5 bands: [B02 Blue, B03 Green, B04 Red, B08 NIR, SCL Quality]

    Applies surface reflectance scaling (divides by 10,000 if raw integer DN).
    """
    if isinstance(source, (bytes, io.BytesIO)):
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        memfile = MemoryFile(source)
        dataset = memfile.open()
    else:
        dataset = rasterio.open(source)

    try:
        profile = dataset.profile.copy()
        band_count = dataset.count

        if band_count < 2:
            raise ValueError(
                f"Dataset must have at least 2 bands (Red & NIR). Found {band_count} bands."
            )

        data = dataset.read()  # Shape: (bands, height, width)

        # Scale Digital Numbers (DN) to surface reflectance [0.0 - 1.0]
        # Sentinel-2 L2A BOA reflectance = DN / 10000.0
        data_f = data.astype(np.float32)
        scl_band = None

        if band_count >= 5:
            # 5th band is SCL (discrete integers 0-11, do not divide)
            scl_band = data[4].astype(np.int32)
            for b in range(4):
                if np.nanmax(data_f[b]) > 2.0:
                    data_f[b] /= 10000.0
            blue = data_f[0]
            green = data_f[1]
            red = data_f[2]
            nir = data_f[3]
        elif band_count == 4:
            for b in range(4):
                if np.nanmax(data_f[b]) > 2.0:
                    data_f[b] /= 10000.0
            blue = data_f[0]
            green = data_f[1]
            red = data_f[2]
            nir = data_f[3]
        elif band_count == 2:
            # Assume Band 1 = Red, Band 2 = NIR
            for b in range(2):
                if np.nanmax(data_f[b]) > 2.0:
                    data_f[b] /= 10000.0
            red = data_f[0]
            nir = data_f[1]
            blue = red.copy()
            green = (red + nir) / 2.0
        else:
            for b in range(min(band_count, 4)):
                if np.nanmax(data_f[b]) > 2.0:
                    data_f[b] /= 10000.0
            blue = data_f[0]
            green = data_f[1] if band_count > 1 else blue
            red = data_f[2] if band_count > 2 else blue
            nir = data_f[3] if band_count > 3 else red

        scene = SentinelScene(
            blue=blue,
            green=green,
            red=red,
            nir=nir,
            scl=scl_band,
            profile=profile,
            acquisition_date=acquisition_date,
            region_name=region_name,
        )
        return scene

    finally:
        dataset.close()
