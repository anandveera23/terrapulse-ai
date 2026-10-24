"""UI components, interactive Folium maps, and Plotly charts.

Provides modular visualization elements following the deep forest green,
muted teal, and warm off-white design system.
"""

from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer
from rasterio.crs import CRS


def get_wgs84_bounds(profile: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    """Compute WGS84 [lat_min, lon_min] and [lat_max, lon_max] bounds from profile.

    Returns:
        Tuple of (bounds_sw_ne, center_lat_lon, [minx, miny, maxx, maxy]).
    """
    transform = profile.get("transform")
    crs = profile.get("crs")
    width = profile.get("width", 100)
    height = profile.get("height", 100)

    # 4 corners in native projection
    xs = [transform[2], transform[2] + width * transform[0]]
    ys = [transform[5] + height * transform[4], transform[5]]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if crs is not None:
        crs_obj = CRS.from_user_input(crs)
        if not crs_obj.is_geographic:
            # Transform all 4 corners from native projected CRS to EPSG:4326 (lat, lon)
            transformer = Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
            corner_xs = [min_x, min_x, max_x, max_x]
            corner_ys = [min_y, max_y, min_y, max_y]
            lons, lats = transformer.transform(corner_xs, corner_ys)
            lat_min, lat_max = float(min(lats)), float(max(lats))
            lon_min, lon_max = float(min(lons)), float(max(lons))
        else:
            lon_min, lat_min = min_x, min_y
            lon_max, lat_max = max_x, max_y
    else:
        # Fallback to Rondônia default
        lat_min, lon_min = -10.50, -62.85
        lat_max, lon_max = -10.40, -62.75

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    return bounds, [center_lat, center_lon]


def create_interactive_map(
    center_lat_lon: list[float],
    zoom_start: int = 14,
    basemap: str = "OpenStreetMap",
) -> folium.Map:
    """Create a standardized base Folium map using public keyless tile providers."""
    try:
        m = folium.Map(
            location=center_lat_lon,
            zoom_start=zoom_start,
            tiles=basemap,
            control_scale=True,
        )
    except Exception:  # noqa: BLE001
        # Graceful fallback to default OpenStreetMap if a custom provider is unavailable
        m = folium.Map(
            location=center_lat_lon,
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
            control_scale=True,
        )

    # Add Esri World Imagery satellite basemap as optional keyless layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite Basemap (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    return m


def overlay_raster_on_map(
    m: folium.Map,
    image_rgba: np.ndarray,
    bounds: list[list[float]],
    layer_name: str,
    opacity: float = 0.85,
    show: bool = True,
):
    """Add an in-memory RGBA raster overlay onto a Folium map."""
    folium.raster_layers.ImageOverlay(
        image=image_rgba,
        bounds=bounds,
        name=layer_name,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        show=show,
    ).add_to(m)


def render_ndvi_colormap(ndvi_array: np.ndarray) -> np.ndarray:
    """Convert a 2D float32 NDVI array [-1.0 to 1.0] to RGBA uint8 for map overlay.

    Uses a diverging ecological palette:
    - Water/Bare (NDVI <= 0.0): Tan / Red-Brown
    - Sparse/Stressed (0.0 < NDVI <= 0.3): Warm Yellow
    - Moderate Canopy (0.3 < NDVI <= 0.6): Light / Yellow-Green
    - Dense Rainforest (NDVI > 0.6): Vibrant Emerald to Deep Forest Green
    """
    h, w = ndvi_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(ndvi_array)

    # Water / Non-vegetated (< 0.1)
    mask_low = valid & (ndvi_array < 0.1)
    rgba[mask_low] = [188, 108, 37, 210]  # Tan / Bare Soil

    # Sparse vegetation (0.1 to 0.3)
    mask_sparse = valid & (ndvi_array >= 0.1) & (ndvi_array < 0.3)
    rgba[mask_sparse] = [221, 161, 94, 215]  # Sand / Ochre

    # Moderate vegetation (0.3 to 0.5)
    mask_mod = valid & (ndvi_array >= 0.3) & (ndvi_array < 0.5)
    rgba[mask_mod] = [167, 201, 87, 225]  # Olive / Lime

    # Healthy Canopy (0.5 to 0.7)
    mask_health = valid & (ndvi_array >= 0.5) & (ndvi_array < 0.7)
    rgba[mask_health] = [82, 183, 136, 235]  # Emerald

    # Dense Rainforest (> 0.7)
    mask_dense = valid & (ndvi_array >= 0.7)
    rgba[mask_dense] = [27, 67, 50, 245]  # Deep Forest

    # Invalid / NaN stays transparent (alpha=0)
    return rgba


def plot_ndvi_distribution_comparison(
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
    t1_label: str = "T1 Baseline",
    t2_label: str = "T2 Follow-up",
) -> go.Figure:
    """Create a high-contrast Plotly histogram comparing NDVI distributions."""
    v1 = ndvi_t1[np.isfinite(ndvi_t1)]
    v2 = ndvi_t2[np.isfinite(ndvi_t2)]

    # Subsample if large for fast responsive rendering
    if len(v1) > 15000:
        v1 = np.random.choice(v1, 15000, replace=False)
    if len(v2) > 15000:
        v2 = np.random.choice(v2, 15000, replace=False)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=v1,
            name=t1_label,
            opacity=0.65,
            marker_color="#2A9D8F",
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=v2,
            name=t2_label,
            opacity=0.65,
            marker_color="#E76F51",
            nbinsx=40,
        )
    )

    fig.update_layout(
        title="NDVI Density Distribution Shift",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,35,27,0.4)",
        font={"family": "Inter, sans-serif", "color": "#F0F5F2", "size": 12},
        xaxis={
            "title": "NDVI Value (-1.0 to +1.0)",
            "gridcolor": "rgba(255,255,255,0.08)",
            "range": [-0.2, 1.0],
        },
        yaxis={"title": "Pixel Frequency", "gridcolor": "rgba(255,255,255,0.08)"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=320,
    )
    return fig


def plot_change_area_breakdown(stats_df: pd.DataFrame) -> go.Figure:
    """Create a styled horizontal bar chart of change categories."""
    df_valid = stats_df[stats_df["Class ID"] != 0].copy()

    fig = go.Figure(
        go.Bar(
            x=df_valid["Area (ha)"],
            y=df_valid["Category"],
            orientation="h",
            marker={"color": df_valid["Color"]},
            text=df_valid["Area (ha)"].apply(lambda x: f"{x:,.1f} ha"),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Vegetation Change Area by Category (Hectares)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,35,27,0.4)",
        font={"family": "Inter, sans-serif", "color": "#F0F5F2", "size": 12},
        xaxis={"title": "Area (ha)", "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.08)", "autorange": "reversed"},
        margin={"l": 100, "r": 30, "t": 50, "b": 40},
        height=300,
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm: list[list[int]], class_names: list[str]
) -> go.Figure:
    """Create a styled Confusion Matrix heatmap for the land cover model."""
    cm_arr = np.array(cm)
    # Row normalize to percentages
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = (cm_arr / row_sums) * 100.0

    text_labels = [
        [f"{cm_arr[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(len(class_names))]
        for i in range(len(class_names))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_pct,
            x=class_names,
            y=class_names,
            text=text_labels,
            texttemplate="%{text}",
            colorscale=[[0, "#0D1B13"], [0.5, "#2A9D8F"], [1.0, "#52B788"]],
            showscale=True,
        )
    )

    fig.update_layout(
        title="Land Cover Model Validation Confusion Matrix",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#F0F5F2", "size": 11},
        xaxis={"title": "Predicted Class", "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": "Actual Ground Truth", "gridcolor": "rgba(255,255,255,0.08)", "autorange": "reversed"},
        margin={"l": 120, "r": 40, "t": 50, "b": 50},
        height=380,
    )
    return fig
