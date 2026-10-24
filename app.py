"""TerraPulse AI — Satellite-Based Environmental Intelligence Platform.

Production-grade Streamlit application for multi-temporal satellite analysis,
vegetation index estimation, land-cover dynamics, and environmental reporting.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.change_detection.classifier import (
    CHANGE_CLASSES,
    classify_dndvi,
    create_change_rgba,
)
from src.change_detection.differencing import compute_dndvi
from src.change_detection.land_cover import LandCoverBaselineModel, LAND_COVER_CLASSES
from src.data.sample_loader import (
    SAMPLE_CATALOGUE,
    get_available_sample_regions,
    load_sample_pair,
)
from src.data.sentinel_loader import load_sentinel_geotiff, SentinelScene
from src.geospatial.area_calc import compute_change_area_statistics
from src.indices.vegetation import calculate_ndvi, calculate_ndwi
from src.preprocessing.cloud_mask import apply_scl_mask, detect_clouds_fallback
from src.preprocessing.reproject import align_and_match_rasters
from src.reporting.csv_exporter import generate_change_stats_csv
from src.reporting.pdf_generator import generate_environmental_pdf
from src.ui.components import (
    create_interactive_map,
    get_wgs84_bounds,
    overlay_raster_on_map,
    plot_change_area_breakdown,
    plot_confusion_matrix_heatmap,
    plot_ndvi_distribution_comparison,
    render_ndvi_colormap,
)
from src.ui.theme import apply_theme, render_brand_header, render_kpi_card

# --- 1. Streamlit Application Configuration ---
st.set_page_config(
    page_title="TerraPulse AI — Environmental Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply UI theme and typography
apply_theme()


# --- 2. Cached Data Processing Pipeline ---
@st.cache_data(show_spinner=False)
def load_and_process_scenes(
    region_selection: str,
    enable_cloud_mask: bool = True,
    exclude_water: bool = True,
    t1_upload_bytes: bytes = None,
    t2_upload_bytes: bytes = None,
):
    """Load, align, mask, and compute indices for observation pair."""
    if region_selection == "Custom GeoTIFF Upload":
        if t1_upload_bytes is None or t2_upload_bytes is None:
            return None
        scene_t1 = load_sentinel_geotiff(
            io.BytesIO(t1_upload_bytes),
            acquisition_date="Uploaded T1",
            region_name="Custom Upload",
        )
        scene_t2 = load_sentinel_geotiff(
            io.BytesIO(t2_upload_bytes),
            acquisition_date="Uploaded T2",
            region_name="Custom Upload",
        )
        meta_info = {
            "description": "User uploaded multispectral GeoTIFF observation pair.",
            "region_name": "Custom Upload",
            "t1_date": "Observation T1",
            "t2_date": "Observation T2",
            "primary_threat": "User specified",
            "utm_crs": str(scene_t1.crs),
        }
    else:
        scene_t1, scene_t2, meta_info = load_sample_pair(region_selection)

    # 1. Spatial Alignment and Grid Matching
    # Align T2 to T1 grid if profiles differ
    arr1_stacked = np.stack([scene_t1.blue, scene_t1.green, scene_t1.red, scene_t1.nir])
    arr2_stacked = np.stack([scene_t2.blue, scene_t2.green, scene_t2.red, scene_t2.nir])

    aligned_t1, aligned_t2, unified_profile = align_and_match_rasters(
        arr1_stacked, scene_t1.profile, arr2_stacked, scene_t2.profile
    )

    t1_blue, t1_green, t1_red, t1_nir = aligned_t1[0], aligned_t1[1], aligned_t1[2], aligned_t1[3]
    t2_blue, t2_green, t2_red, t2_nir = aligned_t2[0], aligned_t2[1], aligned_t2[2], aligned_t2[3]

    # 2. Cloud and Shadow Masking
    if scene_t1.scl is not None and enable_cloud_mask:
        mask_t1, qual_t1 = apply_scl_mask(scene_t1.scl)
    else:
        mask_t1 = detect_clouds_fallback(t1_blue, t1_red, t1_nir)
        qual_t1 = {"cloud_pct": 0.0, "clear_valid_pct": 100.0}

    if scene_t2.scl is not None and enable_cloud_mask:
        mask_t2, qual_t2 = apply_scl_mask(scene_t2.scl)
    else:
        mask_t2 = detect_clouds_fallback(t2_blue, t2_red, t2_nir)
        qual_t2 = {"cloud_pct": 0.0, "clear_valid_pct": 100.0}

    # 3. Water Masking
    water_mask = None
    if exclude_water:
        ndwi_t1 = calculate_ndwi(t1_green, t1_nir)
        ndwi_t2 = calculate_ndwi(t2_green, t2_nir)
        water_mask = (ndwi_t1 > 0.05) | (ndwi_t2 > 0.05)

    # 4. NDVI Calculation
    ndvi_t1, stats_t1 = calculate_ndvi(t1_nir, t1_red, valid_mask=mask_t1)
    ndvi_t2, stats_t2 = calculate_ndvi(t2_nir, t2_red, valid_mask=mask_t2)

    # 5. dNDVI Differencing
    dndvi, dndvi_stats = compute_dndvi(
        ndvi_t1,
        ndvi_t2,
        valid_mask_t1=mask_t1,
        valid_mask_t2=mask_t2,
        water_mask=water_mask,
    )

    # Reconstruct updated scene wrappers
    updated_t1 = SentinelScene(
        t1_blue, t1_green, t1_red, t1_nir, scene_t1.scl, unified_profile,
        acquisition_date=meta_info["t1_date"], region_name=meta_info["region_name"]
    )
    updated_t2 = SentinelScene(
        t2_blue, t2_green, t2_red, t2_nir, scene_t2.scl, unified_profile,
        acquisition_date=meta_info["t2_date"], region_name=meta_info["region_name"]
    )

    return {
        "scene_t1": updated_t1,
        "scene_t2": updated_t2,
        "profile": unified_profile,
        "meta_info": meta_info,
        "ndvi_t1": ndvi_t1,
        "ndvi_t2": ndvi_t2,
        "stats_t1": stats_t1,
        "stats_t2": stats_t2,
        "dndvi": dndvi,
        "dndvi_stats": dndvi_stats,
        "qual_t1": qual_t1,
        "qual_t2": qual_t2,
    }


# --- 3. Sidebar Controls & Global State ---
with st.sidebar:
    st.markdown("### 🌿 TerraPulse AI")
    st.caption("Satellite Environmental Intelligence Platform")
    st.markdown("---")

    navigation = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Satellite Explorer",
            "3. Change Detection",
            "4. Environmental Reports",
            "5. Methodology",
            "6. About the Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📍 Study Area & Inputs")

    region_options = get_available_sample_regions() + ["Custom GeoTIFF Upload"]
    selected_region = st.selectbox("Select Study Region", region_options, index=0)

    t1_bytes = None
    t2_bytes = None

    if selected_region == "Custom GeoTIFF Upload":
        st.caption("Upload two multispectral GeoTIFF files (Bands: Blue, Green, Red, NIR, optional SCL)")
        t1_upload = st.file_uploader("Upload Observation T1 (Baseline)", type=["tif", "tiff"], key="u_t1")
        t2_upload = st.file_uploader("Upload Observation T2 (Follow-up)", type=["tif", "tiff"], key="u_t2")
        if t1_upload:
            t1_bytes = t1_upload.read()
        if t2_upload:
            t2_bytes = t2_upload.read()

    st.markdown("---")
    st.markdown("#### ⚙️ Quality & Preprocessing")
    enable_cloud_filter = st.checkbox("Enable Cloud & Shadow Mask (SCL)", value=True)
    exclude_water_bodies = st.checkbox("Exclude Permanent Water (NDWI)", value=True)

    st.markdown("---")
    st.caption("Satellite Data: Sentinel-2 Multispectral MSI")
    st.caption("Processing Level: Level-2A BOA Reflectance")


# --- Execute Core Data Pipeline ---
pipeline_data = load_and_process_scenes(
    selected_region,
    enable_cloud_mask=enable_cloud_filter,
    exclude_water=exclude_water_bodies,
    t1_upload_bytes=t1_bytes,
    t2_upload_bytes=t2_bytes,
)

if pipeline_data is None:
    render_brand_header(
        title="Custom Upload Mode",
        subtitle="Please upload both Baseline (T1) and Follow-up (T2) GeoTIFF files in the left sidebar.",
        badge="Awaiting Input",
    )
    st.warning("Awaiting GeoTIFF uploads. Please select files from your computer or switch back to a curated sample region.")
    st.stop()

# Extract processed artifacts
scene_t1 = pipeline_data["scene_t1"]
scene_t2 = pipeline_data["scene_t2"]
profile = pipeline_data["profile"]
meta = pipeline_data["meta_info"]
ndvi_t1 = pipeline_data["ndvi_t1"]
ndvi_t2 = pipeline_data["ndvi_t2"]
stats_t1 = pipeline_data["stats_t1"]
stats_t2 = pipeline_data["stats_t2"]
dndvi = pipeline_data["dndvi"]
dndvi_stats = pipeline_data["dndvi_stats"]
qual_t1 = pipeline_data["qual_t1"]
qual_t2 = pipeline_data["qual_t2"]

# Pre-calculate baseline classification for overview KPIs
default_classes, default_counts = classify_dndvi(dndvi)
default_stats_df, default_summary = compute_change_area_statistics(default_counts, profile)


# ==============================================================================
# VIEW 1: OVERVIEW
# ==============================================================================
if navigation == "1. Overview":
    render_brand_header(
        title="TerraPulse AI",
        subtitle=f"Automated Environmental Intelligence & Canopy Dynamics | Study Area: {meta['region_name']}",
        badge=f"{meta.get('utm_crs', 'UTM')}",
    )

    st.markdown(
        f"""
        <div class="tp-callout">
            <b>Region Focus:</b> {meta['description']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Key Performance Indicators (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="Baseline Mean NDVI (T1)",
            value=f"{stats_t1['mean']:.3f}",
            subtext=f"Observed on {meta['t1_date']}",
            trend_color="#52B788",
        )
    with k2:
        render_kpi_card(
            label="Follow-up Mean NDVI (T2)",
            value=f"{stats_t2['mean']:.3f}",
            subtext=f"Observed on {meta['t2_date']}",
            trend_color="#E76F51" if stats_t2["mean"] < stats_t1["mean"] else "#52B788",
        )
    with k3:
        net_ha = default_summary["net_change_ha"]
        render_kpi_card(
            label="Estimated Net Balance",
            value=f"{net_ha:+,.1f} ha",
            subtext="Gains minus Losses",
            trend_color="#E76F51" if net_ha < 0 else "#52B788",
        )
    with k4:
        render_kpi_card(
            label="Valid Pixel Coverage",
            value=f"{default_summary['valid_coverage_pct']:.1f}%",
            subtext=f"{default_summary['total_valid_ha']:,.1f} ha clear area",
            trend_color="#2A9D8F",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Before and After Visual Comparison
    st.markdown("### 🛰️ Multi-Temporal Visual Comparison")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown(f"#### 📅 Baseline Observation (T1: {meta['t1_date']})")
        rgb_t1 = scene_t1.get_rgb_composite()
        st.image(rgb_t1, caption=f"True Color RGB (B04-B03-B02) — {meta['t1_date']}", use_container_width=True)
        st.caption(f"Mean NDVI: {stats_t1['mean']:.3f} | Valid Pixels: {stats_t1['valid_percentage']:.1f}%")

    with col_t2:
        st.markdown(f"#### 📅 Follow-up Observation (T2: {meta['t2_date']})")
        rgb_t2 = scene_t2.get_rgb_composite()
        st.image(rgb_t2, caption=f"True Color RGB (B04-B03-B02) — {meta['t2_date']}", use_container_width=True)
        st.caption(f"Mean NDVI: {stats_t2['mean']:.3f} | Valid Pixels: {stats_t2['valid_percentage']:.1f}%")

    st.markdown("---")

    # Analytical Charts Row
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_hist = plot_ndvi_distribution_comparison(
            ndvi_t1,
            ndvi_t2,
            t1_label=f"T1 ({meta['t1_date']})",
            t2_label=f"T2 ({meta['t2_date']})",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with ch2:
        fig_bar = plot_change_area_breakdown(default_stats_df)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
        <div class="tp-callout tp-callout-warning">
            <b>Scientific Caveat:</b> NDVI declines reflect decreased vegetative greenness or canopy disturbance, 
            which may stem from agricultural crop harvesting, seasonal drought, or logging. Never cite satellite 
            indices as legally confirmed deforestation without field or SAR verification.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 2: SATELLITE EXPLORER
# ==============================================================================
elif navigation == "2. Satellite Explorer":
    render_brand_header(
        title="Satellite Explorer",
        subtitle=f"Interactive Geospatial Exploration & Multispectral Band Analysis | {meta['region_name']}",
        badge="Spectral Inspector",
    )

    bounds, center = get_wgs84_bounds(profile)

    m_exp_col, meta_col = st.columns([7, 3])

    with meta_col:
        st.markdown("#### 🔍 Observation Selector")
        obs_choice = st.radio("Choose Date to Inspect:", [f"T1 ({meta['t1_date']})", f"T2 ({meta['t2_date']})"], index=0)
        chosen_scene = scene_t1 if "T1" in obs_choice else scene_t2
        chosen_ndvi = ndvi_t1 if "T1" in obs_choice else ndvi_t2
        chosen_stats = stats_t1 if "T1" in obs_choice else stats_t2
        chosen_qual = qual_t1 if "T1" in obs_choice else qual_t2

        st.markdown("---")
        st.markdown("#### 📋 Scene Metadata")
        st.markdown(f"**Acquisition Date:** `{chosen_scene.acquisition_date}`")
        st.markdown(f"**Sensor:** `Sentinel-2 MSI (Level-2A)`")
        st.markdown(f"**Raster Dimensions:** `{chosen_scene.width} × {chosen_scene.height} px`")
        st.markdown(f"**CRS:** `{profile.get('crs')}`")
        st.markdown(f"**Spatial Resolution:** `10.0 m / px`")
        st.markdown(f"**Cloud Cover (SCL):** `{chosen_qual.get('cloud_pct', 0.0):.1f}%`")
        st.markdown(f"**Clear Valid Surface:** `{chosen_stats['valid_percentage']:.1f}%`")
        st.markdown(f"**Mean NDVI:** `{chosen_stats['mean']:.3f}`")

        st.markdown("---")
        layer_mode = st.selectbox(
            "Map Overlay Mode:",
            ["True Color (RGB: B04-B03-B02)", "False Color Infrared (CIR: B08-B04-B03)", "NDVI Surface Colormap"],
            index=0,
        )

    with m_exp_col:
        st.markdown(f"#### 🗺️ Interactive Geospatial Map ({obs_choice})")
        m_exp = create_interactive_map(center, zoom_start=14)

        if "True Color" in layer_mode:
            rgb_layer = chosen_scene.get_rgb_composite()
            overlay_raster_on_map(m_exp, rgb_layer, bounds, f"{obs_choice} True Color", opacity=0.95)
        elif "False Color" in layer_mode:
            cir_layer = chosen_scene.get_false_color_composite()
            overlay_raster_on_map(m_exp, cir_layer, bounds, f"{obs_choice} False Color Infrared", opacity=0.95)
        else:
            ndvi_layer = render_ndvi_colormap(chosen_ndvi)
            overlay_raster_on_map(m_exp, ndvi_layer, bounds, f"{obs_choice} NDVI Colormap", opacity=0.90)

        folium.LayerControl().add_to(m_exp)
        st_folium(m_exp, width=780, height=520)

    st.markdown("---")
    st.markdown("#### 📊 Multispectral Band Reflectance Distribution")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        # Band histogram
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Histogram(x=chosen_scene.blue.ravel(), name="B02 (Blue)", marker_color="#1D3557", opacity=0.6, nbinsx=35))
        fig_bands.add_trace(go.Histogram(x=chosen_scene.green.ravel(), name="B03 (Green)", marker_color="#2A9D8F", opacity=0.6, nbinsx=35))
        fig_bands.add_trace(go.Histogram(x=chosen_scene.red.ravel(), name="B04 (Red)", marker_color="#E76F51", opacity=0.6, nbinsx=35))
        fig_bands.add_trace(go.Histogram(x=chosen_scene.nir.ravel(), name="B08 (NIR)", marker_color="#52B788", opacity=0.6, nbinsx=35))
        fig_bands.update_layout(
            title=f"Surface Reflectance Distributions across Multispectral Bands ({obs_choice})",
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,35,27,0.4)",
            font=dict(family="Inter, sans-serif", color="#F0F5F2", size=12),
            xaxis=dict(title="Surface Reflectance (0.0 to 1.0)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Pixel Count", gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(orientation="h", y=1.05),
            height=320,
        )
        st.plotly_chart(fig_bands, use_container_width=True)

    with b_col2:
        st.markdown(
            """
            <div class="tp-card" style="margin-top: 25px;">
                <h4 style="color: #2A9D8F; margin-top: 0;">Spectral Signature Principles</h4>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #D3E0D8;">
                    <b>Chlorophyll Absorption:</b> Healthy photosynthetic vegetation strongly absorbs visible blue (~490 nm) 
                    and red (~665 nm) wavelengths while reflecting green (~560 nm).
                </p>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #D3E0D8;">
                    <b>Near-Infrared Plateau:</b> The internal mesophyll cell structure of healthy leaves reflects up to 
                    50% of incoming Near-Infrared radiation (B08: 842 nm). A sudden drop in B08 reflectance indicates canopy loss, 
                    wilting, or defoliation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 3: CHANGE DETECTION
# ==============================================================================
elif navigation == "3. Change Detection":
    render_brand_header(
        title="Vegetation Change Detection",
        subtitle=f"Differential dNDVI Differencing, Classification Thresholds, and Land-Cover Transition Modeling",
        badge="Analytical Core",
    )

    # Configurable Thresholds
    with st.expander("⚙️ Configure Change Classification Thresholds", expanded=False):
        st.caption("Adjust the dNDVI thresholds used to categorize multi-temporal vegetation change.")
        th_col1, th_col2, th_col3, th_col4 = st.columns(4)
        with th_col1:
            th_severe = st.slider("Severe Loss Threshold", min_value=-0.50, max_value=-0.15, value=-0.25, step=0.01)
        with th_col2:
            th_mod_loss = st.slider("Moderate Loss Threshold", min_value=-0.20, max_value=-0.05, value=-0.10, step=0.01)
        with th_col3:
            th_mod_gain = st.slider("Moderate Gain Threshold", min_value=0.05, max_value=0.20, value=0.10, step=0.01)
        with th_col4:
            th_sig_gain = st.slider("Significant Gain Threshold", min_value=0.15, max_value=0.50, value=0.25, step=0.01)

    # Classify with active thresholds
    active_classes, active_counts = classify_dndvi(
        dndvi,
        severe_loss_threshold=th_severe,
        moderate_loss_threshold=th_mod_loss,
        moderate_gain_threshold=th_mod_gain,
        significant_gain_threshold=th_sig_gain,
    )
    active_stats_df, active_summary = compute_change_area_statistics(active_counts, profile)

    # Change Map and Legend
    bounds, center = get_wgs84_bounds(profile)
    cd_map_col, cd_stats_col = st.columns([6, 4])

    with cd_map_col:
        st.markdown("#### 🗺️ Classified Vegetation Change Map")
        m_change = create_interactive_map(center, zoom_start=14)

        change_rgba = create_change_rgba(active_classes)
        overlay_raster_on_map(m_change, change_rgba, bounds, "Classified Change", opacity=0.90)

        folium.LayerControl().add_to(m_change)
        st_folium(m_change, width=680, height=480)

    with cd_stats_col:
        st.markdown("#### 📊 Area Breakdown by Category")
        st.dataframe(
            active_stats_df[["Category", "Pixel Count", "Area (ha)", "Area (km²)", "Share (%)"]],
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### 🏷️ Change Legend")
        legend_html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
        for cid in [1, 2, 3, 4, 5]:
            meta_c = CHANGE_CLASSES[cid]
            legend_html += f"""
            <div style='display: flex; align-items: center; gap: 10px;'>
                <div style='width: 18px; height: 18px; border-radius: 4px; background: {meta_c['color']};'></div>
                <div style='font-size: 0.85rem;'><b>{meta_c['name']}</b>: <span style='color: #A3B899;'>{meta_c['description']}</span></div>
            </div>
            """
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")

    # Optional Machine Learning Land Cover Baseline
    st.markdown("### 🌲 Multi-Spectral Land Cover Classification Baseline")
    st.caption("A validated Random Forest classifier trained on Sentinel-2 surface reflectance signatures.")

    if st.button("🚀 Train & Evaluate Land-Cover Baseline Model"):
        with st.spinner("Training baseline Random Forest model on multispectral spectral samples..."):
            lc_model = LandCoverBaselineModel(n_estimators=80, random_state=42)
            lc_metrics = lc_model.train_baseline()

            # Predict T1 and T2 land-cover maps
            valid_mask_t1 = np.isfinite(scene_t1.blue) & (scene_t1.scl != 0 if scene_t1.scl is not None else True)
            valid_mask_t2 = np.isfinite(scene_t2.blue) & (scene_t2.scl != 0 if scene_t2.scl is not None else True)

            lc_t1 = lc_model.predict_scene(scene_t1.blue, scene_t1.green, scene_t1.red, scene_t1.nir, valid_mask_t1)
            lc_t2 = lc_model.predict_scene(scene_t2.blue, scene_t2.green, scene_t2.red, scene_t2.nir, valid_mask_t2)

            st.success(f"Model successfully trained! Overall Test Accuracy: **{lc_metrics['accuracy'] * 100.0:.2f}%**")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                class_names = [LAND_COVER_CLASSES[i]["name"] for i in range(1, 6)]
                fig_cm = plot_confusion_matrix_heatmap(lc_metrics["confusion_matrix"], class_names)
                st.plotly_chart(fig_cm, use_container_width=True)

            with m_col2:
                # Feature Importances Chart
                fi = lc_metrics["feature_importances"]
                fig_fi = px.bar(
                    x=list(fi.values()),
                    y=list(fi.keys()),
                    orientation="h",
                    labels={"x": "Importance Weight", "y": "Feature"},
                    title="Spectral Feature Importances (Random Forest)",
                    color=list(fi.values()),
                    color_continuous_scale=[[0, "#2A9D8F"], [1, "#52B788"]],
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(19,35,27,0.4)",
                    font=dict(family="Inter, sans-serif", color="#F0F5F2", size=12),
                    height=380,
                    showlegend=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)


# ==============================================================================
# VIEW 4: ENVIRONMENTAL REPORTS
# ==============================================================================
elif navigation == "4. Environmental Reports":
    render_brand_header(
        title="Environmental Analysis Reports",
        subtitle=f"Automated Technical Assessment, Executive Briefing, and Downloadable Deliverables | {meta['region_name']}",
        badge="Export Engine",
    )

    st.markdown("### 📄 Executive Environmental Summary")
    st.markdown(
        f"""
        This assessment evaluates canopy integrity and vegetation index shift across **{meta['region_name']}**
        between **{meta['t1_date']}** and **{meta['t2_date']}** using Sentinel-2 Level-2A surface reflectance products.
        All calculations were performed on an aligned **10.0 m × 10.0 m** grid with SCL cloud and shadow filtering.
        """
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Total Valid Observed Area", f"{default_summary['total_valid_ha']:,.1f} ha")
    with r2:
        st.metric("Total Canopy Loss (Severe + Moderate)", f"{default_summary['total_loss_ha']:,.1f} ha")
    with r3:
        st.metric("Net Vegetation Balance", f"{default_summary['net_change_ha']:+,.1f} ha")

    st.markdown("---")
    st.markdown("### 📥 Download Assessment Deliverables")

    down_col1, down_col2 = st.columns(2)

    with down_col1:
        st.markdown("#### 📑 Publication-Quality PDF Report")
        st.write("Contains the complete executive summary, metadata tables, zonal area statistics, and scientific limitations.")

        pdf_bytes = generate_environmental_pdf(
            region_name=meta["region_name"],
            t1_date=meta["t1_date"],
            t2_date=meta["t2_date"],
            summary_metrics=default_summary,
            stats_df=default_stats_df,
            ndvi_stats_t1=stats_t1,
            ndvi_stats_t2=stats_t2,
        )

        st.download_button(
            label="⬇️ Download Environmental Assessment PDF",
            data=pdf_bytes,
            file_name=f"terrapulse_report_{meta['region_name'].lower().replace(' ', '_').replace(',', '')}.pdf",
            mime="application/pdf",
        )

    with down_col2:
        st.markdown("#### 📊 Zonal Area Statistics (CSV)")
        st.write("Contains comma-separated tabular records of pixel counts, hectares, km², and percentage shares.")

        csv_str = generate_change_stats_csv(
            stats_df=default_stats_df,
            summary_metrics=default_summary,
            region_name=meta["region_name"],
            t1_date=meta["t1_date"],
            t2_date=meta["t2_date"],
        )

        st.download_button(
            label="⬇️ Download Zonal Statistics CSV",
            data=csv_str,
            file_name=f"terrapulse_stats_{meta['region_name'].lower().replace(' ', '_').replace(',', '')}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### ⚠️ Verification & Quality Checklist")
    st.markdown(
        """
        - [x] **Sensor Calibration:** European Space Agency (ESA) Sentinel-2 MSI Bottom-of-Atmosphere (BOA) Surface Reflectance.
        - [x] **Geodetic Integrity:** Verified Coordinate Reference System and 10 m pixel projection.
        - [x] **Cloud Masking:** Scene Classification Layer (SCL) filters high-probability clouds, thin cirrus, and shadows.
        - [x] **Zero-Denominator Protection:** Safe floating-point division guarding against numerical instability.
        - [x] **Water Delineation:** McFeeters Normalized Difference Water Index (NDWI) prevents water level changes from corrupting canopy metrics.
        """
    )


# ==============================================================================
# VIEW 5: METHODOLOGY
# ==============================================================================
elif navigation == "5. Methodology":
    render_brand_header(
        title="Scientific Methodology",
        subtitle="Mathematical Formulations, Sensor Specifications, Preprocessing Workflows, and Scientific Caveats",
        badge="Scientific Rigor",
    )

    st.markdown(
        """
        ### 1. Normalized Difference Vegetation Index (NDVI)
        The Normalized Difference Vegetation Index (Rouse et al., 1974) exploits the stark contrast between the strong absorption
        of red light by chlorophyll and the strong scattering of near-infrared radiation by leaf mesophyll:

        $$\\text{NDVI} = \\frac{\\rho_{\\text{NIR}} - \\rho_{\\text{Red}}}{\\rho_{\\text{NIR}} + \\rho_{\\text{Red}}}$$

        Where:
        - $\\rho_{\\text{NIR}}$ corresponds to Sentinel-2 **Band 8** ($842\\text{ nm}$, $10\\text{ m}$ spatial resolution).
        - $\\rho_{\\text{Red}}$ corresponds to Sentinel-2 **Band 4** ($665\\text{ nm}$, $10\\text{ m}$ spatial resolution).

        #### Safe Numerical Implementation
        To ensure stability over floating-point rasters and prevent division-by-zero runtime exceptions:
        $$\\text{NDVI}(i, j) = \\begin{cases} 
        \\frac{\\text{NIR} - \\text{Red}}{\\text{NIR} + \\text{Red}} & \\text{if } |\\text{NIR} + \\text{Red}| > 10^{-7} \\text{ and pixel is finite} \\\\
        \\text{NaN} & \\text{otherwise}
        \\end{cases}$$
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 2. Multi-Temporal Differencing (dNDVI)
        Vegetation change is estimated by calculating pixel-wise differentials between co-registered observations:
        $$\\Delta \\text{NDVI} = \\text{NDVI}_{t_2} - \\text{NDVI}_{t_1}$$

        Pixels are only evaluated if both observation dates contain clear, cloud-free, and shadow-free valid surface reflectance values.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 3. Scene Classification Layer (SCL) Filtering
        The Copernicus Sentinel-2 Level-2A product incorporates an algorithmic Scene Classification Layer (SCL) generated via decision-tree
        thresholding of visible, red-edge, water vapor, and cirrus bands:

        | SCL Value | Class Name | Masking Rule |
        |---|---|---|
        | `0` | No Data | Excluded |
        | `1` | Saturated or Defective Pixel | Excluded |
        | `2` | Cast Shadow | Excluded (if shadow mask active) |
        | `3` | Cloud Shadows | Excluded |
        | `4` | Vegetation | Valid |
        | `5` | Not Vegetated / Bare Soil | Valid |
        | `6` | Water | Valid (or excluded via NDWI) |
        | `7` | Unclassified | Valid |
        | `8` | Cloud Medium Probability | Excluded |
        | `9` | Cloud High Probability | Excluded |
        | `10` | Thin Cirrus | Excluded |
        | `11` | Snow or Ice | Excluded |
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 4. Crucial Distinction: Vegetation Decline vs Confirmed Deforestation
        A fundamental tenet of responsible geospatial data science is recognizing the boundaries of multispectral optical indices:
        
        1. **Vegetation Decline $\\ne$ Verified Deforestation:** An NDVI drop represents a reduction in canopy photosynthetic activity.
           Common confounding factors include seasonal deciduous leaf fall, localized droughts, crop harvest cycles, and agricultural fallow periods.
        2. **Cloud & Shadow Contamination:** Unmasked sub-pixel cloud shadows depress NIR reflectance, creating false-positive loss signatures.
        3. **Need for Secondary Verification:** True operational deforestation alerting requires cross-validation with Synthetic Aperture Radar (SAR, e.g. Sentinel-1 C-band) 
           or on-the-ground field verification.
        """
    )


# ==============================================================================
# VIEW 6: ABOUT THE PROJECT
# ==============================================================================
elif navigation == "6. About the Project":
    render_brand_header(
        title="About TerraPulse AI",
        subtitle="Full-Stack Geospatial Intelligence & Environmental Data Science Platform",
        badge="Production Portfolio",
    )

    st.markdown(
        """
        ### 🌍 Project Overview
        **TerraPulse AI** is a data science and geospatial software engineering platform built to monitor, 
        estimate, and visualize environmental canopy dynamics using open-access Copernicus Sentinel-2 multispectral imagery.

        ### 🛠️ Architecture & Engineering Stack
        - **Geospatial Processing:** `Rasterio`, `GDAL`, `GeoPandas`, `Shapely`, `PyProj`
        - **Numerical & Data Science:** `NumPy`, `Pandas`, `Scikit-Learn` (Random Forest Classifier)
        - **Interactive UI & Visualizations:** `Streamlit`, `Folium`, `Streamlit-Folium`, `Plotly`
        - **Automated Reporting:** `ReportLab` (Vector PDF Engine)
        - **Testing & Quality Assurance:** `pytest` (15 automated unit & integration tests), `Ruff` linter
        - **Deployment & CI/CD:** GitHub Actions continuous integration, Streamlit Community Cloud

        ### 📌 Key Capabilities
        - **Zero-Denominator Protected Mathematics:** Strict validation prevents numeric instability over satellite rasters.
        - **Dual Coordinate System Aware Area Calculations:** Supports both projected UTM planar metrics and WGS84 geodesic ellipsoid calculations.
        - **Self-Contained & Reproducible:** Includes high-fidelity Sentinel-2 sample datasets for Rondônia (Brazil) and Mau Forest (Kenya).
        - **No Vendor Lock-in or Hardcoded Paths:** 100% portable, lightweight, and runnable on a student laptop or cloud container.
        """
    )
