"""TerraPulse AI — UI Theme and Custom Styling System.

Provides custom CSS injecting deep forest green, muted teal, warm off-white,
and modern typography to deliver a premium geospatial analytics interface.
"""

import streamlit as st


def apply_theme():
    """Inject custom CSS rules into the Streamlit application."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container background and spacing */
    .stApp {
        background: #0A130E;
        color: #F0F5F2;
    }

    /* Header styling */
    .tp-brand-header {
        background: linear-gradient(135deg, rgba(19, 35, 27, 0.9) 0%, rgba(10, 19, 14, 0.95) 100%);
        border: 1px solid rgba(42, 157, 143, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }
    .tp-brand-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .tp-brand-subtitle {
        font-size: 0.95rem;
        color: #A3B899;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .tp-badge-live {
        background: rgba(42, 157, 143, 0.2);
        color: #52B788;
        border: 1px solid #2A9D8F;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* KPI Cards */
    .tp-kpi-card {
        background: rgba(19, 35, 27, 0.65);
        border: 1px solid rgba(82, 183, 136, 0.18);
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .tp-kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(42, 157, 143, 0.5);
    }
    .tp-kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8FA89B;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .tp-kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .tp-kpi-subtext {
        font-size: 0.78rem;
        color: #6C8276;
        margin-top: 4px;
    }

    /* Cards and containers */
    .tp-card {
        background: #111E17;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Alerts and Callouts */
    .tp-callout {
        border-left: 4px solid #2A9D8F;
        background: rgba(42, 157, 143, 0.08);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #D3E0D8;
        margin: 12px 0;
    }
    .tp-callout-warning {
        border-left-color: #E76F51;
        background: rgba(231, 111, 81, 0.08);
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0A130E;
    }
    ::-webkit-scrollbar-thumb {
        background: #1F362A;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2A9D8F;
    }

    /* Streamlit element overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.92rem;
        color: #8FA89B;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #52B788 !important;
        background-color: rgba(42, 157, 143, 0.12) !important;
        border-bottom: 2px solid #52B788 !important;
    }

    /* Primary buttons */
    .stButton>button {
        background: linear-gradient(135deg, #137547 0%, #0F5E39 100%);
        color: #FFFFFF;
        border: 1px solid rgba(82, 183, 136, 0.3);
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2A9D8F 0%, #137547 100%);
        border-color: #52B788;
        box-shadow: 0 4px 14px rgba(42, 157, 143, 0.35);
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0E1A13;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_brand_header(title="TerraPulse AI", subtitle="Satellite-Based Environmental Intelligence Platform", badge="Sentinel-2 L2A"):
    """Render the standard application brand header."""
    st.markdown(
        f"""
        <div class="tp-brand-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 class="tp-brand-title">
                        <span>🌿</span> {title}
                    </h1>
                    <p class="tp-brand-subtitle">{subtitle}</p>
                </div>
                <div>
                    <span class="tp-badge-live">{badge}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "", trend_color: str | None = None):
    """Render a styled KPI card."""
    color_style = f"style='color: {trend_color};'" if trend_color else ""
    st.markdown(
        f"""
        <div class="tp-kpi-card">
            <div class="tp-kpi-label">{label}</div>
            <div class="tp-kpi-value" {color_style}>{value}</div>
            <div class="tp-kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
