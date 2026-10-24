"""Publication-grade PDF report generator using ReportLab.

Creates structured environmental intelligence reports containing executive KPIs,
satellite observation metadata, zonal statistics tables, and methodological caveats.
"""

import io
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_environmental_pdf(
    region_name: str,
    t1_date: str,
    t2_date: str,
    summary_metrics: dict[str, Any],
    stats_df: pd.DataFrame,
    ndvi_stats_t1: dict[str, float],
    ndvi_stats_t2: dict[str, float],
    report_title: str = "Satellite Environmental Intelligence Assessment",
) -> bytes:
    """Generate a publication-quality PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0D2818")
    accent_color = colors.HexColor("#2A9D8F")
    dark_text = colors.HexColor("#1A1A1A")
    muted_text = colors.HexColor("#555555")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=muted_text,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_text,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=dark_text,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )

    elements = []

    # --- Header Banner ---
    elements.append(Paragraph("TERRAPULSE AI — ENVIRONMENTAL INTELLIGENCE", subtitle_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(report_title, title_style))
    elements.append(
        Paragraph(
            f"Automated Sentinel-2 Multispectral Change Assessment | Study Area: {region_name}",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=14))

    # --- Section 1: Observation & Sensor Metadata ---
    elements.append(Paragraph("1. SATELLITE OBSERVATION METADATA", h2_style))
    meta_data = [
        [
            Paragraph("<b>Study Region:</b>", table_cell_style),
            Paragraph(region_name, table_cell_style),
            Paragraph("<b>Satellite Sensor:</b>", table_cell_style),
            Paragraph("Sentinel-2 MSI Level-2A (BOA)", table_cell_style),
        ],
        [
            Paragraph("<b>Observation T1 (Baseline):</b>", table_cell_style),
            Paragraph(t1_date, table_cell_style),
            Paragraph("<b>Observation T2 (Follow-up):</b>", table_cell_style),
            Paragraph(t2_date, table_cell_style),
        ],
        [
            Paragraph("<b>Spatial Resolution:</b>", table_cell_style),
            Paragraph("10.0 meters / pixel", table_cell_style),
            Paragraph("<b>Cloud Masking:</b>", table_cell_style),
            Paragraph("SCL Scene Classification Layer", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[130, 135, 130, 135])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # --- Section 2: Key Environmental Indicators (KPIs) ---
    elements.append(Paragraph("2. EXECUTIVE ENVIRONMENTAL INDICATORS", h2_style))
    mean_ndvi_1 = ndvi_stats_t1.get("mean", 0.0)
    mean_ndvi_2 = ndvi_stats_t2.get("mean", 0.0)
    net_ha = summary_metrics.get("net_change_ha", 0.0)
    valid_cov = summary_metrics.get("valid_coverage_pct", 0.0)
    total_valid = summary_metrics.get("total_valid_ha", 0.0)
    loss_ha = summary_metrics.get("total_loss_ha", 0.0)
    gain_ha = summary_metrics.get("total_gain_ha", 0.0)

    kpi_data = [
        [
            Paragraph("<b>Indicator</b>", table_header_style),
            Paragraph("<b>Value</b>", table_header_style),
            Paragraph("<b>Baseline Context</b>", table_header_style),
        ],
        [
            Paragraph("Mean NDVI Baseline (T1)", table_cell_style),
            Paragraph(f"{mean_ndvi_1:.3f}", table_cell_style),
            Paragraph("Healthy mature canopy baseline", table_cell_style),
        ],
        [
            Paragraph("Mean NDVI Follow-up (T2)", table_cell_style),
            Paragraph(f"{mean_ndvi_2:.3f}", table_cell_style),
            Paragraph("Post-period canopy condition", table_cell_style),
        ],
        [
            Paragraph("Estimated Total Canopy Loss", table_cell_style),
            Paragraph(f"<b>{loss_ha:,.2f} ha</b>", table_cell_style),
            Paragraph("Sum of Severe & Moderate loss zones", table_cell_style),
        ],
        [
            Paragraph("Estimated Total Canopy Gain", table_cell_style),
            Paragraph(f"<b>{gain_ha:,.2f} ha</b>", table_cell_style),
            Paragraph("Sum of Regrowth & Significant gain zones", table_cell_style),
        ],
        [
            Paragraph("Net Canopy Balance", table_cell_style),
            Paragraph(f"<b>{net_ha:+,.2f} ha</b>", table_cell_style),
            Paragraph("Gains minus Losses across valid pixels", table_cell_style),
        ],
        [
            Paragraph("Valid Pixel Coverage", table_cell_style),
            Paragraph(f"{valid_cov:.1f}% ({total_valid:,.1f} ha)", table_cell_style),
            Paragraph("Cloud-free, non-shadow area", table_cell_style),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[170, 120, 240])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEAEA")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # --- Section 3: Change Category Breakdown ---
    elements.append(Paragraph("3. VEGETATION CHANGE ZONAL BREAKDOWN", h2_style))
    breakdown_data = [
        [
            Paragraph("<b>Category</b>", table_header_style),
            Paragraph("<b>Pixels</b>", table_header_style),
            Paragraph("<b>Area (ha)</b>", table_header_style),
            Paragraph("<b>Area (km²)</b>", table_header_style),
            Paragraph("<b>Share (%)</b>", table_header_style),
        ]
    ]
    for _, row in stats_df.iterrows():
        breakdown_data.append(
            [
                Paragraph(str(row["Category"]), table_cell_style),
                Paragraph(f"{int(row['Pixel Count']):,}", table_cell_style),
                Paragraph(f"{row['Area (ha)']:,.2f}", table_cell_style),
                Paragraph(f"{row['Area (km²)']:,.4f}", table_cell_style),
                Paragraph(f"{row['Share (%)']:.2f}%", table_cell_style),
            ]
        )

    breakdown_table = Table(breakdown_data, colWidths=[180, 80, 90, 90, 90])
    breakdown_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent_color),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEAEA")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(breakdown_table)
    elements.append(Spacer(1, 14))

    # --- Section 4: Scientific Disclaimers & Methodology ---
    elements.append(Paragraph("4. SCIENTIFIC METHODOLOGY & LIMITATIONS", h2_style))
    disclaimer_text = (
        "<b>Important Scientific Distinction:</b> Results produced by this platform represent "
        "NDVI-based vegetation-change estimates and should not be cited as legally verified deforestation without "
        "secondary radar (SAR) or ground-truth confirmation. Multi-spectral index decreases can result from "
        "natural seasonal phenology, localized droughts, agricultural harvest cycles, or unmasked cloud shadows. "
        "All calculations strictly utilize clear, valid pixels filtered through Sentinel-2 Level-2A Scene "
        "Classification Layer (SCL) quality standards."
    )
    elements.append(Paragraph(disclaimer_text, body_style))
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    elements.append(
        Paragraph(
            "TerraPulse AI — Automated Geospatial Environmental Intelligence Platform | Open Source MIT",
            subtitle_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
