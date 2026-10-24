"""CSV export generator for zonal and change area statistics."""

import io
from typing import Any

import pandas as pd


def generate_change_stats_csv(
    stats_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
    region_name: str,
    t1_date: str,
    t2_date: str,
) -> str:
    """Generate a formatted CSV string including metadata header and tabular results."""
    buffer = io.StringIO()

    buffer.write("# TerraPulse AI — Environmental Intelligence Report\n")
    buffer.write(f"# Study Region: {region_name}\n")
    buffer.write(f"# Baseline Date (T1): {t1_date}\n")
    buffer.write(f"# Follow-up Date (T2): {t2_date}\n")
    buffer.write(f"# Total Study Area: {summary_metrics.get('total_study_ha', 0):.2f} ha\n")
    buffer.write(f"# Total Valid Observed Area: {summary_metrics.get('total_valid_ha', 0):.2f} ha\n")
    buffer.write(f"# Net Canopy Change: {summary_metrics.get('net_change_ha', 0):.2f} ha\n")
    buffer.write(f"# Valid Observation Coverage: {summary_metrics.get('valid_coverage_pct', 0):.2f}%\n")
    buffer.write("# ------------------------------------------------------------\n")

    # Export clean DataFrame without color column
    clean_df = stats_df[["Class ID", "Category", "Pixel Count", "Area (ha)", "Area (km²)", "Share (%)"]]
    clean_df.to_csv(buffer, index=False)

    return buffer.getvalue()
