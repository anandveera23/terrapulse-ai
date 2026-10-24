"""Unit tests for reporting modules (PDF and CSV generation)."""

import pandas as pd
import pytest

from src.reporting.csv_exporter import generate_change_stats_csv
from src.reporting.pdf_generator import generate_environmental_pdf


@pytest.fixture
def sample_reporting_data():
    """Create sample tabular records and summary metrics for reporting tests."""
    stats_df = pd.DataFrame(
        [
            {
                "Class ID": 1,
                "Category": "Severe Loss",
                "Pixel Count": 500,
                "Area (ha)": 5.0,
                "Area (km²)": 0.05,
                "Share (%)": 25.0,
                "Color": "#D90429",
            },
            {
                "Class ID": 3,
                "Category": "Relatively Stable",
                "Pixel Count": 1500,
                "Area (ha)": 15.0,
                "Area (km²)": 0.15,
                "Share (%)": 75.0,
                "Color": "#ADB5BD",
            },
        ]
    )

    summary_metrics = {
        "total_study_ha": 20.0,
        "total_valid_ha": 20.0,
        "net_change_ha": -5.0,
        "valid_coverage_pct": 100.0,
        "total_loss_ha": 5.0,
        "total_gain_ha": 0.0,
    }

    ndvi_stats_t1 = {"mean": 0.82}
    ndvi_stats_t2 = {"mean": 0.61}

    return stats_df, summary_metrics, ndvi_stats_t1, ndvi_stats_t2


def test_pdf_generation_validity(sample_reporting_data):
    """Verify PDF generator produces non-empty, valid PDF byte stream."""
    stats_df, summary, ndvi_t1, ndvi_t2 = sample_reporting_data

    pdf_bytes = generate_environmental_pdf(
        region_name="Rondônia, Brazil",
        t1_date="2020-07-18",
        t2_date="2023-07-22",
        summary_metrics=summary,
        stats_df=stats_df,
        ndvi_stats_t1=ndvi_t1,
        ndvi_stats_t2=ndvi_t2,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Standard PDF header magic bytes '%PDF-'
    assert pdf_bytes.startswith(b"%PDF-")


def test_csv_generation_validity(sample_reporting_data):
    """Verify CSV generator formats metadata comments and table records correctly."""
    stats_df, summary, _, _ = sample_reporting_data

    csv_output = generate_change_stats_csv(
        stats_df=stats_df,
        summary_metrics=summary,
        region_name="Mau Forest, Kenya",
        t1_date="2019-01-14",
        t2_date="2024-01-20",
    )

    assert "# Study Region: Mau Forest, Kenya" in csv_output
    assert "# Net Canopy Change: -5.00 ha" in csv_output
    assert "Class ID,Category,Pixel Count,Area (ha),Area (km\xc2\xb2),Share (%)" in csv_output or "Class ID,Category,Pixel Count,Area (ha),Area (km²),Share (%)" in csv_output
    assert "Severe Loss" in csv_output
    assert "Relatively Stable" in csv_output
