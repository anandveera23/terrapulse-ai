"""Unit tests for change detection, differencing, and classification."""

import numpy as np
import pytest

from src.change_detection.classifier import classify_dndvi
from src.change_detection.differencing import compute_dndvi
from src.change_detection.land_cover import LandCoverBaselineModel


def test_dndvi_differencing():
    """Verify dNDVI calculation and statistical aggregations."""
    t1 = np.array([[0.80, 0.40], [0.20, np.nan]], dtype=np.float32)
    t2 = np.array([[0.20, 0.40], [0.60, 0.50]], dtype=np.float32)

    dndvi, stats = compute_dndvi(t1, t2)

    # Pixel (0,0): 0.20 - 0.80 = -0.60 (Decline)
    # Pixel (0,1): 0.40 - 0.40 =  0.00 (Stable)
    # Pixel (1,0): 0.60 - 0.20 = +0.40 (Increase)
    # Pixel (1,1): NaN
    assert pytest.approx(dndvi[0, 0], abs=1e-5) == -0.60
    assert pytest.approx(dndvi[0, 1], abs=1e-5) == 0.00
    assert pytest.approx(dndvi[1, 0], abs=1e-5) == 0.40
    assert np.isnan(dndvi[1, 1])

    assert stats["valid_pixels"] == 3
    assert stats["total_pixels"] == 4
    assert stats["decline_pixel_pct"] == pytest.approx(33.333, rel=1e-2)
    assert stats["increase_pixel_pct"] == pytest.approx(33.333, rel=1e-2)
    assert stats["stable_pixel_pct"] == pytest.approx(33.333, rel=1e-2)


def test_classify_dndvi_thresholds():
    """Verify ecological classification according to configurable thresholds."""
    dndvi = np.array([[-0.30, -0.15, 0.00, 0.15, 0.35, np.nan]], dtype=np.float32)

    classes, counts = classify_dndvi(
        dndvi,
        severe_loss_threshold=-0.25,
        moderate_loss_threshold=-0.10,
        moderate_gain_threshold=0.10,
        significant_gain_threshold=0.25,
    )

    assert classes[0, 0] == 1  # Severe Loss (-0.30 < -0.25)
    assert classes[0, 1] == 2  # Moderate Loss (-0.15)
    assert classes[0, 2] == 3  # Stable (0.00)
    assert classes[0, 3] == 4  # Moderate Gain (+0.15)
    assert classes[0, 4] == 5  # Significant Gain (+0.35)
    assert classes[0, 5] == 0  # Nodata

    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 1
    assert counts[4] == 1
    assert counts[5] == 1
    assert counts[0] == 1


def test_classify_threshold_validation():
    """Verify error when thresholds are improperly ordered."""
    dndvi = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="severe_loss_threshold"):
        classify_dndvi(dndvi, severe_loss_threshold=0.1, moderate_loss_threshold=0.0)


def test_land_cover_model_reproducibility():
    """Verify the Random Forest land cover model trains and achieves valid evaluation metrics."""
    lc_model = LandCoverBaselineModel(n_estimators=40, random_state=42)
    metrics = lc_model.train_baseline()

    assert metrics["accuracy"] >= 0.85
    assert len(metrics["confusion_matrix"]) == 5
    assert "Forest" in str(metrics["classification_report"])
    assert lc_model.is_trained == True
