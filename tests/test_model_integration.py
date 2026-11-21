"""Runs the real trained model. Skipped when the weights are not downloaded (git lfs pull)."""

import pytest

from tests.conftest import model_weights_available, synthetic_fundus

pytestmark = pytest.mark.skipif(not model_weights_available(), reason="model weights not downloaded")


def test_real_model_prediction_and_full_pipeline(tmp_path):
    from src.inference.pipeline import run_full_analysis

    result = run_full_analysis(synthetic_fundus(), results_dir=str(tmp_path))
    assert result["prediction"] in result["classes"]
    assert abs(sum(result["probabilities"]) - 1) < 1e-4
    assert abs(sum(result["region_importance"].values()) - 1) < 1e-4 or \
        sum(result["region_importance"].values()) == 0
