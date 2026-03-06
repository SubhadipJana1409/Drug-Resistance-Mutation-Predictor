"""Tests for DrugResistancePredictor and visualization."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.simulator  import simulate_isolates, DRUGS
from src.models.predictor import DrugResistancePredictor, MODEL_NAMES


@pytest.fixture(scope="module")
def fitted_predictor():
    X, Y = simulate_isolates(n_per_drug=60, resistance_rate=0.4, seed=0)
    strat = Y.max(axis=1).values
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.25, stratify=strat, random_state=0)

    pred = DrugResistancePredictor(drugs=DRUGS, seed=0, cv_folds=3)
    pred.fit(X_tr, Y_tr)
    pred.evaluate(X_te, Y_te)
    return pred, X_tr, X_te, Y_tr, Y_te, X, Y


class TestDrugResistancePredictor:
    def test_results_has_all_drugs(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            assert drug in pred.results_

    def test_results_has_all_models(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            for m in MODEL_NAMES:
                assert m in pred.results_[drug], f"{drug}/{m} missing"

    def test_auc_in_range(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            for m, res in pred.results_[drug].items():
                assert 0.0 <= res["auc_roc"] <= 1.0, f"{drug}/{m} AUC out of range"

    def test_f1_in_range(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            for m, res in pred.results_[drug].items():
                assert 0.0 <= res["f1"] <= 1.0

    def test_confusion_matrix_shape(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            cm = pred.results_[drug]["logistic_l1"]["confusion"]
            assert cm.shape == (2, 2)

    def test_feature_importance_all_drugs(self, fitted_predictor):
        pred = fitted_predictor[0]
        for drug in DRUGS:
            assert drug in pred.feature_importance_
            assert len(pred.feature_importance_[drug]) > 0

    def test_top_resistance_snps_returns_df(self, fitted_predictor):
        pred = fitted_predictor[0]
        top = pred.top_resistance_snps("isoniazid", n=5)
        assert isinstance(top, pd.DataFrame)
        assert "snp" in top.columns
        assert "coefficient" in top.columns
        assert "odds_ratio" in top.columns
        assert len(top) <= 5

    def test_top_snps_invalid_drug_raises(self, fitted_predictor):
        pred = fitted_predictor[0]
        with pytest.raises(KeyError):
            pred.top_resistance_snps("made_up_drug")

    def test_cross_validate_returns_dict(self, fitted_predictor):
        pred, _, _, _, _, X, Y = fitted_predictor
        cv = pred.cross_validate(X, Y, model_name="logistic_l1")
        assert isinstance(cv, dict)
        for drug in DRUGS:
            assert drug in cv
            assert len(cv[drug]) == 3  # cv_folds=3

    def test_cross_validate_auc_reasonable(self, fitted_predictor):
        pred, _, _, _, _, X, Y = fitted_predictor
        cv = pred.cross_validate(X, Y, model_name="logistic_l1")
        for drug, scores in cv.items():
            assert scores.mean() >= 0.5, f"{drug} CV AUC below random"

    def test_save_load(self, fitted_predictor, tmp_path):
        pred, _, X_te, _, Y_te, _, _ = fitted_predictor
        p = tmp_path / "pred.joblib"
        pred.save(p)
        pred2 = DrugResistancePredictor.load(p)
        res2 = pred2.evaluate(X_te, Y_te)
        for drug in DRUGS:
            assert abs(res2[drug]["logistic_l1"]["auc_roc"] -
                       pred.results_[drug]["logistic_l1"]["auc_roc"]) < 1e-6


class TestPlots:
    def test_fig2_creates_file(self, fitted_predictor, tmp_path):
        from src.visualization.plots import fig2_performance_bar
        pred = fitted_predictor[0]
        fig2_performance_bar(pred.results_, DRUGS, Path(tmp_path))
        assert (tmp_path / "fig2_performance_bar.png").exists()

    def test_fig3_creates_file(self, fitted_predictor, tmp_path):
        from src.visualization.plots import fig3_roc_curves
        pred = fitted_predictor[0]
        fig3_roc_curves(pred.results_, "isoniazid", Path(tmp_path))
        assert (tmp_path / "fig3_roc_curves.png").exists()

    def test_fig5_creates_file(self, fitted_predictor, tmp_path):
        from src.visualization.plots import fig5_confusion_matrices
        pred = fitted_predictor[0]
        fig5_confusion_matrices(pred.results_, DRUGS, Path(tmp_path))
        assert (tmp_path / "fig5_confusion_matrices.png").exists()

    def test_fig6_creates_file(self, fitted_predictor, tmp_path):
        from src.visualization.plots import fig6_feature_importance
        pred = fitted_predictor[0]
        fig6_feature_importance(pred, DRUGS, Path(tmp_path))
        assert (tmp_path / "fig6_feature_importance.png").exists()

    def test_fig8_creates_file(self, fitted_predictor, tmp_path):
        from src.visualization.plots import fig8_mdr_heatmap
        _, _, _, _, _, _, Y = fitted_predictor
        fig8_mdr_heatmap(Y, Path(tmp_path))
        assert (tmp_path / "fig8_mdr_heatmap.png").exists()
