"""
Drug Resistance Mutation Predictor.

Per-drug binary classifier: Resistant (1) vs Susceptible (0)
from SNP presence/absence profiles.

Models compared
---------------
1. logistic_l1   – Lasso-penalised LR (feature selection → interpretable)
2. logistic_l2   – Ridge-penalised LR (baseline LR)
3. random_forest – Ensemble for non-linear SNP interactions
4. svm_rbf       – Kernel SVM
5. gradient_boost– XGBoost-style gradient boosting

Feature importance: L1 LR coefficients → directly maps SNPs to resistance odds.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = ["logistic_l1", "logistic_l2", "random_forest", "svm_rbf", "gradient_boost"]
MODEL_LABELS = {
    "logistic_l1":    "Logistic (L1)",
    "logistic_l2":    "Logistic (L2)",
    "random_forest":  "Random Forest",
    "svm_rbf":        "SVM (RBF)",
    "gradient_boost": "Gradient Boosting",
}


class DrugResistancePredictor:
    """
    Multi-drug resistance predictor.
    Trains one model per drug, compares 5 classifiers.

    Parameters
    ----------
    drugs    : list of drug names to predict
    seed     : random seed
    cv_folds : cross-validation folds
    """

    def __init__(
        self,
        drugs: list[str],
        seed: int = 42,
        cv_folds: int = 5,
    ):
        self.drugs    = drugs
        self.seed     = seed
        self.cv_folds = cv_folds

        self._scalers: dict[str, StandardScaler] = {}
        self._models:  dict[str, dict] = {}  # drug → {model_name → fitted model}
        self.results_:  dict[str, dict] = {}  # drug → {model → metrics}
        self.feature_importance_: dict[str, pd.Series] = {}

    # ── Model factory ─────────────────────────────────────────────────────────
    def _make_models(self) -> dict:
        return {
            "logistic_l1": LogisticRegression(
                penalty="l1", solver="liblinear", C=0.5,
                max_iter=1000, random_state=self.seed),
            "logistic_l2": LogisticRegression(
                penalty="l2", solver="lbfgs", C=1.0,
                max_iter=1000, random_state=self.seed),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=8,
                random_state=self.seed, n_jobs=1),
            "svm_rbf": SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=self.seed),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.1,
                random_state=self.seed),
        }

    # ── Train ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
    ) -> "DrugResistancePredictor":
        feature_names = X_train.columns.tolist()

        for drug in self.drugs:
            y = Y_train[drug].values
            if y.sum() == 0 or y.sum() == len(y):
                logger.warning("Drug %s has no variation; skipping", drug)
                continue

            logger.info("Training models for: %s  (n=%d, n_pos=%d)", drug, len(y), y.sum())

            # Scale features (needed for LR, SVM)
            sc = StandardScaler().fit(X_train.values)
            self._scalers[drug] = sc
            Xs = sc.transform(X_train.values)

            self._models[drug] = {}
            self.results_[drug] = {}

            for name, model in self._make_models().items():
                X_input = Xs if name in ("logistic_l1","logistic_l2","svm_rbf") \
                         else X_train.values
                model.fit(X_input, y)
                self._models[drug][name] = model
                logger.info("  ✅ %s", name)

            # Feature importance from L1 logistic regression
            l1_model = self._models[drug]["logistic_l1"]
            coef     = l1_model.coef_[0]
            self.feature_importance_[drug] = pd.Series(
                coef, index=feature_names
            ).sort_values(key=abs, ascending=False)

        return self

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
    ) -> dict[str, dict]:
        for drug in self.drugs:
            if drug not in self._models:
                continue
            y_true = Y_test[drug].values
            Xs     = self._scalers[drug].transform(X_test.values)
            self.results_[drug] = {}

            for name, model in self._models[drug].items():
                X_input = Xs if name in ("logistic_l1","logistic_l2","svm_rbf") \
                         else X_test.values
                proba = model.predict_proba(X_input)[:, 1]
                pred  = (proba >= 0.5).astype(int)
                fpr, tpr, _ = roc_curve(y_true, proba)
                prec_c, rec_c, _ = precision_recall_curve(y_true, proba)

                self.results_[drug][name] = {
                    "accuracy":  round(accuracy_score(y_true, pred), 4),
                    "auc_roc":   round(roc_auc_score(y_true, proba), 4),
                    "auc_pr":    round(average_precision_score(y_true, proba), 4),
                    "f1":        round(f1_score(y_true, pred, zero_division=0), 4),
                    "mcc":       round(matthews_corrcoef(y_true, pred), 4),
                    "confusion": confusion_matrix(y_true, pred),
                    "fpr": fpr, "tpr": tpr,
                    "prec_curve": prec_c, "rec_curve": rec_c,
                    "proba": proba, "pred": pred, "y_true": y_true,
                }

            best = max(self.results_[drug], key=lambda m: self.results_[drug][m]["auc_roc"])
            logger.info(
                "  %s → best=%s  AUC=%.3f  F1=%.3f  MCC=%.3f",
                drug, best,
                self.results_[drug][best]["auc_roc"],
                self.results_[drug][best]["f1"],
                self.results_[drug][best]["mcc"],
            )

        return self.results_

    # ── Cross-validation ──────────────────────────────────────────────────────
    def cross_validate(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        model_name: str = "logistic_l1",
    ) -> dict[str, np.ndarray]:
        """5-fold CV AUC-ROC for primary model per drug."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.seed)
        cv_results = {}
        for drug in self.drugs:
            y = Y[drug].values
            if y.sum() < self.cv_folds:
                continue
            sc = StandardScaler()
            Xs = sc.fit_transform(X.values)
            model = self._make_models()[model_name]
            scores = cross_val_score(model, Xs, y, cv=cv,
                                     scoring="roc_auc", n_jobs=1)
            cv_results[drug] = scores
            logger.info("  CV %s/%s: %.3f ± %.3f",
                        drug, model_name, scores.mean(), scores.std())
        return cv_results

    # ── Top resistance SNPs ───────────────────────────────────────────────────
    def top_resistance_snps(self, drug: str, n: int = 10) -> pd.DataFrame:
        """Return top resistance-associated SNPs for a drug (L1 LR coefficients)."""
        if drug not in self.feature_importance_:
            raise KeyError(f"No model fitted for {drug}")
        imp = self.feature_importance_[drug]
        top = imp[imp > 0].nlargest(n)
        return pd.DataFrame({
            "snp":        top.index,
            "coefficient": top.values,
            "odds_ratio":  np.exp(top.values),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved predictor → %s", path)

    @staticmethod
    def load(path: str | Path) -> "DrugResistancePredictor":
        return joblib.load(path)
