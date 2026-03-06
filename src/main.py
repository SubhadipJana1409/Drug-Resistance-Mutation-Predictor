"""
Day 23 · Drug Resistance Mutation Predictor
==========================================
SNP profiles → antibiotic resistance (Resistant / Susceptible)
Pathogens: M. tuberculosis, E. coli, S. aureus
Drugs:     Isoniazid, Rifampicin, Ciprofloxacin, Ampicillin, Methicillin

Pipeline
--------
1. Simulate SNP profiles + resistance labels (5 drugs × 120 isolates)
2. Train 5 classifiers per drug (L1-LR, L2-LR, RF, SVM, GBM)
3. Evaluate on held-out test set
4. Cross-validate primary model (L1 logistic regression)
5. Extract top resistance SNPs (L1 coefficients)
6. Generate 9 publication-quality figures + save outputs

Usage
-----
    python -m src.main
    python -m src.main --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.simulator      import simulate_isolates, DRUGS, PATHOGENS, make_fasta_record
from src.models.predictor    import DrugResistancePredictor
from src.visualization.plots import generate_all
from src.utils.logger        import setup_logging
from src.utils.config        import load_config

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Drug resistance mutation predictor")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    out  = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Day 23 · Drug Resistance Mutation Predictor")
    logger.info("=" * 60)

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # ── Step 1: Simulate isolates ─────────────────────────────────────────────
    logger.info("[1/5] Simulating SNP profiles …")
    X_df, Y_df = simulate_isolates(
        n_per_drug=data_cfg.get("n_per_drug", 120),
        resistance_rate=data_cfg.get("resistance_rate", 0.35),
        seed=data_cfg.get("seed", 42),
    )
    logger.info("Dataset: %d isolates × %d SNP features",
                len(X_df), X_df.shape[1])
    for drug in DRUGS:
        n_r = Y_df[drug].sum()
        logger.info("  %-15s  resistant=%d / %d  (%.0f%%)",
                    drug, n_r, len(Y_df), 100 * n_r / len(Y_df))

    # Save FASTA-like SNP file (Biopython)
    fasta_path = out / "isolate_snp_profiles.fasta"
    with open(fasta_path, "w") as fh:
        for iso_id in X_df.index[:50]:   # first 50 as example
            fh.write(make_fasta_record(iso_id, X_df.loc[iso_id].values))
    logger.info("Saved FASTA SNP profiles → %s", fasta_path)

    # ── Step 2: Train / test split ────────────────────────────────────────────
    logger.info("[2/5] Splitting data …")
    # Stratify on combined resistance label (any-resistant)
    strat_label = Y_df.max(axis=1).values
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_df, Y_df,
        test_size=data_cfg.get("test_size", 0.25),
        stratify=strat_label,
        random_state=data_cfg.get("seed", 42),
    )
    logger.info("Train: %d | Test: %d", len(X_tr), len(X_te))

    # ── Step 3: Train all models ──────────────────────────────────────────────
    logger.info("[3/5] Training classifiers …")
    predictor = DrugResistancePredictor(
        drugs=DRUGS,
        seed=model_cfg.get("seed", 42),
        cv_folds=model_cfg.get("cv_folds", 5),
    )
    predictor.fit(X_tr, Y_tr)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    logger.info("[4/5] Evaluating …")
    results = predictor.evaluate(X_te, Y_te)

    # Cross-validate primary model
    cv_results = predictor.cross_validate(X_df, Y_df, model_name="logistic_l1")

    # ── Step 5: Save outputs + figures ────────────────────────────────────────
    logger.info("[5/5] Saving outputs and figures …")
    predictor.save(out / "models" / "resistance_predictor.joblib")

    # Metrics CSV
    rows = []
    for drug in DRUGS:
        for mname, mres in results.get(drug, {}).items():
            rows.append({
                "drug": drug, "model": mname,
                "auc_roc":   mres["auc_roc"],
                "auc_pr":    mres["auc_pr"],
                "f1":        mres["f1"],
                "mcc":       mres["mcc"],
                "accuracy":  mres["accuracy"],
            })
    pd.DataFrame(rows).to_csv(out / "model_metrics.csv", index=False)

    # Top SNPs per drug
    for drug in DRUGS:
        try:
            top = predictor.top_resistance_snps(drug, n=15)
            top.to_csv(out / f"top_snps_{drug}.csv", index=False)
        except Exception:
            pass

    # CV results CSV
    cv_df = pd.DataFrame(
        {d: pd.Series(v) for d, v in cv_results.items()}
    )
    cv_df.to_csv(out / "cv_auc_scores.csv", index=False)

    generate_all(X_df, Y_df, predictor, results, cv_results, DRUGS, out)

    # ── Print summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "="*58)
    print("  Day 23 · Drug Resistance Prediction Summary")
    print("="*58)
    print(f"  Isolates      : {len(X_df)} ({len(X_tr)} train / {len(X_te)} test)")
    print(f"  SNP features  : {X_df.shape[1]}")
    print(f"  Drugs         : {len(DRUGS)}")
    print()
    for drug in DRUGS:
        if drug not in results: continue
        best_m   = max(results[drug], key=lambda m: results[drug][m]["auc_roc"])
        best_auc = results[drug][best_m]["auc_roc"]
        best_f1  = results[drug][best_m]["f1"]
        pathogen = PATHOGENS[drug].replace("_", " ")
        print(f"  {drug:<15} ({pathogen:<30})  "
              f"AUC={best_auc:.3f}  F1={best_f1:.3f}  [{best_m}]")
    print(f"\n  Figures  : 9 saved to {out}/")
    print(f"  Elapsed  : {elapsed:.1f}s")
    print("="*58 + "\n")


if __name__ == "__main__":
    main()
