"""
9 publication-quality figures for drug resistance mutation prediction.

fig1 : SNP prevalence heatmap  – resistance vs susceptible isolates per drug
fig2 : Model performance bar   – AUC-ROC / F1 / MCC across all drugs + models
fig3 : ROC curves              – 5 models for best drug (isoniazid)
fig4 : Precision-Recall curves – 5 models for best drug
fig5 : Confusion matrices      – L1 LR for all 5 drugs
fig6 : Feature importance      – top resistance SNPs per drug (L1 coefficients)
fig7 : Cross-validation boxplot– 5-fold AUC per drug
fig8 : Resistance co-occurrence– sample heatmap showing MDR patterns
fig9 : Summary                 – per-drug best AUC, pathogen breakdown, MCC
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DRUG_PALETTE = {
    "isoniazid":     "#E74C3C",
    "rifampicin":    "#E67E22",
    "ciprofloxacin": "#3498DB",
    "ampicillin":    "#2ECC71",
    "methicillin":   "#9B59B6",
}
MODEL_PALETTE = {
    "logistic_l1":    "#E74C3C",
    "logistic_l2":    "#E67E22",
    "random_forest":  "#3498DB",
    "svm_rbf":        "#2ECC71",
    "gradient_boost": "#9B59B6",
}
MODEL_LABELS = {
    "logistic_l1":    "Logistic (L1)",
    "logistic_l2":    "Logistic (L2)",
    "random_forest":  "Random Forest",
    "svm_rbf":        "SVM (RBF)",
    "gradient_boost": "Gradient Boosting",
}
DPI = 150


def _save(fig, out_dir: Path, name: str) -> None:
    p = out_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


# ── Fig 1: SNP prevalence heatmap ─────────────────────────────────────────────
def fig1_snp_prevalence(
    X_df: pd.DataFrame, Y_df: pd.DataFrame,
    drugs: list[str], out_dir: Path, n_top: int = 25,
) -> None:
    fig, axes = plt.subplots(1, len(drugs), figsize=(4 * len(drugs), 7))
    fig.suptitle("Resistance SNP Prevalence: Resistant vs Susceptible",
                 fontsize=13, fontweight="bold")

    for ax, drug in zip(axes, drugs):
        mask_r = Y_df[drug] == 1
        mask_s = Y_df[drug] == 0
        if mask_r.sum() == 0 or mask_s.sum() == 0:
            ax.set_visible(False)
            continue

        # Only resistance SNPs associated with this drug
        drug_short = drug[:3].upper()
        res_cols = [c for c in X_df.columns if "neutral" not in c]
        prev_r = X_df.loc[mask_r, res_cols].mean()
        prev_s = X_df.loc[mask_s, res_cols].mean()
        diff   = (prev_r - prev_s).nlargest(n_top)
        top_cols = diff.index.tolist()

        mat = pd.DataFrame({
            "Resistant":   X_df.loc[mask_r, top_cols].mean(),
            "Susceptible": X_df.loc[mask_s, top_cols].mean(),
        }, index=top_cols)

        sns.heatmap(mat, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 6},
                    linewidths=0.3, cbar=False)
        ax.set_title(drug.capitalize(), fontsize=10, fontweight="bold",
                     color=DRUG_PALETTE.get(drug, "black"))
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels([c.replace("_", " ") for c in top_cols],
                           fontsize=6, rotation=0)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_snp_prevalence.png")


# ── Fig 2: Performance bar ────────────────────────────────────────────────────
def fig2_performance_bar(results: dict, drugs: list[str], out_dir: Path) -> None:
    rows = []
    for drug in drugs:
        for mname, mres in results.get(drug, {}).items():
            rows.append({"drug": drug, "model": mname,
                         "AUC-ROC": mres["auc_roc"],
                         "F1":      mres["f1"],
                         "MCC":     mres["mcc"]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Drug Resistance Prediction Performance",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["AUC-ROC", "F1", "MCC"]):
        pivot = df.pivot(index="drug", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax,
                   color=[MODEL_PALETTE.get(c, "gray") for c in pivot.columns],
                   edgecolor="white", width=0.75)
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel(""); ax.set_ylim(0, 1.15)
        ax.set_xticklabels([d.capitalize() for d in pivot.index],
                           rotation=20, ha="right", fontsize=9)
        ax.legend([MODEL_LABELS.get(c, c) for c in pivot.columns],
                  fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig2_performance_bar.png")


# ── Fig 3: ROC curves (best drug) ────────────────────────────────────────────
def fig3_roc_curves(results: dict, drug: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for mname, mres in results.get(drug, {}).items():
        label = f"{MODEL_LABELS.get(mname, mname)} (AUC={mres['auc_roc']:.3f})"
        ax.plot(mres["fpr"], mres["tpr"],
                color=MODEL_PALETTE.get(mname, "gray"), lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {drug.capitalize()} Resistance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig3_roc_curves.png")


# ── Fig 4: PR curves ──────────────────────────────────────────────────────────
def fig4_pr_curves(results: dict, drug: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for mname, mres in results.get(drug, {}).items():
        label = f"{MODEL_LABELS.get(mname, mname)} (AP={mres['auc_pr']:.3f})"
        ax.plot(mres["rec_curve"], mres["prec_curve"],
                color=MODEL_PALETTE.get(mname, "gray"), lw=2, label=label)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curves — {drug.capitalize()} Resistance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_pr_curves.png")


# ── Fig 5: Confusion matrices (L1 per drug) ───────────────────────────────────
def fig5_confusion_matrices(results: dict, drugs: list[str], out_dir: Path) -> None:
    n = len(drugs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Confusion Matrices — Logistic (L1) per Drug",
                 fontsize=13, fontweight="bold")

    for ax, drug in zip(axes, drugs):
        mres = results.get(drug, {}).get("logistic_l1", {})
        if not mres:
            ax.set_visible(False); continue
        cm = mres["confusion"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Susc.", "Resist."],
                    yticklabels=["Susc.", "Resist."],
                    cbar=False, linewidths=0.5)
        ax.set_title(f"{drug.capitalize()}\nAUC={mres['auc_roc']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "fig5_confusion_matrices.png")


# ── Fig 6: Feature importance ─────────────────────────────────────────────────
def fig6_feature_importance(predictor, drugs: list[str], out_dir: Path,
                             n_top: int = 10) -> None:
    fig, axes = plt.subplots(1, len(drugs), figsize=(4 * len(drugs), 6))
    fig.suptitle("Top Resistance SNPs (L1 Logistic Regression Coefficients)",
                 fontsize=13, fontweight="bold")

    for ax, drug in zip(axes, drugs):
        if drug not in predictor.feature_importance_:
            ax.set_visible(False); continue
        imp = predictor.feature_importance_[drug]
        top = imp[imp > 0].nlargest(n_top)
        if len(top) == 0:
            top = imp.nlargest(n_top)

        colors = ["#E74C3C" if v > 0 else "#3498DB" for v in top.values]
        ax.barh(range(len(top)), top.values[::-1], color=colors[::-1],
                edgecolor="white")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(
            [c.replace("_", " ") for c in top.index[::-1]], fontsize=7)
        ax.set_xlabel("L1 Coefficient", fontsize=9)
        ax.set_title(drug.capitalize(), fontsize=10, fontweight="bold",
                     color=DRUG_PALETTE.get(drug, "black"))
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig6_feature_importance.png")


# ── Fig 7: CV boxplot ────────────────────────────────────────────────────────
def fig7_cv_boxplot(cv_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    drugs  = list(cv_results.keys())
    data   = [cv_results[d] for d in drugs]
    colors = [DRUG_PALETTE.get(d, "gray") for d in drugs]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # Overlay individual points
    for i, (d_data, col) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(i).uniform(-0.1, 0.1, len(d_data))
        ax.scatter(np.full(len(d_data), i) + jitter, d_data,
                   c=col, s=30, zorder=3, alpha=0.85, edgecolors="white")

    ax.set_xticks(range(1, len(drugs) + 1))
    ax.set_xticklabels([d.capitalize() for d in drugs], fontsize=10)
    ax.set_ylabel("5-Fold CV AUC-ROC", fontsize=11)
    ax.set_title("Cross-Validation Performance — Logistic (L1) per Drug",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Random")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig7_cv_boxplot.png")


# ── Fig 8: MDR co-occurrence heatmap ─────────────────────────────────────────
def fig8_mdr_heatmap(Y_df: pd.DataFrame, out_dir: Path) -> None:
    # Show subset of isolates sorted by total resistance burden
    total_r  = Y_df.sum(axis=1).sort_values(ascending=False)
    top_idx  = total_r.index[:60]
    sub      = Y_df.loc[top_idx]

    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(sub.T, cmap="RdYlGn_r", vmin=0, vmax=1,
                xticklabels=False, yticklabels=True, ax=ax,
                linewidths=0.1, cbar_kws={"label": "Resistant (1) / Susceptible (0)"})
    ax.set_yticklabels([d.capitalize() for d in sub.columns], fontsize=11)
    ax.set_xlabel("Isolates (sorted by resistance burden)", fontsize=10)
    ax.set_title("Multi-Drug Resistance (MDR) Co-occurrence Heatmap",
                 fontsize=13, fontweight="bold")

    # Annotate MDR counts
    mdr2  = (Y_df.sum(axis=1) >= 2).sum()
    mdr3  = (Y_df.sum(axis=1) >= 3).sum()
    ax.text(0.98, 0.02,
            f"MDR (≥2 drugs): {mdr2}\nMDR (≥3 drugs): {mdr3}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    _save(fig, out_dir, "fig8_mdr_heatmap.png")


# ── Fig 9: Summary ───────────────────────────────────────────────────────────
def fig9_summary(results: dict, Y_df: pd.DataFrame, drugs: list[str],
                 out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Drug Resistance Prediction — Summary",
                 fontsize=13, fontweight="bold")

    # Panel A: Best AUC per drug
    ax = axes[0]
    best_aucs = {}
    for drug in drugs:
        if drug in results:
            best_aucs[drug] = max(m["auc_roc"] for m in results[drug].values())
    sorted_d = sorted(best_aucs, key=best_aucs.get, reverse=True)
    ax.barh([d.capitalize() for d in sorted_d],
            [best_aucs[d] for d in sorted_d],
            color=[DRUG_PALETTE.get(d, "gray") for d in sorted_d],
            edgecolor="white")
    for i, (d, v) in enumerate(zip(sorted_d, [best_aucs[d] for d in sorted_d])):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Best AUC-ROC (any model)", fontsize=10)
    ax.set_title("Best Model per Drug", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: Resistance prevalence per drug
    ax = axes[1]
    prev = Y_df[drugs].mean()
    ax.bar([d.capitalize() for d in drugs], prev.values,
           color=[DRUG_PALETTE.get(d, "gray") for d in drugs],
           edgecolor="white")
    for i, v in enumerate(prev.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Resistance Prevalence", fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_title("Resistance Rate per Drug", fontsize=11)
    ax.set_xticklabels([d.capitalize() for d in drugs], rotation=15, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: MCC for L1 LR per drug
    ax = axes[2]
    mccs = {drug: results[drug]["logistic_l1"]["mcc"]
            for drug in drugs if drug in results}
    sorted_m = sorted(mccs, key=mccs.get, reverse=True)
    colors_m = ["#2ECC71" if v >= 0.5 else "#E74C3C" if v < 0.3 else "#F39C12"
                for v in [mccs[d] for d in sorted_m]]
    bars = ax.bar([d.capitalize() for d in sorted_m],
                  [mccs[d] for d in sorted_m], color=colors_m, edgecolor="white")
    for b, d in zip(bars, sorted_m):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{mccs[d]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Matthews Correlation Coefficient", fontsize=10)
    ax.set_title("MCC — Logistic (L1)", fontsize=11)
    ax.set_xticklabels([d.capitalize() for d in sorted_m], rotation=15, ha="right")
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig9_summary.png")


# ── Driver ─────────────────────────────────────────────────────────────────────
def generate_all(
    X_df, Y_df, predictor, results, cv_results,
    drugs, out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figures → %s", out_dir)

    best_drug = max(results, key=lambda d: max(
        m["auc_roc"] for m in results[d].values()))

    fig1_snp_prevalence(X_df, Y_df, drugs, out_dir)
    fig2_performance_bar(results, drugs, out_dir)
    fig3_roc_curves(results, best_drug, out_dir)
    fig4_pr_curves(results, best_drug, out_dir)
    fig5_confusion_matrices(results, drugs, out_dir)
    fig6_feature_importance(predictor, drugs, out_dir)
    fig7_cv_boxplot(cv_results, out_dir)
    fig8_mdr_heatmap(Y_df, out_dir)
    fig9_summary(results, Y_df, drugs, out_dir)

    logger.info("All 9 figures saved.")
