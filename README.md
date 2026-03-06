# Day 23 · Drug Resistance Mutation Predictor

**SNP profiles → antibiotic resistance · 5 drugs · 3 pathogens · 9 publication-quality figures**

Part of the [#30DaysOfBioinformatics](https://github.com/SubhadipJana1409) challenge.
Previous: [Day 22 – Anomaly Detection in Microbiome Samples](https://github.com/SubhadipJana1409/Drug-Resistance-Mutation-Predictor.git)

---

## Overview

Antibiotic resistance surveillance relies on the ability to predict resistance phenotypes directly from genomic data — avoiding slow culture-based AST (Antibiotic Susceptibility Testing). This pipeline trains **per-drug binary classifiers** on SNP presence/absence profiles from three WHO-priority pathogens.

**Primary method:** L1-penalised Logistic Regression — interpretable coefficients directly identify which mutations drive resistance (equivalent to the approach used in WHO's catalogue of *M. tuberculosis* mutations).

**3 Pathogens · 5 Drugs:**

| Pathogen | Drug | Key Resistance Genes |
|---|---|---|
| *Mycobacterium tuberculosis* | Isoniazid | *katG* S315T, *inhA* promoter |
| *Mycobacterium tuberculosis* | Rifampicin | *rpoB* S450L, H445Y |
| *Escherichia coli* | Ciprofloxacin | *gyrA* S83L/D87N, *parC* S80I |
| *Escherichia coli* | Ampicillin | *blaTEM*, *blaCTX* gene presence |
| *Staphylococcus aureus* | Methicillin | *mecA* acquisition (MRSA) |

---

## Architecture

```
600 isolates × 56 SNP features  (binary 0/1 presence/absence)
                │
                ▼
   Train/Test split (75/25, stratified)
                │
        ┌───────┼────────────┐
        ▼       ▼            ▼
   L1-LR     L2-LR      Random Forest    SVM (RBF)   Gradient Boosting
  (main)   (baseline)   (interactions)  (kernel)     (boosted trees)
        └───────┴────────────┘
                │
   Per-drug evaluation: AUC-ROC, AUC-PR, F1, MCC
   Feature importance: L1 coefficients → resistance SNP ranking
   Cross-validation: 5-fold stratified CV per drug
```

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_snp_prevalence.png`   | SNP frequency heatmap: resistant vs susceptible per drug |
| `fig2_performance_bar.png`  | AUC-ROC, F1, MCC across all drugs × models |
| `fig3_roc_curves.png`       | ROC curves for 5 models on best drug |
| `fig4_pr_curves.png`        | Precision-Recall curves |
| `fig5_confusion_matrices.png` | L1 LR confusion matrices for all 5 drugs |
| `fig6_feature_importance.png` | Top resistance SNPs (L1 coefficients) per drug |
| `fig7_cv_boxplot.png`       | 5-fold cross-validation AUC boxplot |
| `fig8_mdr_heatmap.png`      | Multi-drug resistance co-occurrence heatmap |
| `fig9_summary.png`          | Best AUC ranking, resistance prevalence, MCC per drug |

---

## Quick Start

```bash
git clone https://github.com/SubhadipJana1409/day23-drug-resistance-predictor
cd day23-drug-resistance-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

### Use on Real Data

```python
import pandas as pd
from src.models.predictor import DrugResistancePredictor

# X: DataFrame (isolates × SNP features), binary 0/1
# Y: DataFrame (isolates × drugs), binary 0=susceptible 1=resistant
X = pd.read_csv("my_snp_matrix.csv", index_col=0)
Y = pd.read_csv("my_phenotypes.csv", index_col=0)

pred = DrugResistancePredictor(drugs=Y.columns.tolist())
pred.fit(X, Y)
results = pred.evaluate(X_test, Y_test)

# Top resistance-conferring SNPs for isoniazid
print(pred.top_resistance_snps("isoniazid", n=10))
```

---

## Project Structure

```
day23-drug-resistance-predictor/
├── src/
│   ├── data/
│   │   └── simulator.py         # SNP profile + resistance label generator
│   ├── models/
│   │   └── predictor.py         # DrugResistancePredictor (5 models, CV)
│   ├── visualization/
│   │   └── plots.py             # All 9 figures
│   └── main.py
├── tests/
│   ├── test_simulator.py        # 13 tests
│   └── test_predictor_plots.py  # 17 tests
├── configs/config.yaml
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 30 passed
```

---

## Methods

**Data simulation:** Binary SNP presence/absence vectors generated with drug-specific resistance mutation probabilities derived from the WHO catalogue of *M. tuberculosis* mutations (2021) and CARD database. Resistance-associated SNPs carry high empirical odds ratios (5–95×); neutral SNPs carry random 5–15% background rates. Linkage disequilibrium between co-occurring mutations is modelled.

**L1 Logistic Regression:** Lasso penalty drives sparse coefficient vectors, directly identifying the minimum set of SNPs that discriminate resistant from susceptible isolates. Coefficients are interpretable as log-odds contributions.

**Multi-drug resistance (MDR):** A sample is MDR if resistant to ≥2 drugs. MDR co-occurrence is visualised in fig8.

---

## References

1. WHO (2021). Catalogue of mutations in *Mycobacterium tuberculosis* complex and their association with drug resistance.
2. Alcock BP et al. (2023). CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database. *NAR*.
3. Earle SG et al. (2016). Identifying lineage effects when controlling for population structure improves power in bacterial association studies. *Nature Microbiology*.

---

## License

MIT
