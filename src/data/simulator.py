"""
Simulate bacterial SNP profiles linked to antibiotic resistance.

Pathogens modelled (WHO priority pathogens):
  - Mycobacterium tuberculosis  (TB)     → Isoniazid, Rifampicin
  - Escherichia coli            (ECOLI)  → Ciprofloxacin, Ampicillin
  - Staphylococcus aureus       (STAPH)  → Methicillin (MRSA)

SNP positions and resistance associations sourced from:
  - ReSeqTB / WHO catalogue of TB mutations (2021)
  - CARD database (AMR gene-SNP associations)
  - Earle et al. 2016 (E. coli GWAS)
  - PATRIC resistome catalogue

Each sample = one bacterial isolate with a binary SNP presence/absence vector.
Label = resistant (1) or susceptible (0) per antibiotic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from Bio.Seq import Seq

# ── Known resistance SNPs per drug ────────────────────────────────────────────
# Format: (gene, position, ref_aa, alt_aa, drug, odds_ratio)
RESISTANCE_MUTATIONS = [
    # ── M. tuberculosis / Isoniazid ──────────────────────────────────────────
    ("katG",  315, "S", "T", "isoniazid",     18.5),
    ("katG",  315, "S", "N", "isoniazid",     12.0),
    ("inhA",   -8, "C", "T", "isoniazid",      6.3),  # promoter
    ("inhA",   -15,"C", "T", "isoniazid",      5.1),
    ("ahpC",  -10, "C", "T", "isoniazid",      3.2),
    ("ndh",   268, "R", "H", "isoniazid",      2.8),
    # ── M. tuberculosis / Rifampicin ─────────────────────────────────────────
    ("rpoB",  450, "S", "L", "rifampicin",    45.0),
    ("rpoB",  445, "H", "Y", "rifampicin",    38.0),
    ("rpoB",  435, "D", "V", "rifampicin",    22.0),
    ("rpoB",  431, "L", "V", "rifampicin",     8.5),
    ("rpoB",  432, "N", "K", "rifampicin",    12.0),
    # ── E. coli / Ciprofloxacin ──────────────────────────────────────────────
    ("gyrA",   83, "S", "L", "ciprofloxacin", 35.0),
    ("gyrA",   87, "D", "N", "ciprofloxacin", 28.0),
    ("parC",   80, "S", "I", "ciprofloxacin", 15.0),
    ("parC",   84, "E", "V", "ciprofloxacin", 12.0),
    ("gyrB",  426, "D", "N", "ciprofloxacin",  6.0),
    ("marA",    1, "M", "V", "ciprofloxacin",  4.0),  # regulator overexp proxy
    # ── E. coli / Ampicillin ─────────────────────────────────────────────────
    ("blaTEM",  1, "*", "E", "ampicillin",    88.0),  # gene presence = SNP1
    ("blaCTX",  1, "*", "K", "ampicillin",    72.0),
    ("ampC",  112, "G", "D", "ampicillin",    18.0),
    ("ompF",   77, "R", "C", "ampicillin",     5.5),  # porin loss
    ("acrB",  672, "G", "V", "ampicillin",     3.2),
    # ── S. aureus / Methicillin (MRSA) ───────────────────────────────────────
    ("mecA",    1, "*", "M", "methicillin",   95.0),  # gene acquisition
    ("pbp2a",  40, "S", "P", "methicillin",   55.0),
    ("femA",  261, "I", "F", "methicillin",   22.0),
    ("femB",  299, "T", "M", "methicillin",   18.0),
    ("blaZ",    1, "*", "B", "methicillin",   12.0),
    ("graSR",  20, "K", "E", "methicillin",    6.0),
]

# Neutral SNPs (not associated with resistance — background variation)
NEUTRAL_SNPS = [
    ("mutS", 450), ("mutL", 340), ("recA", 123), ("dnaA", 89),
    ("rpsL", 200), ("rpsA", 150), ("gidB", 77),  ("tlyA", 88),
    ("atpE",  30), ("ftsZ", 110), ("groL",  55), ("dnaK", 230),
    ("mmpL5",  60),("iniA",  40), ("eis",   25), ("tap",  180),
    ("fabG",   92),("kasA", 269), ("oxyR",  65), ("furA",  33),
    ("rpoC",  516),("rpoA",  80), ("rplC",  69), ("rpsE",  45),
    ("pncA",  162),("embB", 306), ("embC", 270), ("embA", 356),
]

DRUGS    = ["isoniazid", "rifampicin", "ciprofloxacin", "ampicillin", "methicillin"]
PATHOGENS = {
    "isoniazid":     "Mycobacterium_tuberculosis",
    "rifampicin":    "Mycobacterium_tuberculosis",
    "ciprofloxacin": "Escherichia_coli",
    "ampicillin":    "Escherichia_coli",
    "methicillin":   "Staphylococcus_aureus",
}

# Build feature names
def _feature_names() -> list[str]:
    res_feats = [f"{g}_{pos}_{ref}{alt}"
                 for g, pos, ref, alt, drug, _ in RESISTANCE_MUTATIONS]
    neu_feats = [f"{g}_{pos}_neutral"
                 for g, pos in NEUTRAL_SNPS]
    return res_feats + neu_feats

FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)
N_RES_SNPS = len(RESISTANCE_MUTATIONS)


def simulate_isolates(
    n_per_drug: int = 120,
    resistance_rate: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate bacterial isolates with SNP profiles.

    Parameters
    ----------
    n_per_drug       : isolates per drug
    resistance_rate  : fraction of resistant isolates
    seed             : random seed

    Returns
    -------
    X : DataFrame (isolates × SNP features)  binary 0/1
    Y : DataFrame (isolates × drugs)         binary 0=susceptible 1=resistant
    """
    rng = np.random.default_rng(seed)
    all_X, all_y, all_ids = [], [], []

    for drug in DRUGS:
        n_resistant   = int(n_per_drug * resistance_rate)
        n_susceptible = n_per_drug - n_resistant

        drug_muts = [(i, or_) for i, (g, pos, ref, alt, d, or_)
                     in enumerate(RESISTANCE_MUTATIONS) if d == drug]

        for label, n_isolates in [(1, n_resistant), (0, n_susceptible)]:
            for _ in range(n_isolates):
                snp = np.zeros(N_FEATURES, dtype=np.int8)

                # Resistance SNPs
                for idx, odds in drug_muts:
                    if label == 1:
                        # Resistant: high probability of carrying causative SNPs
                        p = min(0.95, odds / (odds + 5))
                    else:
                        # Susceptible: low probability (background rate)
                        p = max(0.02, 1 / (1 + odds))
                    snp[idx] = int(rng.random() < p)

                # Neutral SNPs: random background variation ~5-15%
                for j in range(N_RES_SNPS, N_FEATURES):
                    snp[j] = int(rng.random() < rng.uniform(0.05, 0.15))

                # Linkage disequilibrium: if primary SNP present, secondary likely
                if label == 1 and len(drug_muts) > 1:
                    primary_idx = drug_muts[0][0]
                    if snp[primary_idx] == 1:
                        for idx2, _ in drug_muts[1:]:
                            if rng.random() < 0.35:
                                snp[idx2] = 1

                isolate_id = (f"{PATHOGENS[drug].split('_')[0][:3].upper()}_"
                              f"{drug[:3].upper()}_"
                              f"{'R' if label==1 else 'S'}_"
                              f"{len(all_ids)+1:04d}")
                all_X.append(snp)
                all_y.append({d: (1 if d == drug and label == 1 else 0)
                              for d in DRUGS})
                all_ids.append(isolate_id)

    X_df = pd.DataFrame(all_X, columns=FEATURE_NAMES, index=all_ids)
    Y_df = pd.DataFrame(all_y, index=all_ids)

    return X_df, Y_df


def annotate_snp(snp_name: str) -> dict:
    """Parse SNP feature name into structured annotation."""
    parts = snp_name.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "neutral":
        return {"gene": parts[0], "position": parts[1], "type": "neutral",
                "ref": None, "alt": None}
    if len(parts) >= 3:
        change = parts[2]
        return {"gene": "_".join(parts[:-2]),
                "position": parts[-2],
                "type": "resistance",
                "ref": change[0] if len(change) >= 2 else None,
                "alt": change[-1] if len(change) >= 2 else None}
    return {"gene": snp_name, "position": None, "type": "unknown",
            "ref": None, "alt": None}


def make_fasta_record(isolate_id: str, snp_vector: np.ndarray) -> str:
    """Create a minimal FASTA-like representation of the SNP profile."""
    seq_str = "".join(str(int(v)) for v in snp_vector)
    seq     = Seq(seq_str)
    n_muts  = int(snp_vector[:N_RES_SNPS].sum())
    return (f">{isolate_id} | resistance_SNPs={n_muts} "
            f"| total_SNPs={int(snp_vector.sum())}\n{seq}\n")
