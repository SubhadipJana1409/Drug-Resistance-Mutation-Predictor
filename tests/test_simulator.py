"""Tests for src/data/simulator.py"""
import numpy as np
import pandas as pd
import pytest
from src.data.simulator import (
    simulate_isolates, annotate_snp, make_fasta_record,
    DRUGS, FEATURE_NAMES, N_FEATURES, N_RES_SNPS,
    RESISTANCE_MUTATIONS, PATHOGENS,
)


class TestSimulateIsolates:
    def test_shape(self):
        X, Y = simulate_isolates(n_per_drug=20, seed=0)
        assert X.shape[1] == N_FEATURES
        assert Y.shape[1] == len(DRUGS)
        assert len(X) == len(Y)

    def test_binary_snp_values(self):
        X, _ = simulate_isolates(n_per_drug=20, seed=0)
        assert set(X.values.ravel()).issubset({0, 1})

    def test_binary_label_values(self):
        _, Y = simulate_isolates(n_per_drug=20, seed=0)
        assert set(Y.values.ravel()).issubset({0, 1})

    def test_reproducible(self):
        X1, Y1 = simulate_isolates(n_per_drug=15, seed=7)
        X2, Y2 = simulate_isolates(n_per_drug=15, seed=7)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_resistance_snps_enriched_in_resistant(self):
        X, Y = simulate_isolates(n_per_drug=80, resistance_rate=0.5, seed=42)
        drug = "isoniazid"
        drug_snp_idxs = [
            i for i, (g, pos, ref, alt, d, _) in enumerate(RESISTANCE_MUTATIONS)
            if d == drug
        ]
        r_mask = Y[drug] == 1
        s_mask = Y[drug] == 0
        if r_mask.sum() > 0 and s_mask.sum() > 0:
            mean_r = X.iloc[r_mask.values, drug_snp_idxs].mean().mean()
            mean_s = X.iloc[s_mask.values, drug_snp_idxs].mean().mean()
            assert mean_r > mean_s, "Resistance SNPs should be more prevalent in resistant isolates"

    def test_feature_names_match(self):
        X, _ = simulate_isolates(n_per_drug=10, seed=0)
        assert list(X.columns) == FEATURE_NAMES

    def test_drug_columns(self):
        _, Y = simulate_isolates(n_per_drug=10, seed=0)
        assert set(Y.columns) == set(DRUGS)

    def test_n_res_snps_correct(self):
        assert N_RES_SNPS == len(RESISTANCE_MUTATIONS)

    def test_no_nan(self):
        X, Y = simulate_isolates(n_per_drug=20, seed=1)
        assert not X.isnull().any().any()
        assert not Y.isnull().any().any()


class TestAnnotateSnp:
    def test_resistance_snp(self):
        ann = annotate_snp("katG_315_ST")
        assert ann["gene"] == "katG"
        assert ann["type"] == "resistance"

    def test_neutral_snp(self):
        ann = annotate_snp("mutS_450_neutral")
        assert ann["type"] == "neutral"
        assert ann["gene"] == "mutS"


class TestFastaRecord:
    def test_returns_string(self):
        snp = np.array([1, 0, 1, 0, 0], dtype=np.int8)
        record = make_fasta_record("TEST_001", snp)
        assert isinstance(record, str)

    def test_has_header_line(self):
        snp = np.zeros(10, dtype=np.int8)
        record = make_fasta_record("ISO_001", snp)
        assert record.startswith(">ISO_001")

    def test_sequence_correct_length(self):
        snp = np.array([1, 0, 1, 1, 0], dtype=np.int8)
        record = make_fasta_record("X", snp)
        seq_line = [l for l in record.strip().split("\n") if not l.startswith(">")][0]
        assert len(seq_line) == len(snp)
