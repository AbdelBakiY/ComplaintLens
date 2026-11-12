"""
Tests d'intégrité des splits réels (data/processed/).
Ignorés si les fichiers sont absents : à lancer là où make_splits.py a tourné.
"""
import json
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ["train", "val", "test", "drift_2026"]
CLASSES = {"credit_reporting", "debt_collection", "credit_card", "checking_savings",
           "mortgage", "money_transfer", "student_loan", "vehicle_loan",
           "payday_personal_loan", "prepaid_card"}
YEARS = {"val": {2024}, "test": {2025}, "drift_2026": {2026}}

pytestmark = pytest.mark.skipif(
    not all((PROCESSED / f"{s}.parquet").exists() for s in SPLITS),
    reason="splits absents : exécuter make_splits.py d'abord",
)


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    cols = ["complaint_id", "year", "product", "dup_group", "n_words", "n_chars"]
    return {s: pl.read_parquet(PROCESSED / f"{s}.parquet", columns=cols) for s in SPLITS}


def test_aucun_id_dans_deux_splits(frames):
    ids = pl.concat([f.select("complaint_id") for f in frames.values()])
    assert ids.height == ids["complaint_id"].n_unique()


def test_aucun_groupe_de_doublons_dans_deux_splits(frames):
    groups = pl.concat([f.filter(pl.col("dup_group").is_not_null()).select("dup_group")
                        for f in frames.values()])
    assert groups.height == groups["dup_group"].n_unique()


def test_train_avant_2024(frames):
    assert frames["train"]["year"].max() <= 2023


@pytest.mark.parametrize("split", ["val", "test", "drift_2026"])
def test_annees_des_splits(frames, split):
    assert set(frames[split]["year"].unique().to_list()) == YEARS[split]


def test_seulement_les_10_classes(frames):
    for f in frames.values():
        assert set(f["product"].unique().to_list()) <= CLASSES


def test_recits_non_vides(frames):
    for f in frames.values():
        assert (f["n_chars"] > 0).all()
        assert (f["n_words"] > 0).all()


def test_tailles_conformes_au_resume(frames):
    summary = json.loads((PROCESSED / "splits_summary.json").read_text())
    for split, n in summary["requested"].items():
        assert frames[split].height == n