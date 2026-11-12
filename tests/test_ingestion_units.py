"""
Tests unitaires des fonctions de L0, sur de petites données synthétiques.
Tournent partout, sans les vraies données.
"""
from datetime import date

import polars as pl

from find_duplicates import normalize_for_dedup
from make_splits import assign_split, deduplicate, sample_splits, tempered_targets


# ---------------------------------------------------------------------------
# normalize_for_dedup
# ---------------------------------------------------------------------------
def norm(texts: list[str]) -> list[str]:
    return pl.DataFrame({"t": texts}).select(normalize_for_dedup(pl.col("t")))["t"].to_list()


class TestNormalizeForDedup:
    def test_supprime_masques_montants_dates_ponctuation(self):
        assert norm(["I paid {$1000.00} to XXXX on XX/XX/2019."]) == ["i paid to on"]

    def test_deux_variantes_d_une_lettre_type_sont_identiques(self):
        a, b = norm([
            "I paid {$1000.00} to XXXX on XX/XX/2019.",
            "i  paid {$250.00} to XXXX XXXX on XX/XX/2020 !",
        ])
        assert a == b

    def test_casse_et_ponctuation_ignorees(self):
        assert norm(["Please help", "please HELP!!!"]) == ["please help", "please help"]

    def test_textes_differents_restent_differents(self):
        a, b = norm(["My mortgage was denied.", "My credit card was denied."])
        assert a != b

    def test_mot_contenant_xx_conserve(self):
        assert norm(["Exxon charged me"]) == ["exxon charged me"]

    def test_hash_identique_pour_variantes(self):
        df = pl.DataFrame({"t": ["I paid {$10.00} to XXXX.", "i paid {$99.00} to XXXX XXXX !"]})
        h = df.select(normalize_for_dedup(pl.col("t")).hash(seed=42))["t"].to_list()
        assert h[0] == h[1]


# ---------------------------------------------------------------------------
# tempered_targets
# ---------------------------------------------------------------------------
class TestTemperedTargets:
    def test_somme_exacte(self):
        t = tempered_targets({"a": 5000, "b": 700, "c": 333}, 1000, 0.5)
        assert sum(t.values()) == 1000

    def test_alpha_0_classes_egales(self):
        t = tempered_targets({"a": 1000, "b": 1000, "c": 1000}, 300, 0.0)
        assert t == {"a": 100, "b": 100, "c": 100}

    def test_alpha_1_proportions_naturelles(self):
        t = tempered_targets({"a": 600, "b": 300, "c": 100}, 100, 1.0)
        assert t == {"a": 60, "b": 30, "c": 10}

    def test_alpha_05_entre_les_deux(self):
        t = tempered_targets({"a": 900, "b": 100}, 100, 0.5)
        # naturel : a = 90 ; égal : a = 50 ; tempéré : entre les deux
        assert 50 < t["a"] < 90

    def test_petite_classe_prise_entierement_et_reste_redistribue(self):
        t = tempered_targets({"a": 1000, "b": 100, "c": 10}, 300, 0.5)
        assert t["c"] == 10
        assert sum(t.values()) == 300
        assert t["a"] > t["b"] > t["c"]

    def test_demande_superieure_au_disponible(self):
        t = tempered_targets({"a": 10, "b": 5}, 100, 0.5)
        assert t == {"a": 10, "b": 5}


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------
def test_deduplicate_garde_le_plus_ancien_et_les_uniques():
    df = pl.DataFrame({
        "complaint_id": [1, 2, 3, 4, 5],
        "date_received": [date(2020, 1, 5), date(2019, 1, 1), date(2021, 1, 1),
                          date(2020, 1, 1), date(2022, 1, 1)],
        "year": [2020, 2019, 2021, 2020, 2022],
        "product": ["a"] * 5,
        "dup_group": [1, 1, 1, None, 2],
    })
    kept = set(deduplicate(df)["complaint_id"].to_list())
    # groupe 1 -> id 2 (le plus ancien) ; id 4 unique ; groupe 2 -> id 5
    assert kept == {2, 4, 5}


# ---------------------------------------------------------------------------
# assign_split
# ---------------------------------------------------------------------------
def test_assign_split_par_annee():
    df = pl.DataFrame({"complaint_id": [1, 2, 3, 4, 5],
                       "year": [2015, 2023, 2024, 2025, 2026]})
    out = assign_split(df, seed=42)["split"].to_list()
    assert out == ["train", "train", "val", "test", "drift_2026"]


# ---------------------------------------------------------------------------
# sample_splits, sur un petit corpus synthétique
# ---------------------------------------------------------------------------
def make_synthetic(seed: int = 42) -> pl.DataFrame:
    rows, cid = [], 0
    pools = {
        2020: {"a": 1000, "b": 100, "c": 10},   # train
        2024: {"a": 150, "b": 50},              # val
        2025: {"a": 200, "b": 100},             # test
        2026: {"a": 30, "b": 20},               # drift
    }
    for year, classes in pools.items():
        for product, n in classes.items():
            for _ in range(n):
                cid += 1
                rows.append({"complaint_id": cid, "year": year, "product": product})
    return assign_split(pl.DataFrame(rows), seed=seed)


def ids_by_split(selected: pl.DataFrame, name: str) -> set[int]:
    return set(selected.filter(pl.col("split") == name)["complaint_id"].to_list())


class TestSampleSplits:
    def test_tailles(self):
        sel, _ = sample_splits(make_synthetic(), 300, 100, 200, 0.5)
        sizes = dict(sel.group_by("split").len().iter_rows())
        assert sizes == {"train": 300, "val": 100, "test": 200, "drift_2026": 50}

    def test_cibles_train_respectees(self):
        df = make_synthetic()
        sel, targets = sample_splits(df, 300, 100, 200, 0.5)
        train = df.join(sel.filter(pl.col("split") == "train").select("complaint_id"),
                        on="complaint_id")
        counts = dict(train.group_by("product").len().iter_rows())
        assert counts == targets
        assert counts["c"] == 10

    def test_aucun_id_dans_deux_splits(self):
        sel, _ = sample_splits(make_synthetic(), 300, 100, 200, 0.5)
        assert sel.height == sel["complaint_id"].n_unique()

    def test_reproductible_avec_meme_graine(self):
        a, _ = sample_splits(make_synthetic(42), 300, 100, 200, 0.5)
        b, _ = sample_splits(make_synthetic(42), 300, 100, 200, 0.5)
        assert ids_by_split(a, "train") == ids_by_split(b, "train")
        assert ids_by_split(a, "test") == ids_by_split(b, "test")

    def test_graine_differente_tirage_different(self):
        a, _ = sample_splits(make_synthetic(42), 300, 100, 200, 0.5)
        b, _ = sample_splits(make_synthetic(7), 300, 100, 200, 0.5)
        assert ids_by_split(a, "train") != ids_by_split(b, "train")

    def test_agrandir_val_garde_les_memes_recits(self):
        small, _ = sample_splits(make_synthetic(), 300, 50, 100, 0.5)
        big, _ = sample_splits(make_synthetic(), 300, 100, 200, 0.5)
        assert ids_by_split(small, "val") <= ids_by_split(big, "val")
        assert ids_by_split(small, "test") <= ids_by_split(big, "test")