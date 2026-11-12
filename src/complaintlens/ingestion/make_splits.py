"""
L0 - Étape 8 : sous-corpus et splits temporels.

Règles :
  - 1 récit par groupe de doublons (le plus ancien), tous les récits uniques ;
  - classes __DROP__ exclues ;
  - split par année : train <= 2023 | val 2024 | test 2025 | drift_2026 (à part) ;
  - train : échantillonnage tempéré (part de la classe c proportionnelle à n_c ** alpha) ;
  - val / test : tirage aléatoire, distribution naturelle ;
  - drift_2026 : tout est gardé ;
  - tirage déterministe (hash de complaint_id avec une graine) : mêmes splits à chaque exécution.

Usage (depuis la racine du projet) :
    python src/complaintlens/ingestion/make_splits.py
    python src/complaintlens/ingestion/make_splits.py --alpha 0.5 --n-train 100000

Entrée  : data/interim/narratives_dedup.parquet
Sortie  : data/processed/{train,val,test,drift_2026}.parquet
          data/processed/splits_summary.json
          data/processed/splits_distribution.csv
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402

SRC = Path("data/interim/narratives_dedup.parquet")
OUT = Path("data/processed")
SPLITS = ["train", "val", "test", "drift_2026"]


# ---------------------------------------------------------------------------
# 1. Charger les colonnes légères (sans le texte)
# ---------------------------------------------------------------------------
def load_index() -> pl.DataFrame:
    return (
        pl.scan_parquet(SRC)
        .select("complaint_id", "date_received", "year", "product", "dup_group")
        .filter(pl.col("product") != "__DROP__")
        .collect()
    )


# ---------------------------------------------------------------------------
# 2. Dédoublonner : garder le plus ancien récit de chaque groupe
# ---------------------------------------------------------------------------
def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    uniques = df.filter(pl.col("dup_group").is_null())
    firsts = (
        df.filter(pl.col("dup_group").is_not_null())
        .sort(["date_received", "complaint_id"])
        .unique(subset="dup_group", keep="first", maintain_order=True)
    )
    return pl.concat([uniques, firsts])


# ---------------------------------------------------------------------------
# 3. Assigner chaque récit à un split selon son année
# ---------------------------------------------------------------------------
def assign_split(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("year") <= 2023).then(pl.lit("train"))
        .when(pl.col("year") == 2024).then(pl.lit("val"))
        .when(pl.col("year") == 2025).then(pl.lit("test"))
        .otherwise(pl.lit("drift_2026"))
        .alias("split"),
        # clé aléatoire mais reproductible : ne dépend pas de l'ordre des lignes
        pl.col("complaint_id").hash(seed=seed).alias("_key"),
    )


# ---------------------------------------------------------------------------
# 4. Combien de récits par classe dans le train (échantillonnage tempéré)
# ---------------------------------------------------------------------------
def tempered_targets(counts: dict[str, int], n_total: int, alpha: float) -> dict[str, int]:
    """
    Part de chaque classe proportionnelle à n_c ** alpha.
    Si une classe n'a pas assez de récits, on la prend entièrement
    et on redistribue le reste entre les autres classes.
    """
    n_total = min(n_total, sum(counts.values()))
    targets: dict[str, int] = {}
    active = dict(counts)
    remaining = n_total

    while active:
        weights = {c: n ** alpha for c, n in active.items()}
        s = sum(weights.values())
        ideal = {c: remaining * w / s for c, w in weights.items()}
        capped = [c for c in active if ideal[c] >= active[c]]
        if not capped:
            floors = {c: int(v) for c, v in ideal.items()}
            leftover = remaining - sum(floors.values())
            # on distribue les unités restantes aux plus grandes parties décimales
            for c in sorted(ideal, key=lambda c: ideal[c] - floors[c], reverse=True)[:leftover]:
                floors[c] += 1
            targets.update(floors)
            break
        for c in capped:
            targets[c] = active.pop(c)
            remaining -= targets[c]
    return targets


# ---------------------------------------------------------------------------
# 5. Tirer les échantillons
# ---------------------------------------------------------------------------
def sample_splits(df: pl.DataFrame, n_train: int, n_val: int, n_test: int,
                  alpha: float) -> tuple[pl.DataFrame, dict]:
    # Train : tempéré par classe
    train_pool = df.filter(pl.col("split") == "train")
    counts = dict(train_pool.group_by("product").len().iter_rows())
    targets = tempered_targets(counts, n_train, alpha)
    train = (
        train_pool
        .with_columns(pl.col("_key").rank("ordinal").over("product").alias("_rank"))
        .join(pl.DataFrame({"product": list(targets), "_target": list(targets.values())}),
              on="product")
        .filter(pl.col("_rank") <= pl.col("_target"))
        .drop("_rank", "_target")
    )

    # Val / test : tirage aléatoire, distribution naturelle
    def random_n(name: str, n: int) -> pl.DataFrame:
        return df.filter(pl.col("split") == name).sort("_key").head(n)

    val = random_n("val", n_val)
    test = random_n("test", n_test)
    drift = df.filter(pl.col("split") == "drift_2026")

    selected = pl.concat([train, val, test, drift]).select("complaint_id", "split")
    return selected, targets


# ---------------------------------------------------------------------------
# 6. Écrire chaque split avec toutes les colonnes (texte compris)
# ---------------------------------------------------------------------------
def write_splits(selected: pl.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        ids = selected.filter(pl.col("split") == name)
        (
            pl.scan_parquet(SRC)
            .join(ids.lazy(), on="complaint_id", how="inner")
            .sink_parquet(OUT / f"{name}.parquet", compression="zstd")
        )
        print(f"   {name:<11} -> {OUT / f'{name}.parquet'} ({ids.height:,} récits)")


# ---------------------------------------------------------------------------
# 7. Contrôles et rapport
# ---------------------------------------------------------------------------
def check_and_report(args, n_pool: dict, targets: dict) -> None:
    frames = {
        s: pl.read_parquet(OUT / f"{s}.parquet",
                           columns=["complaint_id", "year", "product", "dup_group"])
        for s in SPLITS
    }
    all_ids = pl.concat([f.select("complaint_id") for f in frames.values()])
    all_groups = pl.concat([
        f.filter(pl.col("dup_group").is_not_null()).select("dup_group") for f in frames.values()
    ])

    checks = {
        "IDs uniques entre splits": all_ids.height == all_ids["complaint_id"].n_unique(),
        "Aucun groupe de doublons dans 2 splits": all_groups.height == all_groups["dup_group"].n_unique(),
        "train <= 2023": frames["train"]["year"].max() <= 2023,
        "val = 2024": set(frames["val"]["year"].unique()) == {2024},
        "test = 2025": set(frames["test"]["year"].unique()) == {2025},
        "drift = 2026": set(frames["drift_2026"]["year"].unique()) == {2026},
        "Aucun __DROP__": all((f["product"] != "__DROP__").all() for f in frames.values()),
    }
    print("\n=== Contrôles ===")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'ÉCHEC'}] {name}")

    dist = (
        pl.concat([f.with_columns(pl.lit(s).alias("split")) for s, f in frames.items()])
        .group_by("split", "product").len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("split") * 100).round(2).alias("pct"))
    )
    table = (
        dist.pivot(on="split", index="product", values="pct")
        .select("product", *SPLITS).fill_null(0).sort("train", descending=True)
    )
    counts = (
        dist.pivot(on="split", index="product", values="len")
        .select("product", *SPLITS).fill_null(0).sort("train", descending=True)
    )
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print("\n=== Nombre de récits par split ===")
        print(counts)
        print("\n=== Part de chaque classe par split (%) ===")
        print(table)

    counts.write_csv(OUT / "splits_distribution.csv")
    summary = {
        "source": str(SRC),
        "seed": args.seed,
        "alpha": args.alpha,
        "requested": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        "obtained": {s: f.height for s, f in frames.items()},
        "pool_after_dedup": n_pool,
        "train_targets": targets,
        "checks": checks,
    }
    (OUT / "splits_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nRésumé -> {OUT / 'splits_summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.5, help="0 = classes égales, 1 = naturel")
    p.add_argument("--n-train", type=int, default=100_000)
    p.add_argument("--n-val", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    t0 = time.time()

    print("1. Chargement de l'index...")
    df = load_index()
    print(f"   {df.height:,} récits (hors __DROP__)")

    print("2. Dédoublonnage (1 récit par groupe)...")
    df = deduplicate(df)
    print(f"   {df.height:,} récits uniques")

    print("3. Assignation des splits par année...")
    df = assign_split(df, args.seed)
    n_pool = dict(df.group_by("split").len().iter_rows())
    for s in SPLITS:
        print(f"   {s:<11} : {n_pool.get(s, 0):>10,} disponibles")

    print(f"4-5. Échantillonnage (alpha = {args.alpha}, seed = {args.seed})...")
    selected, targets = sample_splits(df, args.n_train, args.n_val, args.n_test, args.alpha)
    print("   Cibles train par classe :")
    for c, n in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"     {c:<22} {n:>7,}")

    print("6. Écriture des fichiers...")
    write_splits(selected)

    check_and_report(args, n_pool, targets)
    print(f"Terminé en {time.time() - t0:,.0f} s")


if __name__ == "__main__":
    main()