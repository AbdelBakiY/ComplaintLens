"""
L0 - Étape 6 : harmonisation des produits.

Applique configs/product_mapping.csv aux récits :
  1. correspondance exacte sur (product_raw, sub_product)
  2. sinon correspondance générique sur product_raw (ligne "*")
  3. sinon null  -> détecté par les contrôles

Usage (depuis la racine du projet) :
    python src/complaintlens/ingestion/apply_product_mapping.py

Entrée  : data/interim/narratives.parquet
          configs/product_mapping.csv
Sortie  : data/interim/narratives_mapped.parquet
          data/interim/audit/product_mapping_report.csv
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402

SRC = Path("data/interim/narratives.parquet")
MAPPING = Path("configs/product_mapping.csv")
DST = Path("data/interim/narratives_mapped.parquet")
REPORT = Path("data/interim/audit/product_mapping_report.csv")

WILDCARD = "*"


def load_mapping() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Lit le CSV une seule fois et le sépare en règles exactes / génériques."""
    m = pl.read_csv(MAPPING)

    exact = (
        m.filter(pl.col("sub_product_raw") != WILDCARD)
        .select(
            "product_raw",
            pl.col("sub_product_raw").alias("sub_product"),
            pl.col("product").alias("_product_exact"),
        )
    )
    generic = (
        m.filter(pl.col("sub_product_raw") == WILDCARD)
        .select("product_raw", pl.col("product").alias("_product_generic"))
    )

    # Garde-fou : des clés dupliquées dans le CSV multiplieraient les lignes à la jointure
    dup_exact = exact.filter(pl.struct("product_raw", "sub_product").is_duplicated())
    dup_generic = generic.filter(pl.col("product_raw").is_duplicated())
    if dup_exact.height or dup_generic.height:
        print("ERREUR : clés dupliquées dans le mapping")
        print(dup_exact, dup_generic)
        sys.exit(1)

    return exact, generic


def build_pipeline(exact: pl.DataFrame, generic: pl.DataFrame) -> pl.LazyFrame:
    lf = pl.scan_parquet(SRC).rename({"product": "product_raw"})
    cols = lf.collect_schema().names()

    lf = (
        lf.join(exact.lazy(), on=["product_raw", "sub_product"], how="left")
        .join(generic.lazy(), on="product_raw", how="left")
        .with_columns(
            pl.coalesce("_product_exact", "_product_generic").alias("product")
        )
        .drop("_product_exact", "_product_generic")
    )

    # product juste après product_raw
    i = cols.index("product_raw")
    return lf.select(cols[: i + 1] + ["product"] + cols[i + 1 :])


def main():
    t0 = time.time()
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    exact, generic = load_mapping()
    print(f"Mapping : {exact.height} règles exactes, {generic.height} règles génériques")

    build_pipeline(exact, generic).sink_parquet(DST, compression="zstd", row_group_size=50_000)
    print(f"-> {DST} ({DST.stat().st_size / 1e6:,.0f} Mo)")

    # ================= Contrôles =================
    n_before = pl.scan_parquet(SRC).select(pl.len()).collect().item()
    lf = pl.scan_parquet(DST).select("product_raw", "sub_product", "product")
    small = lf.collect()          # 3 colonnes courtes : tient en RAM
    n_after = small.height

    n_null = small["product"].null_count()
    n_review = (small["product"] == "__REVIEW__").sum()
    n_drop = (small["product"] == "__DROP__").sum()

    print("\n=== Contrôles ===")
    print(f"  [{'OK' if n_before == n_after else 'ÉCHEC'}] 1. Lignes avant = après : {n_before:,} / {n_after:,}")
    print(f"  [{'OK' if n_null == 0 else 'ÉCHEC'}] 2. product nul (libellé absent du CSV) : {n_null:,}")
    print(f"  [{'OK' if n_review == 0 else 'À TRAITER'}] 3. __REVIEW__ : {n_review:,}")
    print(f"  [INFO] 4. __DROP__ : {n_drop:,}")

    if n_null:
        print("\n  Libellés sans correspondance :")
        print(small.filter(pl.col("product").is_null())
                   .group_by("product_raw").len().sort("len", descending=True))

    if n_review:
        print("\n  Couples (product_raw, sub_product) à ajouter au CSV :")
        with pl.Config(tbl_rows=50, fmt_str_lengths=80):
            print(small.filter(pl.col("product") == "__REVIEW__")
                       .group_by("product_raw", "sub_product").len()
                       .sort("len", descending=True))

    # Règles exactes jamais utilisées (sub_product mal orthographié dans le CSV ?)
    used = small.select("product_raw", "sub_product").unique()
    unused = exact.join(used, on=["product_raw", "sub_product"], how="anti")
    if unused.height:
        print("\n  [ATTENTION] Règles exactes qui ne correspondent à aucune ligne :")
        print(unused)

    # 5. Distribution finale
    total_kept = n_after - n_drop
    dist = (
        small.filter(~pl.col("product").is_in(["__DROP__"]))
        .group_by("product").len()
        .sort("len", descending=True)
        .with_columns((pl.col("len") / total_kept * 100).round(2).alias("pct"))
    )
    print(f"\n=== 5. Distribution finale ({dist.height} classes, {total_kept:,} récits) ===")
    print(dist)

    # Rapport détaillé : chaque libellé brut -> classe
    report = (
        small.group_by("product_raw", "product").len()
        .sort("product", "len", descending=[False, True])
    )
    report.write_csv(REPORT)
    print(f"\nRapport -> {REPORT}")
    print(f"Terminé en {time.time() - t0:,.0f} s")


if __name__ == "__main__":
    main()