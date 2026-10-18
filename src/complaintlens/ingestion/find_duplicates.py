"""
L0 - Étape 7 (niveau 1) : détection des doublons exacts après normalisation.

Principe :
  - chaque récit est normalisé (minuscules, sans masques XXXX, sans chiffres
    ni ponctuation, espaces nettoyés) puis réduit à UN hash 64 bits ;
  - même hash = même texte normalisé = même groupe de doublons ;
  - les récits trop courts (< MIN_WORDS vrais mots) ne sont jamais considérés
    comme doublons : deux "please help" identiques ne sont pas une campagne ;
  - AUCUNE ligne n'est supprimée : on ajoute des colonnes.

Colonnes ajoutées :
  text_hash  : hash du texte normalisé
  dup_group  : numéro du groupe (1 = le plus gros), null si pas de doublon
  dup_count  : taille du groupe (1 si pas de doublon)

Usage (depuis la racine du projet) :
    python src/complaintlens/ingestion/find_duplicates.py
    python src/complaintlens/ingestion/find_duplicates.py --min-words 20

Entrée  : data/interim/narratives_mapped.parquet
Sortie  : data/interim/narratives_dedup.parquet
          data/interim/audit/duplicates_report.csv
          data/interim/audit/duplicates_summary.json
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = Path("data/interim/narratives_mapped.parquet")
DST = Path("data/interim/narratives_dedup.parquet")
AUDIT = Path("data/interim/audit")
SEED = 42            # graine du hash : même résultat à chaque exécution
TOP_N_REPORT = 200   # nombre de groupes détaillés dans le rapport CSV


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_for_dedup(expr: pl.Expr) -> pl.Expr:
    """
    Normalise un récit pour la détection de doublons.
    Ne sert QU'À comparer des textes : jamais pour entraîner un modèle.

      1. minuscules
      2. suppression des masques CFPB (xx, xxxx, xxxxxxxx...)
      3. suppression de tout ce qui n'est pas une lettre
      4. espaces multiples -> un seul espace
      5. suppression des espaces en début et fin
    """
    return (
        expr.str.to_lowercase()
        .str.replace_all(r"\bx{2,}\b", " ")
        .str.replace_all(r"[^a-z]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


# ---------------------------------------------------------------------------
# Étape A : un hash par récit (streaming, le texte n'est jamais gardé en RAM)
# ---------------------------------------------------------------------------
BATCH_SIZE = 100_000   # récits traités à la fois ; baisse à 50_000 si la RAM sature encore


def compute_hashes() -> pl.DataFrame:
    """Traite le fichier par tranches de BATCH_SIZE lignes pour borner la RAM."""
    n_total = pl.scan_parquet(SRC).select(pl.len()).collect().item()
    chunks = []
    for offset in range(0, n_total, BATCH_SIZE):
        chunk = (
            pl.scan_parquet(SRC)
            .slice(offset, BATCH_SIZE)
            .select(
                "complaint_id",
                "year",
                "n_words",
                pl.col("product").cast(pl.Utf8),
                normalize_for_dedup(pl.col("narrative")).hash(seed=SEED).alias("text_hash"),
            )
            .collect()
        )
        chunks.append(chunk)
        done = min(offset + BATCH_SIZE, n_total)
        print(f"   {done:>10,} / {n_total:,}", end="\r")
    print()
    return pl.concat(chunks).with_columns(pl.col("product").cast(pl.Categorical))


# ---------------------------------------------------------------------------
# Étape B : groupes de doublons
# ---------------------------------------------------------------------------
def build_groups(h: pl.DataFrame, min_words: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    eligible = pl.col("n_words") >= min_words

    # Groupes = hash présents au moins 2 fois parmi les récits assez longs.
    # Numérotés du plus gros (1) au plus petit.
    groups = (
        h.filter(eligible)
        .group_by("text_hash").len()
        .filter(pl.col("len") > 1)
        .sort(["len", "text_hash"], descending=[True, False])
        .with_row_index("dup_group", offset=1)
        .rename({"len": "dup_count"})
        .with_columns(pl.col("dup_group").cast(pl.UInt32), pl.col("dup_count").cast(pl.UInt32))
    )

    h = (
        h.join(groups, on="text_hash", how="left")
        .with_columns(
            pl.when(eligible).then(pl.col("dup_group")).otherwise(None).alias("dup_group"),
            pl.when(eligible & pl.col("dup_count").is_not_null())
            .then(pl.col("dup_count"))
            .otherwise(1)
            .cast(pl.UInt32)
            .alias("dup_count"),
        )
    )
    return h, groups


# ---------------------------------------------------------------------------
# Étape C : écrire le fichier complet avec les nouvelles colonnes
# ---------------------------------------------------------------------------
def write_output(h: pl.DataFrame) -> None:
    """
    Écrit le fichier final par tranches : pour chaque tranche de récits, on
    recolle les colonnes de doublons puis on l'ajoute au fichier Parquet.
    Une seule tranche de texte en RAM à la fois.
    """
    keys = h.select("complaint_id", "text_hash", "dup_group", "dup_count")
    n_total = h.height
    tmp = DST.with_name(DST.stem + ".tmp.parquet")
    writer = None
    try:
        for offset in range(0, n_total, BATCH_SIZE):
            chunk = (
                pl.scan_parquet(SRC)
                .slice(offset, BATCH_SIZE)
                .collect()
                .join(keys, on="complaint_id", how="left")
            )
            table = chunk.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=50_000)
            del chunk, table
            done = min(offset + BATCH_SIZE, n_total)
            print(f"   {done:>10,} / {n_total:,}", end="\r")
        print()
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(DST)


# ---------------------------------------------------------------------------
# Étape D : statistiques et rapport
# ---------------------------------------------------------------------------
def report(h: pl.DataFrame, groups: pl.DataFrame, min_words: int) -> dict:
    n_total = h.height
    n_eligible = h.filter(pl.col("n_words") >= min_words).height
    n_groups = groups.height
    n_in_groups = int(groups["dup_count"].sum() or 0)
    n_redundant = n_in_groups - n_groups   # copies en trop (on garde 1 par groupe)

    # Pour comparaison : combien de "doublons" parmi les récits courts, ignorés volontairement
    short_dups = (
        h.filter(pl.col("n_words") < min_words)
        .group_by("text_hash").len()
        .filter(pl.col("len") > 1)
    )
    n_short_in_groups = int(short_dups["len"].sum() or 0)

    print("\n=== Résumé ===")
    print(f"Récits                          : {n_total:>10,}")
    print(f"Éligibles (>= {min_words} mots)          : {n_eligible:>10,}")
    print(f"Groupes de doublons             : {n_groups:>10,}")
    print(f"Récits dans un groupe           : {n_in_groups:>10,}  ({n_in_groups / n_total * 100:.2f} %)")
    print(f"Copies en trop (hors 1er/groupe): {n_redundant:>10,}  ({n_redundant / n_total * 100:.2f} %)")
    print(f"Plus gros groupe                : {int(groups['dup_count'].max() or 0):>10,}")
    print(f"[info] récits courts identiques ignorés : {n_short_in_groups:,}")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_fmt_str_lengths(120)

    print("\n=== Part de récits dupliqués par produit ===")
    by_product = (
        h.group_by("product")
        .agg(
            pl.len().alias("n"),
            (pl.col("dup_count") > 1).sum().alias("n_dup"),
        )
        .with_columns((pl.col("n_dup") / pl.col("n") * 100).round(2).alias("pct_dup"))
        .sort("pct_dup", descending=True)
    )
    print(by_product)

    print("\n=== Part de récits dupliqués par année ===")
    by_year = (
        h.group_by("year")
        .agg(pl.len().alias("n"), (pl.col("dup_count") > 1).sum().alias("n_dup"))
        .with_columns((pl.col("n_dup") / pl.col("n") * 100).round(2).alias("pct_dup"))
        .sort("year")
    )
    print(by_year)

    print("\n=== Distribution des tailles de groupes ===")
    sizes = (
        groups.with_columns(
            pl.col("dup_count").cut([2, 5, 10, 100, 1000],
                                    labels=["2", "3-5", "6-10", "11-100", "101-1000", ">1000"])
            .alias("taille")
        )
        .group_by("taille")
        .agg(pl.len().alias("n_groupes"), pl.col("dup_count").sum().alias("n_recits"))
        .sort("taille")
    )
    print(sizes)

    # Top groupes détaillés + un extrait de texte pour chacun
    top = (
        h.filter(pl.col("dup_group") <= TOP_N_REPORT)
        .group_by("dup_group")
        .agg(
            pl.col("dup_count").first(),
            pl.col("product").mode().first().cast(pl.Utf8).alias("main_product"),
            pl.col("product").n_unique().alias("n_products"),
            pl.col("year").min().alias("year_min"),
            pl.col("year").max().alias("year_max"),
            pl.col("n_words").median().alias("median_words"),
            pl.col("complaint_id").min().alias("example_id"),
        )
        .sort("dup_group")
    )
    excerpts = (
        pl.scan_parquet(SRC)
        .filter(pl.col("complaint_id").is_in(top["example_id"].to_list()))
        .select(
            pl.col("complaint_id").alias("example_id"),
            pl.col("narrative").str.slice(0, 150).str.replace_all(r"\s+", " ").alias("excerpt"),
        )
        .collect()
    )
    top = top.join(excerpts, on="example_id", how="left").sort("dup_group")

    print("\n=== Top 10 des plus gros groupes ===")
    print(top.head(10).select("dup_group", "dup_count", "main_product", "n_products",
                              "year_min", "year_max", "excerpt"))

    top.write_csv(AUDIT / "duplicates_report.csv")

    summary = {
        "min_words": min_words,
        "hash_seed": SEED,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_groups": n_groups,
        "n_in_groups": n_in_groups,
        "n_redundant_copies": n_redundant,
        "pct_in_groups": round(n_in_groups / n_total * 100, 3),
        "largest_group": int(groups["dup_count"].max() or 0),
        "n_short_identical_ignored": n_short_in_groups,
    }
    (AUDIT / "duplicates_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-words", type=int, default=10,
                   help="nb minimal de vrais mots pour qu'un récit puisse être un doublon")
    args = p.parse_args()

    t0 = time.time()
    AUDIT.mkdir(parents=True, exist_ok=True)

    print(f"A. Calcul des hash (seuil = {args.min_words} mots)...")
    h = compute_hashes()
    print(f"   {h.height:,} récits hashés ({h.estimated_size('mb'):.0f} Mo en RAM)")

    print("B. Construction des groupes...")
    h, groups = build_groups(h, args.min_words)

    print("C. Écriture du fichier...")
    write_output(h)
    n_out = pl.scan_parquet(DST).select(pl.len()).collect().item()
    status = "OK" if n_out == h.height else "ÉCHEC"
    print(f"   [{status}] {DST} : {n_out:,} lignes ({DST.stat().st_size / 1e6:,.0f} Mo)")

    report(h, groups, args.min_words)
    print(f"\nRapport -> {AUDIT / 'duplicates_report.csv'}")
    print(f"Résumé  -> {AUDIT / 'duplicates_summary.json'}")
    print(f"Terminé en {time.time() - t0:,.0f} s")


if __name__ == "__main__":
    main()