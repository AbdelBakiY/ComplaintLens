"""
L0 - Étape 4 : audit des Parquet bruts (lecture lazy, économe en RAM).

Usage :
    python audit_l0.py

Sortie : affichage console + data/interim/audit/*.csv
"""
from pathlib import Path

import polars as pl

PARTS = "data/interim/parts/*.parquet"
OUT = Path("data/interim/audit")
OUT.mkdir(parents=True, exist_ok=True)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(80)

lf = pl.scan_parquet(PARTS)

# Dates : format inconnu a priori -> on essaie plusieurs formats
lf = lf.with_columns(
    pl.coalesce(
        pl.col("date_received").str.to_date("%Y-%m-%d", strict=False),
        pl.col("date_received").str.to_date("%m/%d/%Y", strict=False),
        pl.col("date_received").str.to_date("%m/%d/%y", strict=False),
        pl.col("date_received").str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
    ).alias("date"),
    (pl.col("consumer_complaint_narrative").str.strip_chars().str.len_chars().fill_null(0)).alias("narr_len"),
)
lf = lf.with_columns(
    (pl.col("narr_len") > 0).alias("has_narr"),
    pl.col("date").dt.year().alias("year"),
)


def show(title, df, name):
    print(f"\n===== {title} =====")
    print(df)
    df.write_csv(OUT / f"{name}.csv")


# 0. Exemples bruts de dates (pour vérifier le format)
show("Exemples de date_received brutes",
     pl.scan_parquet(PARTS).select("date_received").head(5).collect(), "00_date_examples")

# 1. Par export : lignes, récits, dates
by_export = (
    lf.group_by("source_export")
    .agg(
        pl.len().alias("rows"),
        pl.col("has_narr").sum().alias("with_narrative"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        pl.col("date").is_null().sum().alias("date_unparsed"),
    )
    .with_columns((pl.col("with_narrative") / pl.col("rows") * 100).round(1).alias("pct_narr"))
    .sort("source_export")
    .collect(engine="streaming")
)
show("Par export", by_export, "01_by_export")

# 2. Totaux et doublons d'ID entre exports
ids = lf.select(pl.col("complaint_id")).collect(engine="streaming")
tot, uniq = ids.height, ids["complaint_id"].n_unique()
print(f"\n===== Doublons =====\nlignes totales : {tot:,}\nIDs uniques   : {uniq:,}\ndoublons      : {tot - uniq:,}")
del ids

dup_ids_narr = (
    lf.filter(pl.col("has_narr"))
    .group_by("complaint_id").len()
    .filter(pl.col("len") > 1)
    .select(pl.len())
    .collect(engine="streaming").item()
)
print(f"IDs avec récit présents dans >1 export : {dup_ids_narr:,}")

# 3. Récits par année
by_year = (
    lf.group_by("year")
    .agg(pl.len().alias("rows"), pl.col("has_narr").sum().alias("with_narrative"))
    .with_columns((pl.col("with_narrative") / pl.col("rows") * 100).round(1).alias("pct_narr"))
    .sort("year")
    .collect(engine="streaming")
)
show("Par année", by_year, "03_by_year")

# 4. Produits par année (plaintes avec récit uniquement)
prod_year = (
    lf.filter(pl.col("has_narr"))
    .group_by(["product", "year"]).len()
    .collect(engine="streaming")
    .pivot(on="year", index="product", values="len", sort_columns=True)
    .fill_null(0)
    .sort("product")
)
show("Produits x année (avec récit)", prod_year, "04_product_by_year")

# 5. Longueur des récits
lengths = (
    lf.filter(pl.col("has_narr"))
    .select(
        pl.col("narr_len").min().alias("min"),
        pl.col("narr_len").quantile(0.05).alias("p05"),
        pl.col("narr_len").median().alias("median"),
        pl.col("narr_len").quantile(0.95).alias("p95"),
        pl.col("narr_len").max().alias("max"),
        (pl.col("narr_len") < 50).sum().alias("n_below_50_chars"),
    )
    .collect(engine="streaming")
)
show("Longueur des récits (caractères)", lengths, "05_narrative_length")

# 6. Valeurs manquantes (plaintes avec récit)
nulls = (
    lf.filter(pl.col("has_narr"))
    .select(pl.all().null_count())
    .collect(engine="streaming")
    .transpose(include_header=True, header_name="column", column_names=["nulls"])
)
show("Valeurs manquantes (avec récit)", nulls, "06_nulls")

# 7. Trois récits au hasard pour voir à quoi ça ressemble
sample = (
    lf.filter(pl.col("has_narr"))
    .select("complaint_id", "date", "product", "consumer_complaint_narrative")
    .head(200_000).collect()
    .sample(3, seed=42)
)
print("\n===== Exemples de récits =====")
for r in sample.iter_rows(named=True):
    print(f"\n[{r['complaint_id']}] {r['date']} | {r['product']}")
    print(r["consumer_complaint_narrative"][:600], "...")