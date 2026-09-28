"""
L0 - Étape 5 : extraire les plaintes avec récit exploitable.

Lit les 21 Parquet bruts une seule fois (streaming), garde les récits
non vides contenant au moins un vrai mot, type les colonnes, ajoute des
colonnes de longueur, et écrit un seul fichier intermédiaire.

Usage (depuis la racine du projet) :
    python src/complaintlens/ingestion/filter_narratives.py

Entrée  : data/interim/parts/export_*.parquet
Sortie  : data/interim/narratives.parquet
          data/interim/audit/funnel_step5.json
"""
import json
import time
from pathlib import Path

import polars as pl

PARTS = "data/interim/parts/export_*.parquet"
OUT = Path("data/interim/narratives.parquet")
FUNNEL = Path("data/interim/audit/funnel_step5.json")

MASK = r"X{2,}"          # masques PII du CFPB : XX, XXXX, XXXXXXXX...
WORD = r"[A-Za-z]{2,}"   # un "vrai mot" = au moins 2 lettres


def build_pipeline() -> pl.LazyFrame:
    return (
        pl.scan_parquet(PARTS)
        # 1. Récit sans espaces parasites aux extrémités (le contenu n'est pas modifié)
        .with_columns(
            pl.col("consumer_complaint_narrative").str.strip_chars().alias("narrative")
        )
        .drop("consumer_complaint_narrative")
        # 2. Filtre : récit non nul et non vide
        .filter(pl.col("narrative").str.len_chars() > 0)
        # 3. Colonnes de longueur (le texte reste brut)
        .with_columns(
            pl.col("narrative").str.len_chars().cast(pl.UInt32).alias("n_chars"),
            pl.col("narrative").str.count_matches(MASK).cast(pl.UInt32).alias("n_masks"),
            pl.col("narrative")
            .str.replace_all(MASK, " ")
            .str.count_matches(WORD)
            .cast(pl.UInt32)
            .alias("n_words"),
        )
        # 4. Filtre : au moins un vrai mot
        .filter(pl.col("n_words") > 0)
        # 5. Typage
        .with_columns(
            pl.col("complaint_id").cast(pl.Int64),
            pl.col("date_received").str.to_date("%Y-%m-%d"),
            pl.col("date_sent_to_company").str.to_date("%Y-%m-%d", strict=False),
            pl.col("source_export").cast(pl.Int16),
        )
        .with_columns(pl.col("date_received").dt.year().cast(pl.Int16).alias("year"))
        # 6. Ordre des colonnes : identifiants, texte, cibles, contexte
        .select(
            "complaint_id", "date_received", "year", "narrative",
            "n_chars", "n_words", "n_masks",
            "product", "sub_product", "issue", "sub_issue",
            "company", "company_response_to_consumer", "timely_response",
            "company_public_response", "state", "zip_code", "tags",
            "submitted_via", "date_sent_to_company", "source_export",
        )
    )


def count(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect(engine="streaming").item()


def main():
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FUNNEL.parent.mkdir(parents=True, exist_ok=True)

    # --- Entonnoir : comptages avant écriture ---
    raw = pl.scan_parquet(PARTS)
    n_raw = count(raw)
    n_nonempty = count(
        raw.filter(pl.col("consumer_complaint_narrative").str.strip_chars().str.len_chars() > 0)
    )
    print(f"Lignes brutes             : {n_raw:>12,}")
    print(f"Récit non vide            : {n_nonempty:>12,}")

    # --- Écriture en streaming ---
    print("\nÉcriture en cours...")
    build_pipeline().sink_parquet(OUT, compression="zstd")
    print(f"-> {OUT} ({OUT.stat().st_size / 1e6:,.0f} Mo)")

    # --- Contrôles ---
    lf = pl.scan_parquet(OUT)
    checks = lf.select(
        pl.len().alias("rows"),
        pl.col("complaint_id").n_unique().alias("unique_ids"),
        (pl.col("narrative").str.len_chars() == 0).sum().alias("empty_narratives"),
        pl.col("date_received").null_count().alias("null_dates"),
        (pl.col("n_words") == 0).sum().alias("zero_word_narratives"),
        pl.col("date_received").min().alias("date_min"),
        pl.col("date_received").max().alias("date_max"),
    ).collect(engine="streaming").row(0, named=True)

    n_final = checks["rows"]
    print(f"Au moins un vrai mot      : {n_final:>12,}")
    print(f"  (retirés car 0 mot réel : {n_nonempty - n_final:,})")

    print("\n=== Contrôles ===")
    ok = True
    for name, cond in [
        ("IDs uniques", checks["unique_ids"] == n_final),
        ("Aucun récit vide", checks["empty_narratives"] == 0),
        ("Aucune date nulle", checks["null_dates"] == 0),
        ("Aucun récit à 0 mot", checks["zero_word_narratives"] == 0),
    ]:
        print(f"  [{'OK' if cond else 'ÉCHEC'}] {name}")
        ok &= cond
    print(f"  Période : {checks['date_min']} -> {checks['date_max']}")

    print("\n=== Schéma ===")
    for col, dtype in lf.collect_schema().items():
        print(f"  {col:<32} {dtype}")

    funnel = {
        "raw_rows": n_raw,
        "non_empty_narrative": n_nonempty,
        "at_least_one_real_word": n_final,
        "removed_zero_word": n_nonempty - n_final,
        "date_min": str(checks["date_min"]),
        "date_max": str(checks["date_max"]),
        "all_checks_passed": ok,
    }
    FUNNEL.write_text(json.dumps(funnel, indent=2))
    print(f"\nEntonnoir -> {FUNNEL}")
    print(f"Terminé en {time.time() - t0:,.0f} s")


if __name__ == "__main__":
    main()