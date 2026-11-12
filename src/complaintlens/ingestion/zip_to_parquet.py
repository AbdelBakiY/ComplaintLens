"""
L0 - Étape 3 : ZIP -> Parquet (un fichier par export), économe en RAM.

Usage :
    python zip_to_parquet.py

Entrée  : data/raw/ccdb_exports/*.zip
Sortie  : data/interim/parts/export_XX.parquet
          data/interim/schema_report.csv   (colonnes présentes dans chaque export)
"""
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import polars as pl

RAW = Path("data/raw/ccdb_exports")
OUT = Path("data/interim/parts")
REPORT = Path("data/interim/schema_report.csv")


def export_num(name: str) -> int:
    return int(re.search(r"CCDB_Export_(\d+)_", name).group(1))


def normalize(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def convert_member(src: Path, dst: Path, num: int) -> None:
    """Lit un fichier extrait en streaming et l'écrit en Parquet."""
    low = src.name.lower()
    if low.endswith(".csv"):
        lf = pl.scan_csv(src, infer_schema=False)            # tout en texte
    elif low.endswith((".json", ".jsonl", ".ndjson")):
        try:
            lf = pl.scan_ndjson(src, infer_schema_length=None)
        except Exception:
            lf = pl.read_json(src).lazy()
        lf = lf.select(pl.all().cast(pl.Utf8))
    else:
        print(f"    ignoré (format non géré) : {src.name}")
        return

    cols = lf.collect_schema().names()
    lf = lf.rename({c: normalize(c) for c in cols})
    lf = lf.with_columns(pl.lit(num, dtype=pl.Int16).alias("source_export"))
    lf.sink_parquet(dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zips = sorted(RAW.glob("CCDB_Export_*.zip"), key=lambda p: export_num(p.name))
    print(f"{len(zips)} exports trouvés")

    for zp in zips:
        num = export_num(zp.name)
        dst = OUT / f"export_{num:02d}.parquet"
        if dst.exists():
            print(f"[{num:02d}] déjà converti, skip")
            continue

        print(f"[{num:02d}] {zp.name}")
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zp) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            for m in members:
                print(f"    - {m.filename} ({m.file_size / 1e6:.0f} Mo décompressé)")
            if len(members) > 1:
                print("    ATTENTION : plusieurs fichiers dans ce ZIP, seul le plus gros est converti.")
            m = max(members, key=lambda x: x.file_size)
            extracted = Path(z.extract(m, tmp))
            convert_member(extracted, dst, num)

        n = pl.scan_parquet(dst).select(pl.len()).collect().item()
        print(f"    -> {dst.name} : {n:,} lignes")

    # Rapport de schéma : quelles colonnes dans quel export
    rows = []
    for p in sorted(OUT.glob("export_*.parquet")):
        for c in pl.scan_parquet(p).collect_schema().names():
            rows.append({"export": p.stem, "column": c})
    report = (
        pl.DataFrame(rows)
        .with_columns(pl.lit(1).alias("present"))
        .pivot(on="export", index="column", values="present")
        .fill_null(0)
        .sort("column")
    )
    report.write_csv(REPORT)
    print(f"\nRapport de schéma -> {REPORT}")
    with pl.Config(tbl_rows=100, tbl_cols=30):
        print(report)


if __name__ == "__main__":
    main()