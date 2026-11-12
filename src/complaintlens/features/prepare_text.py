"""
L2 - Étape 1 : préparer une fois pour toutes les textes des splits.

Pour chaque récit, 3 versions du texte :
  - text_light : clean_expr(..., "light")                      -> transformers (L5)
  - text_full  : clean_expr(..., "full")                       -> TF-IDF sur caractères
  - text_tfidf : clean_expr(..., "full") + linguistic (spaCy)  -> TF-IDF sur mots

Usage (depuis la racine du projet) :
    python src/complaintlens/features/prepare_text.py --splits train val test --n-process 4

Entrée  : data/processed/{split}.parquet
Sortie  : data/features/{split}_text.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "complaintlens" / "preprocessing"))

from clean import clean_expr  # noqa: E402
from linguistic import process_texts  # noqa: E402

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/features")
KEEP = ["complaint_id", "year", "product", "issue", "sub_issue"]


def prepare(split: str, n_process: int, chunk: int) -> None:
    t0 = time.time()
    df = (
        pl.read_parquet(IN_DIR / f"{split}.parquet", columns=KEEP + ["narrative"])
        .with_columns(
            clean_expr(pl.col("narrative"), "light").alias("text_light"),
            clean_expr(pl.col("narrative"), "full").alias("text_full"),
        )
        .drop("narrative")
    )

    texts = df["text_full"].to_list()
    processed = []
    for i in range(0, len(texts), chunk):
        processed.extend(process_texts(texts[i:i + chunk], n_process=n_process))
        done = min(i + chunk, len(texts))
        elapsed = time.time() - t0
        eta = elapsed / done * (len(texts) - done)
        print(f"   [{split}] {done:>7,} / {len(texts):,}  ({elapsed / 60:.1f} min, reste ≈ {eta / 60:.1f} min)",
              flush=True)

    df = df.with_columns(pl.Series("text_tfidf", processed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{split}_text.parquet"
    df.write_parquet(out, compression="zstd")

    empty = df.filter(pl.col("text_tfidf").str.len_chars() == 0).height
    print(f"-> {out} : {df.height:,} récits, {out.stat().st_size / 1e6:.0f} Mo, "
          f"{empty} textes vides après traitement, {(time.time() - t0) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--n-process", type=int, default=4)
    p.add_argument("--chunk", type=int, default=10_000)
    args = p.parse_args()
    for split in args.splits:
        print(f"== {split}")
        prepare(split, args.n_process, args.chunk)


if __name__ == "__main__":
    main()