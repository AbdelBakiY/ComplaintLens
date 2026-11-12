"""
L1 - Nettoyage des récits.

Deux niveaux :
  - "light" : pour les transformers. On garde la casse, la ponctuation et les nombres.
  - "full"  : pour TF-IDF. "light" + minuscules, nombres -> [NUM], ponctuation répétée réduite.

Règles communes (décisions du L1) :
  - apostrophes, guillemets, tirets et puces typographiques normalisés ;
  - URL -> [URL], email -> [EMAIL] ;
  - dates masquées ou non (XX/XX/XXXX, XX/XX/2020, 03/12/19) -> [DATE], sans l'année ;
  - montants {$790.00} -> [MONEY] ;
  - masques XXXX (simples, longs, XXXX/XXXX) -> [MASK], et une rafale de masques -> un seul [MASK] ;
  - références de loi (FCRA, 15 USC 1681...) gardées telles quelles ;
  - espaces et retours à la ligne -> un seul espace.

Le texte brut des splits n'est jamais modifié : ces fonctions produisent une nouvelle colonne.
"""
import polars as pl

SPECIAL_TOKENS = ["[MASK]", "[DATE]", "[MONEY]", "[URL]", "[EMAIL]", "[NUM]"]
LEVELS = ("light", "full")

TYPO_FROM = ["\u2019", "\u2018", "\u201c", "\u201d", "\u2013", "\u2014", "\u2022", "\u2026"]
TYPO_TO = ["'", "'", '"', '"', "-", "-", " ", "..."]

URL = r"https?\s*:\s*//\S+|www\.\S+"
EMAIL = r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"
DATE = r"\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:X{2,}|\d{2,4})\b"
MONEY = r"\{\$[^}]*\}"
MASK = r"X{2,}(?:/X{2,})*"
MASK_RUN = r"\[MASK\](?:\s+\[MASK\])+"
NUMBER = r"\b\d[\d,.]*\b"
REPEATED_PUNCT = r"([!?.])[!?.]+"


def clean_expr(expr: pl.Expr, level: str = "light") -> pl.Expr:
    """Expression Polars qui nettoie une colonne de récits."""
    if level not in LEVELS:
        raise ValueError(f"level doit être dans {LEVELS}, reçu : {level!r}")

    out = (
        expr.str.replace_many(TYPO_FROM, TYPO_TO)
        # l'ordre compte : URL et email avant tout le reste, dates avant les masques
        .str.replace_all(URL, " [URL] ")
        .str.replace_all(EMAIL, " [EMAIL] ")
        .str.replace_all(DATE, " [DATE] ")
        .str.replace_all(MONEY, " [MONEY] ")
        .str.replace_all(MASK, " [MASK] ")
        .str.replace_all(MASK_RUN, "[MASK]")
    )

    if level == "full":
        out = (
            out.str.to_lowercase()
            # remettre les tokens spéciaux en majuscules après le passage en minuscules
            .str.replace_many([t.lower() for t in SPECIAL_TOKENS], SPECIAL_TOKENS)
            .str.replace_all(NUMBER, " [NUM] ")
            .str.replace_all(REPEATED_PUNCT, "${1}")
        )

    return out.str.replace_all(r"\s+", " ").str.strip_chars()


def clean_text(text: str, level: str = "light") -> str:
    """Nettoie un seul texte (tests, application)."""
    return pl.DataFrame({"t": [text]}).select(clean_expr(pl.col("t"), level))["t"][0]