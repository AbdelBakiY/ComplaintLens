"""
L1 - Traitements linguistiques (spaCy) pour les modèles à base de mots (TF-IDF, L2).

Entrée : texte nettoyé au niveau "full" (clean.py).
Sortie : une chaîne de tokens séparés par des espaces.

Options (toutes activées par défaut) :
  - lemmatize        : chaque mot ramené à sa forme de base (disputed -> dispute) ;
  - remove_stopwords : mots vides retirés (the, of, to...), SAUF les négations ;
  - remove_punct     : ponctuation retirée.

Les tokens spéciaux ([PII], [DATE], [MONEY], [URL], [EMAIL], [NUM]) et les
références de loi (usc_1681...) ne sont jamais modifiés.

Limite connue : le texte "full" est en minuscules, ce qui rend l'étiquetage
grammatical de spaCy un peu moins précis qu'avec la casse d'origine.
"""
from functools import lru_cache

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from clean import SPECIAL_TOKENS

# Négations : des mots vides classiques, mais qui inversent le sens d'une plainte
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody",
             "cannot", "n't", "n\u2019t", "without"}
STOPWORDS = frozenset(STOP_WORDS - NEGATIONS)
PROTECTED = frozenset(SPECIAL_TOKENS)


@lru_cache(maxsize=1)
def build_nlp(model: str = "en_core_web_sm"):
    """Charge spaCy une seule fois, sans parser ni NER (inutiles ici, et lents)."""
    nlp = spacy.load(model, disable=["parser", "ner"])
    for tok in SPECIAL_TOKENS:
        nlp.tokenizer.add_special_case(tok, [{"ORTH": tok}])
    return nlp


def _is_protected(tok) -> bool:
    return tok.text in PROTECTED or tok.lower_.startswith("usc_")


def _keep(tok, remove_stopwords: bool, remove_punct: bool) -> bool:
    if tok.is_space:
        return False
    if _is_protected(tok):
        return True
    if remove_punct and (tok.is_punct or tok.is_quote or tok.is_bracket):
        return False
    if remove_stopwords and tok.lower_ in STOPWORDS:
        return False
    return True


def _form(tok, lemmatize: bool) -> str:
    if _is_protected(tok):
        return tok.text
    return tok.lemma_.lower() if lemmatize else tok.lower_


def process_texts(texts: list[str], lemmatize: bool = True, remove_stopwords: bool = True,
                  remove_punct: bool = True, batch_size: int = 500, n_process: int = 1) -> list[str]:
    """Traite une liste de textes. n_process > 1 pour utiliser plusieurs cœurs."""
    nlp = build_nlp()
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        out.append(" ".join(_form(t, lemmatize) for t in doc
                            if _keep(t, remove_stopwords, remove_punct)))
    return out


def process_text(text: str, **options) -> str:
    """Traite un seul texte (tests, application)."""
    return process_texts([text], **options)[0]