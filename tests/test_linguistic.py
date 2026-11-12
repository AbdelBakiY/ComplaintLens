import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("modèle spaCy en_core_web_sm absent", allow_module_level=True)

from linguistic import process_text


def tokens(text, **options):
    return process_text(text, **options).split()


def test_lemmatisation():
    t = tokens("i disputed the charges")
    assert "dispute" in t and "charge" in t


def test_mots_vides_retires():
    t = tokens("i disputed the charges on my account")
    for w in ("the", "on", "my", "i"):
        assert w not in t


def test_negation_gardee():
    t = tokens("i did not authorize this charge")
    assert "not" in t and "authorize" in t


def test_negation_contractee_gardee():
    t = tokens("i didn't authorize this charge")
    assert "not" in t


def test_tokens_speciaux_intacts():
    t = tokens("paid [MONEY] to [PII] on [DATE] for [NUM] months")
    for sp in ("[MONEY]", "[PII]", "[DATE]", "[NUM]"):
        assert sp in t
    assert "pay" in t


def test_reference_de_loi_intacte():
    assert "usc_1681" in tokens("violation under usc_1681")


def test_ponctuation_retiree():
    t = tokens("late fee! again, really?")
    assert not any(w in t for w in ("!", ",", "?"))


def test_sans_lemmatisation():
    assert "disputed" in tokens("i disputed the charges", lemmatize=False)


def test_sans_retrait_mots_vides():
    assert "the" in tokens("i disputed the charges", remove_stopwords=False)