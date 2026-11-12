import pytest

from clean import clean_text


class TestCleanLight:
    def test_rafale_de_masques_un_seul_token(self):
        assert clean_text("paid XXXX XXXX XXXX on time") == "paid [MASK] on time"

    def test_masque_long(self):
        assert clean_text("account XXXXXXXX closed") == "account [MASK] closed"

    def test_masques_separes_par_virgule_restent_distincts(self):
        assert clean_text("to XXXX, XXXX and") == "to [MASK] , [MASK] and"

    def test_dates_sans_annee(self):
        assert clean_text("on XX/XX/2020 and XX/XX/XXXX") == "on [DATE] and [DATE]"

    def test_date_non_masquee(self):
        assert clean_text("on 03/12/19 I called") == "on [DATE] I called"

    def test_montant(self):
        assert clean_text("I owe {$790.00} now") == "I owe [MONEY] now"

    def test_url_avec_espaces_du_cfpb(self):
        assert clean_text("go to https : //www.ftc.gov now") == "go to [URL] now"

    def test_email(self):
        assert clean_text("write to help@bank.com today") == "write to [EMAIL] today"

    def test_apostrophe_typographique(self):
        assert clean_text("my daughter\u2019s account") == "my daughter's account"

    def test_retours_a_la_ligne(self):
        assert clean_text("first line\n\nsecond   line") == "first line second line"

    def test_reference_de_loi_gardee(self):
        assert clean_text("under 15 USC 1681 ( FCRA )") == "under 15 USC 1681 ( FCRA )"

    def test_casse_et_nombres_gardes(self):
        assert clean_text("I waited 3 YEARS!!!") == "I waited 3 YEARS!!!"

    def test_mot_avec_x_minuscules_intact(self):
        assert clean_text("Exxon charged me") == "Exxon charged me"


class TestCleanFull:
    def test_minuscules_nombres_ponctuation(self):
        assert clean_text("I OWE {$10.00} for 3 years!!!", "full") == "i owe [MONEY] for [NUM] years!"

    def test_tokens_speciaux_restent_en_majuscules(self):
        assert clean_text("Paid XXXX on XX/XX/2020", "full") == "paid [MASK] on [DATE]"

    def test_reference_usc_simple(self):
        assert clean_text("under 15 USC 1681", "full") == "under usc_1681"

    def test_reference_us_code_avec_sous_section(self):
        assert clean_text("15 U.S. Code 1681c2 - Block", "full") == "usc_1681 - block"

    def test_reference_avec_parenthese_numerotee(self):
        assert clean_text("15 US Code 1692a ( 6 )", "full") == "usc_1692 ( [NUM] )"

    def test_reference_avec_paragraphe(self):
        assert clean_text("15 U.S.C. \u00a7 1692e", "full") == "usc_1692"


def test_niveau_inconnu():
    with pytest.raises(ValueError):
        clean_text("abc", "medium")