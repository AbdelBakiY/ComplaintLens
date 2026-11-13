# ComplaintLens — état d'avancement

> Document de passation : où en est le projet au **13/11/2026**, ce qui est fait, les décisions prises et la suite.
> Spécification complète : [Cahier des charges](Cahier%20des%20charges%20—%20ComplaintLens.pdf).
> Fiche du corpus : [docs/datasheet.md](docs/datasheet.md).
> Auteur : **Abdel** — projet personnel.

## 1. Le projet en bref

Une plateforme NLP de bout en bout sur les **plaintes clients du CFPB** (Consumer Financial Protection Bureau, régulateur financier américain) : classification du produit et du motif, extraction d'entités, thèmes, résumé, RAG réglementaire et application déployée.

- **Corpus** : archive FOIA du CFPB, 21 ZIP (déc. 2011 → août 2026), récits libres inclus. Le corpus est figé : le CFPB a cessé de publier les récits le 14/08/2026.
- **Objectifs** : Product (10 classes) macro-F1 ≥ 0,85 ; Issue (hiérarchique) macro-F1 ≥ 0,60 ; NER (8 types) F1 ≥ 0,75 ; RAG ≥ 80 % de réponses correctement sourcées.
- **Découpage** : 10 lots, de L0 (ingestion) à L10 (explicabilité et équité), plus un lot MLOps transverse.

**Statut :**
- **L0 — Ingestion et data engineering** : terminé (points ouverts listés en section 10).
- **L1 — Prétraitement et analyse linguistique** : terminé.
- **L2 — Baselines classiques** : terminé pour la cible Product (macro-F1 val 0,763). Cibles Issue et Company response à faire.
- **L3 — Embeddings statiques et modèles de langue** : terminé.
- Prochain lot : **L4 — Deep learning from scratch**.

## 2. Environnement et workflow

| Où | Quoi |
|---|---|
| **PC** (Windows, VS Code, Python 3.13) | Écriture du code, tests unitaires, `git push` |
| **Kaggle** (notebooks, ~30 Go de RAM, GPU 30 h/semaine) | `git pull`, exécution sur les vraies données, création des datasets |
| **GitHub** | Code, tests, configs, docs, rapports, notebooks. **Jamais de données ni de modèles.** |

- Datasets Kaggle privés :
  - `complaintlens-data` : `narratives_dedup.parquet` ;
  - `complaintlens-splits` : les 4 splits ;
  - `complaintlens-features` : textes prétraités `{train,val,test}_text.parquet` ;
  - `complaintlens-l2-cache` : cache des calculs et modèle L2 (`svm_tfidf_v1.joblib`) ;
  - `complaintlens-l3-cache` : cache L3 et embeddings (`w2v_complaints_d100.txt`, `ft_complaints_d100.kv`).
- Traitement en **Polars** (2.1.0, version figée sur Kaggle) : lecture lazy, traitement par tranches, jamais la colonne `narrative` entière en mémoire sur le PC.
- **spaCy** avec le modèle `en_core_web_sm` (`python -m spacy download en_core_web_sm`).
- Tokenizers Hugging Face (`transformers`, `tokenizers`) et téléchargements (GloVe via `gensim.downloader`) : le notebook doit avoir **Internet : ON** sur Kaggle.
- `data/` est dans le `.gitignore`. DVC est dans les dépendances mais n'est pas encore initialisé.
- Tous les scripts se lancent **depuis la racine du projet** (chemins relatifs).
- **Le CPU suffit pour L0 à L3.** Le GPU de Kaggle est réservé à l'entraînement des réseaux (L4, L5, L9).
- **Cache dans les notebooks Kaggle** : chaque calcul long passe par une fonction `cached(nom, fonction)` qui sauvegarde son résultat avec `joblib`. Au Save & Run suivant, le résultat est rechargé depuis l'Output de la version précédente, ajouté en input. Changer le nom du cache dès que le code de l'étape change.

## 3. Pipeline

### L0 — Ingestion (`src/complaintlens/ingestion/`)

| # | Étape | Script | Sortie |
|---|---|---|---|
| 0 | Téléchargement des 21 ZIP + base courante, checksums SHA-256 | manuel | `data/raw/ccdb_exports/`, `data/raw/ccdb_current/`, `data/raw/manifest.json` |
| 1 | ZIP → Parquet (un fichier par export, tout en texte, colonnes en snake_case, `source_export`) | `zip_to_parquet.py` | `data/interim/parts/export_XX.parquet` |
| 2 | Audit des données brutes | `audit_l0.py` | `data/interim/audit/0*.csv` |
| 3 | Filtrage : récit non vide, au moins 1 vrai mot ; typage ; `n_chars`, `n_words`, `n_masks`, `year` | `filter_narratives.py` | `data/interim/narratives.parquet` |
| 4 | Harmonisation des produits (21 libellés → 10 classes) | `apply_product_mapping.py` + `configs/product_mapping.csv` | `data/interim/narratives_mapped.parquet` |
| 5 | Doublons exacts : hash 64 bits du texte normalisé, marquage sans suppression | `find_duplicates.py` | `data/interim/narratives_dedup.parquet` |
| 6 | Dédoublonnage, split temporel, échantillonnage | `make_splits.py` | `data/processed/{train,val,test,drift_2026}.parquet` |

```bash
python src/complaintlens/ingestion/zip_to_parquet.py
python src/complaintlens/ingestion/audit_l0.py
python src/complaintlens/ingestion/filter_narratives.py
python src/complaintlens/ingestion/apply_product_mapping.py
python src/complaintlens/ingestion/find_duplicates.py --min-words 10
python src/complaintlens/ingestion/make_splits.py --alpha 0.5 --n-val 50000 --n-test 100000 --seed 42
```

### L1 — Prétraitement (`src/complaintlens/preprocessing/`)

Ces modules ne modifient jamais les données : ils produisent une version transformée du texte, à la demande.

| Module | Rôle | Fonctions |
|---|---|---|
| `clean.py` | Nettoyage en 2 niveaux : `light` (transformers) et `full` (TF-IDF) | `clean_expr(colonne, niveau)`, `clean_text(texte, niveau)`, `SPECIAL_TOKENS` |
| `linguistic.py` | Lemmatisation, retrait des mots vides (sauf négations) et de la ponctuation, avec spaCy | `process_texts(textes, ...)`, `process_text(texte, ...)` |

Chaîne pour les modèles à base de mots (L2, L3) : `narrative` → `clean_expr(..., "full")` → `process_texts(...)` → TF-IDF ou embeddings.
Chaîne pour les transformers (L5) : `narrative` → `clean_expr(..., "light")` → tokenizer du modèle, avec les `SPECIAL_TOKENS` déclarés.

L'analyse du L1 est dans `notebooks/complaintlens-l1.ipynb` (exécuté sur Kaggle), ses résultats dans `reports/l1/`.

### L2 — Features et baselines (`src/complaintlens/features/`)

| # | Étape | Où | Sortie |
|---|---|---|---|
| 1 | Prétraitement unique des 250 000 récits (train, val, test) : `text_light`, `text_full`, `text_tfidf` | `prepare_text.py` (Kaggle, ≈ 34 min, 4 cœurs) | `data/features/{train,val,test}_text.parquet` → dataset `complaintlens-features` |
| 2 | TF-IDF, comparaison des modèles, réglage, ablations, analyse d'erreurs | `notebooks/complaintlens-l2-models.ipynb` (Kaggle) | `reports/l2/`, modèle `models/svm_tfidf_v1.joblib` dans l'Output Kaggle |

```bash
python src/complaintlens/features/prepare_text.py --splits train val test --n-process 4
```

Colonnes produites : `complaint_id`, `year`, `product`, `issue`, `sub_issue`, `text_light` (nettoyage léger), `text_full` (nettoyage complet, sans lemmes), `text_tfidf` (nettoyage complet, lemmes, sans mots vides).

### L3 — Embeddings et modèle de langue

Tout est dans `notebooks/complaintlens-l3.ipynb` (Kaggle, CPU, ≈ 25 min au premier passage, ≈ 5 min ensuite grâce au cache).

| # | Étape | Entrée | Sortie |
|---|---|---|---|
| 1 | Word2Vec et FastText (skip-gram, 100 dimensions) sur le train | `text_tfidf` | `models/w2v_complaints_d100.txt`, `models/ft_complaints_d100.kv` |
| 2 | Évaluation intrinsèque : voisins, analogies métier, intrus | embeddings | `analogies_*.csv`, `odd_one_out.csv` |
| 3 | Évaluation extrinsèque : moyenne des vecteurs + régression logistique | embeddings | `extrinsic_*.csv` |
| 4 | Comparaison avec GloVe 100d (Wikipédia + Gigaword) | embeddings + GloVe | `*_glove_vs_w2v.csv`, `domain_shift_neighbors.csv` |
| 5 | Trigramme Kneser-Ney par produit : perplexité, classification, génération | `text_full` | `ngram_perplexity_matrix.csv`, `ngram_generations.csv` |

### Tests

```bash
pytest -v
```

**58 tests** dans `tests/` :

| Fichier | Tests | Ce qui est vérifié |
|---|---|---|
| `test_ingestion_units.py` | 20 | Normalisation des doublons, échantillonnage tempéré, dédoublonnage, tirage des splits |
| `test_splits_data.py` | 9 | Intégrité des splits réels (ne tournent que là où `data/processed/` existe, donc sur Kaggle) |
| `test_clean.py` | 20 | Masques, dates, montants, URL, emails, rafales, références de loi, niveaux `light` / `full` |
| `test_linguistic.py` | 9 | Lemmes, mots vides, négations gardées, tokens spéciaux protégés, options (ignorés si `en_core_web_sm` est absent) |

## 4. Chiffres clés — L0

**Entonnoir**

| Étape | Récits |
|---|---|
| Lignes brutes (21 exports) | 17 533 918 |
| Récit non vide | 3 851 415 |
| Au moins un vrai mot | 3 850 263 |
| Sans les classes `__DROP__` | 3 844 365 |
| Après dédoublonnage exact (1 récit par groupe) | 2 316 381 |
| **Sous-corpus final (4 splits)** | **356 310** |

- Récits du **19/03/2015 au 11/08/2026**.
- Longueur : médiane 670 caractères, p5 127, p95 3 007, max 35 984.

**Doublons exacts** (seuil : 10 mots)
- 262 765 groupes, 1 790 903 récits (**46,5 %**), soit 1 528 138 copies en trop.
- Plus gros groupe : 31 053 récits identiques. Ce sont surtout des lettres types (« credit repair ») en `credit_reporting` : 65,5 % de ses récits sont des copies.
- Part de copies : 3,6 % en 2015, 62,6 % en 2025.

**Récits uniques disponibles par période** (après dédoublonnage)

| Classe | ≤ 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| credit_reporting | 569 176 | 237 325 | 302 422 | 2 218 |
| debt_collection | 217 966 | 47 215 | 73 485 | 29 413 |
| credit_card | 121 859 | 31 248 | 38 940 | 20 182 |
| mortgage | 112 942 | 12 187 | 14 013 | 7 768 |
| checking_savings | 101 743 | 29 409 | 46 805 | 21 288 |
| student_loan | 39 948 | 8 009 | 10 637 | 4 046 |
| money_transfer | 38 190 | 10 202 | 36 896 | 9 718 |
| vehicle_loan | 33 968 | 7 453 | 11 597 | 5 876 |
| payday_personal_loan | 23 654 | 5 149 | 7 761 | 4 342 |
| prepaid_card | 12 417 | 3 639 | 3 816 | 1 459 |
| **Total** | **1 271 863** | **391 836** | **546 372** | **106 310** |

**Splits finaux** (`data/processed/`, dataset Kaggle `complaintlens-splits`)

| Classe | train | val | test | drift_2026 |
|---|---|---|---|---|
| credit_reporting | 24 573 (24,6 %) | 30 275 (60,6 %) | 55 624 (55,6 %) | 2 218 (2,1 %) |
| debt_collection | 15 206 (15,2 %) | 6 010 (12,0 %) | 13 302 (13,3 %) | 29 413 (27,7 %) |
| credit_card | 11 370 (11,4 %) | 3 980 (8,0 %) | 7 072 (7,1 %) | 20 182 (19,0 %) |
| mortgage | 10 946 (10,9 %) | 1 510 (3,0 %) | 2 543 (2,5 %) | 7 768 (7,3 %) |
| checking_savings | 10 389 (10,4 %) | 3 843 (7,7 %) | 8 563 (8,6 %) | 21 288 (20,0 %) |
| student_loan | 6 510 (6,5 %) | 1 010 (2,0 %) | 2 003 (2,0 %) | 4 046 (3,8 %) |
| money_transfer | 6 365 (6,4 %) | 1 325 (2,7 %) | 6 644 (6,6 %) | 9 718 (9,1 %) |
| vehicle_loan | 6 003 (6,0 %) | 954 (1,9 %) | 2 122 (2,1 %) | 5 876 (5,5 %) |
| payday_personal_loan | 5 009 (5,0 %) | 631 (1,3 %) | 1 436 (1,4 %) | 4 342 (4,1 %) |
| prepaid_card | 3 629 (3,6 %) | 462 (0,9 %) | 691 (0,7 %) | 1 459 (1,4 %) |
| **Total** | **100 000** | **50 000** | **100 000** | **106 310** |

## 5. Chiffres clés — L1

**Rupture `credit_reporting` 2026 (expliquée)**
- Les récits passent de 38 858 en décembre 2025 à 862 en janvier 2026 (÷ 45), puis restent sous 900 par mois.
- La chute touche toutes les entreprises dans les mêmes proportions (Equifax, TransUnion, Experian et les autres : ÷ 100 à 370 entre 2025 et 2026), autant les récits uniques que les copies.
- Les autres produits n'ont qu'un creux ponctuel en janvier 2026 et retrouvent leur niveau dès février.
- Conclusion : changement de publication propre à ce produit, pas une évolution des plaintes.

**Campagne de janvier 2025** : pic simultané en `money_transfer` (≈ 42 000 récits dans le mois, contre ≈ 1 000 habituellement), `checking_savings` et `student_loan`, visible aussi sur les récits uniques. Campagne à textes variants, que la dédup exacte ne détecte pas.

**Inventaire du texte** (20 000 récits du train)

| Élément | % des récits | Traitement |
|---|---|---|
| Masques `XXXX` | 86,1 % (17,7 par récit) | `[PII]`, une rafale = un seul token |
| Dates masquées | 44,6 % | `[DATE]`, sans l'année |
| Montants `{$...}` | 36,5 % | `[MONEY]` |
| Nombres | 63,3 % | gardés (`light`) / `[NUM]` (`full`) |
| Retours à la ligne | 39,3 % | un seul espace |
| Mots en majuscules | 31,6 % | gardés (`light`) / minuscules (`full`) |
| Références de loi | 9,9 % | gardées (`light`) / `usc_XXXX` (`full`) |
| URL, emails | < 1 % | `[URL]`, `[EMAIL]` |
| Encodage cassé | 0 % | rien à faire |

Après nettoyage : 0 % de masques restants, 5,8 `[PII]`, 1,6 `[DATE]` et 1,1 `[MONEY]` par récit en moyenne.

**Comparaison des tokenizers** (vocabulaire appris sur 20 000 récits du train, mesures sur 5 000 de la val, niveau `light`)

| Tokenizer | Vocabulaire vu | % OOV | Tokens / récit | Fertilité | % récits > 512 |
|---|---|---|---|---|---|
| Espaces | 68 016 | 1,01 | 211,0 | 1,00 | 7,5 |
| spaCy | 41 161 | 0,48 | 228,1 | 1,08 | 8,8 |
| BPE GPT-2 | 25 008 | 0,12 | 243,6 | 1,15 | 9,8 |
| WordPiece BERT | 15 282 | 0,05 | 246,6 | 1,17 | 10,3 |

Aucun token `[UNK]` chez BERT. Environ **10 % des récits dépassent 512 tokens**.

**BPE entraîné sur les plaintes** (100 000 récits du train)

| Tokenizer | Vocabulaire | Tokens / récit | Gain vs GPT-2 |
|---|---|---|---|
| GPT-2 (générique) | 50 263 | 243,6 | — |
| BPE plaintes 8k | 8 006 | 245,5 | −0,8 % |
| BPE plaintes 16k | 16 006 | 235,9 | +3,2 % |
| BPE plaintes 30k | 30 528 | 232,8 | +4,4 % |
| BPE plaintes 50k | 50 263 | 231,8 | +4,8 % |

Sur 12 termes métier testés, GPT-2 en découpe 10 (`FDCPA` → `F` `DC` `PA`, `Equifax` → `Equ` `ifax`) ; notre BPE les garde tous entiers.

**Traitements linguistiques** (5 000 récits du train, niveau `full`)

| Configuration | Vocabulaire | Tokens / récit |
|---|---|---|
| Ponctuation retirée seulement | 15 086 | 201,7 |
| + mots vides retirés | 14 795 (−2 %) | 94,5 (−53 %) |
| + lemmatisation | 11 908 (−21 %) | 201,7 |
| + les deux | 11 701 (−22 %) | 94,5 (−53 %) |

Temps : ≈ 48 s pour 5 000 récits avec 4 cœurs. `not` est le 2e token le plus fréquent après traitement.

**Analyse linguistique** (train)
- **Zipf** (50 000 récits, 10,4 M de tokens, 41 542 mots différents) : pente −1,71, plus raide que la valeur classique (≈ −1) à cause de la normalisation des nombres et des entités masquées. **45 % du vocabulaire est constitué d'hapax.**
- **Heaps** : V = 30,8 · N^0,45. Le vocabulaire croît sans saturer (doubler le corpus ≈ +37 % de vocabulaire).
- **Collocations (PMI)** : trois familles. Des noms d'entreprises (Goldman Sachs, Penny Mac, Navy Federal…), un jargon par produit (grace period, co signer, wear and tear, child support…) et des formules juridiques de lettres types (et seq, willful noncompliance). Certaines expressions sont transverses (social security, identity theft).
- **Lisibilité** : niveau scolaire médian 9,3, facilité 61,8 (anglais standard). Le `credit_reporting` se distingue (10,5 ; 52,4), en cohérence avec la forte part de lettres types juridiques.

## 6. Chiffres clés — L2 (cible Product)

**Tâche** : prédire le produit (10 classes) à partir du seul récit. Métrique principale : macro-F1 (chaque classe pèse autant). Toutes les mesures sont faites sur la **validation 2024** ; le test 2025 n'a pas encore été utilisé.

**Représentation** : TF-IDF sur `text_tfidf`, unigrammes + bigrammes, `min_df=5`, `max_df=0.5`, `sublinear_tf`, 200 000 termes (`token_pattern=\S+`, le texte étant déjà tokenisé par spaCy).

**Comparaison des modèles** (val 2024)

| Modèle | Features | Accuracy | Macro-F1 | Entraînement |
|---|---|---|---|---|
| Classe majoritaire | — | 0,606 | 0,075 | — |
| Naive Bayes (Complement) | mots | 0,838 | 0,714 | 1 s |
| Régression logistique (C = 4) | mots | 0,848 | 0,752 | ≈ 2 min |
| SVM linéaire | caractères 3-5 | 0,845 | 0,747 | ≈ 8 min |
| SVM linéaire | mots + caractères | 0,847 | 0,752 | ≈ 2 min |
| LightGBM | mots (20k, chi²) | 0,848 | 0,755 | ≈ 25 min |
| SVM linéaire (C = 0,5) | mots | 0,848 | 0,754 | 15 s |
| **SVM linéaire réglé** | **mots** | **0,856** | **0,763** | 15 s |

- Seul le SVM a été réglé (Optuna, 30 essais) : la comparaison est légèrement favorable au SVM.
- Avant réglage, les modèles linéaires plafonnent tous autour de 0,75. Les arbres (LightGBM, et Random Forest à 0,719) n'apportent rien, pour un coût 100 fois supérieur.

**Réglage** : C = 0,16 (régularisation forte : 200 000 colonnes pour 100 000 récits) et α simulé = 0,75 via des poids de classe (le train a été tiré avec α = 0,5).
- Gain : +0,0085 de macro-F1, IC 95 % par bootstrap apparié [0,0050 ; 0,0124] ; le SVM réglé gagne dans 100 % des 1 000 tirages.
- Réglé et mesuré sur la même val : léger biais optimiste, atténué par un plateau large (C entre 0,16 et 0,42, α entre 0,70 et 0,84, tous à moins de 0,001 du meilleur).

**Effet de α** (C = 0,16)

| α | Macro-F1 | Précision petites classes | Rappel petites classes | Rappel credit_reporting |
|---|---|---|---|---|
| 0 | 0,737 | 0,512 | 0,754 | 0,883 |
| 0,25 | 0,751 | 0,570 | 0,726 | 0,897 |
| 0,5 | 0,757 | 0,624 | 0,686 | 0,909 |
| **0,75** | **0,762** | **0,686** | **0,641** | **0,919** |
| 1 | 0,758 | 0,742 | 0,584 | 0,927 |

Petites classes : `payday_personal_loan`, `prepaid_card`, `vehicle_loan`. Monter α échange du rappel contre de la précision sur ces classes.

**F1 par classe** (SVM réglé) : credit_reporting 0,922 ; mortgage 0,873 ; student_loan 0,868 ; checking_savings 0,812 ; credit_card 0,745 ; prepaid_card 0,735 ; debt_collection 0,715 ; vehicle_loan 0,708 ; money_transfer 0,706 ; **payday_personal_loan 0,542** (catégorie hétérogène : payday, title, prêt personnel, avance).

**Ablations** (SVM réglé)

| Configuration | Macro-F1 |
|---|---|
| Référence (lemmes, sans mots vides, mots 1-2) | 0,763 |
| Unigrammes seuls (13 570 termes) | 0,752 |
| Sans lemmatisation ni retrait des mots vides (`text_full`) | 0,754 |
| Noms d'entreprises masqués | 0,743 |

- Les bigrammes et le prétraitement linguistique apportent chacun environ 1 point.
- **Masquage des entreprises** : 208 tokens, tirés des noms des entreprises ayant au moins 30 plaintes en train et filtrés par fréquence en anglais courant (`wordfreq`, zipf < 3,5), plus des marques-mots courants et des variantes d'écriture. Coût : 1,9 point. Le modèle repose surtout sur le contenu.
  - La dépendance est maximale pour `student_loan` (−4,4 : marché concentré sur quelques gestionnaires), `money_transfer` (−3,6) et `payday_personal_loan` (−2,8), et minimale pour `credit_reporting` (−0,4).
  - Après masquage, les termes décisifs deviennent du vocabulaire métier (forbearance, pslf, dispute, investigation, refinance, heloc).
  - Le masquage étant imparfait (`capital`, `trans`, `mastercard` passent), ce chiffre est une borne basse.

**Termes les plus discriminants** (extrait) : mortgage → escrow, modification, foreclosure, appraisal ; checking_savings → overdraft, deposit, branch ; credit_reporting → equifax, experian, inquiry, `usc_1681` ; prepaid_card → prepaid, netspend, unemployment, edd (artefact Covid : allocations chômage versées sur cartes prépayées).

**Analyse d'erreurs**
- Confusions principales, quasi symétriques : `debt_collection` ↔ `credit_reporting` (1 337 / 1 335), `credit_card` ↔ `credit_reporting` (554 / 449), `money_transfer` ↔ `checking_savings` (261 / 272).
- Lecture manuelle de 100 erreurs tirées dans les principales confusions (un annotateur, échantillon non représentatif) :

| Cause | Part |
|---|---|
| Récit relevant de deux produits | 46 % |
| Étiquette vraisemblablement erronée | 29 % |
| Texte inexploitable | 11 % |
| Vraie erreur du modèle | 10 % |
| Raccourci par nom d'entreprise | 4 % |

- L'essentiel des erreurs vient de l'ambiguïté du découpage par produit et du choix du consommateur, ce qui borne le score atteignable.
- Le modèle est confiant quand l'étiquette est douteuse (écart de score médian 1,03) et hésitant sur ses vraies erreurs (0,27). Cet écart pourra servir à signaler les cas incertains (L8) et à repérer le bruit d'étiquette.
- Plusieurs erreurs sont des variantes d'une même lettre type, ce qui confirme l'intérêt du MinHash.

## 7. Chiffres clés — L3 (embeddings et modèle de langue)

Tout est appris sur le train uniquement.

**Embeddings** appris sur les 100 000 récits du train (`text_tfidf`, 9,8 M de tokens, 15 093 mots vus au moins 5 fois) : skip-gram, 100 dimensions, fenêtre 5, 10 époques, sous-échantillonnage 1e-3, 10 exemples négatifs. FastText : n-grammes de caractères 3 à 6, 200 000 buckets. Apprentissage : 6 min (Word2Vec), 9 min (FastText).

| Test | Word2Vec | FastText | GloVe 100d (Wikipédia, 6 Md de mots) |
|---|---|---|---|
| Intrus (10 listes métier) | 10/10 | 10/10 | 6/6 (seules listes connues) |
| Analogies métier, top 5 (104 questions) | 25 % | 17 % | 1,6 % (64 communes ; Word2Vec : 18,8 %) |
| Classification : moyenne des vecteurs + régression logistique (macro-F1 val) | 0,700 | 0,702 | 0,586 |

- **Similarité** : voisins pertinents sans supervision (car → vehicle, truck, suv ; equifax → experian, transunion, lexisnexis ; overdraft → nsf, overdrawn ; foreclosure → modification, eviction, tro). Word2Vec regroupe aussi les fautes de frappe (fraudulent → fradulent, fruadulant). FastText privilégie la forme (equifax → equifax-, /equifax) mais place les mots inconnus (equifaxx → equifax ; repossesed → repossess).
- **Analogies** : 3 relations construites à la main (entreprise → produit, problème → produit, loi → domaine). Échec relatif, la formule 3CosMul n'aide pas (21 % au top 5). Les erreurs restent plausibles (midland : debt :: ocwen : ? → sps, un autre gestionnaire de prêts immobiliers ; santander → scusa) : le modèle renvoie le voisin de *c* au lieu d'appliquer la relation, car une entreprise et son produit sont déjà voisins sur la carte. Les embeddings capturent la proximité thématique, pas les relations structurées.
- **Classification** : 6 points sous TF-IDF (0,763), avec 2 000 fois moins de dimensions (92 % de la performance). La pondération IDF de la moyenne dégrade le score (0,680) : elle donne du poids à des mots rares, mal positionnés. La perte est maximale pour les classes dont le signal tient en quelques mots (`payday_personal_loan` −13,8, `vehicle_loan` −9,8, `prepaid_card` −7,5) et minimale quand il est diffus (`credit_reporting` −1,2).
- **Décalage de domaine** : GloVe couvre 82 % de notre vocabulaire (88 % des occurrences) ; il ignore les tokens spéciaux, le jargon (furnisher, fdcpa, pslf, 605b, autopay) et les entités postérieures à 2014 (covid, venmo, coinbase, navient, truist). Les mots ambigus y prennent leur sens général : forbearance → repentance, magnanimity ; chime → bells, peal ; charge → criminal, guilty ; collection → artwork, catalogue ; statement → official, announcement. Exception : default (debt, borrower), le sens financier étant courant dans la presse. Des données du domaine (10 M de mots) battent un corpus général 600 fois plus grand : argument pour le DAPT du L5.

**Modèle de langue** : trigramme à lissage Kneser-Ney interpolé (remise d = 0,75, implémenté à la main ; somme des probabilités vérifiée à 1), un modèle par produit, chacun appris sur 3 600 récits (`text_full`, mots outils gardés), vocabulaire commun de 16 310 mots (vus au moins 3 fois, les autres → `<unk>`), 8,1 M de tokens au total, évaluation sur 200 récits par produit de la val.

**Perplexité** (ligne = modèle, colonne = produit des récits lus ; la plus basse de chaque colonne est en gras)

| Modèle \ récits | checking | credit_card | credit_rep. | debt_coll. | money_tr. | mortgage | payday | prepaid | student | vehicle |
|---|---|---|---|---|---|---|---|---|---|---|
| checking_savings | **98,0** | 112,9 | 257,7 | 205,9 | 116,5 | 189,6 | 149,9 | 99,2 | 196,4 | 159,5 |
| credit_card | 121,2 | **91,2** | 134,0 | 152,4 | 140,6 | 186,0 | 132,9 | 99,1 | 173,7 | 135,1 |
| credit_reporting | 203,4 | 157,5 | **29,8** | 104,5 | 231,5 | 226,1 | 176,1 | 190,2 | 193,0 | 160,6 |
| debt_collection | 179,9 | 144,7 | 78,1 | **83,9** | 198,9 | 191,2 | 152,0 | 161,0 | 183,4 | 137,7 |
| money_transfer | 108,9 | 126,6 | 298,4 | 214,0 | **94,4** | 202,6 | 161,3 | 102,0 | 216,6 | 171,1 |
| mortgage | 150,8 | 140,1 | 234,7 | 169,7 | 171,1 | **106,5** | 135,3 | 160,1 | 150,2 | 135,1 |
| payday_personal_loan | 141,7 | 115,6 | 123,8 | 141,7 | 156,7 | 158,6 | **109,0** | 127,3 | 156,6 | 110,9 |
| prepaid_card | 124,9 | 127,0 | 280,2 | 255,0 | 133,0 | 238,1 | 178,4 | **73,2** | 229,8 | 187,3 |
| student_loan | 181,1 | 151,0 | 197,8 | 173,1 | 206,6 | 180,0 | 145,2 | 170,5 | **91,6** | 157,8 |
| vehicle_loan | 152,4 | 126,5 | 98,8 | 128,2 | 175,7 | 162,8 | 123,3 | 142,2 | 160,5 | **84,8** |

- Chaque modèle est le moins surpris par son propre produit (10 colonnes sur 10).
- `credit_reporting` est extrêmement prévisible (29,8, contre 73 à 109 pour les autres produits) : signe des lettres types et des quasi-doublons.
- **Familles de produits** (perplexités croisées basses) : checking_savings, money_transfer et prepaid_card ; credit_reporting et debt_collection ; payday_personal_loan et vehicle_loan. Ce sont les mêmes que les confusions du L2.
- `payday_personal_loan` est la classe la moins distinctive (109,0 pour son modèle, contre 110,9 pour celui de vehicle_loan).
- **Classification par perplexité minimale** : accuracy 0,707, macro-F1 0,704 sur un échantillon équilibré (non comparable directement au L2).
- **Génération** (amorce « i called the ») : vocabulaire propre à chaque produit (loss mitigation, promissory note, loan forgiveness, cash app) et cohérence locale, mais aucune cohérence au-delà de 2 mots. C'est la limite qui motive les réseaux récurrents (L4) et l'attention (L5).

## 8. Décisions prises (à ne pas remettre en cause sans raison)

### L0
1. **Le texte brut est conservé tel quel** dans `narrative` (seuls les espaces de début et de fin sont retirés).
2. **Taxonomie produit alignée sur la version 2023**, puisque le modèle classera des plaintes futures. Les renommages sont regroupés ; les fusions 2017–2023 sont redécoupées via `sub_product` (`Credit card or prepaid card` → `credit_card` / `prepaid_card` ; `Consumer Loan` → `vehicle_loan` / `payday_personal_loan`).
3. **Classes exclues (`__DROP__`, 5 898 récits)** : `Debt or credit management` (n'existe que depuis 2023, donc absente du train) et `Other financial service` (292 récits, sans équivalent). Aucun cas `__REVIEW__`.
4. **Doublons** : marqués, jamais supprimés dans les fichiers intermédiaires. Un récit de moins de 10 vrais mots n'est jamais considéré comme un doublon. Pour les splits, on garde **le récit le plus ancien** de chaque groupe.
5. **Split temporel** : train ≤ 2023, val 2024, test 2025. **2026 est mis à part** (`drift_2026`) et s'évalue classe par classe, à cause de la rupture de publication du credit reporting.
6. **Train rééquilibré, val et test naturels.** Le train utilise un échantillonnage tempéré (part de la classe *c* ∝ n_c^α, **α = 0,5**). Val et test gardent la distribution réelle. α est un hyperparamètre à régler **sur la val**, jamais sur le test.
7. **Tailles** : train 100 000, val 50 000, test 100 000 (la plus petite classe a ≥ 460 exemples en val et ≥ 690 en test).
8. **Tirage reproductible** : clé = hash de `complaint_id` avec la graine 42.

### L1
9. **Tout ce qui apprend sur les données apprend sur le train** (vocabulaires, BPE, seuils, embeddings, IDF). Val et test servent seulement à mesurer.
10. **Deux niveaux de nettoyage** : `light` pour les transformers (casse, ponctuation et nombres gardés), `full` pour TF-IDF (minuscules, `[NUM]`, `usc_XXXX`, ponctuation répétée réduite).
11. **Tokens spéciaux** : `[PII]`, `[DATE]`, `[MONEY]`, `[URL]`, `[EMAIL]`, `[NUM]`. Le masque s'appelle `[PII]` et **non `[MASK]`**, qui est un token réservé de BERT.
12. **Une rafale de masques = un seul `[PII]`.** **`[DATE]` sans l'année**, pour que le modèle ne devine pas l'époque du récit (le split est temporel).
13. **Références au Code américain unifiées** en niveau `full` : `15 U.S. Code 1681c2`, `15 USC 1681`… → `usc_1681`. Les numéros d'articles portent du sens (1681 = FCRA, 1692 = FDCPA).
14. **Les tokens spéciaux sont déclarés à chaque tokenizer** (spaCy : `add_special_case` ; Hugging Face : `additional_special_tokens`, avec `lstrip=True` pour GPT-2), pour ne jamais être découpés.
15. **Négations gardées** (`not`, `no`, `never`, `n't`…) lors du retrait des mots vides.
16. **Tokenizers retenus** : spaCy (par mots, niveau `full`, après `linguistic.py`) pour TF-IDF et les embeddings ; le tokenizer de chaque transformer, avec nos tokens spéciaux, pour le L5 ; notre **BPE 30k** pour les modèles entraînés depuis zéro (L4).

### L2
17. **Métrique principale : macro-F1.** L'accuracy n'est donnée qu'en complément (le modèle « classe majoritaire » atteint 0,606).
18. **Sélection des modèles sur la val 2024 plutôt que par validation croisée stratifiée** (prévue au cahier des charges) : avec un split temporel, la validation croisée mélangerait passé et futur. **Le test 2025 n'est utilisé qu'une fois**, à l'évaluation finale.
19. **Entrée = récit seul.** `sub_product`, `issue` et `sub_issue` sont exclus (fuite de cible), ainsi que `company` (raccourci et biais) et les métadonnées (état, date, canal).
20. **Modèle retenu** : SVM linéaire sur TF-IDF mots 1-2, C = 0,16, poids de classe équivalents à α = 0,75. Les n-grammes de caractères sont écartés (aucun gain, coût élevé, coefficients illisibles).
21. **α se règle par poids de classe**, sans retirer les splits : poids de la classe *c* ∝ n_c^(α − 0,5), normalisés.
22. **Modèle sauvegardé** dans l'Output Kaggle et le dataset `complaintlens-l2-cache` (`svm_tfidf_v1.joblib` : vectoriseur sans `stop_words_` + SVM + paramètres + classes). Jamais sur Git.

### L3
23. **Embeddings sur `text_tfidf`** (lemmes, sans mots vides) : voisins plus informatifs, et comparaison directe avec le TF-IDF du L2 sur le même texte.
24. **Word2Vec skip-gram 100d est l'embedding de référence pour le L4** (meilleur sur le sens). FastText est conservé pour les mots inconnus et les fautes de frappe.
25. **Le modèle de langue utilise `text_full`** (mots outils gardés, nécessaires à la grammaire), avec **la même quantité de texte par produit** (3 600 récits) pour que les perplexités soient comparables.
26. **Kneser-Ney implémenté à la main** (≈ 30 lignes, vérifié par la somme des probabilités) : l'implémentation NLTK est trop lente à cette échelle.

## 9. Points d'attention

- **Rupture `credit_reporting` 2026 (expliquée)** : artefact de publication, pas une dérive réelle. `drift_2026` s'évalue classe par classe.
- **Campagne de janvier 2025 (à traiter)** : récits à textes variants, invisibles pour la dédup exacte. **Le MinHash devient prioritaire.**
- **Quasi-doublons non traités** : des variantes proches d'une même lettre type peuvent se trouver à la fois en train et en test. L'analyse d'erreurs L2 en a trouvé plusieurs, et la perplexité de `credit_reporting` (29,8) le confirme.
- **Écart de distribution voulu** : 25 % de credit reporting en train, 56 % en test. Traité au L2 par des poids de classe (α = 0,75). La correction des logits n'a pas été testée.
- **Plafond lié aux étiquettes** : 75 % des erreurs analysées viennent de récits ambigus ou mal étiquetés. L'objectif de 0,85 en macro-F1 sur Product sera probablement hors d'atteinte avec ce découpage, quel que soit le modèle. À confirmer au L5.
- **Dépendance aux noms d'entreprises** : environ 2 points de macro-F1 (borne basse), jusqu'à 4,4 pour `student_loan`. À surveiller sur `drift_2026` (gestionnaires de prêts étudiants qui changent) et au L10 (équité).
- **Artefacts d'époque** : `unemployment` et `edd` tirent vers `prepaid_card` (cartes d'allocations chômage de la période Covid). Risque de dérive sur 2025-2026.
- **Comparaison L2 non équitable** : seul le SVM a été réglé.
- **Tokens collés** : ponctuation collée aux mots (`/equifax`, `vehicle-`, `fraudulent/`) et mots sans espace (`ireceive`, `balanceowe`, `originalcreditor`), visibles dans les voisins Word2Vec. Impact faible ; à corriger dans `clean.py`.
- **Textes longs** : environ 10 % des récits dépassent 512 tokens. Stratégie (troncature, découpage, modèle long) à choisir au L5.
- **Hapax et mots trop fréquents** : gérés au L2 par `min_df = 5` et `max_df = 0,5`, et au L3 par `min_count = 5` et le sous-échantillonnage.
- **Notre BPE n'est pas compatible avec les modèles pré-entraînés** (BERT, GPT-2) : leurs poids dépendent de leur propre vocabulaire. L'écart de vocabulaire mesuré, et le décalage de domaine de GloVe, justifient le DAPT au L5.
- **spaCy sur texte en minuscules** (niveau `full`) : étiquetage grammatical un peu moins précis.
- **Bruit d'étiquette** : les étiquettes sont choisies par le consommateur ; certains groupes de copies identiques portent plusieurs produits.

## 10. Reste à faire

### Points ouverts du L0
- [x] Investiguer l'anomalie `credit_reporting` 2026
- [ ] Quasi-doublons MinHash/LSH (`datasketch`) — **prioritaire** (campagne de janvier 2025, lettres types vues en L2 et L3) ; réutilisable en L7
- [ ] Harmoniser la taxonomie `issue` / `sub_issue` (`configs/issue_mapping.csv`) — **indispensable avant la cible Issue**
- [ ] Reproductibilité : `dvc init`, `dvc.yaml`, `Makefile`, script de vérification des SHA-256
- [ ] Hygiène : ruff, pre-commit
- [ ] Vérifier la licence de réutilisation des données CFPB

### L1
- [x] Anomalie 2026 expliquée
- [x] `clean.py` (20 tests)
- [x] Comparaison des 4 tokenizers
- [x] BPE entraîné sur les plaintes (30k et 50k, dans l'Output Kaggle du notebook L1)
- [x] `linguistic.py` (9 tests)
- [x] Analyse linguistique : Zipf, Heaps, collocations, lisibilité
- [ ] Séparer la ponctuation collée et signaler les mots collés dans `clean.py`

### L2
- [x] Prétraitement unique des 250 000 récits (`prepare_text.py`, dataset `complaintlens-features`)
- [x] TF-IDF (mots + caractères) avec Naive Bayes, régression logistique, SVM linéaire, LightGBM (et Random Forest)
- [x] Réglage Optuna du SVM sur la val
- [x] Effet de α et ablations du prétraitement
- [x] Bootstrap apparié, termes discriminants, matrice de confusion, analyse de 100 erreurs
- [x] Ablation par masquage des noms d'entreprises
- [ ] Tests unitaires de `prepare_text.py`
- [ ] Cible 2 : Issue (après `issue_mapping.csv`)
- [ ] Cible 3 : Company response
- [ ] Évaluation finale sur le test 2025 (une seule fois, avec les modèles du L5)

### L3
- [x] Word2Vec et FastText entraînés sur le corpus
- [x] Évaluation intrinsèque : analogies métier construites à la main, intrus, voisins
- [x] Évaluation extrinsèque : moyenne d'embeddings + régression logistique, comparée au TF-IDF du L2
- [x] Comparaison avec GloVe pré-entraîné : décalage de domaine
- [x] Modèle n-gramme Kneser-Ney : perplexité par produit, classification, génération, limites

### L4 — Deep learning from scratch (prochain lot)
- [ ] TextCNN et BiLSTM + attention en PyTorch pur
- [ ] Couches d'embedding initialisées aléatoirement contre Word2Vec du L3
- [ ] Visualisation des poids d'attention sur des exemples
- [ ] Sentiment : lexique VADER contre modèle transféré ; corrélation avec « Company response » et « Timely response »
- [ ] ABSA : sentiment par aspect (frais, service client, délai, fraude)

### Ensuite
- **L5 à L10** et MLOps : voir le cahier des charges.

## 11. Arborescence

```
ComplaintLens/
├── README.md                          # ce document
├── requirements.txt
├── Cahier des charges — ComplaintLens.pdf
├── 0_import_data.ipynb                # exploration L0 (hors pipeline)
├── configs/
│   └── product_mapping.csv            # mapping produit ancien → nouveau
├── src/complaintlens/
│   ├── ingestion/                     # L0
│   │   ├── zip_to_parquet.py
│   │   ├── audit_l0.py
│   │   ├── filter_narratives.py
│   │   ├── apply_product_mapping.py
│   │   ├── find_duplicates.py
│   │   └── make_splits.py
│   ├── preprocessing/                 # L1
│   │   ├── __init__.py
│   │   ├── clean.py
│   │   └── linguistic.py
│   └── features/                      # L2
│       ├── __init__.py
│       └── prepare_text.py
├── tests/
│   ├── conftest.py
│   ├── test_ingestion_units.py        # 20 tests
│   ├── test_splits_data.py            # 9 tests
│   ├── test_clean.py                  # 20 tests
│   └── test_linguistic.py             # 9 tests
├── notebooks/
│   ├── complaintlens-l1.ipynb         # analyse du L1 (Kaggle)
│   ├── complaintlens-l2-models.ipynb  # baselines L2 (Kaggle)
│   └── complaintlens-l3.ipynb         # embeddings et n-grammes L3 (Kaggle)
├── docs/
│   └── datasheet.md                   # fiche du corpus (Gebru et al.)
├── reports/
│   ├── l0/
│   │   ├── splits_summary.json
│   │   └── splits_distribution.csv
│   ├── l1/
│   │   ├── cleaning_inventory.csv
│   │   ├── tokenizers_comparison.csv
│   │   ├── bpe_comparison.csv
│   │   ├── linguistic_comparison.csv
│   │   ├── zipf_top1000.csv
│   │   ├── heaps_curve.csv
│   │   ├── collocations_pmi.csv
│   │   ├── readability.csv
│   │   ├── zipf.png
│   │   ├── heaps.png
│   │   └── readability.png
│   ├── l2/
│   │   ├── results_val.csv            # tous les modèles
│   │   ├── f1_per_class_val.csv
│   │   ├── optuna_svm_trials.csv
│   │   ├── best_svm_params.json
│   │   ├── alpha_effect.csv
│   │   ├── alpha_effect.png
│   │   ├── top_features_svm.csv
│   │   ├── confusion_svm.png
│   │   ├── ablation_preprocessing.csv
│   │   ├── ablation_company_mask.csv
│   │   └── errors_sample_annotated.csv
│   └── l3/
│       ├── analogies_detail.csv       # 3CosAdd et 3CosMul, Word2Vec et FastText
│       ├── analogies_summary.csv
│       ├── odd_one_out.csv
│       ├── extrinsic_comparison.csv
│       ├── extrinsic_per_class.csv
│       ├── analogies_glove_vs_w2v.csv
│       ├── odd_one_out_glove_vs_w2v.csv
│       ├── domain_shift_neighbors.csv
│       ├── ngram_perplexity_matrix.csv
│       └── ngram_generations.csv
└── data/                              # hors Git
    ├── raw/{ccdb_exports,ccdb_current}/, manifest.json
    ├── interim/{parts/, narratives*.parquet, audit/}
    ├── processed/{train,val,test,drift_2026}.parquet
    └── features/{train,val,test}_text.parquet
```