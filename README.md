# ComplaintLens — état d'avancement

> Document de passation : où en est le projet au **12/11/2026**, ce qui est fait, les décisions prises et la suite.
> Spécification complète : [Cahier des charges](Cahier%20des%20charges%20—%20ComplaintLens.pdf).
> Fiche du corpus : [docs/datasheet.md](docs/datasheet.md).
> Auteur : **Abdel** — projet personnel.

## 1. Le projet en bref

Une plateforme NLP de bout en bout sur les **plaintes clients du CFPB** (Consumer Financial Protection Bureau, régulateur financier américain) : classification du produit et du motif, extraction d'entités, thèmes, résumé, RAG réglementaire et application déployée.

- **Corpus** : archive FOIA du CFPB, 21 ZIP (déc. 2011 → août 2026), récits libres inclus. Le corpus est figé : le CFPB a cessé de publier les récits le 14/08/2026.
- **Objectifs** : Product (10 classes) macro-F1 ≥ 0,85 ; Issue (hiérarchique) macro-F1 ≥ 0,60 ; NER (8 types) F1 ≥ 0,75 ; RAG ≥ 80 % de réponses correctement sourcées.
- **Découpage** : 10 lots, de L0 (ingestion) à L10 (explicabilité et équité), plus un lot MLOps transverse.

**Statut :**
- **L0 — Ingestion et data engineering** : terminé.
- **L1 — Prétraitement et analyse linguistique** : terminé.
- Prochain lot : **L2 — Baselines de classification** (TF-IDF).

## 2. Environnement et workflow

| Où | Quoi |
|---|---|
| **PC** (Windows, VS Code, Python 3.13) | Écriture du code, tests unitaires, `git push` |
| **Kaggle** (notebooks, ~30 Go de RAM, GPU 30 h/semaine) | `git pull`, exécution sur les vraies données, création des datasets |
| **GitHub** | Code, tests, configs, docs, rapports, notebooks. **Jamais de données.** |

- Données de référence : datasets Kaggle privés `complaintlens-data` (`narratives_dedup.parquet`) et `complaintlens-splits` (les 4 splits).
- Traitement en **Polars** (2.1.0, version figée sur Kaggle) : lecture lazy, traitement par tranches, jamais la colonne `narrative` entière en mémoire sur le PC.
- **spaCy** avec le modèle `en_core_web_sm` (`python -m spacy download en_core_web_sm`).
- Tokenizers Hugging Face (`transformers`, `tokenizers`) : le notebook doit avoir **Internet : ON** sur Kaggle.
- `data/` est dans le `.gitignore`. DVC est dans les dépendances mais n'est pas encore initialisé.
- Tous les scripts se lancent **depuis la racine du projet** (chemins relatifs).
- **Le CPU suffit pour L0 et L1.** Le GPU de Kaggle est réservé à l'entraînement (L5, L9).

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

Chaîne pour les modèles à base de mots (L2) : `narrative` → `clean_expr(..., "full")` → `process_texts(...)` → TF-IDF.
Chaîne pour les transformers (L5) : `narrative` → `clean_expr(..., "light")` → tokenizer du modèle, avec les `SPECIAL_TOKENS` déclarés.

L'analyse du L1 est dans `notebooks/complaintlens-l1.ipynb` (exécuté sur Kaggle), ses résultats dans `reports/l1/`.

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

## 6. Décisions prises (à ne pas remettre en cause sans raison)

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
9. **Tout ce qui apprend sur les données apprend sur le train** (vocabulaires, BPE, seuils). Val et test servent seulement à mesurer.
10. **Deux niveaux de nettoyage** : `light` pour les transformers (casse, ponctuation et nombres gardés), `full` pour TF-IDF (minuscules, `[NUM]`, `usc_XXXX`, ponctuation répétée réduite).
11. **Tokens spéciaux** : `[PII]`, `[DATE]`, `[MONEY]`, `[URL]`, `[EMAIL]`, `[NUM]`. Le masque s'appelle `[PII]` et **non `[MASK]`**, qui est un token réservé de BERT.
12. **Une rafale de masques = un seul `[PII]`.** **`[DATE]` sans l'année**, pour que le modèle ne devine pas l'époque du récit (le split est temporel).
13. **Références au Code américain unifiées** en niveau `full` : `15 U.S. Code 1681c2`, `15 USC 1681`… → `usc_1681`. Les numéros d'articles portent du sens (1681 = FCRA, 1692 = FDCPA).
14. **Les tokens spéciaux sont déclarés à chaque tokenizer** (spaCy : `add_special_case` ; Hugging Face : `additional_special_tokens`, avec `lstrip=True` pour GPT-2), pour ne jamais être découpés.
15. **Négations gardées** (`not`, `no`, `never`, `n't`…) lors du retrait des mots vides.
16. **Tokenizers retenus** : spaCy (par mots, niveau `full`, après `linguistic.py`) pour TF-IDF ; le tokenizer de chaque transformer, avec nos tokens spéciaux, pour le L5 ; notre **BPE 30k** pour les modèles entraînés depuis zéro (L3, L4).

## 7. Points d'attention

- **Rupture `credit_reporting` 2026 (expliquée)** : artefact de publication, pas une dérive réelle. `drift_2026` s'évalue classe par classe.
- **Campagne de janvier 2025 (à traiter)** : récits à textes variants, invisibles pour la dédup exacte. **Le MinHash devient prioritaire.**
- **Quasi-doublons non traités** : des variantes proches d'une même lettre type peuvent se trouver à la fois en train et en test.
- **Écart de distribution voulu** : 25 % de credit reporting en train, 56 % en test. Le modèle tendra à sous-prédire cette classe : à étudier au L2 (comparaison de α, correction des logits).
- **Textes longs** : environ 10 % des récits dépassent 512 tokens. Stratégie (troncature, découpage, modèle long) à choisir au L5.
- **Mots trop fréquents dans le domaine** (`[PII]`, `account`, `credit`) : à gérer par l'IDF et `max_df` au L2, pas par une liste de mots vides.
- **Hapax** : 45 % du vocabulaire n'apparaît qu'une fois. À filtrer avec `min_df` au L2.
- **Notre BPE n'est pas compatible avec les modèles pré-entraînés** (BERT, GPT-2) : leurs poids dépendent de leur propre vocabulaire. L'écart de vocabulaire mesuré justifie le DAPT au L5.
- **spaCy sur texte en minuscules** (niveau `full`) : étiquetage grammatical un peu moins précis.
- **Bruit d'étiquette** : les étiquettes sont choisies par le consommateur ; certains groupes de copies identiques portent plusieurs produits.

## 8. Reste à faire

### Points ouverts du L0 (non bloquants pour L2)
- [x] Investiguer l'anomalie `credit_reporting` 2026
- [ ] Quasi-doublons MinHash/LSH (`datasketch`) — **prioritaire** depuis la campagne de janvier 2025 ; réutilisable en L7
- [ ] Harmoniser la taxonomie `issue` / `sub_issue` (`configs/issue_mapping.csv`) — **indispensable avant la tâche 2**
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

### L2 — Baselines (prochain lot)
- [ ] Traiter une fois pour toutes les 250 000 récits (train, val, test) avec `clean` + `linguistic`, et enregistrer le résultat en Parquet (≈ 40 min sur Kaggle)
- [ ] TF-IDF (mots + caractères) avec Naive Bayes, régression logistique, SVM linéaire, LightGBM
- [ ] Recherche d'hyperparamètres (Optuna), réglée sur la val
- [ ] Comparaison de α ∈ {0 ; 0,5 ; 1} et ablations du prétraitement (lemmes, mots vides)
- [ ] Macro-F1, intervalles de confiance par bootstrap, analyse des mots les plus discriminants par classe

### Ensuite
- **L3 à L10** et MLOps : voir le cahier des charges.

## 9. Arborescence

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
│   └── preprocessing/                 # L1
│       ├── __init__.py
│       ├── clean.py
│       └── linguistic.py
├── tests/
│   ├── conftest.py
│   ├── test_ingestion_units.py        # 20 tests
│   ├── test_splits_data.py            # 9 tests
│   ├── test_clean.py                  # 20 tests
│   └── test_linguistic.py             # 9 tests
├── notebooks/
│   └── complaintlens-l1.ipynb         # analyse du L1 (Kaggle)
├── docs/
│   └── datasheet.md                   # fiche du corpus (Gebru et al.)
├── reports/
│   ├── l0/
│   │   ├── splits_summary.json
│   │   └── splits_distribution.csv
│   └── l1/
│       ├── cleaning_inventory.csv
│       ├── tokenizers_comparison.csv
│       ├── bpe_comparison.csv
│       ├── linguistic_comparison.csv
│       ├── zipf_top1000.csv
│       ├── heaps_curve.csv
│       ├── collocations_pmi.csv
│       ├── readability.csv
│       ├── zipf.png
│       ├── heaps.png
│       └── readability.png
└── data/                              # hors Git
    ├── raw/{ccdb_exports,ccdb_current}/, manifest.json
    ├── interim/{parts/, narratives*.parquet, audit/}
    └── processed/{train,val,test,drift_2026}.parquet
```