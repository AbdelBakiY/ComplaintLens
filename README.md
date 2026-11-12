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

**Statut : L0 (ingestion et data engineering) terminé.** Prochain lot : L1 (prétraitement et analyse linguistique).

## 2. Environnement et workflow

| Où | Quoi |
|---|---|
| **PC** (Windows, VS Code, Python 3.13) | Écriture du code, tests unitaires, `git push` |
| **Kaggle** (notebooks, ~30 Go de RAM, GPU 30 h/semaine) | `git pull`, exécution sur les vraies données, création des datasets |
| **GitHub** | Code, tests, configs, docs, rapports. **Jamais de données.** |

- Données de référence : datasets Kaggle privés `complaintlens-data` (`narratives_dedup.parquet`) et `complaintlens-splits` (les 4 splits).
- Traitement en **Polars** (2.1.0, version figée sur Kaggle) : lecture lazy, traitement par tranches, jamais la colonne `narrative` entière en mémoire sur le PC.
- `data/` est dans le `.gitignore`. DVC est dans les dépendances mais n'est pas encore initialisé.
- Tous les scripts se lancent **depuis la racine du projet** (chemins relatifs).

## 3. Pipeline L0

| # | Étape | Script | Sortie |
|---|---|---|---|
| 0 | Téléchargement des 21 ZIP + base courante, checksums SHA-256 | manuel | `data/raw/ccdb_exports/`, `data/raw/ccdb_current/`, `data/raw/manifest.json` |
| 1 | ZIP → Parquet (un fichier par export, tout en texte, colonnes en snake_case, `source_export`) | `zip_to_parquet.py` | `data/interim/parts/export_XX.parquet` |
| 2 | Audit des données brutes | `audit_l0.py` | `data/interim/audit/0*.csv` |
| 3 | Filtrage : récit non vide, au moins 1 vrai mot ; typage ; `n_chars`, `n_words`, `n_masks`, `year` | `filter_narratives.py` | `data/interim/narratives.parquet` |
| 4 | Harmonisation des produits (21 libellés → 10 classes) | `apply_product_mapping.py` + `configs/product_mapping.csv` | `data/interim/narratives_mapped.parquet` |
| 5 | Doublons exacts : hash 64 bits du texte normalisé, marquage sans suppression | `find_duplicates.py` | `data/interim/narratives_dedup.parquet` |
| 6 | Dédoublonnage, split temporel, échantillonnage | `make_splits.py` | `data/processed/{train,val,test,drift_2026}.parquet` |

Tous les scripts sont dans `src/complaintlens/ingestion/`.

```bash
python src/complaintlens/ingestion/zip_to_parquet.py
python src/complaintlens/ingestion/audit_l0.py
python src/complaintlens/ingestion/filter_narratives.py
python src/complaintlens/ingestion/apply_product_mapping.py
python src/complaintlens/ingestion/find_duplicates.py --min-words 10
python src/complaintlens/ingestion/make_splits.py --alpha 0.5 --n-val 50000 --n-test 100000 --seed 42
pytest -v
```

**Tests** : 29 tests (`tests/`). 20 tests unitaires (normalisation, échantillonnage tempéré, dédoublonnage, tirage) tournent partout. 9 tests d'intégrité des splits tournent là où `data/processed/` existe (Kaggle). Dernière exécution : **29 passed**.

## 4. Chiffres clés

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

## 5. Décisions prises (à ne pas remettre en cause sans raison)

1. **Le texte brut est conservé tel quel** dans `narrative` (seuls les espaces de début et de fin sont retirés). La normalisation de `find_duplicates.py` ne sert qu'à comparer les textes, jamais à entraîner un modèle.
2. **Taxonomie produit alignée sur la version 2023**, puisque le modèle classera des plaintes futures. Les renommages sont regroupés ; les fusions 2017–2023 sont redécoupées via `sub_product` (`Credit card or prepaid card` → `credit_card` / `prepaid_card` ; `Consumer Loan` → `vehicle_loan` / `payday_personal_loan`).
3. **Classes exclues (`__DROP__`, 5 898 récits)** : `Debt or credit management` (n'existe que depuis 2023, donc absente du train) et `Other financial service` (292 récits, sans équivalent). Aucun cas `__REVIEW__`.
4. **Doublons** : marqués, jamais supprimés dans les fichiers intermédiaires. Un récit de moins de 10 vrais mots n'est jamais considéré comme un doublon. Pour les splits, on garde **le récit le plus ancien** de chaque groupe.
5. **Split temporel** : train ≤ 2023, val 2024, test 2025. **2026 est mis à part** (`drift_2026`) : effondrement des récits et quasi-disparition du credit reporting, donc non représentatif comme test.
6. **Train rééquilibré, val et test naturels.** Le train utilise un échantillonnage tempéré (part de la classe *c* ∝ n_c^α, **α = 0,5**). Val et test gardent la distribution réelle, pour mesurer honnêtement. α est un hyperparamètre à régler **sur la val**, jamais sur le test.
7. **Tailles** : train 100 000, val 50 000, test 100 000 (la plus petite classe a ≥ 460 exemples en val et ≥ 690 en test).
8. **Tirage reproductible** : clé = hash de `complaint_id` avec la graine 42. Les splits ne dépendent pas de l'ordre des lignes, et agrandir val ou test garde les mêmes récits.

## 6. Points d'attention

- **Rupture `credit_reporting` 2026 (expliquée)** : au 1er janvier 2026, la publication des récits de credit reporting s'arrête quasiment (÷ 45 en un mois, toutes entreprises confondues), alors que les autres produits continuent. Artefact de publication, pas une dérive réelle : `drift_2026` s'évalue classe par classe. Détail dans la datasheet §5.
- ⚠️ **Campagne de janvier 2025** : pic massif de récits à textes variants en `money_transfer`, `checking_savings` et `student_loan`, invisible pour la dédup exacte. Le MinHash devient prioritaire.
- **Quasi-doublons non traités** : la dédup est exacte. Des variantes proches d'une même lettre type peuvent encore se trouver à la fois en train et en test.
- **Écart de distribution voulu** : 25 % de credit reporting en train, 56 % en test. Le modèle tendra à sous-prédire cette classe : à étudier en L2 (comparaison de α, correction des logits).
- **Bruit d'étiquette** : les étiquettes sont choisies par le consommateur ; certains groupes de copies identiques portent plusieurs produits.

## 7. Reste à faire

### Clôture de L0 (non bloquant pour L1)
- [x] Dédoublonnage définitif et splits (`make_splits.py`)
- [x] Tests (29) et datasheet (`docs/datasheet.md`)
- [x] Investiguer l'anomalie `credit_reporting` 2026
- [ ] Harmoniser la taxonomie `issue` / `sub_issue` (`configs/issue_mapping.csv`) — **indispensable avant la tâche 2**
- [ ] Quasi-doublons MinHash/LSH (`datasketch`), réutilisable en L7
- [ ] Reproductibilité : `dvc init`, `dvc.yaml`, `Makefile`, script de vérification des SHA-256
- [ ] Hygiène : ruff, pre-commit
- [ ] Vérifier la licence de réutilisation des données CFPB

### Ensuite
- **L1** : pipeline spaCy, token spécial pour les masques `XXXX`, comparaison de 4 tokenizers, EDA linguistique (Zipf, Heaps, PMI, lisibilité), BPE entraîné sur le corpus.
- **L2** : baselines TF-IDF (Naive Bayes, régression logistique, SVM, LightGBM) avec Optuna ; comparaison de α ∈ {0 ; 0,5 ; 1} sur la val.
- **L3 à L10** et MLOps : voir le cahier des charges.

## 8. Arborescence

```
ComplaintLens/
├── README.md                          # ce document
├── requirements.txt
├── Cahier des charges — ComplaintLens.pdf
├── 0_import_data.ipynb                # exploration (hors pipeline)
├── configs/
│   └── product_mapping.csv            # mapping produit ancien → nouveau
├── src/complaintlens/ingestion/
│   ├── zip_to_parquet.py              # étape 1
│   ├── audit_l0.py                    # étape 2
│   ├── filter_narratives.py           # étape 3
│   ├── apply_product_mapping.py       # étape 4
│   ├── find_duplicates.py             # étape 5
│   └── make_splits.py                 # étape 6
├── tests/
│   ├── conftest.py
│   ├── test_ingestion_units.py        # 20 tests unitaires
│   └── test_splits_data.py            # 9 tests d'intégrité des splits
├── docs/
│   └── datasheet.md                   # fiche du corpus (Gebru et al.)
├── reports/l0/
│   ├── splits_summary.json
│   └── splits_distribution.csv
└── data/                              # hors Git
    ├── raw/{ccdb_exports,ccdb_current}/, manifest.json
    ├── interim/{parts/, narratives*.parquet, audit/}
    └── processed/{train,val,test,drift_2026}.parquet
```