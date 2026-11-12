# Datasheet — Corpus ComplaintLens v1

> Format : *Datasheets for Datasets* (Gebru et al., 2021).
> Version du corpus : **v1 — 12/11/2026** · Graine : **42** · Code : `src/complaintlens/ingestion/`
> Auteur : **Abdel** — projet personnel.

---

## 1. Motivation

**Pourquoi ce corpus a-t-il été créé ?**
Pour construire et évaluer ComplaintLens, une chaîne NLP de bout en bout sur des réclamations clients du secteur financier : classification du produit et du motif, extraction d'entités, détection de thèmes, résumé et assistant RAG réglementaire. Le corpus sert de base commune à tous les lots du projet (L1 à L10).

**Quelles tâches vise-t-il ?**
- Tâche 1 : classifier le produit concerné (10 classes) à partir du seul récit.
- Tâche 2 : classifier le motif (`issue`), de manière hiérarchique. ⚠️ Taxonomie `issue` pas encore harmonisée, voir §5.
- Tâche 3 : prédire la réponse de l'entreprise (`company_response_to_consumer`).
- Tâches non supervisées : thèmes, dérive temporelle, détection de campagnes de lettres types.

**Qui l'a créé ?**
Abdel, dans le cadre d'un projet personnel. Les données sources sont publiées par le CFPB.

**Financement :** aucun.

---

## 2. Composition

**Que représente une instance ?**
Une plainte de consommateur déposée auprès du Consumer Financial Protection Bureau (CFPB, régulateur financier américain) contre une entreprise financière. Elle contient un récit libre (`narrative`) et des champs structurés.

**Volumes**

| Niveau | Plaintes |
|---|---|
| Plaintes brutes (21 exports CFPB) | 17 533 918 |
| Avec un récit non vide | 3 851 415 |
| Avec au moins un vrai mot | 3 850 263 |
| Hors classes exclues (`__DROP__`) | 3 844 365 |
| Après dédoublonnage exact (1 récit par groupe) | 2 316 381 |
| **Sous-corpus final (4 splits)** | **356 310** |

**Période :** récits du **19/03/2015 au 11/08/2026**. Aucune plainte antérieure à 2015 n'a de récit publié.

**Champs**

| Famille | Colonnes | Rôle |
|---|---|---|
| Texte | `narrative` | Entrée des modèles |
| Étiquettes (choisies par le consommateur) | `product`, `product_raw`, `sub_product`, `issue`, `sub_issue` | Cibles |
| Entreprise | `company`, `company_response_to_consumer`, `company_public_response`, `timely_response` | Cible n°3, tests de robustesse |
| Consommateur (anonymisé) | `state`, `zip_code`, `tags`, `submitted_via` | Analyse de biais |
| Identité et dates | `complaint_id`, `date_received`, `year`, `date_sent_to_company` | Identifiant, split temporel |
| Ajoutés par le pipeline | `n_chars`, `n_words`, `n_masks`, `source_export`, `text_hash`, `dup_group`, `dup_count`, `split` | Traçabilité, filtrage, dédoublonnage |

**Les 10 classes de produit**, après harmonisation (voir §4) :

| Classe | Récits (avant dédup) | Part avant dédup | Récits uniques | Part après dédup |
|---|---|---|---|---|
| credit_reporting | 2 510 397 | 65,3 % | 1 111 141 | 48,0 % |
| debt_collection | 445 229 | 11,6 % | 368 079 | 15,9 % |
| credit_card | 228 184 | 5,9 % | 212 229 | 9,2 % |
| checking_savings | 204 227 | 5,3 % | 199 245 | 8,6 % |
| mortgage | 147 249 | 3,8 % | 146 910 | 6,3 % |
| money_transfer | 123 514 | 3,2 % | 95 006 | 4,1 % |
| student_loan | 63 196 | 1,6 % | 62 640 | 2,7 % |
| vehicle_loan | 59 758 | 1,6 % | 58 894 | 2,5 % |
| payday_personal_loan | 41 188 | 1,1 % | 40 906 | 1,8 % |
| prepaid_card | 21 423 | 0,6 % | 21 331 | 0,9 % |

**Splits livrés** (dans `data/processed/`)

| Split | Période | Récits | Règle d'échantillonnage |
|---|---|---|---|
| train | 2015–2023 | 100 000 | Tempéré par classe : part ∝ n_c^α, α = 0,5 |
| val | 2024 | 50 000 | Aléatoire, distribution naturelle |
| test | 2025 | 100 000 | Aléatoire, distribution naturelle |
| drift_2026 | 2026 | 106 310 | Totalité des récits uniques |

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

**Longueur des récits** (avant dédoublonnage) : médiane 670 caractères, p5 127, p95 3 007, max 35 984. En approximation (1 token ≈ 4 caractères), environ 5 % des récits dépassent 512 tokens.

**Valeurs manquantes** (sur les 3,85 M de récits) :
- `sub_product` : 52 205. Structurel : certains produits n'ont pas de sous-produit.
- `sub_issue` : 337 449. Structurel : certains motifs n'ont pas de sous-motif.
- `company_public_response` : 1 807 070. Réponse publique facultative.
- `tags` : 3 481 435. **Ce n'est pas une donnée manquante** : un champ vide signifie « aucun tag ».
- `state` : 13 816 · `zip_code` : 377 · `company_response_to_consumer` : 9.

**Données personnelles**
Le CFPB masque les informations identifiantes avant publication (séquences `XXXX`, `XX/XX/2019`) et met les montants entre accolades (`{$1000.00}`). Le masquage est **imparfait** : des noms d'entreprises et parfois de personnes restent visibles. ⚠️ Le taux de fuite résiduelle n'est pas encore mesuré (prévu au §7 du cahier des charges, avec Presidio).

**Le corpus contient-il des informations sensibles ?**
Oui, potentiellement : des situations financières difficiles (dettes, faillites, saisies), des mentions d'âge (`tags` = Older American) et de statut militaire (`tags` = Servicemember), et des États de résidence.

---

## 3. Collecte

**Source :** archive FOIA du CFPB, *Consumer Complaint Database Narratives Archive* : 21 fichiers ZIP couvrant les plaintes reçues du 01/12/2011 au 14/08/2026.
https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/

**Pourquoi cette archive et pas la base courante ?**
Le 14/08/2026, le CFPB a cessé de publier les récits dans sa base publique. La base courante (`complaints.csv.zip`) ne contient plus de texte. Elle n'a servi qu'à des contrôles de cohérence. Le corpus est donc **figé** : il ne recevra plus de nouveaux récits.

**Comment les plaintes sont-elles collectées par le CFPB ?**
Le consommateur remplit un formulaire : il choisit le produit et le motif, puis rédige un récit. Le CFPB transmet la plainte à l'entreprise, qui répond. La plainte est publiée après la réponse de l'entreprise, ou au bout de 15 jours. **Le récit n'est publié que si le consommateur y consent**, après retrait des informations personnelles par le CFPB.

**Intégrité :** les sommes SHA-256 des 22 ZIP téléchargés sont enregistrées dans `data/raw/manifest.json`.

**Licence :** ⚠️ À VÉRIFIER. Données publiées par une agence fédérale américaine. Confirmer les conditions de réutilisation sur le site du CFPB avant toute diffusion du corpus dérivé.

---

## 4. Prétraitement

Chaque étape lit un fichier et en écrit un nouveau ; aucun fichier n'est modifié sur place.

| # | Étape | Script | Effet |
|---|---|---|---|
| 1 | ZIP → Parquet | `zip_to_parquet.py` | 21 Parquet, colonnes en snake_case, ajout de `source_export`. Schéma identique dans les 21 exports. |
| 2 | Audit | `audit_l0.py` | Contrôles : 0 date illisible, 0 doublon d'identifiant entre exports. |
| 3 | Filtrage | `filter_narratives.py` | Récit non vide et au moins 1 vrai mot (≥ 2 lettres hors masques) : 1 152 récits retirés. Typage des colonnes, ajout de `n_chars`, `n_words`, `n_masks`, `year`. |
| 4 | Harmonisation des produits | `apply_product_mapping.py` + `configs/product_mapping.csv` | 21 libellés → 10 classes (détail ci-dessous). |
| 5 | Doublons | `find_duplicates.py` | Marquage des copies exactes (détail ci-dessous). |
| 6 | Splits | `make_splits.py` | Dédoublonnage, split temporel, échantillonnage. |

**Le texte n'est jamais modifié** : seuls les espaces de début et de fin sont retirés. Minuscules, masques, ponctuation : tout reste tel quel. Le nettoyage linguistique relève du lot L1.

**Harmonisation des produits**
Le CFPB a renommé ses catégories en 2017 et en 2023. Toutes les étiquettes sont alignées sur la **taxonomie de 2023**, puisque le modèle servira à classer des plaintes futures.
- Les simples renommages sont regroupés (3 variantes de *Credit reporting* → `credit_reporting`, par exemple).
- Les catégories fusionnées entre 2017 et 2023 sont redécoupées grâce à `sub_product` : *Credit card or prepaid card* est réparti entre `credit_card` et `prepaid_card` ; *Consumer Loan* entre `vehicle_loan` et `payday_personal_loan`.
- Deux classes sont exclues (`__DROP__`, 5 898 récits) : *Debt or credit management* (5 606), qui n'existe que depuis 2023 et serait donc absente du train ; *Other financial service* (292), sans équivalent actuel.
- Aucune ligne n'est restée sans correspondance (0 cas `__REVIEW__`).

**Doublons exacts**
- Chaque récit est normalisé (minuscules, sans masques, sans chiffres ni ponctuation, espaces réduits), puis réduit à un hash 64 bits (graine 42).
- Les récits de moins de 10 vrais mots ne sont jamais considérés comme des doublons : 26 747 récits courts identiques ont été ignorés.
- Résultat : 262 765 groupes rassemblent 1 790 903 récits (46,5 %), soit 1 528 138 copies en trop (39,7 %). Le plus gros groupe compte 31 053 récits identiques.
- Ce sont surtout des **lettres types** déposées en masse (sociétés de « credit repair ») : 65,5 % des récits `credit_reporting` sont des copies, contre 0,3 % pour `mortgage`. La part de copies passe de 3,6 % en 2015 à 62,6 % en 2025.
- **Aucune ligne n'est supprimée** dans le fichier intermédiaire : les copies sont marquées (`dup_group`, `dup_count`).

**Construction des splits**
1. On garde **un récit par groupe de doublons** (le plus ancien) et tous les récits uniques. Une même lettre ne peut donc pas apparaître dans deux splits.
2. Chaque récit est affecté à un split selon son année.
3. Train : échantillonnage tempéré, avec une part de la classe *c* proportionnelle à n_c^0,5. Le poids du credit reporting passe de 45 % à 25 %, celui de `prepaid_card` de 1 % à 3,6 %.
4. Val et test : tirage aléatoire, **distribution naturelle conservée**, pour mesurer la performance dans les conditions réelles.
5. Le tirage repose sur le hash de `complaint_id` avec la graine 42 : il est entièrement reproductible et ne dépend pas de l'ordre des lignes.
6. Contrôles validés (29 tests pytest) : aucun identifiant ni groupe de doublons dans deux splits, années conformes, aucun `__DROP__`, tailles conformes.

**Les données brutes sont-elles conservées ?** Oui : les 21 ZIP et tous les fichiers intermédiaires.

---

## 5. Biais, limites et points de vigilance

1. **Étiquettes déclaratives.** Le produit et le motif sont choisis par le consommateur, pas par un expert. Le bruit d'étiquette est réel : certains groupes de copies identiques portent plusieurs produits différents. Aucun modèle ne peut atteindre 100 %.
2. **Allégations non vérifiées.** Un récit est la version du consommateur. Il ne prouve aucune faute de l'entreprise citée.
3. **Biais d'autosélection.** Seuls les consommateurs qui se plaignent, et qui consentent à publier leur récit, sont représentés (environ 22 % des plaintes). Le corpus n'est pas un échantillon représentatif de la clientèle.
4. **Campagnes de lettres types.** Près de la moitié des récits sont des copies. La dédup exacte les traite, mais **pas les quasi-doublons** (lettres modifiées d'une phrase). ⚠️ Des variantes proches peuvent encore apparaître à la fois en train et en test. Une détection MinHash/LSH est prévue.
5. **Changement de distribution entre train et test.** Train rééquilibré (25 % de credit reporting) contre test naturel (56 %). C'est voulu, mais le modèle tendra à sous-prédire le credit reporting.
6. **Dérive temporelle.** La part du credit reporting passe de 55,6 % (2025) à 2,1 % (2026), alors que les autres classes gardent des volumes cohérents. ⚠️ Cause non identifiée : changement de politique de publication propre à ce produit ? À investiguer avant toute conclusion sur `drift_2026`.
7. **Effondrement des récits en 2026.** 2,5 % des plaintes de 2026 ont un récit, et un seul récit en août 2026. C'est la conséquence de l'arrêt de la publication.
8. **Taxonomie `issue` non harmonisée.** Les motifs ont aussi changé en 2017 et 2023. ⚠️ À traiter avant la tâche 2.
9. **Langue.** Récits quasi exclusivement en anglais américain, avec des fautes d'orthographe et des styles très variables. ⚠️ La détection de langue n'a pas été appliquée.
10. **Géographie.** Uniquement des plaintes adressées au régulateur américain : pas de transposition directe à un autre cadre réglementaire.

---

## 6. Usages

**Usages prévus**
- Recherche et apprentissage en NLP : classification, extraction d'information, thèmes, résumé, RAG.
- Étude de la dérive temporelle et des campagnes de plaintes coordonnées.

**Usages déconseillés**
- Présenter une plainte comme un fait établi sur une entreprise nommée.
- Classer, évaluer ou comparer des entreprises à partir de ces récits.
- Prendre une décision automatique sur un client réel sans validation humaine.
- Tenter de réidentifier des consommateurs à partir des récits.

---

## 7. Distribution et maintenance

- **Diffusion :** privée (datasets Kaggle `complaintlens-data` et `complaintlens-splits`). Le code est sur GitHub ; les données ne sont jamais versionnées dans Git.
- **Statut :** corpus **figé**. La source ne publie plus de récits.
- **Régénération complète** (depuis la racine du projet, ZIP présents dans `data/raw/ccdb_exports/`) :

```bash
python src/complaintlens/ingestion/zip_to_parquet.py
python src/complaintlens/ingestion/audit_l0.py
python src/complaintlens/ingestion/filter_narratives.py
python src/complaintlens/ingestion/apply_product_mapping.py
python src/complaintlens/ingestion/find_duplicates.py --min-words 10
python src/complaintlens/ingestion/make_splits.py --alpha 0.5 --n-val 50000 --n-test 100000 --seed 42
pytest -v
```

- **Paramètres de la v1 :** seuil de doublon = 10 mots, graine du hash = 42, α = 0,5, graine des splits = 42, polars 2.1.0.
- **Contact :** Abdel.

| Version | Date | Changement |
|---|---|---|
| v1 | 12/11/2026 | Première version : 10 classes, dédup exacte, splits temporels |