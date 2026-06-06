# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "seaborn==0.13.2",
# ]
# ///
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ANALYSE DE DONNÉES — Séance 1 · Introduction à l'Analyse de Données       ║
║  Institut Polytechnique de Saint-Louis (IPSL) · Version 1                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PLAN                                                                        ║
║  PARTIE 1 — CM (2h)                                                          ║
║    1.1  Qu'est-ce que l'analyse de données ?                                ║
║    1.2  Les 4 stratégies de la data science                                 ║
║    1.3  Domaines d'application                                              ║
║    1.4  Taxonomie des données                                               ║
║    1.5  Sources et formats de données                                       ║
║    1.6  Cycle de vie — CRISP-DM                                             ║
║    1.7  Écosystème Python (pandas, NumPy, matplotlib…)                     ║
║    1.8  Éthique, RGPD et biais                                              ║
║    1.9  Introduction à marimo & uv                                          ║
║  PARTIE 2 — TP (1h)                                                          ║
║    2.1  Chargement des données                                              ║
║    2.2  head, tail, shape                                                   ║
║    2.3  info()                                                              ║
║    2.4  describe() + formules                                               ║
║    2.5  Sélection, filtrage, groupement                                     ║
║    2.6  Visualisations exploratoires                                        ║
║    2.7  Widgets réactifs marimo                                             ║
║    — Jeu de données : Titanic (Kaggle)                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Séance 1 — Introduction à l'Analyse de Données",
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from io import StringIO
    import warnings
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120
    return mo, mpatches, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📊 Séance 1 — Introduction à l'Analyse de Données
    **Institut Polytechnique de Saint-Louis (IPSL) · Année 2025–2026**

    ---

    | | |
    |---|---|
    | **Durée** | 3 heures (CM 2h + TP 1h) |
    | **Outils** | Python 3.11+, pandas, polars, marimo, uv |
    | **Dataset TP** | Titanic (1309 passagers — `datasets/raw/titanic.csv`) |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🎓 PARTIE 1 — Cours Magistral (CM · 2h)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 · Analyse de données vs Science des données

    #### Définitions fondamentales

    > *« Data analytics is the science of analyzing raw data to make conclusions
    > about that information. »*
    > — **Igual, L. & Seguí, S. (2017). Introduction to Data Science . Springer**

    > *« Data science is commonly defined as a methodology by which actionable
    > insights can be inferred from data. »*
    > — **Igual, L. & Seguí, S. (2017). Introduction to Data Science . Springer**

    En pratique, l'analyse de données et la science de données sont deux disciplines distinctes,
    bien qu'elles se chevauchent partiellement et partagent certains outils communs.

    #### Points importants

    | | **Analyse de données** (*Data Analytics*) | **Science des données** (*Data Science*) |
    |---|---|---|
    | **Objectif** | Tirer des conclusions à partir de données existantes | Extraire des connaissances et construire des modèles prédictifs |
    | **Question(s)** | *Que s'est-il passé ? Pourquoi ?* | *Que va-t-il se passer ? Comment l'automatiser ?* |
    | **Horizon** | Court/Moyen terme — décision opérationnelle | Moyen/Long terme — modélisation et mise en production |
    | **Outils typiques** | Tableurs, SQL, Python, R, BI tools, Data Viz | Python, R, scikit-learn, TensorFlow, Spark, MLflow |
    | **Profil** | Data Analyst, Statisticien | Data Scientist, ML Engineer |

    #### Pourquoi l'analyse de données est-elle stratégique ?

    L'analyse de données aide une organisation à :

    - **optimiser ses performances** et ses processus internes
    - **maximiser ses profits** en identifiant les leviers d'action
    - **prendre des décisions guidées par les faits** plutôt que par l'intuition

    Les techniques et processus d'analyse sont aujourd'hui largement **automatisés**
    via des algorithmes travaillant sur des données brutes — c'est précisément
    ce que vous allez apprendre à construire dans ce cours.

    #### Donnée vs Information

    | Concept | Définition | Exemple |
    |---|---|---|
    | **Donnée** (*data*) | Observation brute, objective, sans contexte | `38.0`, `"female"`, `71.28` |
    | **Information** | Donnée interprétée par un humain, avec sens | *« Une femme adulte de 1ʳᵉ classe »* |

    La donnée devient information lorsqu'on lui attribue un **contexte métier**.
    La data science automatise cette transformation à grande échelle.

    #### Pourquoi maintenant ? L'ère du Big Data

    Le succès de **Google, Amazon, Facebook, Apple, Microsoft (GAFAM)** est directement
    attribuable à leur maîtrise de la data science, de l'IA et du Big Data.

    Une organisation qui n'exploite pas ses données perd en compétitivité face à celles qui le font.

    Quatre conditions sont nécessaires à un projet data réussi :

    1. Des **objectifs quantifiables** clairs
    2. Une **méthodologie rigoureuse** (ex. : CRISP-DM)
    3. Une **équipe interdisciplinaire** (voir section 1.2b)
    4. Un **workflow reproductible** (code versionné, environnements figés)

    #### Le continuum analytique — les 4 approches

    | Approche | Question posée | Exemple |
    |---|---|---|
    | **Descriptive** | *Que s'est-il passé ?* | Tableau de bord des ventes mensuelles |
    | **Diagnostique** | *Pourquoi cela s'est-il passé ?* | Analyse des causes d'une chute de CA |
    | **Prédictive** | *Que va-t-il se passer ?* | Prévision du taux de désabonnement |
    | **Prescriptive** | *Que faut-il faire ?* | Optimisation automatique du stock |

    > 💡 **Exemple fil rouge** : Un opérateur télécom analyse son churn (*descriptif*),
    > identifie les segments à risque (*diagnostique*), prédit les résiliations du mois
    > prochain (*prédictif*) et décide d'envoyer une offre ciblée (*prescriptif*).

    #### Outils selon le type d'analyse

    | Niveau | Outils typiques |
    |---|---|
    | Tableurs & reporting | Excel, Google Sheets, Power BI, Tableau |
    | Visualisation | matplotlib, seaborn, Plotly, Grafana |
    | Exploration & wrangling | pandas, SQL, polar, dplyr (R) |
    | Mining & ML | scikit-learn, XGBoost, TensorFlow, PyTorch |
    | Langages open-source | **Python** ✓, R, Julia |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("#### 📌 L'importance de la donnée dans l'organisation"),
        mo.hstack([mo.image(src="images/importance_of_data.png", width=600)], justify="center"),
        mo.md("""
    *La donnée est au cœur de chaque étape de la chaîne de valeur organisationnelle :
    elle alimente la connaissance, éclaire les décisions, déclenche les actions et
    permet d'en mesurer l'impact — créant ainsi une boucle d'amélioration continue.*
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 · Les 4 stratégies pour explorer le monde par les données


    **1. Sonder la réalité (Probing reality)**
    Les données représentent la réponse du monde à nos actions.
    Exemple : l'**A/B testing** — quelle couleur de bouton convertit le mieux ?
    La seule réponse valide vient de l'expérimentation sur des données réelles.

    **2. Découverte de patterns (Pattern discovery)**
    Analyser automatiquement des problèmes pour découvrir des clusters naturels.
    Exemple : **profilage d'utilisateurs** en marketing digital ou publicité programmatique.

    **3. Prédiction d'événements futurs (Predicting future events)**
    Construire des **modèles prédictifs** robustes. Exemple : optimiser les plannings
    du personnel d'un magasin en analysant météo, historique des ventes, trafic.

    **4. Comprendre les personnes et le monde**
    Investissement en NLP, vision par ordinateur, neurosciences.
    Le deep learning pour la compréhension du langage en est l'exemple type.

    📌 **Distinction prédiction vs prévision** :

    > « it will rain tomorrow » = prédiction (un résultat unique) ;

    > « 40% chance of rain » = prévision (distribution de résultats possibles).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2b · Les rôles dans une équipe Data Science

    Un projet data mobilise généralement **6 profils complémentaires**.
    En pratique, les rôles se chevauchent — le « *data scientist unicorn* » qui
    maîtrise tout seul l'ensemble du cycle est exceptionnel.

    | Rôle | Responsabilités principales |
    |---|---|
    | **Chef de projet** | Coordination, planning, communication avec les métiers |
    | **Data Architect** | Conception de l'infrastructure et des pipelines de données |
    | **Data Engineer** | Construction et maintenance des pipelines ETL / ELT |
    | **Data Analyst** | Exploration, dashboards, reporting, aide à la décision |
    | **Data Scientist** | Modélisation ML, expérimentation, feature engineering |
    | **ML Engineer** | Déploiement, monitoring et industrialisation des modèles |

    > ⚡ Dans ce cours, vous allez progressivement couvrir les rôles
    > **Data Analyst** (séances 1–3) et **Data Scientist** (séances 4–7).

    ### 1.2c · Types d'apprentissage automatique (Machine Learning)

    La définition de référence (Tom Mitchell, 1997) :
    > *"A program learns from experience **E** regarding task **T** with performance
    > measure **P**, if its performance on **T** (measured by **P**) improves with experience **E**."*

    #### Apprentissage supervisé vs non-supervisé

    ```
    Machine Learning
    ├── Supervisé (données étiquetées — on connaît la réponse)
    │   ├── Régression    → sortie numérique    ex: prédire un prix, un revenu
    │   └── Classification → sortie catégorielle  ex: spam/non-spam, churn
    │
    └── Non-supervisé (données non étiquetées — on cherche la structure)
        ├── Clustering          → regrouper des individus similaires  ex: segmentation client
        ├── Réduction de dimension → compresser l'espace de features   ex: PCA, UMAP
        └── Règles d'association  → trouver des co-occurrences        ex: panier d'achat
    ```

    #### ML vs Programmation classique

    | Approche | Entrées | Sortie |
    |---|---|---|
    | **Classique** | Données + Règles manuelles | Résultats |
    | **ML** | Données + Exemples (résultats connus) | **Règles apprises** → appliquées à de nouvelles données |

    > 💡 Dans ce cours : séances 5–6 = supervisé (clustering, régression, classification) ;
    > séance 4 = non-supervisé (PCA, UMAP, K-Means).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1.3 · Domaines d'application
    """)
    return


@app.cell(hide_code=True)
def _(mpatches, plt):
    domaines = {
        "Finance\n& Banque": ("Détection de fraude\nScoring crédit\nAlgo-trading", "#4C72B0"),
        "Santé\n& Médecine": ("Diagnostic assisté\nGénomique\nEpidémiologie", "#DD8452"),
        "Industrie\n& IoT": ("Maintenance prédictive\nContrôle qualité\nOptimisation", "#55A868"),
        "Marketing\n& E-commerce": ("Segmentation client\nRecommandation\nChurn", "#C44E52"),
        "Transport\n& Logistique": ("GPS optimisé\nDemande prédictive\nCovoiturage", "#8172B2"),
        "Éducation": ("Apprentissage adaptatif\nDétection décrochage\nEvaluation auto.", "#937860"),
    }
    fig_dom, axes_dom = plt.subplots(2, 3, figsize=(12, 6))
    fig_dom.suptitle("Domaines d'application",
                     fontsize=12, fontweight="bold")
    for _ax, (_titre, (_cas, _color)) in zip(axes_dom.flat, domaines.items()):
        _ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
            linewidth=2, edgecolor=_color, facecolor=_color + "28"))
        _ax.text(0.5, 0.78, _titre, ha="center", va="center", fontsize=11,
                fontweight="bold", color=_color, transform=_ax.transAxes)
        _ax.text(0.5, 0.36, _cas, ha="center", va="center", fontsize=8.5,
                color="#333", transform=_ax.transAxes, linespacing=1.7)
        _ax.set_xlim(0, 1); _ax.set_ylim(0, 1); _ax.axis("off")
    plt.tight_layout()
    fig_dom
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 · Taxonomie des données

    #### Par structure

    Exemple : les **images** sont des tableaux 2D de valeurs de pixels.

    | Type | Définition | Exemples |
    |---|---|---|
    | **Structurées** | Schéma fixe, lignes/colonnes | Tables SQL, CSV, Excel |
    | **Semi-structurées** | Hiérarchie flexible | JSON, XML, HTML |
    | **Non-structurées** | Aucun schéma | Images, vidéos, textes libres |
    | **Multimodales** | Combinaison de types | Réseaux sociaux (texte + image) |

    #### Par nature statistique

    ```
    Variables
    ├── Qualitatives (catégorielles)
    │   ├── Nominales  → sans ordre   ex: couleur des yeux, pays d'origine
    │   └── Ordinales  → avec ordre   ex: niveau d'études (L1 < L2 < M1 < M2)
    │
    └── Quantitatives (numériques)
        ├── Discrètes  → entiers      ex: nombre d'enfants, nombre de clics
        └── Continues  → réels        ex: taille, température, revenu, âge
    ```

    #### Étapes de préparation des données

    1. **Obtention** : lecture depuis un fichier ou scraping web
    2. **Parsing** : selon le format (CSV, XML, HTML, texte brut…)
    3. **Nettoyage** : les jeux de données sont *"almost always incomplete"*.
       Stratégie simple : supprimer ou ignorer les enregistrements incomplets
    4. **Construction de structures** : DataFrame si données tiennent en mémoire,
       base de données sinon (mapping clés → valeurs)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 · Sources et formats de données

    Quatre façons de collecter des données :
    - **Collecter de nouvelles données** depuis internet et autres sources
    - **Utiliser des données déjà stockées**
    - **Réutiliser les données de quelqu'un d'autre** (open data, Kaggle, UCI)
    - **Acheter des données** (fournisseurs spécialisés)

    #### Sources techniques — panorama

    | Source | Mécanisme | Exemple |
    |---|---|---|
    | **Fichiers plats** | CSV, Excel, JSON, Parquet sur disque | Dataset Titanic |
    | **Bases relationnelles** | SQL via SQLAlchemy (`pd.read_sql`) | PostgreSQL, MySQL, SQLite |
    | **Bases NoSQL** | Connecteurs spécialisés | MongoDB (documents), Redis (clé-valeur), Cassandra (colonnes) |
    | **APIs REST** | HTTP GET/POST → JSON | Twitter API, OpenWeather, INSEE |
    | **Web scraping** | `BeautifulSoup`, `Scrapy` | Prix immobiliers, avis clients |
    | **Streaming** | Kafka, Spark Streaming | Logs temps réel, IoT |

    ```python
    # Exemple — requête API REST avec requests
    import requests, pandas as pd

    resp = requests.get("https://api.example.com/data?year=2024")
    df = pd.DataFrame(resp.json()["results"])

    # Exemple — lecture depuis une base SQL
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///magasin.db")
    df = pd.read_sql("SELECT * FROM ventes WHERE annee = 2024", engine)
    ```

    #### Fonctions de chargement avec pandas

    | Fonction | Format | Description |
    |---|---|---|
    | `pd.read_csv()` | CSV | Données délimitées — **le plus fréquent** |
    | `pd.read_json()` | JSON | JavaScript Object Notation |
    | `pd.read_parquet()` | Parquet | Format binaire Apache — très performant |
    | `pd.read_feather()` | Arrow/Feather | Format binaire ultra-rapide en mémoire |
    | `pd.read_excel()` | Excel | Fichiers XLS/XLSX |
    | `pd.read_sql()` | SQL | Résultat d'une requête (via SQLAlchemy) |
    | `pd.read_html()` | HTML | Tableaux dans une page web |

    > Certains formats (HDF5, ORC, Parquet) intègrent les types
    > directement dans le fichier, évitant l'inférence coûteuse du CSV.

    #### Formats modernes — comparatif

    | Format | Stockage | Vitesse lecture | Colonnes sélectives | Usage recommandé |
    |---|:---:|:---:|:---:|---|
    | **CSV** | Texte | Lente | Non | Échange, petits datasets |
    | **JSON** | Texte | Lente | Non | APIs REST, données hiérarchiques |
    | **Parquet** | Binaire colonnaire | Très rapide | Oui | Analyses big data, stockage long terme |
    | **Arrow / Feather** | Binaire en-mémoire | Ultra-rapide | Oui | Échange entre outils (pandas ↔ polars ↔ Spark) |

    #### Apache Arrow — le standard inter-outils 2025

    **Arrow** est un format mémoire standardisé (colonnaire, zero-copy) qui permet
    à **pandas**, **polars**, **DuckDB**, **Spark** et **R** de partager des données
    sans copie ni sérialisation. `pd.read_feather()` et `pd.read_parquet()` utilisent Arrow en interne.

    ```python
    # Sauvegarder en Arrow/Feather (échange rapide entre outils)
    df.to_feather("titanic.feather")
    df2 = pd.read_feather("titanic.feather")  # 10× plus rapide que CSV
    ```

    #### Parquet vs CSV — lecture columnar

    ```
    CSV (ligne par ligne) :
    PassengerId | Survived | Pclass | Age | Fare  ← toutes les colonnes lues
        1       |    0     |   3    | 22  | 7.25
        2       |    1     |   1    | 38  | 71.28

    Parquet (colonne par colonne) :
    [PassengerId: 1,2,3...] [Age: 22,38,26...]   ← seulement les colonnes demandées
    → Gain : 5–10× moins d'espace, 10× plus rapide pour requêtes sélectives
    ```

    ```python
    # [R2] McKinney, Chap. 6 — Principaux arguments de read_csv (Table 6-2)
    import pandas as pd

    df = pd.read_csv(
        "titanic.csv",
        sep=",",                         # délimiteur (défaut virgule)
        header=0,                        # ligne d'en-tête (0 = 1ère ligne)
        index_col=None,                  # colonne à utiliser comme index
        na_values=["NA", "NULL", "?"],   # valeurs → NaN
        nrows=1000                       # lire seulement N lignes
    )

    # Parquet — lecture sélective de colonnes uniquement
    df2 = pd.read_parquet("titanic.parquet", columns=["Age", "Survived"])
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 · Cycle de vie des données — CRISP-DM

    > **CRISP-DM** (*Cross-Industry Standard Processfor Data Mining*)
    > comme la méthodologie de référence pour résoudre un problème data
    > de bout en bout.
    > C'est un processus **itératif** — on revient souvent en arrière entre les phases.

    ---

    #### Phase 1 · Compréhension métier (*Business Understanding*)

    Avant toute ligne de code, il faut comprendre le **contexte organisationnel** :

    | Étape | Description |
    |---|---|
    | Comprendre le processus métier | Quelle activité est concernée ? Quelles décisions sont prises ? |
    | Définir et cadrer le problème | Formuler précisément la question à résoudre |
    | Définir l'objectif métier | Quel résultat concret est attendu ? (réduction du churn, détection de fraude…) |
    | Fixer les critères de succès | Quels indicateurs valideront que le projet a réussi ? (KPIs, seuils) |

    ---

    #### Phase 2 · Compréhension des données (*Data Understanding*)

    | Étape | Description |
    |---|---|
    | **Identifier les sources** | Où les données sont-elles générées, stockées, et comment circulent-elles ? |
    | **Comprendre l'origine** | D'où viennent les données, comment sont-elles traitées ? |
    | **Analyser la sémantique** | Que signifient les colonnes dans le contexte métier ? |
    | **Évaluer les sources externes** | Existe-t-il des données sectorielles complémentaires utilisables ? |
    | **Vérifier les labels (ML)** | La variable cible est-elle disponible ? Y a-t-il des labels tardifs (*late labels*) ? |

    ---

    #### Phase 3 · Collecte des données (*Data Collection*)

    Quatre stratégies pour obtenir les données :

    - 🌐 **Collecter de nouvelles données** depuis internet ou d'autres sources
    - 🗄️ **Utiliser des données déjà stockées** dans les systèmes existants
    - ♻️ **Réutiliser les données de quelqu'un d'autre** (open data, Kaggle, UCI)
    - 💳 **Acheter des données** auprès de fournisseurs spécialisés

    ---

    #### Phase 4 · Préparation des données (*Data Preparation*)

    ```
    Découverte         Collecte &      Nettoyage      Transformation    Validation &
    & Profilage   →   Enrichissement  →  des données  →  & Structuration →  Publication
    ```

    | Étape | Actions typiques |
    |---|---|
    | **Découverte & profilage** | Statistiques descriptives, détection des anomalies |
    | **Collecte & enrichissement** | Jointure avec sources externes, ajout de features |
    | **Nettoyage** | Traitement des valeurs manquantes, doublons, outliers |
    | **Transformation & structuration** | Encodage, normalisation, feature engineering |
    | **Validation & publication** | Contrôle qualité, mise à disposition du dataset final |

    ---

    #### Phase 5 · Modélisation (*Modelling*)

    > *"Building a model is a very iterative process because there is no such
    > thing as a final and perfect solution."*

    On sélectionne et entraîne les algorithmes adaptés au problème (classification,
    régression, clustering…). De nombreuses techniques ML et statistiques sont
    disponibles dans les plateformes modernes (scikit-learn, XGBoost, etc.).

    ---

    #### Phase 6 · Évaluation (*Model Evaluation*)

    Le choix des métriques dépend du **type de problème** :

    | Classification | Régression |
    |---|---|
    | Matrice de confusion | MAE (Mean Absolute Error) |
    | Accuracy, Précision, Rappel | MSE / RMSE |
    | F-score (F1) | RMSLE |
    | AUC-ROC | R² et R² ajusté |
    | Log loss, Gini coefficient | |

    > Ces métriques seront détaillées en **Séance 6**.

    ---

    #### Phase 7 · Déploiement (*Deployment*)

    La mise en production s'inscrit dans une démarche **MLOps** :

    | Étape MLOps | Description |
    |---|---|
    | **Packaging & déploiement** | Sérialisation du modèle (joblib, ONNX), exposition via API |
    | **Lignage du modèle** | Traçabilité complète : données, code, paramètres (MLflow) |
    | **Monitoring & logging** | Surveillance des performances en production, détection de dérive |
    | **Ré-entraînement** | Déclenchement automatique si les performances chutent |
    | **Versioning** | Code, données et pipeline versionnés (git, DVC) |

    > Ces outils seront mis en pratique en **Séance 7** (MLflow, Streamlit).

    ---

    #### Cas d'usage complet — Analyse du Churn Client

    | Étape CRISP-DM | Actions concrètes |
    |---|---|
    | **Compréhension métier** | Définir le churn, identifier son coût pour l'entreprise |
    | **Compréhension des données** | Explorer les données d'appels, factures, réclamations |
    | **Collecte** | Extraire les données CRM, ajouter des tendances mensuelles |
    | **Préparation** | Nettoyer, agréger, standardiser → profils clients prêts |
    | **Modélisation** | Construire un modèle prédictif (ex : Random Forest) |
    | **Évaluation** | Identifier les drivers de churn, valider les KPIs de suivi |
    | **Déploiement** | Générer des listes d'abonnés ciblés pour des campagnes anti-churn |

    > ⚠️ **CRISP-DM est itératif** — la modélisation peut révéler un besoin
    > de features supplémentaires → retour immédiat à la phase de préparation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack([mo.image(src="images/crisp_dm.png", width=400)], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.7 · Écosystème Python pour l'analyse de données

    #### Pourquoi Python ?

    - *L'analyse de données est une discipline de décision guidée par les faits*,
    - Python fournit un des environnements les plus développés pour l'analyse et la science de données.
    - Les bases s'acquièrent facilement
    - **open-source** : les algorithmes cruciaux sont librement disponibles


    #### Bibliothèques fondamentales

    | Bibliothèque | Rôle |
    |---|---|
    | **NumPy** | Calcul numérique, tableaux N-D |
    | **SciPy** | Calcul scientifique (stats, optimisation) |
    | **pandas** | DataFrames, I/O, manipulation |
    | **polars** | DataFrames ultra-rapides (Rust, Arrow-natif) |
    | **scikit-learn** | Machine learning en Python |
    | **statsmodels** | Tests statistiques, modèles économétriques |
    | **matplotlib** | Visualisation 2D de référence |
    | **seaborn/matplotlib** | Visualisation statistique |

    #### pandas vs polars — quand utiliser quoi ?

    | Critère | pandas | polars |
    |---|---|---|
    | Maturité / écosystème | Très mature (2008) | Récent (2021), croissance rapide |
    | Syntaxe | Familière, large documentation | Expressions lazy, plus verbeux |
    | Performance (>1M lignes) | Lente (GIL Python) | 5–20× plus rapide (Rust, multi-thread) |
    | Intégration Arrow | Via `pyarrow` | Natif |
    | Usage dans ce cours | **Principal** | Mentionné pour les gros volumes |

    ```python
    # polars — syntaxe (pour référence)
    import polars as pl

    df_pl = pl.read_csv("titanic.csv")
    df_pl.filter(pl.col("Survived") == 1).group_by("Pclass").agg(
        pl.col("Age").mean().alias("age_moy")
    )
    # → équivalent de df[df.Survived==1].groupby("Pclass")["Age"].mean() en pandas
    ```

    #### pandas — les deux structures clés

    - *DataFrame* : « Un DataFrame représente un tableau de données rectangulaire […] On peut le concevoir comme un dictionnaire de Series partageant toutes le même index. »

    - *Series* : « Tableau unidimensionnel indexé pouvant contenir n'importe quel type de données. Contrairement à un tableau NumPy, chaque valeur est associée à un label d'index explicite, ce qui permet un accès et une manipulation des données par nom plutôt que par position.
     »




    ```python
    import pandas as pd

    # ── pd.Series : tableau 1D indexé ─────────────────────────────────────
    # [R3] : "A Pandas Series is a one-dimensional array of indexed data"
    s = pd.Series([0.25, 0.5, 0.75, 1.0])

    # Index explicite personnalisé — [R3] Chap. 3
    pop = pd.Series(
        {"Dakar": 3_800_000, "Abidjan": 5_300_000, "Lagos": 15_000_000},
        name="population"
    )
    print(pop)
    # Dakar       3800000
    # Abidjan     5300000
    # Lagos      15000000

    # ── pd.DataFrame : tableau 2D à index flexible ────────────────────────
    df = pd.DataFrame({
        "ville":  ["Dakar", "Abidjan", "Lagos"],
        "pays":   ["Sénégal", "Côte d'Ivoire", "Nigeria"],
        "pop_M":  [3.8, 5.3, 15.0],
    })
    # df.shape → (3, 3) : 3 lignes × 3 colonnes  — [R1] Chap. 2.6
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 · Éthique, RGPD et biais dans les données

    #### Biais algorithmiques et la loi de Campbell

    **Loi de Campbell** :
    > *"The more any quantitative social indicator is used for social decision-making,
    > the more subject it will be to corruption pressures and the more apt it will be
    > to distort and corrupt the social processes it is intended to monitor."*

    > *"Plus un indicateur social quantitatif est utilisé comme aide à la décision en matière de politique sociale,
    > plus cet indicateur est susceptible d'être manipulé et d'agir comme facteur de distorsion,
    > faussant ainsi les processus sociaux qu'il est censé surveiller."*


    #### Corrélation ≠ Causalité

    **Exemple fil rouge — Google Flu Trends (2008–2013)**

    Google a observé que le nombre de recherches du mot *"flu"* (grippe)
    augmentait chaque hiver en même temps que le nombre de cas de grippe déclarés.
    Corrélation forte → Google en a conclu qu'on pouvait **prédire** l'épidémie
    en temps réel, avant même les rapports officiels des centres de santé.

    Mais attention : les recherches Google ne *causent* pas la grippe.
    Les gens cherchent *"flu"* **parce qu'ils ont déjà des symptômes** ou
    entendent parler de l'épidémie autour d'eux.
    La vraie chaîne causale est :

    ```
    Virus grippal → symptômes → gens malades cherchent "flu" sur Google
         ↑                              ↑
      CAUSE                        EFFET (observable rapidement)
    ```

    Les recherches sont un **signal précoce** de l'épidémie, pas sa cause.
    Le modèle a d'ailleurs échoué en 2013 en sur-estimant massivement l'épidémie —
    preuve qu'une corrélation sans compréhension causale est fragile.

    **Règle d'or** : avant de conclure que A *cause* B parce qu'ils sont corrélés,
    demandez-vous toujours :

    | Question | Exemple |
    |---|---|
    | Y a-t-il une **cause commune** cachée ? | La saison hivernale cause *à la fois* la grippe et les recherches |
    | Le sens est-il **inversé** ? | C'est la grippe qui provoque les recherches, pas l'inverse |
    | Est-ce une **coïncidence** statistique ? | La consommation de glaces est corrélée aux noyades (cause commune : l'été) |

    > ⚠️ Ce piège (corrélation ≠ causalité) sera approfondi en **Séance 3**.

    #### Le RGPD pour le Data Scientist

    | Principe | Signification pratique |
    |---|---|
    | **Licéité** | Traitement sur base légale (consentement, intérêt légitime…) |
    | **Finalité** | But précis et légitime lors de la collecte |
    | **Minimisation** | Ne collecter que les données strictement nécessaires |
    | **Exactitude** | Données à jour, corrigées si besoin |
    | **Limitation de conservation** | Durée de vie définie pour chaque donnée |
    | **Sécurité** | Protection contre accès non autorisés et pertes |
    | **Responsabilité** | Pouvoir démontrer la conformité (*accountability*) |

    #### Types de biais courants

    | Type | Description | Exemple concret |
    |---|---|---|
    | **Sélection** | Échantillon non représentatif | Sondage web → exclut les non-connectés |
    | **Confirmation** | Chercher à valider une hypothèse | Ne tester que les cas favorables |
    | **Label** | Annotations humaines avec préjugés | Algorithme de recrutement biaisé |
    | **Historique** | Données passées → inégalités perpétuées | Crédit refusé selon quartier |
    | **Simpson** | Tendance inversée à l'agrégation | Admissions universitaires USA |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.9 · Introduction à marimo & uv ( plus de détails sur la ***seance1-1***)

    #### Contexte

    **Jupyter** est l'environnement de référence pour la data science interactive. **marimo** en est le successeur moderne (2025) :

    | Problème Jupyter | Solution marimo |
    |---|---|
    | Ordre d'exécution non déterministe | Graphe de dépendances réactif |
    | État caché dangereux | Chaque cellule = fonction Python pure |
    | Reproductibilité difficile | `.py` valide, versionnable avec git |
    | Widgets nécessitent un backend | `mo.ui.slider`, `mo.ui.dropdown` natifs |

    #### marimo — Fonctionnement réactif

    ```python
    import marimo as mo

    # Un slider — quand sa valeur change, toutes les cellules
    # qui utilisent `n` sont AUTOMATIQUEMENT recalculées
    n = mo.ui.slider(1, 100, value=10, label="Nombre de points")

    # Cellule dépendante : se met à jour sans clic
    import numpy as np
    data = np.random.randn(n.value)   # recalculé dès que n change
    ```

    #### uv — Setup du projet

    ```bash
    # 1. Installer uv (une seule fois, ~10 secondes)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 2. Créer le projet de cours
    uv init data_analysis
    cd data_analysis

    # 3. Ajouter les dépendances du cours
    uv add pandas numpy matplotlib seaborn scikit-learn marimo

    # 4. Lancer marimo
    uv run marimo edit seance1-0_introduction_analyse_donnees.py

    # Structure générée :
    # seance1_data/
    # ├── pyproject.toml   ← dépendances et config
    # ├── uv.lock          ← versions exactes (reproductibilité)
    # └── seance1-0_introduction_analyse_donnees.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🧪 PARTIE 2 — Travaux Pratiques
    ### Dataset : Titanic — Kaggle
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Le dataset **Titanic** (Kaggle) contient des informations sur les passagers
    du navire naufragé en 1912. Il est classiquement utilisé pour les premières
    explorations pandas.
    """)
    return


@app.cell
def _(pd):
    _TITANIC_PATH = "datasets/raw/titanic.csv"
    df = pd.read_csv(_TITANIC_PATH, na_values=["", "NA", "?"])
    df = df.rename(columns={
        "pclass": "Pclass", "survived": "Survived", "name": "Name",
        "sex": "Sex", "age": "Age", "sibsp": "SibSp", "parch": "Parch",
        "ticket": "Ticket", "fare": "Fare", "cabin": "Cabin",
        "embarked": "Embarked", "boat": "Boat", "body": "Body",
        "home.dest": "HomeDest",
    })
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Source : {_TITANIC_PATH}")
    print(df[["Survived","Pclass","Sex","Age","Fare","Cabin","Embarked"]].head().to_string())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🚢 pandas vs polars — Comparaison syntaxe

    | Opération | pandas | polars |
    |---|---|---|
    | Chargement | `pd.read_csv("file.csv")` | `pl.read_csv("file.csv")` |
    | Sélection colonne | `df["Age"]` | `df["Age"]` ou `df.select(pl.col("Age"))` |
    | Filtrage | `df[df["Survived"]==1]` | `df.filter(pl.col("survived")==1)` |
    | Groupby | `df.groupby("Pclass")["Fare"].mean()` | `df.group_by("Pclass").agg(pl.col("Fare").mean())` |
    | Null count | `df.isnull().sum()` | `df.null_count()` |
    | Types | `df.dtypes` | `df.schema` |

    > polars utilise des **expressions lazy** chaînables — plus expressif pour les transformations complexes.
    """)
    return


@app.cell
def _():
    try:
        import polars as _pl
        _df_pl = _pl.read_csv(
            "datasets/raw/titanic.csv",
            null_values=["", "NA", "?"],
            schema_overrides={"pclass": _pl.Int8, "survived": _pl.Int8},
        )
        print("=== 🚢 TITANIC — Polars ===")
        print(f"Schema : {_df_pl.schema}")
        print(f"\nDimensions : {_df_pl.shape}")
        print("\nNull count par colonne :")
        print(_df_pl.null_count())
        print("\nHead (5 lignes) :")
        print(_df_pl.select(["survived","pclass","sex","age","fare","cabin","embarked"]).head())
        df_titanic_pl = _df_pl
    except ImportError:
        print("💡 polars non installé — uv add polars")
        print("   Polars lit directement le même CSV, sans renommage de colonnes")
        print("\n# Syntaxe polars (référence) :")
        print("# df_pl = pl.read_csv('datasets/raw/titanic.csv', null_values=[''])")
        print("# df_pl.filter(pl.col('survived') == 1).group_by('pclass').agg(pl.col('fare').mean())")
        df_titanic_pl = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1b · Classification des variables du dataset Titanic

    Appliquer la taxonomie de la **section 1.4** aux colonnes réelles — étape
    incontournable avant toute analyse ou modélisation.

    | Colonne | Type statistique | Détail |
    |---|---|---|
    | `Survived` | Nominale binaire | **Variable cible** : 0 = décédé, 1 = survivant |
    | `Pclass` | **Ordinale** | Classe de voyage : 1 > 2 > 3 (ordre a du sens) |
    | `Name` | Nominale textuelle | Texte libre — potentiel NLP (titre : Mr, Mrs…) |
    | `Sex` | Nominale | 2 catégories : male / female |
    | `Age` | **Continue** | En années décimales — nombreuses valeurs manquantes |
    | `SibSp` | **Discrète** | Nb de frères/sœurs ou conjoints à bord (entier) |
    | `Parch` | **Discrète** | Nb de parents ou enfants à bord (entier) |
    | `Ticket` | Nominale mixte | Alphanumérique — peu exploitable brut |
    | `Fare` | **Continue** | Prix du billet en £ — distribution très asymétrique |
    | `Cabin` | Nominale mixte | Lettre (pont) + numéro — 77 % de valeurs manquantes |
    | `Embarked` | Nominale | Port : S (Southampton), C (Cherbourg), Q (Queenstown) |
    | `Boat` | Nominale | Numéro du canot de sauvetage — lié à la survie |
    | `Body` | Discrète | Numéro du corps récupéré — uniquement pour décédés |
    | `HomeDest` | Nominale textuelle | Lieu de résidence / destination — peu renseigné |

    > 🔑 **À retenir** : `Pclass` est **ordinale** (pas juste un entier),
    > `Age` et `Fare` sont **continues**, `SibSp` et `Parch` sont **discrètes**.
    > Cette distinction détermine quelle visualisation et quel test statistique utiliser.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 · `head()`, `tail()`, `shape`


    > *« Pour visualiser l'aspect des données, on peut utiliser la méthode `head()`,
    > qui affiche uniquement les cinq premières lignes. Si l'on passe un nombre en argument,
    > ce nombre correspond aux lignes qui seront affichées. »*

    `shape` renvoie `(n_lignes, n_colonnes)` : *« La commande shape donne exactement le nombre d'échantillons
    de données et de variables. »*
    """)
    return


@app.cell
def _(df):
    print("── df.head() ───────────────────────────────────────────")
    print(df.head())
    print(f"\n── df.shape ── {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return


@app.cell
def _(df):
    print("── df.tail(3) ── 3 dernières lignes ────────────────────")
    df.tail(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 · `df.info()` — Structure, types, valeurs non-nulles

    `info()` affiche les **dtypes** (types de données), le nombre de valeurs
    non-nulles et la mémoire utilisée.
    """)
    return


@app.cell
def _(df):
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())
    return


@app.cell
def _(df, plt):
    try:
        import missingno as msno
        _fig_msno, _axes_msno = plt.subplots(1, 2, figsize=(13, 4))
        msno.matrix(df, ax=_axes_msno[0], sparkline=False, color=(0.2, 0.5, 0.8))
        _axes_msno[0].set_title("Matrix missingno — zones de données manquantes\n(blanc = manquant)", fontweight="bold")
        msno.bar(df, ax=_axes_msno[1], color=(0.2, 0.5, 0.8))
        _axes_msno[1].set_title("Complétude par colonne (%)", fontweight="bold")
        plt.suptitle("🚢 Titanic — Visualisation des données manquantes (missingno)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("/tmp/titanic_s1_missingno.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("💡 Cabin : ~77% manquants (MNAR) | Age : ~20% (MAR) | Embarked : ~0.2% (MCAR)")
        print("   Ces mécanismes seront approfondis en Séance 2.")
    except ImportError:
        print("⚠️ missingno non installé — uv add missingno")
        print("   Visualise les patterns de données manquantes : matrix(), bar(), heatmap(), dendrogram()")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 · `df.describe()` — Statistiques descriptives

    > *« Si l'on souhaite obtenir rapidement des informations statistiques sur toutes les colonnes
    > numériques d'un DataFrame, on peut utiliser la fonction `describe()`. Le résultat affiche
    > le nombre de valeurs, la moyenne, l'écart-type, le minimum et le maximum,
    > ainsi que les percentiles. »*

    **Sample and Estimated Mean, Variance and Standard Scores** :

    | Statistique | Formule | Description |
    |:---:|:---:|---|
    | count | $n$ | Nombre de valeurs non-nulles |
    | mean | $\bar{x} = \dfrac{1}{n}\displaystyle\sum_{i=1}^{n} x_i$ | Moyenne arithmétique |
    | std | $s = \sqrt{\dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}$ | Écart-type (corrigé de Bessel) |
    | min / max | $\min(x_i)$, $\max(x_i)$ | Valeurs extrêmes |
    | 25%, 50%, 75% | $Q_1$, $Q_2$ (médiane), $Q_3$ | Quartiles |
    """)
    return


@app.cell
def _(df):
    df.describe().round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 · Sélection, filtrage, groupement
    """)
    return


@app.cell
def _(df):
    # [R1] Chap. 2.6.2 — Sélection d'une colonne → retourne une pd.Series
    print("── Sélection d'une colonne → pd.Series ─────────────────")
    print(df["Age"].head())
    print(f"\ntype : {type(df['Age'])}")
    return


@app.cell
def _(df):
    # [R1] Chap. 2.6.3 — Filtrage conditionnel
    # [R1] Exemple : df[(df.sex == 'Male') & (df.income == '>50K')]
    print("── Filtrage : survivants de 1ère classe ─────────────────")
    df_elite = df[(df["Survived"] == 1) & (df["Pclass"] == 1)]
    print(f"Nombre : {len(df_elite)} passagers")
    df_elite[["Name", "Sex", "Age", "Fare"]].head()
    return


@app.cell
def _(df, pd):
    # [R1] Chap. 2.6.4 — Valeurs manquantes
    print("── Valeurs manquantes — isna() ─────────────────────────")
    miss = df.isna().sum()
    miss_pct = (df.isna().mean() * 100).round(1)
    res_miss = pd.DataFrame({"N manquants": miss, "% manquants": miss_pct})
    res_miss = res_miss[res_miss["N manquants"] > 0].sort_values("N manquants", ascending=False)
    print(res_miss)
    return


@app.cell
def _(df):
    print("── Groupement : taux de survie par Pclass ───────────────")
    surv_class = df.groupby("Pclass")["Survived"].mean().mul(100).round(1)
    print(surv_class.rename("Taux survie (%)"))
    print()
    print("── Groupement : statistiques d'âge par sexe ─────────────")
    print(df.groupby("Sex")["Age"].describe().round(1))
    return


@app.cell
def _():
    print("=== 🚢 TITANIC — Équivalents polars pour sélection et groupement ===\n")

    try:
        import polars as _pl
        _df_pl = _pl.read_csv("datasets/raw/titanic.csv", null_values=["", "NA", "?"])

        print("# Filtrage pandas :")
        print("df[(df['Survived']==1) & (df['Pclass']==1)]")
        print("# Équivalent polars :")
        print("df_pl.filter((pl.col('survived')==1) & (pl.col('pclass')==1))")

        _surv_pl = _df_pl.filter(
            (_pl.col("survived") == 1) & (_pl.col("pclass") == 1)
        ).select(["name","sex","age","fare"]).head(3)
        print(f"\nPolars — survivants 1ère classe (3 premières lignes) :")
        print(_surv_pl)

        print("\n# Groupby polars :")
        _grp = _df_pl.group_by("pclass").agg(
            _pl.col("survived").mean().mul(100).round(1).alias("taux_survie_%"),
            _pl.col("fare").mean().round(2).alias("fare_moyen"),
            _pl.len().alias("n_passagers"),
        ).sort("pclass")
        print(_grp)

        print("\n# Lazy evaluation polars (plus efficace sur gros volumes) :")
        print("# result = df_pl.lazy().filter(...).group_by(...).agg(...).collect()")
    except ImportError:
        print("# polars non installé — voici les équivalents pour référence :\n")
        print("import polars as pl")
        print("df_pl = pl.read_csv('datasets/raw/titanic.csv', null_values=[''])")
        print()
        print("# Filtrage")
        print("df_pl.filter((pl.col('survived')==1) & (pl.col('pclass')==1))")
        print()
        print("# Groupby")
        print("df_pl.group_by('pclass').agg(")
        print("    pl.col('survived').mean().mul(100).alias('taux_survie_%'),")
        print("    pl.col('fare').mean().alias('fare_moyen'),")
        print(")")
        print()
        print("# Lazy (optimisé)")
        print("df_pl.lazy().filter(...).group_by(...).agg(...).collect()")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6 · Visualisations exploratoires

    Importance de visualiser les distributions et relations avant
    toute modélisation.

    > 💡 Le **boxplot** — Outlier Treatment) révèle les outliers
    > sur les tarifs — nous apprendrons à les traiter en **Séance 2**.
    """)
    return


@app.cell
def _(df, plt):
    fig_eda, axes_eda = plt.subplots(2, 3, figsize=(14, 8))
    fig_eda.suptitle(
        "Exploration Titanic — [R1] Chap. 3.3 · [R2] Chap. 9",
        fontsize=13, fontweight="bold")

    # 1. Survie
    sc_v = df["Survived"].value_counts()
    axes_eda[0, 0].bar(["Décédé (0)", "Survivant (1)"], sc_v.values,
                       color=["#C44E52", "#55A868"], edgecolor="white", lw=1.5)
    axes_eda[0, 0].set_title("Survie globale")
    for i, v in enumerate(sc_v.values):
        axes_eda[0, 0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    # 2. Classe
    pc_v = df["Pclass"].value_counts().sort_index()
    axes_eda[0, 1].bar([f"Cl. {c}" for c in pc_v.index], pc_v.values,
                       color=["#4C72B0", "#DD8452", "#8172B2"], edgecolor="white")
    axes_eda[0, 1].set_title("Classe de voyage (Pclass)")

    # 3. Histogramme des âges — [R1] Chap. 3.3.2
    ages_v = df["Age"].dropna()
    axes_eda[0, 2].hist(ages_v, bins=14, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes_eda[0, 2].axvline(ages_v.mean(), color="#C44E52", ls="--",
                           label=f"Moy. {ages_v.mean():.1f} ans")
    axes_eda[0, 2].axvline(ages_v.median(), color="#55A868", ls="-.",
                           label=f"Méd. {ages_v.median():.1f} ans")
    axes_eda[0, 2].set_xlabel("Âge"); axes_eda[0, 2].set_title("Distribution des âges")
    axes_eda[0, 2].legend(fontsize=8)

    # 4. Taux survie par sexe
    ss_v = df.groupby("Sex")["Survived"].mean() * 100
    axes_eda[1, 0].bar(ss_v.index, ss_v.values,
                       color=["#4C72B0", "#DD8452"], edgecolor="white")
    axes_eda[1, 0].set_ylabel("Taux de survie (%)")
    axes_eda[1, 0].set_title("Taux de survie par sexe")
    for i, v in enumerate(ss_v.values):
        axes_eda[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    # 5. Boxplot tarifs — [R1] Chap. 3.3.3 Outlier Treatment
    axes_eda[1, 1].boxplot(
        [df[df["Pclass"] == c]["Fare"].dropna() for c in [1, 2, 3]],
        labels=["1ère", "2ème", "3ème"])
    axes_eda[1, 1].set_title("Tarifs par classe (boxplot)\n[R1] Chap. 3.3.3 — Outliers")
    axes_eda[1, 1].set_ylabel("Fare (£)")

    # 6. Taux survie par classe
    spc_v = df.groupby("Pclass")["Survived"].mean() * 100
    axes_eda[1, 2].bar([f"Cl. {c}" for c in spc_v.index], spc_v.values,
                       color=["#4C72B0", "#DD8452", "#8172B2"], edgecolor="white")
    axes_eda[1, 2].set_ylabel("Taux de survie (%)")
    axes_eda[1, 2].set_title("Taux de survie par classe")
    for i, v in enumerate(spc_v.values):
        axes_eda[1, 2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    fig_eda
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6b · Violin plots, corrélation et visualisations par catégorie

    Le **violin plot** combine le boxplot et la densité de distribution — il révèle
    la forme réelle de la distribution là où le boxplot la masque (bimodalité, asymétrie).

    La **corrélation de Pearson** $\rho$ mesure la relation linéaire entre deux variables continues :

    $$\rho = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y} \in [-1,\, 1]$$

    - $\rho \approx 1$ : relation positive forte
    - $\rho \approx 0$ : pas de relation linéaire
    - $\rho \approx -1$ : relation négative forte

    > ⚠️ $\rho$ mesure uniquement la **linéarité** — deux variables peuvent être
    > fortement liées non-linéairement avec $\rho \approx 0$.
    > Corrélation ≠ causalité (voir section 1.8).
    """)
    return


@app.cell
def _(df, plt, sns):
    fig_viol, axes_viol = plt.subplots(1, 2, figsize=(12, 5))
    fig_viol.suptitle("Violin plots — distribution par catégorie", fontweight="bold")

    # Violin : âge selon survie
    sns.violinplot(x="Survived", y="Age", data=df,
                   palette=["#C44E52", "#55A868"], ax=axes_viol[0])
    axes_viol[0].set_xticklabels(["Décédé (0)", "Survivant (1)"])
    axes_viol[0].set_title("Distribution des âges selon la survie")
    axes_viol[0].set_ylabel("Âge (années)")

    # Violin : tarif selon classe
    sns.violinplot(x="Pclass", y="Fare", data=df,
                   palette=["#4C72B0", "#DD8452", "#8172B2"], ax=axes_viol[1])
    axes_viol[1].set_xticklabels(["1ère", "2ème", "3ème"])
    axes_viol[1].set_title("Distribution des tarifs par classe\n(noter l'asymétrie en 1ère)")
    axes_viol[1].set_ylabel("Fare (£)")

    plt.tight_layout()
    fig_viol
    return


@app.cell
def _(df):
    # Corrélation de Pearson entre variables numériques
    print("── Corrélation de Pearson — variables numériques ────────")
    corr_matrix = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr().round(3)
    print(corr_matrix)
    print()
    print("── Corrélations avec Survived (variable cible) ──────────")
    print(corr_matrix["Survived"].drop("Survived").sort_values())
    return


@app.cell
def _(df, np, plt, sns):
    _fig_corr, _axes_corr = plt.subplots(1, 2, figsize=(13, 5))

    _num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    _corr = df[_num_cols].corr()
    _mask = np.triu(np.ones_like(_corr, dtype=bool))
    sns.heatmap(
        _corr, ax=_axes_corr[0], annot=True, fmt=".2f", cmap="RdYlGn",
        mask=_mask, vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    _axes_corr[0].set_title("Matrice de corrélation de Pearson\n(triangle inférieur)", fontweight="bold")

    _corr_surv = _corr["Survived"].drop("Survived").sort_values()
    _colors_corr = ["#e74c3c" if v < 0 else "#2ecc71" for v in _corr_surv.values]
    _axes_corr[1].barh(_corr_surv.index, _corr_surv.values, color=_colors_corr, alpha=0.85, edgecolor="white")
    _axes_corr[1].axvline(0, color="black", linewidth=0.8)
    _axes_corr[1].set_title("Corrélations avec 'Survived'\n(variable cible)", fontweight="bold")
    _axes_corr[1].set_xlabel("ρ de Pearson")
    for _i, _v in enumerate(_corr_surv.values):
        _axes_corr[1].text(_v + (0.01 if _v >= 0 else -0.01), _i,
                           f"{_v:+.2f}", va="center", ha="left" if _v >= 0 else "right", fontsize=9)

    plt.suptitle("🚢 Titanic — Corrélations entre variables numériques", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/titanic_s1_correlation.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("💡 Pclass négativement corrélé : 3ème classe (=3) → moins de survie")
    print("💡 Fare positivement corrélé : tarif élevé → 1ère classe → plus de survie")

    try:
        import polars as _pl
        _df_pl = _pl.read_csv("datasets/raw/titanic.csv", null_values=["", "NA", "?"])
        _corr_pl = _df_pl.select(["survived","pclass","age","sibsp","parch","fare"]).drop_nulls().to_pandas().corr()
        print("\n📊 Polars — corrélation (via pandas pour l'affichage) :")
        print(_corr_pl["survived"].drop("survived").sort_values().round(3).to_string())
    except ImportError:
        print("\n# Polars — corrélation :")
        print("# df_pl.select([...]).drop_nulls().to_pandas().corr()")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6d · Feature Engineering — Avant-goût (Séance 2)

    Le **feature engineering** consiste à créer de nouvelles variables à partir des données brutes pour améliorer l'analyse et la modélisation. Voici un aperçu des techniques détaillées en **Séance 2**.

    | Feature créé | Source | Intérêt |
    |---|---|---|
    | `FamilySize` | `SibSp + Parch + 1` | Familles de 2-4 survivent mieux |
    | `IsAlone` | `FamilySize == 1` | Passagers seuls → moins de survie |
    | `AgeGroup` | `pd.cut(Age, ...)` | Enfants → taux de survie élevé |
    | `Title` | Extraction depuis `Name` (Mr, Mrs, Miss…) | Proxy du statut social |
    | `HasCabin` | `Cabin.notna()` | Indicatrice — 1ère classe en général |
    """)
    return


@app.cell
def _(df, pd, plt):
    print("=== 🚢 TITANIC — Feature Engineering (avant-goût Séance 2) ===\n")
    _df_fe = df.copy()

    _df_fe["FamilySize"] = _df_fe["SibSp"] + _df_fe["Parch"] + 1
    _df_fe["IsAlone"] = (_df_fe["FamilySize"] == 1).astype(int)

    _df_fe["AgeGroup"] = pd.cut(
        _df_fe["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Enfant\n(0-12)", "Ado\n(13-18)", "Adulte\n(19-35)", "Mûr\n(36-60)", "Senior\n(60+)"]
    )

    _df_fe["Title"] = _df_fe["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    _df_fe["Title"] = _df_fe["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    _df_fe["Title"] = _df_fe["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    _df_fe["HasCabin"] = _df_fe["Cabin"].notna().astype(int)

    print("Nouvelles features créées :")
    _new_features = ["FamilySize", "IsAlone", "AgeGroup", "Title", "HasCabin"]
    for _f in _new_features:
        print(f"  {_f}: {_df_fe[_f].value_counts().to_dict()}")

    _fig_fe, _axes_fe = plt.subplots(2, 3, figsize=(15, 8))

    _fs = _df_fe.groupby("FamilySize")["Survived"].agg(["mean","count"]).reset_index()
    _fs = _fs[_fs["count"] >= 5]
    _axes_fe[0,0].bar(_fs["FamilySize"].astype(str), _fs["mean"]*100,
                      color=["#2ecc71" if v > 0.5 else "#e74c3c" for v in _fs["mean"]],
                      alpha=0.8, edgecolor="white")
    _axes_fe[0,0].set_title("Taux de survie\npar taille de famille", fontweight="bold")
    _axes_fe[0,0].set_ylabel("% survie")
    _axes_fe[0,0].set_xlabel("FamilySize")
    _axes_fe[0,0].axhline(50, color="gray", linestyle="--", alpha=0.5)

    _alone = _df_fe.groupby("IsAlone")["Survived"].mean() * 100
    _axes_fe[0,1].bar(["Famille", "Seul"], _alone.values,
                      color=["#2ecc71", "#e74c3c"], alpha=0.8, edgecolor="white")
    _axes_fe[0,1].set_title("Taux de survie\n(seul vs famille)", fontweight="bold")
    for _i, _v in enumerate(_alone.values):
        _axes_fe[0,1].text(_i, _v+1, f"{_v:.1f}%", ha="center", fontweight="bold")

    _ag = _df_fe.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
    _axes_fe[0,2].bar(_ag.index.astype(str), _ag.values,
                      color=["#2ecc71" if v > 50 else "#e74c3c" for v in _ag.values],
                      alpha=0.8, edgecolor="white")
    _axes_fe[0,2].set_title("Taux de survie\npar groupe d'âge (pd.cut)", fontweight="bold")
    _axes_fe[0,2].axhline(50, color="gray", linestyle="--", alpha=0.5)

    _tt = _df_fe.groupby("Title")["Survived"].agg(["mean","count"])
    _tt = _tt[_tt["count"] >= 5].sort_values("mean")
    _axes_fe[1,0].barh(_tt.index, _tt["mean"]*100,
                       color=["#2ecc71" if v > 0.5 else "#e74c3c" for v in _tt["mean"]],
                       alpha=0.8, edgecolor="white")
    _axes_fe[1,0].axvline(50, color="gray", linestyle="--", alpha=0.5)
    _axes_fe[1,0].set_title("Taux de survie par Title\n(extrait du nom)", fontweight="bold")
    _axes_fe[1,0].set_xlabel("% survie")

    _hc = _df_fe.groupby("HasCabin")["Survived"].mean() * 100
    _axes_fe[1,1].bar(["Sans cabine", "Avec cabine"], _hc.values,
                      color=["#e74c3c", "#2ecc71"], alpha=0.8, edgecolor="white")
    _axes_fe[1,1].set_title("Taux de survie\n(indicatrice Cabin)", fontweight="bold")
    for _i, _v in enumerate(_hc.values):
        _axes_fe[1,1].text(_i, _v+1, f"{_v:.1f}%", ha="center", fontweight="bold")

    _axes_fe[1,2].axis("off")
    _txt = (
        "BILAN FEATURE ENGINEERING\n"
        "─────────────────────────────\n"
        "✅ FamilySize — familles 2-4\n"
        "   survivent mieux que seuls\n\n"
        "✅ Title — Mr très défavorisé\n"
        "   Mrs/Miss bien mieux\n\n"
        "✅ AgeGroup — enfants\n"
        "   ont priorité sur les canots\n\n"
        "✅ HasCabin — proxy pclass 1\n"
        "   forte corrélation survie\n\n"
        "→ Approfondissement : Séance 2\n"
        "  (MCAR/MAR/MNAR, ColumnTransformer)"
    )
    _axes_fe[1,2].text(0.05, 0.95, _txt, transform=_axes_fe[1,2].transAxes, va="top",
                        fontsize=10, fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="#eaf4fb", alpha=0.9))

    plt.suptitle("🚢 Titanic — Feature Engineering (avant-goût Séance 2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/titanic_s1_fe.png", dpi=120, bbox_inches="tight")
    plt.show()

    try:
        import polars as _pl
        _df_pl = _pl.read_csv("datasets/raw/titanic.csv", null_values=["", "NA", "?"])
        _df_pl = _df_pl.with_columns([
            (_pl.col("sibsp") + _pl.col("parch") + 1).alias("family_size"),
            ((_pl.col("sibsp") + _pl.col("parch")) == 0).cast(_pl.Int8).alias("is_alone"),
            _pl.col("cabin").is_not_null().cast(_pl.Int8).alias("has_cabin"),
        ])
        print("\n📊 Polars — feature engineering en une expression :")
        print(_df_pl.select(["survived","pclass","family_size","is_alone","has_cabin"]).head(5))
    except ImportError:
        print("\n# Polars — feature engineering :")
        print("# df_pl.with_columns([")
        print("#     (pl.col('sibsp') + pl.col('parch') + 1).alias('family_size'),")
        print("#     pl.col('cabin').is_not_null().cast(pl.Int8).alias('has_cabin'),")
        print("# ])")
    return


@app.cell
def _(df, plt):
    # FacetGrid — distribution des âges par classe ET survie
    fig_fg, axes_fg = plt.subplots(1, 2, figsize=(13, 4))
    fig_fg.suptitle("FacetGrid — âge et tarif selon survie et classe", fontweight="bold")

    # Distribution des âges par survie (histogramme facetté)
    for _survived, _color, _label in zip(
        [0, 1], ["#C44E52", "#55A868"], ["Décédé", "Survivant"]
    ):
        _data = df[df["Survived"] == _survived]["Age"].dropna()
        axes_fg[0].hist(_data, bins=12, alpha=0.6, color=_color, label=_label, edgecolor="white")
    axes_fg[0].set_xlabel("Âge"); axes_fg[0].set_ylabel("Effectif")
    axes_fg[0].set_title("Distribution des âges\nselon la survie")
    axes_fg[0].legend()

    # Tarif moyen par classe et sexe (barplot)
    _pivot = df.groupby(["Pclass", "Sex"])["Fare"].mean().reset_index()
    _width = 0.35
    _classes = [1, 2, 3]
    _x = range(len(_classes))
    _men = [_pivot[(_pivot.Pclass == c) & (_pivot.Sex == "male")]["Fare"].values[0]
            for c in _classes]
    _women = [_pivot[(_pivot.Pclass == c) & (_pivot.Sex == "female")]["Fare"].values[0]
              for c in _classes]
    axes_fg[1].bar([i - _width/2 for i in _x], _men, _width,
                   label="Homme", color="#4C72B0", edgecolor="white")
    axes_fg[1].bar([i + _width/2 for i in _x], _women, _width,
                   label="Femme", color="#DD8452", edgecolor="white")
    axes_fg[1].set_xticks(list(_x))
    axes_fg[1].set_xticklabels(["1ère", "2ème", "3ème"])
    axes_fg[1].set_ylabel("Tarif moyen (£)")
    axes_fg[1].set_title("Tarif moyen par classe et sexe\n(les femmes de 1ère payaient plus)")
    axes_fg[1].legend()

    plt.tight_layout()
    fig_fg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6c · Train/Test Split — Préparer la modélisation

    Avant de construire un modèle (séances 5–6), on sépare le dataset en deux :

    | Ensemble | Proportion | Rôle |
    |---|:---:|---|
    | **Train** | 80 % | Apprentissage du modèle |
    | **Test** | 20 % | Évaluation sur données jamais vues |

    **Pourquoi `stratify` ?**
    Sans stratification, le hasard pourrait créer un test set avec
    70 % de survivants alors que le dataset réel en contient 38 %.
    Le modèle serait évalué sur une distribution biaisée.
    `stratify=df["Survived"]` garantit que la proportion 62/38 est maintenue
    dans les deux ensembles.
    """)
    return


@app.cell
def _(df, pd):
    from sklearn.model_selection import train_test_split as _tts

    df_train, df_test = _tts(
        df, test_size=0.2, random_state=42, stratify=df["Survived"]
    )

    print("── Taille des ensembles ─────────────────────────────────")
    print(f"Train : {len(df_train)} lignes ({len(df_train)/len(df)*100:.0f} %)")
    print(f"Test  : {len(df_test)}  lignes ({len(df_test)/len(df)*100:.0f} %)")

    print("\n── Vérification de la stratification (taux de survie) ──")
    _check = pd.DataFrame({
        "Dataset complet": df["Survived"].value_counts(normalize=True).mul(100).round(1),
        "Train": df_train["Survived"].value_counts(normalize=True).mul(100).round(1),
        "Test": df_test["Survived"].value_counts(normalize=True).mul(100).round(1),
    })
    print(_check.rename(index={0: "Décédé (%)", 1: "Survivant (%)"}))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.7 · Widgets réactifs marimo

    Contrairement à Jupyter, marimo
    intègre l'interactivité nativement : modifier un widget **recalcule
    automatiquement** toutes les cellules dépendantes.
    """)
    return


@app.cell
def _(mo):
    filtre_classe = mo.ui.dropdown(
        options={"Toutes classes": 0, "1ère classe": 1, "2ème classe": 2, "3ème classe": 3},
        value="Toutes classes",
        label="🎟️ Classe"
    )
    filtre_sexe = mo.ui.dropdown(
        options=["Tous", "male", "female"],
        value="Tous",
        label="👤 Sexe"
    )
    mo.hstack([filtre_classe, filtre_sexe])
    return filtre_classe, filtre_sexe


@app.cell
def _(df, filtre_classe, filtre_sexe, plt):
    dff = df.copy()
    if filtre_classe.value != 0:
        dff = dff[dff["Pclass"] == filtre_classe.value]
    if filtre_sexe.value != "Tous":
        dff = dff[dff["Sex"] == filtre_sexe.value]
    n_pax = len(dff)
    taux_surv = dff["Survived"].mean() * 100 if n_pax > 0 else 0

    fig_w, axes_w = plt.subplots(1, 2, figsize=(10, 4))
    fig_w.suptitle(
        f"Filtre actif — {n_pax} passagers | Taux de survie : {taux_surv:.1f}%",
        fontweight="bold")
    if n_pax > 0:
        sv = dff["Survived"].value_counts()
        axes_w[0].pie(sv.values, labels=["Décédé", "Survivant"][:len(sv)],
                      colors=["#C44E52", "#55A868"], autopct="%1.1f%%", startangle=90)
        axes_w[0].set_title("Répartition survie")
        ag = dff["Age"].dropna()
        if len(ag) > 0:
            axes_w[1].hist(ag, bins=10, color="#4C72B0", edgecolor="white")
            axes_w[1].set_xlabel("Âge"); axes_w[1].set_title("Distribution des âges")
    else:
        for axx in axes_w:
            axx.text(0.5, 0.5, "Aucune donnée", ha="center", va="center", fontsize=12)
            axx.axis("off")
    plt.tight_layout()
    fig_w
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 🏋️ Exercices — À faire en binôme

    *Créer un nouveau notebook marimo appélé TP_seance1-0. Faire le tuto pas à pas et prennez le temps de comprendre les output avant d'avancer.*

    **Exercice 1 — Exploration**
    ```python
    # a) Affichez les types de chaque colonne
    print(df.dtypes)

    # b) Comptez par port d'embarquement
    print(df.groupby("Embarked").size())

    # c) Tarif moyen par classe
    print(df.groupby("Pclass")["Fare"].mean().round(2))

    # d) Taux de survie global
    print(f"Taux global : {df['Survived'].mean()*100:.1f}%")
    ```

    **Exercice 2 — Manipulation**
    ```python
    # a) Créer une colonne catégorie d'âge (variable ordinale)
    df["Cat_age"] = pd.cut(df["Age"],
                           bins=[0, 12, 18, 60, 100],
                           labels=["Enfant", "Ado", "Adulte", "Senior"])

    # b) Remplacer les âges manquants par la médiane
    df["Age_imputed"] = df["Age"].fillna(df["Age"].median())

    # c) Taux de survie par catégorie d'âge
    print(df.groupby("Cat_age")["Survived"].mean().mul(100).round(1))
    ```

    **Exercice 3 — Visualisation**
    ```python
    import matplotlib.pyplot as plt

    # Boxplot âge selon survie — [R1] Chap. 3.3.3
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column="Age", by="Survived", ax=ax)
    ax.set_title("Distribution des âges selon la survie")
    ax.set_xlabel("Survived (0=Décédé, 1=Survivant)")
    plt.suptitle("\")
    plt.show()
    ```

    ---
    *→ **Séance 1-1** : Prise en main de Python pour l'analyse de données*
    *Outils : pandas, ydata-profiling · Dataset : Online Retail (UCI)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 📚 Références bibliographiques

    **[R1]** Igual, L. & Seguí, S. (2017). *Introduction to Data Science — A Python Approach to Concepts, Techniques and Applications*. Springer.

    **[R2]** McKinney, W. (2022). *Python for Data Analysis* (3ᵉ éd.). O'Reilly Media.


    **[R3]** VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. *(disponible librement en ligne)*


    **Jeu de données** · Titanic Passenger Dataset — Kaggle Competition *"Titanic: Machine Learning from Disaster"*
    Source : [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) · Licence : Open

    **Outils** · Python 3.11+ · [marimo](https://marimo.io) · [uv](https://docs.astral.sh/uv) ·
    pandas · NumPy · matplotlib · seaborn · scikit-learn

    ---
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
