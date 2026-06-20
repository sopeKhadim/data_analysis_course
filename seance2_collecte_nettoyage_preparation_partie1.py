# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib==3.11.0",
#     "missingno==0.5.2",
#     "numpy==2.4.6",
#     "openpyxl==3.1.5",
#     "pandas==3.0.3",
#     "scikit-learn==1.9.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Séance 2 — Collecte, Nettoyage et Préparation des données",
)

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    sns.set_palette("husl")
    print("✅ Imports OK")
    print(f"   pandas {pd.__version__}  |  numpy {np.__version__}")


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _header(mo):
    mo.md(r"""
    # 📦 Séance 2 — Collecte, Nettoyage, Préparation des données et Echatillonnage.

    **Cours : Analyse de données · 1ère année ingénieur · 2025–2026**

    ---

    > *"Data scientists spend 80% of their time cleaning and preparing data."*
    > — Wes McKinney, *Python for Data Analysis* (O'Reilly, 3e éd., 2022)

    ## Objectifs de la séance

    | Partie | Durée | Contenu |
    |--------|-------|---------|
    | **CM** | 3h00 | Collecte · Valeurs manquantes · Outliers · Normalisation · Profiling · Echantillonage |
    | **TD** | 1h00 | Exercices sur dataset e-commerce |
    | **TP** | 30min | Pipeline de nettoyage complet sur Online Retail |

    ## Dataset utilisé

    | Dataset | Source | Rôle pédagogique |
    |---------|--------|-----------------|
    | **Online Retail** | UCI ML Repository (synthétique) | Fil rouge — exemples pandas & polars dans chaque section + TD et TP complets |

    ## Plan

    1. Collecte des données — CSV, JSON, Excel, API REST, SQL · **🛒 Online Retail pandas vs polars**
    2. Qualité des données — Doublons, cohérences, types · **🛒 Audit manquants Online Retail**
    3. Valeurs manquantes — MCAR / MAR / MNAR · Stratégies · **🛒 Imputation UnitPrice, CustomerID**
    4. Détection des outliers — IQR · Z-score · Isolation Forest
    5. Normalisation et standardisation — Min-Max · Z-score · Robust · **🛒 Quantity et UnitPrice**
    6. Encodage des variables catégorielles — Label · OneHot · Target · **🛒 Country, Description**
    7. Profiling automatique — ydata-profiling
    8. Feature Engineering — Extraction dates · Binning · Combinaison
    9. Pipeline complet — ColumnTransformer
    10. Échantillonnage — SRS · Stratifié · Grappes · Bootstrap · Classes déséquilibrées · SMOTE
    11. TD — Exercices e-commerce
    12. TP — Pipeline de nettoyage complet

    > **Note :** Ce cours fait suite à la Séance 1. Les concepts de `.head()`, `.info()`, `.describe()`, et la manipulation de base avec pandas sont considérés comme acquis. La Séance 1 a introduit `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder` et les pipelines scikit-learn. Nous allons ici **approfondir** avec des méthodes plus robustes (MCAR/MAR/MNAR, KNN, Isolation Forest, RobustScaler) et introduire **polars** comme alternative moderne à pandas.
    """)
    return


@app.cell(hide_code=True)
def _section1(mo):
    mo.md(r"""
    ---
    ## 1. Collecte des données

    La première étape de tout projet data est **d'acquérir les données** depuis leurs sources.

    | Source | Format typique | Outil Python |
    |--------|---------------|--------------|
    | Fichiers plats | CSV, TSV | `pd.read_csv()` |
    | Excel | XLSX, XLS | `pd.read_excel()` |
    | JSON / APIs | JSON | `pd.read_json()`, `requests` |
    | Bases de données SQL | SQL | `pd.read_sql()`, SQLAlchemy |
    | Parquet / Arrow | Colonnes compressées | `pd.read_parquet()` |

    ### 1.1 Chargement CSV — Options avancées

    ```python
    df = pd.read_csv(
    "data.csv",
    sep=";",               # séparateur point-virgule (format FR)
    decimal=",",           # virgule décimale
    encoding="latin-1",    # encodage Windows
    parse_dates=["date"],  # conversion datetime automatique
    na_values=["N/A", "?", "-"],  # valeurs → NaN
    dtype={"code_postal": str},    # forcer le type
    )
    ```

    ### 1.2 Collecte via API REST

    ```python
    import requests
    import pandas as pd

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
    "latitude": 14.6928,   # Dakar
    "longitude": -17.4467,
    "daily": "temperature_2m_max,temperature_2m_min",
    "forecast_days": 7,
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
    data = response.json()
    df_meteo = pd.DataFrame(data["daily"])
    ```

    **Codes HTTP importants :** `200` OK · `404` Non trouvé · `429` Trop de requêtes · `403` Interdit

    > ⚠️ **Sécurité :** Stocker les clés API dans des variables d'environnement (`os.environ`), jamais dans le code.

    ### 1.3 Collecte depuis une base SQL

    ```python
    import sqlalchemy as sa
    import pandas as pd

    engine = sa.create_engine("postgresql://user:password@localhost:5432/mydb")

    # Requête paramétrée (évite les injections SQL)
    query = sa.text("SELECT * FROM commandes WHERE region = :region")
    df = pd.read_sql(query, engine, params={"region": "Dakar"})
    ```
    """)
    return


@app.cell(hide_code=True)
def _retail_collecte_theory(mo):
    mo.md(r"""
    ### 🛒 Exemple fil rouge — Online Retail (pandas vs polars)

    Le dataset **Online Retail** (UCI ML Repository) est notre référence tout au long du cours.
    Il illustre chaque concept dans un contexte métier concret : transactions e-commerce d'un grossiste britannique
    de décembre 2010 à décembre 2011 (~541 909 lignes, 8 colonnes).

    **pandas** est la bibliothèque historique, **polars** est une alternative moderne (Rust, lazy evaluation, API expressif).

    | | **pandas** | **polars** |
    |---|---|---|
    | Langage interne | Python/NumPy | Rust |
    | Évaluation | Eager (immédiate) | Lazy ou Eager |
    | Syntaxe sélection | `df["col"]` / `.loc` | `df.select(pl.col("col"))` |
    | Chaînage | Limité | Natif (`.pipe`, `.lazy()`) |
    | Vitesse | Référence | 5–20× plus rapide sur gros volumes |

    ```python
    # pandas — options adaptées au fichier Online Retail (Excel d'origine)
    import pandas as pd

    df = pd.read_excel("Online Retail.xlsx",
                       dtype={"CustomerID": str},
                       parse_dates=["InvoiceDate"])

    # Ou depuis un CSV exporté
    df = pd.read_csv("online_retail.csv",
                     encoding="latin-1",        # encodage Windows fréquent
                     parse_dates=["InvoiceDate"],
                     dtype={"CustomerID": str},
                     na_values=["\", "NA", "?"])

    # polars — chargement direct, très rapide sur ~500k lignes
    import polars as pl
    df_pl = pl.read_csv("online_retail.csv",
                        null_values=["\", "NA", "?"],
                        try_parse_dates=True)
    ```
    """)
    return


@app.cell
def _():
    path_retail = "datasets/online_retail/Online_Retail.xlsx"
    df_ret = pd.read_excel(path_retail, 
                            parse_dates=['InvoiceDate'],
                            dtype={'CustomerID': str},   # éviter la conversion float → perte du zéro de tête
                            na_values=['', 'NA', '?'],
                           )
    print(df_ret.dtypes.to_string())
    print(df_ret.describe())
    return (df_ret,)


@app.cell
def _(df_ret, mo):
    # Vous pouvez utiliser transform pour manipuler le dataframe.
    mo.ui.dataframe(df_ret.head())
    return


@app.cell
def _retail_chargement_demo():
    """Démonstration du chargement — utilise le dataset synthétique créé en mémoire."""
    import io as _io

    # Simulation d'un chargement CSV avec options avancées (identique à un vrai fichier Online Retail)
    print("=== 🛒 ONLINE RETAIL — Chargement pandas (options avancées) ===")
    print()
    print("Exemple de commande réelle pour le fichier UCI :")
    print("""  df = pd.read_csv(
      'online_retail.csv',
      encoding='latin-1',          # encodage Windows fréquent
      parse_dates=['InvoiceDate'],
      dtype={'CustomerID': str},   # éviter la conversion float → perte du zéro de tête
      na_values=['', 'NA', '?'],
      )""")
    print()
    print("=== Dataset synthétique chargé en mémoire (500 lignes, 8 colonnes) ===")
    print("Colonnes : InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")
    print()
    print("Types attendus après chargement :")
    _schema = {
        "InvoiceNo"  : "object  — numéro de facture (peut contenir 'C' pour retour)",
        "StockCode"  : "object  — code article (alphanumérique)",
        "Description": "object  — libellé produit (peut être NaN)",
        "Quantity"   : "int64   — quantité (négatif = retour/annulation)",
        "InvoiceDate": "datetime64 — horodatage de la transaction",
        "UnitPrice"  : "float64 — prix unitaire en £ (peut être NaN ou 0)",
        "CustomerID" : "object  — identifiant client (~20% manquant = guest checkout)",
        "Country"    : "object  — pays de livraison (65% UK)",
    }
    for _col, _desc in _schema.items():
        print(f"  {_col:<14} → {_desc}")

    # Polars (si installé)
    print()
    try:
        import polars as _pl
        import pandas as _pd_demo
        import numpy as _np_demo

        _n_demo = 10
        _df_demo = _pd_demo.DataFrame({
            "InvoiceNo"  : [f"INV{i:05d}" for i in range(_n_demo)],
            "StockCode"  : ["85123A", "71053", "84406B", "84029G", "84029E",
                            "22752", "21730", "85123A", None, "71053"],
            "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN",
                             None, "KNITTED UNION FLAG HOT WATER BOTTLE", "RED WOOLLY HOTTIE",
                             "CREAM CUPID HEARTS COAT HANGER", "GLASS STAR FROSTED T-LIGHT HOLDER",
                             "WHITE HANGING HEART T-LIGHT HOLDER", "SAMPLE", None],
            "Quantity"   : [6, 6, 8, 6, 6, 2, 6, -1, 32, 6],
            "UnitPrice"  : [2.55, 3.39, 2.75, 3.39, 3.39, None, 3.25, 2.55, 5.95, 3.39],
            "CustomerID" : ["17850", "13047", None, "13047", "17850",
                             "17850", None, "17850", "13047", None],
            "Country"    : ["United Kingdom", "United Kingdom", "France", "United Kingdom", "United Kingdom",
                             "Germany", "United Kingdom", "United Kingdom", None, "France"],
        })
        _df_pl_demo = _pl.from_pandas(_df_demo)
        print("=== Polars — schema() et head() ===")
        print(_df_pl_demo.schema)
        print(_df_pl_demo.head(5))
    except ImportError:
        print("💡 polars non installé (uv add polars) — exemples polars affichés en commentaire")
    return


@app.cell
def _create_dataset():
    """
    Dataset synthétique Online Retail-like avec imperfections réalistes.
    Source d'inspiration : Online Retail Dataset (UCI ML Repository)
    https://archive.ics.uci.edu/dataset/352/online+retail
    """
    np.random.seed(42)
    _n = 500

    _invoice_no = [f"INV{str(i).zfill(5)}" for i in range(1, _n + 1)]
    _stock_codes = np.random.choice(
        ["85123A", "71053", "84406B", "84029G", "84029E", "22752", "21730", None],
        _n, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]
    )
    _descriptions = np.random.choice(
        ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN",
         "CREAM CUPID HEARTS COAT HANGER", "KNITTED UNION FLAG HOT WATER BOTTLE",
         "RED WOOLLY HOTTIE WHITE HEART", None, "???", "SAMPLE"],
        _n, p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05]
    )
    _quantities = np.concatenate([
        np.random.randint(1, 100, int(_n * 0.85)),
        np.random.randint(-50, 0, int(_n * 0.05)),
        np.array([500, 600, 800, 1000, 2000]),
        np.zeros(_n - int(_n * 0.85) - int(_n * 0.05) - 5, dtype=int)
    ])[:_n]

    _unit_prices = np.abs(np.random.exponential(5, _n))
    _unit_prices[np.random.choice(_n, 30, replace=False)] = np.nan
    _unit_prices = np.concatenate([_unit_prices[:495], [0.0, 0.0, 150.0, 200.0, 500.0]])[:_n]

    _customer_ids = np.where(
        np.random.random(_n) > 0.20,
        np.random.choice(range(12000, 18000), _n),
        None
    )
    _countries = np.random.choice(
        ["United Kingdom", "France", "Germany", "Spain", "Netherlands", "EIRE", None],
        _n, p=[0.65, 0.10, 0.08, 0.07, 0.05, 0.04, 0.01]
    )
    _dates = pd.date_range("2010-12-01", "2011-12-09", periods=_n)

    df_retail = pd.DataFrame({
        "InvoiceNo": _invoice_no,
        "StockCode": _stock_codes,
        "Description": _descriptions,
        "Quantity": _quantities,
        "InvoiceDate": _dates,
        "UnitPrice": _unit_prices,
        "CustomerID": _customer_ids,
        "Country": _countries,
    })

    # Ajout de doublons intentionnels
    _doublons = df_retail.sample(15, random_state=1)
    df_retail = pd.concat([df_retail, _doublons], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Dataset créé : {df_retail.shape[0]} lignes × {df_retail.shape[1]} colonnes")
    print(df_retail.head(5).to_string())
    return (df_retail,)


@app.cell(hide_code=True)
def _section2(mo):
    mo.md(r"""
    ---
    ## 2. Qualité des données

    Avant de traiter les données, il faut **diagnostiquer leur état de santé**.

    | Problème | Description | Exemple |
    |----------|-------------|---------|
    | **Doublons** | Lignes identiques | Même commande enregistrée deux fois |
    | **Valeurs manquantes** | `NaN`, `None` | Client sans identifiant |
    | **Types incohérents** | Nombre stocké en texte | `"123"` au lieu de `123` |
    | **Valeurs aberrantes** | Hors plage normale | Prix = -5€ |
    | **Incohérences métier** | Logique invalide | Date fin < Date début |
    """)
    return


@app.cell
def _audit(df_retail):
    print("=" * 60)
    print("AUDIT RAPIDE DU DATASET")
    print("=" * 60)
    print(f"\n📐 Dimensions : {df_retail.shape[0]:,} lignes × {df_retail.shape[1]} colonnes")
    print("\n📋 Types de données :")
    print(df_retail.dtypes.to_string())
    _missing = df_retail.isnull().sum()
    _missing_pct = (_missing / len(df_retail) * 100).round(1)
    _missing_df = pd.DataFrame({"Manquants": _missing, "%": _missing_pct})
    print("\n🔍 Valeurs manquantes :")
    print(_missing_df[_missing_df["Manquants"] > 0].to_string())
    print(f"\n🔁 Doublons : {df_retail.duplicated().sum()} lignes")
    print("\n📊 Statistiques (numériques) :")
    print(df_retail.describe().round(2).to_string())
    return


@app.cell
def _plot_manquants(df_retail):
    import seaborn as _sns2
    _fig1, _axes1 = plt.subplots(1, 2, figsize=(14, 5))

    _mp = (df_retail.isnull().sum() / len(df_retail) * 100).sort_values(ascending=True)
    _mp = _mp[_mp > 0]
    _colors1 = ["#e74c3c" if p > 30 else "#f39c12" if p > 10 else "#3498db" for p in _mp]
    _mp.plot(kind="barh", ax=_axes1[0], color=_colors1)
    _axes1[0].set_xlabel("% de valeurs manquantes")
    _axes1[0].set_title("Valeurs manquantes par colonne", fontweight="bold")
    _axes1[0].axvline(x=30, color="red", linestyle="--", alpha=0.5, label="Seuil 30%")
    _axes1[0].axvline(x=10, color="orange", linestyle="--", alpha=0.5, label="Seuil 10%")
    _axes1[0].legend(fontsize=9)
    for _i1, (_idx1, _v1) in enumerate(_mp.items()):
        _axes1[0].text(_v1 + 0.3, _i1, f"{_v1:.1f}%", va="center", fontsize=9)

    _sample1 = df_retail.head(100).isnull().astype(int)
    _sns2.heatmap(_sample1.T, ax=_axes1[1], cmap="Blues", cbar=False,
                  yticklabels=_sample1.columns, xticklabels=False)
    _axes1[1].set_title("Pattern valeurs manquantes\n(100 premières lignes)", fontweight="bold")

    plt.tight_layout()
    plt.suptitle("Diagnostic des valeurs manquantes", y=1.02, fontsize=13, fontweight="bold")
    plt.savefig("/tmp/manquants.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Rouge = critique (>30%) | Orange = modéré (>10%) | Bleu = faible")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [**missingno**](https://github.com/residentmario/missingno) est une librairie Python qui vous permet de d'analyser rapidement les valeurs manquantes du dataset. Elle permet d'analyser la corrélation entre les colonnes manquantes
    """)
    return


@app.cell
def _missingno_viz(df_retail):
    try:
        import missingno as msno
        _fig_msno, _axes_msno = plt.subplots(1, 3, figsize=(18, 5))

        # Matrix — zones contiguës = manquants non aléatoires
        msno.matrix(df_retail, ax=_axes_msno[0], sparkline=False, color=(0.2, 0.5, 0.8))
        _axes_msno[0].set_title("Matrix — zones contiguës\n→ manquants non aléatoires", fontweight="bold")

        # Heatmap — corrélations entre colonnes manquantes
        msno.heatmap(df_retail, ax=_axes_msno[1], cmap="RdYlGn")
        _axes_msno[1].set_title("Heatmap — corrélations\nentre colonnes manquantes", fontweight="bold")

        # Dendrogram — regroupements de colonnes
        msno.dendrogram(df_retail, ax=_axes_msno[2], orientation="top")
        _axes_msno[2].set_title("Dendrogramme — colonnes\nà manquants similaires", fontweight="bold")

        plt.suptitle("Analyse structurelle des données manquantes — missingno", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("/tmp/missingno.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("💡 Heatmap : corrélation ≈ +1 → manquants ensemble | ≈ -1 → l'un présent implique l'autre absent") #  corrélations dans les endroits où les données sont manquantes
        print("💡 Dendrogramme : colonnes proches = patterns de manquants similaires → possible MAR") # Les feuilles qui sont au même niveau prédisent la présence de l’autre (vide ou pleine). Les bras verticaux sont utilisés pour indiquer les différents groupes. Les bras courts signifient que les branches sont similaires
        print("")
    except ImportError:
        print("⚠️  missingno non installé — lancez : uv add missingno")
        print("   La librairie génère des visualisations matricielles des patterns de données manquantes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Suppression des doublons.
    """)
    return


@app.cell
def _doublons(df_retail):
    print("=== GESTION DES DOUBLONS ===")
    print(f"Taille initiale : {len(df_retail):,}")
    print(f"Doublons : {df_retail.duplicated().sum()}")
    df_clean = df_retail.drop_duplicates()
    print(f"Après suppression : {len(df_clean):,} lignes")
    return (df_clean,)


@app.cell(hide_code=True)
def _section3(mo):
    mo.md(r"""
    ---
    ## 3. Valeurs manquantes — MCAR / MAR / MNAR

    Toutes les valeurs manquantes ne sont pas équivalentes. Rubin (1976) distingue trois mécanismes fondamentaux :

    ### 3.1 Taxonomie

    #### MCAR — Missing Completely At Random

    La probabilité qu'une valeur soit manquante est **indépendante** de toutes les données :

    $$P(\text{manquant} \mid X_{\text{obs}}, X_{\text{manq}}) = P(\text{manquant})$$

    **Exemple :** Panne aléatoire d'un capteur de température. Aucun biais introduit.
    **Stratégie :** Imputation simple (médiane, moyenne) acceptable.

    #### MAR — Missing At Random

    La probabilité de manquant **dépend des données observées**, pas des valeurs manquantes :

    $$P(\text{manquant} \mid X_{\text{obs}}, X_{\text{manq}}) = P(\text{manquant} \mid X_{\text{obs}})$$

    **Exemple :** Les hommes répondent moins aux questions sur leur poids (dépend du sexe — observé). Les clients `guest checkout` n'ont pas de `CustomerID` (dépend du mode d'achat — observable).
    **Stratégie :** KNN Imputer, IterativeImputer (MICE), ou indicatrice.

    #### MNAR — Missing Not At Random

    La probabilité de manquant **dépend de la valeur manquante elle-même** :

    $$P(\text{manquant} \mid X_{\text{obs}}, X_{\text{manq}}) = f(X_{\text{manq}})$$

    **Exemple :** Les patients très malades ne remplissent pas le questionnaire de santé — les cas les plus graves sont sous-représentés.
    **Stratégie :** Indicatrice de manquant + modèle de sélection. Expertise métier indispensable.

    > ⚠️ **Le MNAR ne peut pas être détecté uniquement à partir des données observées.** Il nécessite une connaissance du processus de collecte.
    """)
    return


@app.cell
def _retail_audit_manquants(df_retail):
    print("=== 🛒 ONLINE RETAIL — Audit des valeurs manquantes ===\n")
    _df_r = df_retail.copy()

    _mp_r = _df_r.isnull().sum()
    _pct_r = (_mp_r / len(_df_r) * 100).round(1)
    _diag = pd.DataFrame({"Manquants": _mp_r, "%": _pct_r}).sort_values("%", ascending=False)
    print(_diag.to_string())
    print()

    # Classification MCAR / MAR / MNAR
    _analyse = {
        "CustomerID" : ("~20%", "MAR",  "Guest checkout — dépend du mode d'achat (observable)"),
        "UnitPrice"  : ("~6%",  "MCAR", "Panne système aléatoire — indépendant des autres variables"),
        "Description": ("~5%",  "MCAR", "Libellé absent aléatoirement (problème export)"),
        "StockCode"  : ("~5%",  "MCAR", "Code article manquant aléatoirement"),
        "Country"    : ("~1%",  "MCAR", "Pays non renseigné (faible taux, négligeable)"),
    }
    print(f"{'Colonne':<14} {'% manq.':<10} {'Mécanisme':<10} Interprétation")
    print("─" * 78)
    for _col_r, (_pct, _mec, _expl) in _analyse.items():
        print(f"{_col_r:<14} {_pct:<10} {_mec:<10} {_expl}")

    # Visualisation
    _fig_r, _axes_r = plt.subplots(1, 2, figsize=(14, 4))
    _mp_nz = _pct_r[_pct_r > 0].sort_values()
    _colors_r = ["#e74c3c" if p > 30 else "#f39c12" if p > 10 else "#3498db" for p in _mp_nz]
    _mp_nz.plot(kind="barh", ax=_axes_r[0], color=_colors_r)
    _axes_r[0].axvline(30, color="red", linestyle="--", alpha=0.5, label="Seuil 30%")
    _axes_r[0].axvline(10, color="orange", linestyle="--", alpha=0.5, label="Seuil 10%")
    _axes_r[0].set_title("Online Retail — % valeurs manquantes", fontweight="bold")
    _axes_r[0].set_xlabel("% manquants")
    _axes_r[0].legend(fontsize=8)
    for _i_r, (_idx_r, _v_r) in enumerate(_mp_nz.items()):
        _axes_r[0].text(_v_r + 0.3, _i_r, f"{_v_r:.1f}%", va="center", fontsize=9)

    # CustomerID manquant par pays → preuve MAR
    _top5_r = _df_r["Country"].value_counts().head(5).index
    _sub5 = _df_r[_df_r["Country"].isin(_top5_r)]
    _pct_cid = _sub5.groupby("Country")["CustomerID"].apply(lambda x: x.isnull().mean() * 100).sort_values()
    _colors_cid = ["#2ecc71" if v < 15 else "#f39c12" if v < 25 else "#e74c3c" for v in _pct_cid]
    _axes_r[1].barh(_pct_cid.index, _pct_cid.values, color=_colors_cid, alpha=0.8, edgecolor="white")
    _axes_r[1].set_title("CustomerID manquant par pays — preuve MAR\n(dépend du marché → observable)", fontweight="bold")
    _axes_r[1].set_xlabel("% CustomerID manquant")
    for _i_r2, _v_r2 in enumerate(_pct_cid.values):
        _axes_r[1].text(_v_r2 + 0.3, _i_r2, f"{_v_r2:.1f}%", va="center", fontsize=9)
    _note_r = "Taux varie selon le pays\n→ dépend d'une variable observée\n→ mécanisme MAR confirmé"
    _axes_r[1].text(0.97, 0.97, _note_r, transform=_axes_r[1].transAxes,
                    va="top", ha="right", fontsize=9, style="italic",
                    bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.9))

    plt.suptitle("🛒 Online Retail — Diagnostic et mécanismes de données manquantes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/retail_manquants.png", dpi=120, bbox_inches="tight")
    plt.show()

    # Polars (si installé)
    try:
        import polars as _pl
        _df_pl_r = _pl.from_pandas(_df_r)
        print("\n📊 Polars — null_count() :")
        print(_df_pl_r.null_count())
    except ImportError:
        print("\n# Polars équivalent :")
        print("# df_pl.null_count()")
        print("# df_pl.select(pl.all().is_null().sum())")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Illustration des concepts MCAR, MAR et MNAR dans le dataset ci-dessous
    """)
    return


@app.cell
def _plot_mcar_mar_mnar():
    np.random.seed(0)
    _n2 = 200
    _age2 = np.random.normal(40, 12, _n2).clip(18, 80)
    _salaire2 = 1500 + 60 * _age2 + np.random.normal(0, 500, _n2)

    _fig2, _axes2 = plt.subplots(1, 3, figsize=(15, 5))
    _titres2 = ["MCAR\n(aléatoire complet)", "MAR\n(aléatoire conditionnel)", "MNAR\n(non aléatoire)"]
    _subtitres2 = [
        "Manquant indépendant de tout",
        "Manquant si âge < 30 (observé)",
        "Manquant si salaire élevé (manquant)"
    ]
    _mask_mcar2 = np.random.random(_n2) < 0.35
    _mask_mar2 = _age2 < 30
    _mask_mnar2 = _salaire2 > np.percentile(_salaire2, 65)

    for _ax2, _mask2, _titre2, _sous2 in zip(_axes2, [_mask_mcar2, _mask_mar2, _mask_mnar2], _titres2, _subtitres2):
        _ax2.scatter(_age2[~_mask2], _salaire2[~_mask2], alpha=0.6, s=25, color="#3498db", label="Observé")
        _ax2.scatter(_age2[_mask2], _salaire2[_mask2], alpha=0.6, s=25, color="#e74c3c", marker="x", label="Manquant")
        _ax2.set_xlabel("Âge (observé)")
        _ax2.set_ylabel("Salaire")
        _ax2.set_title(f"{_titre2}", fontweight="bold", fontsize=11)
        _ax2.text(0.5, -0.18, _sous2, transform=_ax2.transAxes, ha="center",
                  fontsize=9, color="gray", style="italic")
        _ax2.legend(fontsize=8)
        _n_m2 = _mask2.sum()
        _ax2.text(0.02, 0.97, f"Manquants : {_n_m2} ({_n_m2/_n2*100:.0f}%)",
                  transform=_ax2.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Trois mécanismes de données manquantes (Rubin, 1976)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("/tmp/mcar_mar_mnar.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _strategies_imputation(mo):
    mo.md(r"""
    ### 3.2 Stratégies d'imputation

    | Stratégie | Mécanisme adapté | Avantages | Inconvénients |
    |-----------|-----------------|-----------|---------------|
    | **Suppression** (listwise) | MCAR uniquement | Simple | Perte d'information |
    | **Médiane / Moyenne** | MCAR, MAR simple | Rapide | Réduit la variance |
    | **Mode** (catégoriel) | MCAR | Simple | Peut déséquilibrer |
    | **Forward/Backward fill** | Séries temporelles | Préserve la tendance | Mauvais sur longs gaps |
    | **KNN Imputer** | MAR | Utilise les voisins | Lent (grands datasets) |
    | **IterativeImputer** (MICE) | MAR | Modélise les relations | Complexe |
    | **Indicatrice de manquant** | MNAR | Capture le signal | Combinaison nécessaire |

    **Formule de l'imputation par la médiane :**

    $$x_{\text{imputé}} = \text{médiane}(x_1, x_2, \ldots, x_n)$$

    La médiane est préférée à la moyenne car elle est **robuste aux outliers**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sklearn.impute
    Le module  [**impute**](https://scikit-learn.org/stable/modules/impute.html) de sklearn offre un ensemble de fonctionnalités qui permet de traiter les valeurs manquantes : ***SimpleImputer***, ***KNNImputer***, ***IterativeImputer***.

     ***SimpleImputer*** : Cette classe applique une stratégie simple pour remplacer les valeurs manquantes par une statistique calculée sur chaque colonne (*mean*, *median*, *most_frequent*). Si la stratégie est *constant*, il faut ajouter le paramètre fill_value avec la valeur souhaitée.
    **Paramètres clés** : *strategy(stragie de remplacement)

    ***KNNImputer*** : Cette classe impute les valeurs manquantes à partir des k plus proches voisins. Pour chaque valeur manquante, elle calcule la moyenne (simple ou pondérée par la distance) des valeurs observées chez les *n_neighbors* échantillons les plus proches.
    **Paramètres clés** : *n_neighbors* (nombre de voisins, défaut 5) et *weights* (uniform ou distance).

    ***IterativeImputer*** : Cette classe  modélise chaque colonne avec valeurs manquantes comme une fonction des autres colonnes, via un modèle de régression (par défaut BayesianRidge), de manière itérative jusqu'à convergence.
    **Paramètres clés** : *estimator* (modèle de régression), *max_iter* (nombre d'itérations, défaut 10), *initial_strategy* (stratégie d'amorçage).
    """)
    return


@app.cell
def _demo_imputation(df_clean):
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    print("=== STRATÉGIES D'IMPUTATION ===\n")
    _prix3 = df_clean["UnitPrice"].copy() 
    print(f"UnitPrice — manquants : {_prix3.isnull().sum()} ({_prix3.isnull().sum()/len(_prix3)*100:.1f}%)")

    # 1. Médiane
    _imp_med = SimpleImputer(strategy="median")
    _imp_med.fit_transform(_prix3.values.reshape(-1, 1))
    print(f"\n1. Médiane  → valeur imputée = {_imp_med.statistics_[0]:.4f} €")

    # 2. Moyenne
    _imp_mean = SimpleImputer(strategy="mean")
    _imp_mean.fit_transform(_prix3.values.reshape(-1, 1))
    print(f"2. Moyenne  → valeur imputée = {_imp_mean.statistics_[0]:.4f} €")

    # 3. KNN (multi-colonnes pour illustration)
    _X3 = df_clean[["Quantity", "UnitPrice"]].copy().head(200)
    _imp_knn = KNNImputer(n_neighbors=5)
    _X3_knn = _imp_knn.fit_transform(_X3)
    print(f"3. KNN(k=5) → imputation basée sur les 5 voisins les plus proches")

    # 4. MICE (IterativeImputer)
    _imp_iter = IterativeImputer(max_iter=10, random_state=42)
    _X3_mice = _imp_iter.fit_transform(_X3)
    print(f"4. MICE     → modélisation itérative des relations entre variables")

    print("\n📌 Toujours fitter sur le TRAIN SET uniquement, jamais sur le test set !")

    imp_median_val = _imp_med.statistics_[0]
    return (imp_median_val,)


@app.cell
def _plot_imputation(df_clean, imp_median_val):
    _prix_obs4 = df_clean["UnitPrice"]
    _prix_med4 = _prix_obs4.fillna(imp_median_val)
    _prix_mean4 = _prix_obs4.fillna(_prix_obs4.mean())

    _fig4, _axes4 = plt.subplots(1, 3, figsize=(15, 4))
    _bins4 = np.linspace(0, 30, 40)

    for _ax4, _d4, _lbl4, _col4 in zip(
        _axes4,
        [_prix_obs4, _prix_med4, _prix_mean4],
        ["Originale", "Imputation médiane", "Imputation moyenne"],
        ["#95a5a6", "#2ecc71", "#e67e22"]
    ):
        _ax4.hist(_d4[_d4 <= 30], bins=_bins4, color=_col4, alpha=0.7, edgecolor="white")
        _ax4.axvline(_d4.median(), color="red", linestyle="--", label=f"Médiane={_d4.median():.2f}")
        _ax4.axvline(_d4.mean(), color="purple", linestyle=":", label=f"Moyenne={_d4.mean():.2f}")
        _ax4.set_title(_lbl4, fontweight="bold")
        _ax4.legend(fontsize=8)

    plt.suptitle("Impact de la stratégie d'imputation — UnitPrice", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/imputation.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _retail_imputation(df_clean):
    from sklearn.impute import SimpleImputer as _SI_t, KNNImputer as _KNN_t

    print("=== 🛒 ONLINE RETAIL — Stratégies d'imputation par colonne ===\n")
    _df_ri = df_clean[["Quantity", "UnitPrice", "CustomerID", "Country"]].copy()

    print(f"Manquants initiaux :")
    print(f"  UnitPrice  = {_df_ri['UnitPrice'].isnull().sum()} ({_df_ri['UnitPrice'].isnull().mean()*100:.1f}%) → MCAR → médiane")
    print(f"  CustomerID = {_df_ri['CustomerID'].isnull().sum()} ({_df_ri['CustomerID'].isnull().mean()*100:.1f}%) → MAR  → indicatrice")
    print(f"       = {_df_ri['Country'].isnull().sum()} ({_df_ri['Country'].isnull().mean()*100:.1f}%) → MCAR → mode")
    print()

    # UnitPrice — MCAR → médiane (simple, non biaisée pour MCAR)
    _prix_avant = _df_ri["UnitPrice"].copy()
    _imp_med_r = _SI_t(strategy="median")
    _df_ri["UnitPrice"] = _imp_med_r.fit_transform(_df_ri[["UnitPrice"]]).flatten()
    print(f"UnitPrice — Médiane : valeur imputée = {_imp_med_r.statistics_[0]:.4f} £")

    # UnitPrice — KNN pour comparaison (utilise Quantity comme voisin)
    _X_knn_r = _df_ri[["Quantity", "UnitPrice"]].copy()
    _X_knn_r["UnitPrice"] = _prix_avant  # restaurer les manquants
    _knn_r = _KNN_t(n_neighbors=5)
    _prix_knn = _knn_r.fit_transform(_X_knn_r)[:, 1]
    print(f"UnitPrice — KNN(k=5) : médiane avant={_prix_avant.median():.4f} | médiane après={pd.Series(_prix_knn).median():.4f}")

    #CustomerID — MAR → indicatrice (ne pas imputer une ID fictive)
    _df_ri["HasCustomerID"] = _df_ri["CustomerID"].notna().astype(int)
    print(f"\nCustomerID — Indicatrice créée : HasCustomerID")
    print(f"  Clients identifiés : {_df_ri['HasCustomerID'].sum()} ({_df_ri['HasCustomerID'].mean()*100:.1f}%)")
    print(f"  Guest checkout     : {(1-_df_ri['HasCustomerID']).sum()} ({(1-_df_ri['HasCustomerID']).mean()*100:.1f}%)")

    # Country — MCAR → mode
    _mode_country = _df_ri["Country"].mode()[0]
    _df_ri["Country"] = _df_ri["Country"].fillna(_mode_country)
    print(f"\nCountry — Mode : valeur imputée = '{_mode_country}' (MCAR)")
    print(f"\nManquants après imputation : {_df_ri.isnull().sum().sum()} (hors CustomerID conservé comme objet)")

    # Visualisation : Médiane vs KNN pour UnitPrice
    _fig_ri, _axes_ri = plt.subplots(1, 3, figsize=(15, 4))
    _bins_ri = np.linspace(0, 20, 40)

    _axes_ri[0].hist(_prix_avant.dropna()[_prix_avant.dropna() <= 20], bins=_bins_ri,
                     color="#95a5a6", alpha=0.8, edgecolor="white")
    _axes_ri[0].axvline(_prix_avant.median(), color="red", linestyle="--",
                        label=f"Médiane={_prix_avant.median():.2f} £")
    _axes_ri[0].set_title("UnitPrice — original\n(avec manquants)", fontweight="bold")
    _axes_ri[0].set_xlabel("UnitPrice (£)")
    _axes_ri[0].legend(fontsize=8)

    _prix_med_full = _prix_avant.fillna(_prix_avant.median())
    _axes_ri[1].hist(_prix_med_full[_prix_med_full <= 20], bins=_bins_ri,
                     color="#e67e22", alpha=0.8, edgecolor="white")
    _axes_ri[1].axvline(_prix_med_full.median(), color="red", linestyle="--",
                        label=f"Médiane={_prix_med_full.median():.2f} £")
    _axes_ri[1].set_title("UnitPrice — imputation médiane\n(pic artificiel à la médiane)", fontweight="bold")
    _axes_ri[1].set_xlabel("UnitPrice (£)")
    _axes_ri[1].legend(fontsize=8)

    _prix_knn_s = pd.Series(_prix_knn)
    _axes_ri[2].hist(_prix_knn_s[_prix_knn_s <= 20], bins=_bins_ri,
                     color="#2ecc71", alpha=0.8, edgecolor="white")
    _axes_ri[2].axvline(_prix_knn_s.median(), color="red", linestyle="--",
                        label=f"Médiane={_prix_knn_s.median():.2f} £")
    _axes_ri[2].set_title("UnitPrice — KNN imputation\n(distribution préservée)", fontweight="bold")
    _axes_ri[2].set_xlabel("UnitPrice (£)")
    _axes_ri[2].legend(fontsize=8)

    plt.suptitle("🛒 Online Retail — KNN vs Médiane pour l'imputation du UnitPrice", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/retail_imputation.png", dpi=120, bbox_inches="tight")
    plt.show()

    # Polars équivalent
    try:
        import polars as _pl
        _df_pl_ri = _pl.from_pandas(df_clean[["UnitPrice", "Country"]].head(100))
        _mode_pl = _df_pl_ri["Country"].drop_nulls().mode()[0]
        _df_pl_ri = _df_pl_ri.with_columns(
            _pl.col("UnitPrice").fill_null(_pl.col("UnitPrice").median()),
            _pl.col("Country").fill_null(_mode_pl),
        )
        print("\n📊 Polars — imputation en une expression :")
        print(_df_pl_ri.null_count())
    except ImportError:
        print("\n# Polars — imputation en une expression :")
        print("# df_pl.with_columns(")
        print("#     pl.col('UnitPrice').fill_null(pl.col('UnitPrice').median()),")
        print("#     pl.col('Country').fill_null(pl.col('Country').drop_nulls().mode().first()),")
        print("# )")

    df_retail_clean = _df_ri
    return (df_retail_clean,)


@app.cell
def _(df_retail_clean):
    df_retail_clean.head()
    return


@app.cell(hide_code=True)
def _retail_cat_imputation_theory(mo):
    mo.md(r"""
    ### 3.3 Imputation des variables catégorielles — Online Retail

    Les variables catégorielles ne peuvent pas être imputées par la moyenne ou la médiane (ces notions ne s'appliquent pas à des catégories textuelles). La stratégie standard est le **mode** (`most_frequent`) : on remplace chaque valeur manquante par la catégorie la plus fréquente dans la colonne.

    **Colonnes catégorielles du dataset Online Retail :**

    | Colonne | Type | Manquants | Stratégie |
    |---------|------|-----------|-----------|
    | `Country` | Pays de livraison | ~1% | `most_frequent` → **United Kingdom** (MCAR) |
    | `Description` | Libellé produit | ~5% | `most_frequent` ou suppression (MCAR) |
    | `StockCode` | Code article | ~5% | `most_frequent` ou suppression (MCAR) |
    | `InvoiceNo` | Numéro de facture | 0 | Pas de manquants |
    | `CustomerID` | Identifiant client | ~20% | Indicatrice `HasCustomerID` (MAR — ne pas imputer une ID fictive) |

    > **Pourquoi `SimpleImputer(strategy="most_frequent")` ?**
    > `sklearn` uniformise le traitement : même API pour les numériques et les catégorielles.
    > Cela facilite l'intégration dans un `Pipeline` ou un `ColumnTransformer` en production.

    > ⚠️ Pour `CustomerID`, **ne jamais imputer** une valeur fictive — une fausse ID regroupera artificiellement des clients distincts. La bonne approche est de créer une indicatrice `HasCustomerID` (section 3) et de traiter séparément les transactions anonymes.

    > ⚠️ Pour `Description`, le mode (`most_frequent`) est acceptable si le taux de manquants est < 5 %. Au-delà, préférez la suppression ou une jointure sur `StockCode` pour récupérer le libellé depuis une table produit.
    """)
    return


@app.cell
def _retail_cat_imputation(df_clean):
    from sklearn.impute import SimpleImputer as _SI_cat

    print("=== 🛒 ONLINE RETAIL — Imputation des variables catégorielles ===\n")

    _cat_features = ["StockCode", "Description", "Country"]
    _retail_cat = df_clean[_cat_features].copy()

    print("Aperçu des données catégorielles (5 premières lignes) :")
    print(_retail_cat.head().to_string())

    print("\nValeurs manquantes AVANT imputation :")
    print(_retail_cat.isnull().sum().to_string())

    # Imputation par le mode (valeur la plus fréquente)
    _imputer_cat = _SI_cat(strategy="most_frequent")
    _retail_cat_imp = _imputer_cat.fit_transform(_retail_cat)
    _retail_cat_imp = pd.DataFrame(
        _retail_cat_imp,
        columns=_retail_cat.columns,
        index=_retail_cat.index,
    )

    print("\nValeurs manquantes APRÈS imputation :")
    print(_retail_cat_imp.isnull().sum().to_string())

    print("\nValeurs imputées par colonne (mode) :")
    for _col_c, _stat_c in zip(_cat_features, _imputer_cat.statistics_):
        print(f"  {_col_c:<14} → mode = '{_stat_c}'")

    print("\nAperçu après imputation :")
    print(_retail_cat_imp.head().to_string())

    print("\n⚠️ Note : CustomerID n'est PAS imputé ici — on crée une indicatrice HasCustomerID.")
    print("   Imputer une ID fictive regrouperait artificiellement des clients distincts.")
    return


@app.cell(hide_code=True)
def _section4(mo):
    mo.md(r"""
    ---
    ## 4. Détection des outliers

    Un **outlier** est une observation significativement différente des autres, souvent ce sont des valeurs abérantes extrêmes.

    Les outliers peuvent être des erreurs de saisie (à corriger), des événements rares légitimes (fraude, pic de ventes) ou des signaux importants à analyser séparément.

    ### 4.1 Méthode IQR (robuste, sans hypothèse de distribution)

    $$IQR = Q_3 - Q_1$$

    $$\text{Borne inf} = Q_1 - 1.5 \times IQR \qquad \text{Borne sup} = Q_3 + 1.5 \times IQR$$

    Toute valeur hors de cet intervalle est un outlier. Utiliser $3 \times IQR$ pour une détection plus conservatrice.

    ### 4.2 Z-score (pour distributions approximativement normales)

    $$z_i = \frac{x_i - \bar{x}}{\sigma}$$

    Avec $\bar{x}$ : la moyenne
    Et $\sigma$ : L'écar type

    **Règle :** Si $|z_i| > 3$, la valeur est un outlier (règle des 3 sigma). Cette méthode suppose une distribution normale — elle est sensible aux outliers eux-mêmes (problème de circularité).

    ### 4.3 Isolation Forest

    Algorithme non supervisé basé sur des arbres de décision aléatoires.
    Les outliers sont **isolés plus rapidement** que les points normaux car ils sont dans des régions peu denses.
    Particulièrement adapté aux données **haute dimension** ou aux distributions non normales.
    """)
    return


@app.cell
def _demo_outliers(df_clean):
    from scipy import stats as _scipy_stats
    from sklearn.ensemble import IsolationForest

    print("=== DÉTECTION DES OUTLIERS — Quantity ===\n")
    _qty5 = df_clean["Quantity"].dropna()

    # IQR
    _Q1_5 = _qty5.quantile(0.25)
    _Q3_5 = _qty5.quantile(0.75)
    _IQR5 = _Q3_5 - _Q1_5
    _b_inf5 = _Q1_5 - 1.5 * _IQR5
    _b_sup5 = _Q3_5 + 1.5 * _IQR5
    _out_iqr5 = (_qty5 < _b_inf5) | (_qty5 > _b_sup5)
    print(f"IQR = {_IQR5:.1f}  |  Q1 = {_Q1_5:.1f}  |  Q3 = {_Q3_5:.1f}")
    print(f"Bornes IQR : [{_b_inf5:.1f}, {_b_sup5:.1f}]")
    print(f"Outliers IQR : {_out_iqr5.sum()} ({_out_iqr5.sum()/len(_qty5)*100:.1f}%)")

    # Z-score
    _z5 = np.abs(_scipy_stats.zscore(_qty5))
    _out_z5 = _z5 > 3
    print(f"\nOutliers Z-score (|z|>3) : {_out_z5.sum()} ({_out_z5.sum()/len(_qty5)*100:.1f}%)")

    # Isolation Forest
    _iso5 = IsolationForest(contamination=0.05, random_state=42)
    _pred5 = _iso5.fit_predict(_qty5.values.reshape(-1, 1))
    _out_iso5 = _pred5 == -1
    print(f"Outliers Isolation Forest : {_out_iso5.sum()} ({_out_iso5.sum()/len(_qty5)*100:.1f}%)")

    qty_clean = _qty5
    out_iqr = _out_iqr5
    b_inf = _b_inf5
    b_sup = _b_sup5
    out_iso = _out_iso5
    Q1_qty = _Q1_5
    Q3_qty = _Q3_5
    return b_inf, b_sup, out_iqr, out_iso, qty_clean


@app.cell
def _plot_outliers(b_inf, b_sup, out_iqr, out_iso, qty_clean):
    _fig6, _axes6 = plt.subplots(1, 3, figsize=(16, 5))

    # Boxplot
    _ax6a = _axes6[0]
    _ax6a.boxplot(qty_clean, vert=True, patch_artist=True,
                  boxprops=dict(facecolor="#3498db", alpha=0.6),
                  flierprops=dict(marker="o", markerfacecolor="#e74c3c", markeredgecolor="#e74c3c", markersize=4, alpha=0.7))
    _ax6a.set_title("Boxplot", fontweight="bold")
    _ax6a.set_ylabel("Quantity")
    _ax6a.axhline(b_inf, color="orange", linestyle="--", alpha=0.7, label="Bornes IQR")
    _ax6a.axhline(b_sup, color="orange", linestyle="--", alpha=0.7)
    _outlier_patch = plt.matplotlib.lines.Line2D(
        [], [], marker="o", color="w", markerfacecolor="#e74c3c",
        markersize=6, label=f"Outliers IQR (n={out_iqr.sum()})")
    _iqr_line = plt.matplotlib.lines.Line2D(
        [], [], color="orange", linestyle="--", label="Bornes IQR")
    _ax6a.legend(handles=[_outlier_patch, _iqr_line], fontsize=8)

    # Distribution + seuils IQR
    _ax6b = _axes6[1]
    _in6 = qty_clean[~out_iqr]
    _out6 = qty_clean[out_iqr]
    _ax6b.hist(_in6, bins=40, color="#3498db", alpha=0.7, label="Normal")
    _ax6b.hist(_out6, bins=20, color="#e74c3c", alpha=0.9, label=f"Outliers IQR (n={out_iqr.sum()})")
    _ax6b.axvline(b_inf, color="orange", linestyle="--", linewidth=2, label="Bornes IQR")
    _ax6b.axvline(b_sup, color="orange", linestyle="--", linewidth=2)
    _ax6b.set_title("Distribution des quantités", fontweight="bold")
    _ax6b.set_xlabel("Quantity")
    _ax6b.set_xlim(-100, 250)
    _ax6b.legend(fontsize=8)

    # Comparaison méthodes
    _ax6c = _axes6[2]
    _methods6 = ["IQR", "Z-score\n(|z|>3)", "Isolation\nForest"]
    _counts6 = [out_iqr.sum(),
                int((np.abs((qty_clean - qty_clean.mean()) / qty_clean.std()) > 3).sum()),
                out_iso.sum()]
    _bars6 = _ax6c.bar(_methods6, _counts6, color=["#3498db", "#2ecc71", "#9b59b6"], alpha=0.8, edgecolor="white")
    _ax6c.set_title("Outliers détectés par méthode", fontweight="bold")
    _ax6c.set_ylabel("Nombre")
    for _b6, _c6 in zip(_bars6, _counts6):
        _ax6c.text(_b6.get_x() + _b6.get_width()/2, _b6.get_height() + 0.5,
                   str(_c6), ha="center", fontweight="bold")

    plt.suptitle("Comparaison des méthodes de détection d'outliers — Quantity", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/outliers.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _section_td(mo):
    mo.md(r"""
    ---
    ## 5. TD/TP A Rendre — Exercices de nettoyage sur le dataset Titanic

    **A Rendre au plus tard lundi 22/06/2026 avant 23h 59.***
    **Mail** : mboup.djibril@ugb.edu.sn

    ### Contexte métier

    Vous êtes data analyst pour une compagnie maritime. On vous fournit le manifeste des passagers du RMS Titanic (1 309 passagers, 14 colonnes). Votre mission : **nettoyer et préparer les données dans le but d'analyser les chances de survie de ce nauffrage**.

    | Colonne | Description |
    |---------|-------------|
    | `pclass` | Classe du billet (1 = 1ère, 2 = 2ème, 3 = 3ème) |
    | `survived` | Survie (1 = oui, 0 = non) |
    | `name` | Nom complet du passager |
    | `sex` | Sexe |
    | `age` | Âge en années |
    | `sibsp` | Nombre de frères/sœurs ou conjoint(e)s à bord |
    | `parch` | Nombre de parents/enfants à bord |
    | `ticket` | Numéro de ticket |
    | `fare` | Tarif payé (£) |
    | `cabin` | Numéro de cabine |
    | `embarked` | Port d'embarquement (S = Southampton, C = Cherbourg, Q = Queenstown) |
    | `boat` | Numéro du canot de sauvetage (si survivant) |
    | `body` | Numéro d'identification du corps (si décédé retrouvé) |
    | `home.dest` | Ville/destination de résidence |

    ---

    ### Exercice 1 — Audit du dataset

    Chargez le dataset et analysez son état de santé :

    ```python
    df_titanic = pd.read_csv("datasets/raw/titanic.csv")
    ```

    Répondez aux questions suivantes :

    1. Combien de lignes et de colonnes contient le dataset ?
    2. Quelle colonne présente le taux de valeurs manquantes le plus élevé ? Pourquoi est-ce logique ?
    3. Quels sont les types de données de chaque colonne ? Y a-t-il des incohérences ?
    4. Combien de doublons y a-t-il ?

    ---

    ### Exercice 2 — Mécanismes de données manquantes

    Pour chaque colonne ci-dessous, identifiez le mécanisme (MCAR / MAR / MNAR) et proposez une stratégie d'imputation :

    | Colonne | % Manquants | Contexte | Mécanisme ? | Votre stratégie ? |
    |---------|------------|----------|-------------|-------------------|
    | `age` | ~20% | Âge non renseigné sur le manifeste | ? | ? |
    | `cabin` | ~77% | Numéro de cabine | ? | ? |
    | `embarked` | <1% | Port d'embarquement | ? | ? |
    | `fare` | <1% | Tarif du billet | ? | ? |
    | `boat` | ~63% | Numéro du canot de sauvetage | ? | ? |
    | `body` | ~90% | Numéro du corps retrouvé | ? | ? |

    > **Indice pour `boat` et `body` :** réfléchissez à ce que signifie une valeur manquante dans ces colonnes par rapport à la variable `survived`.

    ---

    ### Exercice 3 — Comparer deux stratégies de nettoyage

    **Stratégie A :** Supprimer toutes les lignes avec au moins une valeur manquante (sur les colonnes `age`, `embarked`, `fare`).

    **Stratégie B :**
    - Imputer `age` par la **médiane par classe et par sexe** (MAR)
    - Imputer `embarked` par le **mode** (MCAR, 2 valeurs seulement)
    - Imputer `fare` par la **médiane** (MCAR, 1 valeur seulement)
    - Créer une indicatrice `has_cabin` (1 si `cabin` renseigné, 0 sinon)

    Pour chaque stratégie, calculez :
    - Le nombre de passagers conservés et le % de données perdues
    - Le taux de survie global (`survived.mean()`)
    - La répartition homme/femme conservée

    > Comparez les deux taux de survie : la stratégie A introduit-elle un biais ?

    ---

    ### Exercice 4 — Détection et traitement des outliers

    1. Appliquez la méthode IQR sur la colonne `fare`. Combien d'outliers détectez-vous ?
    2. Ces outliers sont-ils des erreurs ou des valeurs légitimes ? Justifiez.
    3. Comparez avec le Z-score (seuil |z| > 3). Y a-t-il une différence ?
    4. Quelle stratégie adopteriez-vous : suppression, remplacement, ou conservation ?

    ---

    ### Exercice 5 — Discussion

    1. Les données manquantes dans `body` sont-elles MCAR, MAR ou MNAR ? Expliquez le lien avec `survived`.
    2. Si on supprime tous les passagers sans `cabin`, quel biais introduit-on par rapport à `pclass` ?
    3. Faut-il conserver les colonnes `boat`, `body`, `ticket`, `name` pour un modèle de prédiction ? Pourquoi ?
    4. L'imputation de `age` par la médiane globale vs la médiane par `pclass`+`sex` — quelle différence observez-vous ?
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
