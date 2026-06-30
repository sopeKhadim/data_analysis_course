# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.9",
#     "setuptools",
#     "scikit-learn",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="TP/TD Séance 2 — Automobile Dataset",
)


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _header(mo):
    mo.md(r"""
    # 🚗 TP/TD — A Rendre
    ## Dataset : Automobile — UCI Machine Learning Repository

    **Cours : Collecte, Nettoyage, Préparation et Échantillonnage**

    L'objectif de ce TP est de vous permettre à apprendre et comprendre les concepts oborder dans le cours. Vous pouvez travailler par binome et soumettre le projet dans un repos GiHub que vous allez me partager avec la liste des participants.

    **Note Important** : Chaque réponse aux questions doit être commentée et justifiée pour un resultat basé sur un code python qui sera produit.


    **Date limite de rendu : lundi 06/07/2026 avant 23h59**

    **Mail** : mboup.djibril@ugb.edu.sn

    ---


    ## Contexte métier

    Vous êtes data analyst dans une entreprise d'assurance automobile. On vous fournit le dataset **Automobile** du UCI ML Repository, qui recense les caractéristiques techniques et tarifaires de **205 véhicules** de différentes marques (1985).

    Votre mission : **nettoyer, préparer et analyser ces données** pour permettre à l'équipe actuarielle de modéliser le risque d'assurance (`symboling`) en fonction des caractéristiques du véhicule.

    ## Colonnes du dataset

    | Colonne | Type | Description |
    |---------|------|-------------|
    | `symboling` | int | Risque d'assurance (-3 = sûr, +3 = risqué) |
    | `normalized-losses` | float | Pertes normalisées annuelles (valeur cible actuarielle) |
    | `make` | str | Marque du véhicule |
    | `fuel-type` | str | Type de carburant (`gas` / `diesel`) |
    | `aspiration` | str | Aspiration moteur (`std` / `turbo`) |
    | `num-of-doors` | str | Nombre de portes (`two` / `four`) |
    | `body-style` | str | Type de carrosserie |
    | `drive-wheels` | str | Roues motrices (`fwd` / `rwd` / `4wd`) |
    | `engine-location` | str | Emplacement moteur (`front` / `rear`) |
    | `wheel-base` | float | Empattement (pouces) |
    | `length` / `width` / `height` | float | Dimensions (pouces) |
    | `curb-weight` | int | Poids à vide (livres) |
    | `engine-type` | str | Type de moteur |
    | `num-of-cylinders` | str | Nombre de cylindres (en lettres) |
    | `engine-size` | int | Cylindrée (cm³) |
    | `fuel-system` | str | Système d'alimentation |
    | `bore` / `stroke` | float | Alésage / Course (pouces) |
    | `compression-ratio` | float | Taux de compression |
    | `horsepower` | float | Puissance (chevaux) |
    | `peak-rpm` | float | Régime maximal (tr/min) |
    | `city-mpg` / `highway-mpg` | int | Consommation ville / route (miles/gallon) |
    | `price` | float | Prix du véhicule ($) |

    > ⚠️ **Particularité :** Les valeurs manquantes sont encodées par `"?"` dans ce dataset — à gérer au chargement.

    ## Plan du TP/TD

    | Partie | Contenu |
    |--------|---------|
    | **Exercice 1** | Chargement et audit des données |
    | **Exercice 2** | Mécanismes de données manquantes (MCAR / MAR / MNAR) |
    | **Exercice 3** | Stratégies d'imputation |
    | **Exercice 4** | Détection et traitement des outliers |
    | **Exercice 5** | Normalisation et standardisation |
    | **Exercice 6** | Pipeline complet (ColumnTransformer) |
    | **Exercice 7** | Échantillonnage |
    | **Exercice 8** | Discussion et synthèse |



    ---

    ## Chargement du dataset

    Téléchargez le dataset depuis :
    `https://archive.ics.uci.edu/static/public/10/automobile.zip`

    Extrayez le fichier `imports-85.data` puis chargez-le comme suit :

    ```python
    COLUMNS = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
        "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "wheel-base", "length", "width", "height", "curb-weight",
        "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
        "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
        "city-mpg", "highway-mpg", "price"
    ]

    df = pd.read_csv(
        "imports-85.data",
        names=COLUMNS,        # pas d'en-tête dans le fichier
        na_values=["?"],      # "?" → NaN
        sep=","
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _ex1_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 1 — Audit du dataset

    ### Questions

    **1.1** Affichez les dimensions du dataset (nombre de lignes et de colonnes).

    **1.2** Quels sont les types de données (`dtypes`) de chaque colonne ? Y a-t-il des colonnes
    dont le type vous semble incohérent avec le contenu attendu ? Lesquelles et pourquoi ?

    **1.3** Calculez le nombre et le pourcentage de valeurs manquantes par colonne.
    Identifiez les 3 colonnes les plus touchées.

    **1.4** Combien de doublons y a-t-il dans le dataset ?

    **1.5** Affichez les statistiques descriptives  pour les colonnes numériques.
    Relevez une observation sur la colonne `compression-ratio`.

    > **Indice 1.2 :** Les colonnes `horsepower`, `peak-rpm`, `bore`, `stroke` et `price` devraient
    > être numériques. Vérifiez leur type après chargement avec `na_values=["?"]`.

    > **Indice 1.5 :** Comparez `min`, `max` et `mean` de `compression-ratio`. Une bimodalité
    > peut indiquer deux populations distinctes dans les données.
    """)
    return


@app.cell(hide_code=True)
def _ex2_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 2 — Mécanismes de données manquantes

    Pour chaque colonne ci-dessous, **identifiez le mécanisme** (MCAR / MAR / MNAR) et
    **proposez une stratégie d'imputation**. Justifiez votre réponse avec des calculs.

    | Colonne | % Manquants | Contexte | Mécanisme ? | Stratégie ? |
    |---------|:-----------:|----------|:-----------:|:-----------:|
    | `normalized-losses` | ~20% | Pertes normalisées (calcul actuariel complexe) | ? | ? |
    | `num-of-doors` | ~1% | Nombre de portes (`two`/`four`) | ? | ? |
    | `bore` | ~2% | Alésage moteur (donnée technique précise) | ? | ? |
    | `stroke` | ~2% | Course moteur (donnée technique précise) | ? | ? |
    | `horsepower` | ~1% | Puissance moteur | ? | ? |
    | `peak-rpm` | ~1% | Régime maximal | ? | ? |
    | `price` | ~2% | Prix catalogue ($) | ? | ? |

    ### Questions

    **2.1** Vérifiez si `normalized-losses` est manquant de façon homogène entre les marques
    (`make`). Calculez le taux de manquants par marque (`groupby`). Que concluez-vous sur
    le mécanisme ?

    **2.2** Vérifiez si `price` manque de façon corrélée avec `make` ou `body-style`.
    Le manquant dans `price` dépend-il d'une variable observée ?

    **2.3** Les colonnes `bore`, `stroke`, `horsepower` et `peak-rpm` présentent-elles
    des patterns communs dans leurs données manquantes ? Utilisez une corrélation de
    co-occurrence (`df.isnull().corr()`).

    > **Rappel :**
    > - **MCAR** — manquant indépendant de tout → imputation simple (médiane/mode)
    > - **MAR** — manquant dépend d'une variable observée → KNN, MICE, ou médiane par groupe
    > - **MNAR** — manquant dépend de la valeur elle-même → indicatrice + expertise métier
    """)
    return


@app.cell(hide_code=True)
def _ex3_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 3 — Stratégies d'imputation

    Appliquez les stratégies identifiées dans l'Exercice 2.

    ### Questions

    **3.1** Imputation de `normalized-losses` :
    - Imputez par la **médiane par marque** (`groupby + transform`)
    - Si une marque n'a aucune valeur observée (médiane de groupe = NaN), imputez par la médiane globale
    - Comparez la distribution avant et après imputation (histogramme)

    **3.2** Imputation de `num-of-doors` :
    - Identifiez le mode de `num-of-doors`
    - Imputez les 2 valeurs manquantes par ce mode
    - Vérifiez qu'il ne reste plus de manquants

    **3.3** Imputation de `horsepower` et `peak-rpm` :
    - Utilisez `SimpleImputer(strategy="median")` de scikit-learn
    - Comparez avec `KNNImputer(n_neighbors=5)` en utilisant `["horsepower", "peak-rpm", "engine-size"]`
    - Affichez la valeur imputée par chaque méthode

    **3.4** Imputation de `price` :
    - Imputez par la **médiane par `make`**
    - Créez une indicatrice `price_imputed` (1 si la valeur était manquante, 0 sinon) *avant* l'imputation
    - Pourquoi cette indicatrice peut être utile pour un modèle ?

    **3.5** Créez le dataframe `df_imputed` qui rassemble toutes les imputations ci-dessus.
    Vérifiez que `df_imputed.isnull().sum().sum() == 0` (hors colonnes volontairement exclues).

    > **Rappel :** Toujours `.fit()` sur le train set uniquement. Ici, pour simplifier, on
    > travaille sur l'ensemble complet — en production, on séparerait train/test avant d'imputer.
    """)
    return


@app.cell(hide_code=True)
def _ex4_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 4 — Détection et traitement des outliers

    ### Questions

    **4.1** Appliquez la méthode **IQR** sur les colonnes `price`, `horsepower` et
    `normalized-losses`. Pour chacune :
    - Calculez Q1, Q3 et IQR
    - Calculez les bornes inférieure et supérieure
    - Comptez le nombre d'outliers détectés

    **4.2** Appliquez le **Z-score** (seuil |z| > 3) sur les mêmes colonnes.
    Comparez les résultats avec l'IQR. Quelle méthode détecte le plus d'outliers ?
    Pourquoi ?

    **4.3** Appliquez l'**Isolation Forest** (`contamination=0.05`) sur les colonnes
    numériques `["price","horsepower","engine-size","curb-weight","normalized-losses"]`.
    Combien d'outliers multivariés détecte-t-il ?

    **4.4** Analysez les outliers de `price` :
    - À quelles marques appartiennent-ils ?
    - Sont-ils des erreurs ou des valeurs légitimes ?
    - Quelle stratégie adoptez-vous : suppression, remplacement ou conservation ?

    **4.5** Visualisez la distribution de `price` avant et après traitement des outliers
    (boxplot + histogramme).
    """)
    return


@app.cell(hide_code=True)
def _ex5_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 5 — Normalisation et Standardisation

    ### Questions

    **5.1** Sélectionnez les colonnes numériques continues :
    `["price", "horsepower", "engine-size", "curb-weight", "normalized-losses",
    "city-mpg", "highway-mpg", "wheel-base", "length", "width", "height"]`

    Affichez leurs plages de valeurs (`min`, `max`, `std`). Pourquoi la mise à l'échelle
    est-elle nécessaire pour ces variables ?

    **5.2** Appliquez les trois scalers sur `price` :
    - `MinMaxScaler` → plage [0, 1]
    - `StandardScaler` → μ = 0, σ = 1
    - `RobustScaler` → médiane/IQR

    Comparez les distributions obtenues (histogramme, moyenne, écart-type).

    **5.3** Pour le dataset Automobile (avec des outliers légitimes comme les voitures de luxe),
    quel scaler recommandez-vous ? Justifiez.

    **5.4** Appliquez le `RobustScaler` sur toutes les colonnes numériques continues
    (listées en 5.1). Vérifiez que les données scalées ont une médiane ≈ 0 pour chaque colonne.

    **5.5** La colonne `city-mpg` est négativement corrélée avec `horsepower` dans les données
    originales. Vérifiez que cette corrélation est **préservée** après scaling (calcul de
    corrélation de Pearson avant et après).
    """)
    return


@app.cell(hide_code=True)
def _ex6_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 6 — Pipeline complet (ColumnTransformer)

    En production, le nettoyage doit être **reproductible** et **sans data leakage**.
    On utilise un `Pipeline` scikit-learn.

    ### Questions

    **6.1** Définissez deux listes de colonnes :
    - `num_features` : toutes les colonnes numériques continues (voir Exercice 5)
    - `cat_features` : `["fuel-type", "aspiration", "body-style", "drive-wheels", "num-of-doors"]`

    **6.2** Construisez un `ColumnTransformer` avec :
    - **Sous-pipeline numérique** : `SimpleImputer(median)` → `RobustScaler()`
    - **Sous-pipeline catégoriel** : `SimpleImputer(most_frequent)`

    **6.3** Séparez le dataset en train (80%) et test (20%) avec `train_test_split(random_state=42)`.
    Fittez le pipeline sur le **train set uniquement**, puis transformez train et test.

    **6.4** Sauvegarder les données préparées en format csv.

    **6.5** Vérifiez que :
    - Il n'y a plus de valeurs manquantes dans les sets transformés
    - Les dimensions sont cohérentes (nombre de colonnes attendu après OneHotEncoding)

    **6.5** Expliquez pourquoi on ne doit jamais fitter le pipeline sur le test set.
    Quel est le risque (terme technique) ?
    """)
    return


@app.cell(hide_code=True)
def _ex7_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 7 — Échantillonnage

    Le dataset Automobile est petit (205 lignes). Pour illustrer les méthodes d'échantillonnage,
    nous allons simuler une situation réaliste.

    ### Questions

    **7.1 — Échantillonnage aléatoire simple (SRS)**

    Tirez un échantillon de **n = 50** véhicules sans remise (`replace=False`).
    - Comparez la distribution de `symboling` (risque d'assurance) dans l'échantillon vs la population
    - Calculez le taux de couverture par `fuel-type` (essence vs diesel)
    - L'échantillon est-il représentatif ? Pourquoi peut-il ne pas l'être ?

    **7.2 — Échantillonnage stratifié**

    Tirez un échantillon stratifié de **n = 50** véhicules en stratifiant sur `fuel-type`.
    Utilisez `train_test_split` avec `stratify=df["fuel-type"]` ou implémentez manuellement.
    - Comparez la répartition `fuel-type` avec le SRS de la question 7.1
    - Dans quel cas l'échantillonnage stratifié est-il préférable au SRS ?

    **7.3 — Échantillonnage par grappes**

    Considérez que chaque **marque** (`make`) est une grappe.
    - Il y a combien de marques distinctes ?
    - Tirez aléatoirement **5 marques** (sans remise), puis récupérez tous les véhicules
      de ces marques. Combien de véhicules obtenez-vous ?
    - Calculez le taux de couverture de chaque `body-style` dans l'échantillon
    - Quel est l'inconvénient majeur de cet échantillonnage ?

    **7.4 — Biais d'échantillonnage**

    Imaginez qu'un enquêteur ne collecte que les véhicules dont `price > 20 000 $`.
    - Quel pourcentage du dataset cela représente-t-il ?
    - Comparez le taux de `symboling ≥ 2` (très risqué) dans cet échantillon vs la population
    - Quel type de biais cela introduit-il ?

    **7.5 — Classes déséquilibrées**

    La variable `symboling` contient des classes déséquilibrées.
    - Affichez la distribution de `symboling`
    - Quelle est la proportion de la classe majoritaire ?
    - Proposez deux stratégies pour gérer ce déséquilibre lors de la modélisation
      (sans SMOTE — algorithme vu en cours)
    """)
    return


@app.cell(hide_code=True)
def _ex8_enonce(mo):
    mo.md(r"""
    ---
    ## Exercice 8 — Discussion et Synthèse

    ### Questions de réflexion

    **8.1** La colonne `num-of-cylinders` contient des valeurs textuelles (`"four"`, `"six"`, etc.)
    alors qu'elle représente un nombre. Proposez deux approches pour le convertir en valeur numérique.


    **8.2** La colonne `compression-ratio` présente une distribution bimodale (pic autour de 9
    pour les moteurs essence, pic autour de 22 pour les diesels). Faut-il la normaliser avec
    `StandardScaler` ou `RobustScaler` ? Ou proposez une meilleure transformation.

    **8.3** On souhaite prédire le `symboling` (risque d'assurance) d'un véhicule.
    Parmi les colonnes suivantes, lesquelles faut-il **exclure** du modèle et pourquoi ?

    | Colonne | Exclure ? | Raison ? |
    |---------|:---------:|----------|
    | `normalized-losses` | ? | ? |
    | `price` | ? | ? |
    | `make` | ? | ? |
    | `city-mpg` | ? | ? |
    | `horsepower` | ? | ? |

    **8.4** Vous avez appliqué un `RobustScaler` sur `price`. Un nouveau véhicule arrive avec
    `price = 45 000 $`. Comment appliquer la transformation sans re-fitter le scaler ?
    Quelle erreur classique faut-il éviter ?

    **8.5** Résumez en 5 étapes le pipeline de nettoyage complet que vous avez construit
    sur ce dataset. Pour chaque étape, précisez l'outil scikit-learn utilisé.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
