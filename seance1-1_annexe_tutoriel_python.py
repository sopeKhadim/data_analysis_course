import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial — Quickstart with Python

    **Cours : Analyse de données — IPSL · 2025–2026**

    Ce tutoriel est organisé en trois parties :

    | Partie | Contenu |
    |--------|---------|
    | **Partie 1** | Marimo — concepts clés du notebook réactif |
    | **Partie 2** | Bases de Python — types, structures, fonctions, POO |
    | **Partie 3** | Librairies — numpy, pandas, polars, seaborn, statsmodels, scikit-learn |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Partie 1 · Marimo — Le Notebook Réactif

    ## Pourquoi Marimo ?

    Marimo est une alternative moderne à Jupyter conçue pour corriger ses défauts fondamentaux.

    | Problème Jupyter | Solution Marimo |
    |-----------------|-----------------|
    | Cellules exécutées dans n'importe quel ordre → résultats incohérents | Exécution **déterministe** : chaque cellule s'exécute dans l'ordre du graphe de dépendances |
    | État caché entre les cellules | Pas d'état global partagé : chaque variable n'appartient qu'à **une seule cellule** |
    | Pas de réactivité native | Re-exécution **automatique** des cellules dépendantes quand une valeur change |
    | Difficile à versionner (format JSON) | Fichier `.py` pur, lisible, diffable avec `git` |

    ## Concepts fondamentaux

    ### 1 · Les cellules et le graphe de dépendances

    Marimo analyse les **entrées** (variables consommées) et **sorties** (variables produites)
    de chaque cellule pour construire un **DAG** (graphe acyclique dirigé).

    ```
    Cellule A  →  produit  x
    Cellule B  →  consomme x, produit y
    Cellule C  →  consomme y
    ```
    Si A change → B et C sont **automatiquement** re-exécutées.

    ### 2 · Règle : une variable = une cellule

    Une variable ne peut être **définie que dans une seule cellule**.
    Si deux cellules définissent `x`, marimo signale une erreur `multiple-definitions`.

    > **Astuce :** Préfixer avec `_` (ex: `_temp`) rend la variable **privée** à sa cellule
    > et évite les conflits.

    ### 3 · `mo` — L'objet marimo

    `mo` est l'objet central de marimo. Il fournit :
    - `mo.md()` : rendu Markdown / LaTeX
    - `mo.ui.*` : widgets interactifs (slider, dropdown, checkbox…)
    - `mo.stat()` : affichage de statistiques
    - `mo.hstack()` / `mo.vstack()` : mise en page des éléments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.1 · `mo.md()` — Texte, Markdown et LaTeX
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    `mo.md()` permet d'écrire du **Markdown enrichi** avec du LaTeX dans n'importe quelle cellule.

    Exemples de rendu :

    - **Gras**, *italique*, `code inline`
    - Liste à puces ou numérotée
    - Tableau Markdown
    - Formule inline : $\bar{x} = \frac{1}{n}\sum x_i$
    - Formule bloc :

    $$\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

    > **Astuce :** Utiliser `r"\"\"..."\"\"` (raw string) pour éviter les conflits
    > entre les `\` LaTeX et les séquences d'échappement Python.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.2 · `mo.ui` — Widgets Interactifs
    """)
    return


@app.cell
def _(mo):
    # Widgets de base — ils produisent des variables réactives
    slider_bins = mo.ui.slider(5, 50, value=20, step=5, label="Nombre de bins")
    dropdown_var = mo.ui.dropdown(
        options=["total_bill", "tip", "size"],
        value="total_bill",
        label="Variable"
    )
    checkbox_kde = mo.ui.checkbox(value=True, label="Afficher la courbe KDE")

    mo.vstack([
        mo.md("**Contrôles interactifs :**"),
        slider_bins,
        dropdown_var,
        checkbox_kde,
    ])
    return checkbox_kde, dropdown_var, slider_bins


@app.cell
def _(checkbox_kde, dropdown_var, plt, slider_bins, sns):
    # Cette cellule se re-exécute automatiquement quand un widget change
    import seaborn as sns_mo
    tips_mo = sns_mo.load_dataset("tips")

    fig_mo, ax_mo = plt.subplots(figsize=(9, 4))
    sns.histplot(
        tips_mo[dropdown_var.value],
        bins=slider_bins.value,
        kde=checkbox_kde.value,
        color="steelblue", alpha=0.7,
        ax=ax_mo
    )
    ax_mo.set_title(
        f"Distribution de `{dropdown_var.value}` — {slider_bins.value} bins"
        f"{' + KDE' if checkbox_kde.value else ''}",
        fontweight="bold"
    )
    ax_mo.set_xlabel(dropdown_var.value)
    plt.tight_layout()
    fig_mo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **Réactivité :** Modifie le slider ou le dropdown ci-dessus —
    > le graphique se met à jour **instantanément** sans cliquer sur "Run".
    > C'est la différence fondamentale avec Jupyter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.3 · Autres Widgets Utiles
    """)
    return


@app.cell
def _(mo):
    # Sélection multiple, bouton radio, champ texte
    multiselect_jours = mo.ui.multiselect(
        options=["Thur", "Fri", "Sat", "Sun"],
        value=["Sat", "Sun"],
        label="Jours à afficher"
    )
    radio_palette = mo.ui.radio(
        options=["Set2", "Set1", "Blues", "RdBu"],
        value="Set2",
        label="Palette de couleurs"
    )
    number_alpha = mo.ui.number(0.1, 1.0, value=0.7, step=0.1, label="Transparence (alpha)")

    mo.hstack([
        mo.vstack([mo.md("**Sélection**"), multiselect_jours]),
        mo.vstack([mo.md("**Palette**"), radio_palette]),
        mo.vstack([mo.md("**Alpha**"), number_alpha]),
    ])
    return multiselect_jours, number_alpha, radio_palette


@app.cell
def _(mo, multiselect_jours, number_alpha, plt, radio_palette, sns, tips):
    _jours_sel = multiselect_jours.value if multiselect_jours.value else ["Sat"]
    _df_filtre = tips[tips["day"].isin(_jours_sel)]

    if len(_df_filtre) == 0:
        _out = mo.md("⚠️ Sélectionne au moins un jour.")
    else:
        _fig, _ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(
            data=_df_filtre,
            x="day",
            y="tip",
            order=[d for d in ["Thur", "Fri", "Sat", "Sun"] if d in _jours_sel],
            palette=radio_palette.value,
            ax=_ax,
        )
        for patch in _ax.patches:
            patch.set_alpha(number_alpha.value)
        _ax.set_title(f"Pourboires — {', '.join(_jours_sel)}", fontweight="bold")
        plt.tight_layout()
        _out = _fig
    
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.4 · `mo.stat()` et `mo.hstack` / `mo.vstack` — Mise en page
    """)
    return


@app.cell
def _(mo, tips):
    # mo.stat : affiche une statistique mise en valeur
    stats_display = mo.hstack([
        mo.stat(
            value=f"{tips['tip'].mean():.2f} $",
            label="Pourboire moyen",
            caption="sur 244 repas",
        ),
        mo.stat(
            value=f"{tips['tip'].median():.2f} $",
            label="Pourboire médian",
            caption="robuste aux outliers",
        ),
        mo.stat(
            value=f"{(tips['tip']/tips['total_bill']*100).mean():.1f} %",
            label="Taux moyen",
            caption="tip / total_bill",
        ),
        mo.stat(
            value=str(len(tips)),
            label="Observations",
            caption="dataset tips",
        ),
    ])
    stats_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.5 · Bonnes Pratiques Marimo

    ### Structure recommandée d'un notebook

    ```
    Cellule 1 : imports globaux  (numpy, pandas, etc.)
    Cellule 2 : chargement des données
    Cellule 3 : nettoyage / préparation
    Cellules 4+ : analyses, visualisations, widgets
    ```

    ### Règles à respecter

    | Règle | Pourquoi |
    |-------|----------|
    | Une variable définie dans **une seule** cellule | Évite `multiple-definitions` |
    | Préfixer les variables de boucle avec `_` | Les rend privées à la cellule |
    | Ne pas modifier un DataFrame importé dans une autre cellule | Utiliser `.copy()` |
    | Les cellules `hide_code=True` servent aux explications | Séparer texte et code |
    | Retourner la figure (`fig`) ou la laisser en dernière expression | Pour l'afficher |

    ### Raccourcis clavier utiles

    | Raccourci | Action |
    |-----------|--------|
    | `Ctrl + Enter` | Exécuter la cellule courante |
    | `Shift + Enter` | Exécuter et passer à la suivante |
    | `Ctrl + Shift + R` | Re-exécuter tout le notebook |
    | `Ctrl + /` | Commenter / décommenter |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Partie 2 · Bases de Python

    ## Pourquoi Python pour l'Analyse de Données ?

    Apparu en 1991, Python s'est imposé comme **le** langage de référence pour la data science.
    Plusieurs raisons expliquent ce succès :

    | Atout | Explication |
    |-------|-------------|
    | **Écosystème riche** | `numpy`, `pandas`, `scikit-learn`, `statsmodels`… couvrent tous les besoins |
    | **Lisibilité** | Syntaxe proche du pseudo-code — rapide à apprendre, facile à maintenir |
    | **Langage universel** | Prototypage *et* mise en production dans le même environnement |
    | **Interopérabilité** | Intègre nativement du code C/C++/Fortran via des extensions (NumPy en est un exemple) |

    ### Python est un langage interprété

    Contrairement à C++ ou Java qui sont **compilés** (le code source est traduit en langage machine
    avant l'exécution), Python est **interprété** : un programme appelé *interpréteur* lit et exécute
    le code **ligne par ligne**, à la volée.

    ```
    Code source (.py)  →  Interpréteur Python  →  Exécution
                              (CPython)
    ```

    **Conséquences pratiques :**

    - ✅ **Interactivité** : on peut tester une ligne de code immédiatement (REPL, notebooks)
    - ✅ **Portabilité** : le même fichier `.py` tourne sur Windows, Linux, macOS
    - ⚠️ **Vitesse** : plus lent que C/C++ pour les calculs intensifs — compensé par des librairies
      compilées (`numpy` exécute ses opérations en C, pas en Python pur)
    - ⚠️ **GIL** : le *Global Interpreter Lock* limite le vrai parallélisme multi-thread,
      mais les extensions C (comme NumPy) y échappent

    > **En pratique :** le "vrai" calcul est délégué à des librairies compilées.
    > Python joue le rôle de **langage de colle** (*glue language*) — expressif, lisible,
    > et suffisamment rapide pour orchestrer des opérations numériques très performantes.

    Cette partie couvre les fondamentaux du langage Python nécessaires
    pour l'analyse de données. Python est un langage **interprété, dynamiquement typé,
    à indentation significative**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 · Types de Base et Variables

    Python est **dynamiquement typé** : le type d'une variable est inféré automatiquement
    à l'assignation.

    | Type | Exemple | Remarque |
    |------|---------|----------|
    | `int` | `42` | Entier de précision arbitraire |
    | `float` | `3.14` | Nombre décimal (64 bits) |
    | `bool` | `True`, `False` | Sous-classe de `int` |
    | `str` | `"bonjour"` | Chaîne immuable, Unicode |
    | `NoneType` | `None` | Absence de valeur |
    """)
    return


@app.cell
def _():
    entier   = 42
    flottant = 3.14159
    booleen  = True
    texte    = "Analyse de données"
    vide     = None

    print(f"entier   : {entier!r:20}  type → {type(entier).__name__}")
    print(f"flottant : {flottant!r:20}  type → {type(flottant).__name__}")
    print(f"booleen  : {booleen!r:20}  type → {type(booleen).__name__}")
    print(f"texte    : {texte!r:20}  type → {type(texte).__name__}")
    print(f"vide     : {vide!r:20}  type → {type(vide).__name__}")

    print(f"\nCasting   : int('7')={int('7')} | float(3)={float(3)} | str(42)={str(42)!r}")
    print(f"Opérations: 17//3={17//3} | 17%3={17%3} | 2**10={2**10}")
    return


@app.cell
def _():
    # ── Chaînes de caractères ─────────────────────────────────────
    s_str = "Analyse de Données"
    print(f"upper      : {s_str.upper()}")
    print(f"lower      : {s_str.lower()}")
    print(f"len        : {len(s_str)}")
    print(f"slice[0:7] : {s_str[0:7]}")
    print(f"split      : {s_str.split()}")
    print(f"replace    : {s_str.replace('Données', 'Data')}")

    nom_str, note_str = "Alice", 17.5
    print(f"\nf-string   : {nom_str} a obtenu {note_str:.1f}/20 (soit {note_str/20*100:.0f}%)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 · Structures de Données

    | Structure | Syntaxe | Ordonné | Mutable | Doublons |
    |-----------|---------|---------|---------|----------|
    | `list` | `[1, 2, 3]` | ✅ | ✅ | ✅ |
    | `tuple` | `(1, 2, 3)` | ✅ | ❌ | ✅ |
    | `dict` | `{"a": 1}` | ✅ (3.7+) | ✅ | clés ❌ |
    | `set` | `{1, 2, 3}` | ❌ | ✅ | ❌ |
    """)
    return


@app.cell
def _():
    notes_list = [12, 15, 9, 18, 14, 11]
    print(f"liste          : {notes_list}")
    print(f"index [0]/-[1] : {notes_list[0]} / {notes_list[-1]}")
    print(f"slice [1:4]    : {notes_list[1:4]}")
    print(f"sorted         : {sorted(notes_list)}")
    print(f"sum/len/max    : {sum(notes_list)} / {len(notes_list)} / {max(notes_list)}")

    etudiant_dict = {"nom": "Alice", "note": 17.5, "cours": ["Maths", "Stats"]}
    print(f"\ndict['nom']        : {etudiant_dict['nom']}")
    print(f"dict.get('age','?'): {etudiant_dict.get('age', 'N/A')}")
    etudiant_dict["promo"] = 2025
    print(f"après ajout        : {etudiant_dict}")
    return


@app.cell
def _():
    # ── Compréhensions ────────────────────────────────────────────
    notes_comp2 = [12, 15, 9, 18, 14, 11, 7, 16]

    pcts_comp    = [round(n/20*100, 1) for n in notes_comp2]
    admis_comp   = [n for n in notes_comp2 if n >= 10]
    labels_comp  = {n: ("✅" if n >= 10 else "❌") for n in notes_comp2}

    print(f"list  comprehension : {pcts_comp}")
    print(f"list  filtrage      : {admis_comp}")
    print(f"dict  comprehension : {labels_comp}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 · Structures de Contrôle

    Python délimite les blocs par l'**indentation** (4 espaces), pas d'accolades.

    ```python
    if condition:       for element in iterable:       while condition:
        ...                 ...                            ...
    elif ...:           else:                          break / continue
        ...                 ...
    else:
        ...
    ```
    """)
    return


@app.cell
def _():
    def mention_ctrl(note_c):
        if note_c >= 16:   return "Très Bien"
        elif note_c >= 14: return "Bien"
        elif note_c >= 12: return "Assez Bien"
        elif note_c >= 10: return "Passable"
        else:              return "Insuffisant"

    eleves_ctrl  = ["Alice", "Bob", "Clara", "David"]
    scores_ctrl  = [17, 9, 14, 11]

    print("── for + enumerate + zip ──")
    for _i, (_nom_c, _s) in enumerate(zip(eleves_ctrl, scores_ctrl)):
        print(f"  {_i+1}. {_nom_c:<8} {_s}/20  {mention_ctrl(_s)}")

    print("\n── while ──")
    _total_c, _k = 0, 0
    while _k < len(scores_ctrl):
        _total_c += scores_ctrl[_k]; _k += 1
    print(f"  Moyenne (while) : {_total_c/len(scores_ctrl):.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 · Fonctions

    | Style | Exemple | Remarque |
    |-------|---------|----------|
    | Positionnel | `f(1, 2)` | Ordre important |
    | Nommé | `f(a=1, b=2)` | Ordre libre |
    | Valeur par défaut | `def f(x, n=2)` | Argument optionnel |
    | `*args` | `def f(*args)` | Nombre variable de positionnels |
    | `**kwargs` | `def f(**kwargs)` | Nombre variable de nommés |
    | `lambda` | `lambda x: x**2` | Fonction anonyme en une ligne |
    """)
    return


@app.cell
def _():
    import math as _math

    def statistiques_fn(valeurs, ddof=1):
        """Calcule les statistiques de base sans librairie externe."""
        n_fn = len(valeurs)
        moy  = sum(valeurs) / n_fn
        var  = sum((v - moy)**2 for v in valeurs) / (n_fn - ddof)
        tri  = sorted(valeurs)
        med  = tri[n_fn//2] if n_fn % 2 else (tri[n_fn//2-1] + tri[n_fn//2]) / 2
        return {"n": n_fn, "moyenne": round(moy, 3),
                "mediane": med, "ecart_type": round(_math.sqrt(var), 3)}

    _data_fn = [12, 15, 9, 18, 14, 11, 7, 16, 10, 13]
    _res_fn  = statistiques_fn(_data_fn)
    print("Statistiques sans librairie :")
    for _k_fn, _v_fn in _res_fn.items():
        print(f"  {_k_fn:<12} : {_v_fn}")

    # Lambda + map + filter + sorted
    notes_fn = [12, 15, 9, 18, 14, 11, 7, 16]
    print(f"\nmap (→ %)  : {list(map(lambda n: round(n/20*100,1), notes_fn))}")
    print(f"filter ≥10 : {list(filter(lambda n: n >= 10, notes_fn))}")
    etudiants_fn = [("Alice",17),("Bob",9),("Clara",14)]
    print(f"sorted     : {sorted(etudiants_fn, key=lambda e: e[1], reverse=True)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.5 · Classes et Programmation Orientée Objet

    ```python
    class NomClasse:
        def __init__(self, ...):   # constructeur
            self.attribut = ...
        def methode(self): ...     # méthode d'instance
    ```

    > En analyse de données vous **utilisez** des classes en permanence :
    > `DataFrame`, `LinearRegression`, `StandardScaler` sont toutes des classes.
    """)
    return


@app.cell
def _():
    class Etudiant_cls:
        universite = "IPSL"

        def __init__(self, nom_cls, notes_cls):
            self.nom   = nom_cls
            self.notes = notes_cls

        def moyenne(self):
            return round(sum(self.notes) / len(self.notes), 2)

        def mention(self):
            m = self.moyenne()
            if m >= 16: return "Très Bien"
            elif m >= 14: return "Bien"
            elif m >= 12: return "Assez Bien"
            elif m >= 10: return "Passable"
            else: return "Insuffisant"

        def __repr__(self):
            return f"Etudiant({self.nom!r}, moy={self.moyenne()})"

    class EtudiantBoursier_cls(Etudiant_cls):
        def __init__(self, nom_cls, notes_cls, bourse):
            super().__init__(nom_cls, notes_cls)
            self.bourse = bourse

        def eligible(self):
            return "✅" if self.moyenne() >= 12 else "❌"

        def __repr__(self):
            return f"Boursier({self.nom!r}, moy={self.moyenne()}, {self.bourse}€ {self.eligible()})"

    alice_cls = Etudiant_cls("Alice", [15, 17, 14, 18, 16])
    bob_cls   = Etudiant_cls("Bob",   [9, 11, 8, 12, 10])
    clara_cls = EtudiantBoursier_cls("Clara", [13, 16, 14, 15, 12], 3000)

    print(f"{alice_cls}  →  {alice_cls.mention()}")
    print(f"{bob_cls}  →  {bob_cls.mention()}")
    print(f"{clara_cls}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.6 · Gestion des Erreurs

    | Exception | Cause fréquente |
    |-----------|-----------------|
    | `ValueError` | `int("abc")` |
    | `TypeError` | `"a" + 1` |
    | `KeyError` | Clé absente dans un dict |
    | `IndexError` | Indice hors limites |
    | `ZeroDivisionError` | Division par zéro |
    | `FileNotFoundError` | Fichier inexistant |
    """)
    return


@app.cell
def _():
    def division_sure(a_e, b_e):
        try:
            return a_e / b_e
        except ZeroDivisionError:
            print(f"  ❌ Division par zéro : {a_e}/{b_e}")
        except TypeError as e_e:
            print(f"  ❌ TypeError : {e_e}")

    def convertir_note(val_e):
        try:
            n_e = float(val_e)
            if not 0 <= n_e <= 20:
                raise ValueError(f"hors intervalle [0,20] : {n_e}")
            return n_e
        except (ValueError, TypeError) as e_e:
            print(f"  ⚠️  {val_e!r} invalide → {e_e}")
            return None

    print("── division_sure ──")
    for _a, _b in [(10, 3), (5, 0), ("dix", 2)]:
        _r = division_sure(_a, _b)
        if _r is not None: print(f"  {_a}/{_b} = {_r:.4f}")

    print("\n── convertir_note ──")
    for _v in ["15.5", "abc", 25, None, "12"]:
        print(f"  {_v!r:8} → {convertir_note(_v)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Partie 3 · Librairies Python pour l'Analyse de Données

    | # | Librairie | Rôle |
    |---|-----------|------|
    | 1 | `numpy` | Calcul numérique, tableaux multidimensionnels |
    | 2 | `pandas` | Manipulation de données tabulaires |
    | 3 | `polars` | Alternative rapide à pandas (lazy evaluation) |
    | 4 | `seaborn` / `matplotlib` | Visualisation statistique |
    | 5 | `statsmodels` | Modélisation statistique avancée |
    | 6 | `scikit-learn` | Machine learning |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0 · Imports
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, r2_score, classification_report
    import warnings
    warnings.filterwarnings("ignore")

    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 110

    # Dataset fil rouge
    tips = sns.load_dataset("tips")
    print("✅ Imports réussis")
    print(f"Dataset tips : {tips.shape[0]} lignes × {tips.shape[1]} colonnes")
    return (
        LabelEncoder,
        LinearRegression,
        LogisticRegression,
        StandardScaler,
        classification_report,
        mean_squared_error,
        np,
        pd,
        pl,
        plt,
        r2_score,
        smf,
        sns,
        tips,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1 · NumPy — Calcul Numérique

    NumPy est la brique de base de tout l'écosystème scientifique Python.
    Il fournit le type `ndarray` — un tableau multidimensionnel **typé et continu en mémoire**,
    beaucoup plus rapide qu'une liste Python pour les opérations numériques.

    ### Concepts clés

    | Concept | Description |
    |---------|-------------|
    | `ndarray` | Tableau N-dimensionnel homogène |
    | `dtype` | Type des éléments (`float64`, `int32`…) |
    | Broadcasting | Opérations entre tableaux de formes différentes |
    | Vectorisation | Appliquer une opération à tout un tableau sans boucle |
    """)
    return


@app.cell
def _(np):
    # ── Création de tableaux ──────────────────────────────────────
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
    c = np.linspace(0, 1, 5)          # 5 points équidistants entre 0 et 1
    matrice = np.zeros((3, 4))         # Matrice 3×4 de zéros

    print("Tableau a :", a)
    print("arange   :", b)
    print("linspace :", c)
    print("Matrice  :\n", matrice)
    return


@app.cell
def _(np, tips):
    # ── Opérations vectorisées ────────────────────────────────────
    tarifs = tips["total_bill"].values   # Convertir Series pandas → ndarray

    print("Type         :", type(tarifs), tarifs.dtype)
    print("Forme        :", tarifs.shape)
    print("Moyenne      :", np.mean(tarifs).round(2))
    print("Écart-type   :", np.std(tarifs, ddof=1).round(2))
    print("Percentiles  :", np.percentile(tarifs, [25, 50, 75]).round(2))

    # Broadcasting : normalisation z-score sans boucle
    z_scores = (tarifs - tarifs.mean()) / tarifs.std()
    print(f"\nZ-scores : min={z_scores.min():.2f}, max={z_scores.max():.2f}, mean≈{z_scores.mean():.4f}")
    return


@app.cell
def _(np, plt):
    # ── Broadcasting illustré ─────────────────────────────────────
    rng_np = np.random.default_rng(42)
    x_np = rng_np.normal(0, 1, 300)

    fig_np, axes_np = plt.subplots(1, 2, figsize=(13, 4))

    axes_np[0].hist(x_np, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    axes_np[0].set_title("Distribution N(0,1) — 300 tirages", fontweight="bold")
    axes_np[0].set_xlabel("Valeur"); axes_np[0].set_ylabel("Effectif")

    # Règle empirique 68-95-99.7 c'est à dire : 
    # 68 % des données se situent à ±1σ de la moyenne.
    # 95 % des données se situent à ±2σ de la moyenne.
    # 99,7 % des données se situent à ±3σ de la moyenne.

    for n_sig, col in [(1,"red"), (2,"orange"), (3,"gold")]:
        axes_np[1].axvspan(-n_sig, n_sig, alpha=0.25, color=col,
                           label=f"±{n_sig}σ ({np.mean(np.abs(x_np) <= n_sig)*100:.0f}%)")
    axes_np[1].hist(x_np, bins=30, color="steelblue", alpha=0.6, edgecolor="white", density=True)
    axes_np[1].set_title("Règle empirique 68-95-99.7", fontweight="bold")
    axes_np[1].legend()

    plt.suptitle("NumPy — Génération aléatoire et vectorisation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_np
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2 · Pandas — Manipulation de Données Tabulaires

    Pandas est la librairie de référence pour manipuler des **données tabulaires** en Python.
    Ses deux structures centrales sont :

    - **`Series`** : tableau 1D indexé (colonne)
    - **`DataFrame`** : tableau 2D (lignes × colonnes), analogue à une feuille Excel ou une table SQL

    ### Opérations fondamentales

    | Opération | Méthode |
    |-----------|---------|
    | Exploration | `.head()`, `.info()`, `.describe()` |
    | Sélection | `df["col"]`, `df.loc[]`, `df.iloc[]` |
    | Filtrage | `df[condition]` |
    | Agrégation | `.groupby().agg()` |
    | Gestion NA | `.isnull()`, `.fillna()`, `.dropna()` |
    | Fusion | `pd.merge()`, `pd.concat()` |
    """)
    return


@app.cell
def _(tips):
    # ── Exploration rapide ────────────────────────────────────────
    print("── Infos générales ──")
    print(f"Dimensions : {tips.shape}")
    print(f"Types :\n{tips.dtypes}\n")
    print("── Statistiques descriptives ──")
    tips.describe().round(2)
    return


@app.cell
def _(tips):
    # ── Sélection et filtrage ─────────────────────────────────────
    # Sélection de colonnes
    cols = tips[["total_bill", "tip", "sex"]]

    # Filtrage conditionnel
    gros_pourboires = tips[(tips["tip"] > 5) & (tips["sex"] == "Female")]

    # loc (labels) vs iloc (indices entiers)
    print("loc  — ligne 0 :", tips.loc[0, ["total_bill", "tip"]].to_dict())
    print("iloc — ligne 0 :", tips.iloc[0, [0, 1]].to_dict())
    print(f"\nFiltrage tip > 5 & Female : {len(gros_pourboires)} lignes")

    # Nouvelle colonne calculée
    tips_pd = tips.copy()
    tips_pd["tip_rate"] = (tips_pd["tip"] / tips_pd["total_bill"] * 100).round(2)
    tips_pd[["total_bill", "tip", "tip_rate"]].head(4)
    return (tips_pd,)


@app.cell
def _(tips_pd):
    # ── Groupby + agrégations multiples ───────────────────────────
    resume = (
        tips_pd
        .groupby(["day", "sex"])["tip_rate"]
        .agg(
            moyenne="mean",
            mediane="median",
            ecart_type="std",
            n="count"
        )
        .round(2)
        .reset_index()
    )
    print("Taux de pourboire moyen par jour et sexe :")
    resume
    return


@app.cell
def _(pd, tips):
    # ── Gestion des valeurs manquantes ────────────────────────────
    # Simulation de NaN
    tips_na = tips.copy()
    tips_na.loc[[3, 7, 15, 42], "tip"] = float("nan")

    print(f"Valeurs manquantes avant : {tips_na['tip'].isnull().sum()}")

    # Stratégies d'imputation
    tips_na["tip_mean"]   = tips_na["tip"].fillna(tips_na["tip"].mean())
    tips_na["tip_median"] = tips_na["tip"].fillna(tips_na["tip"].median())

    # Imputation par groupe (plus fine)
    tips_na["tip_group"] = tips_na["tip"].fillna(
        tips_na.groupby("sex")["tip"].transform("median")
    )

    print(f"Valeurs manquantes après  : {tips_na['tip_mean'].isnull().sum()}")
    print(f"\nComparaison des imputations :")
    print(pd.DataFrame({
        "Originale": tips["tip"].describe(),
        "Par moyenne": tips_na["tip_mean"].describe(),
        "Par médiane groupe": tips_na["tip_group"].describe()
    }).round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3 · Polars — La Nouvelle Génération de DataFrames

    Polars est une alternative **très rapide** à pandas, écrite en Rust.
    Ses atouts principaux :

    - **Lazy evaluation** : les opérations sont optimisées avant exécution
    - **Multi-threading** : parallélisation automatique sur tous les cœurs
    - **Expressif** : API chainable et cohérente (pas d'index implicite)

    ### Pandas vs Polars — Syntaxe comparée

    | Opération | Pandas | Polars |
    |-----------|--------|--------|
    | Filtrage | `df[df["col"] > 5]` | `df.filter(pl.col("col") > 5)` |
    | Nouvelle colonne | `df["new"] = ...` | `df.with_columns(...)` |
    | Agrégation | `df.groupby().agg()` | `df.group_by().agg()` |
    | Lazy mode | ✗ | `df.lazy().collect()` |
    """)
    return


@app.cell
def _(pl, tips):
    # ── Conversion pandas → polars ────────────────────────────────
    tips_pl = pl.from_pandas(tips)

    print("Type :", type(tips_pl))
    print("Schéma :")
    print(tips_pl.schema)
    return (tips_pl,)


@app.cell
def _(pl, tips_pl):
    # ── Opérations de base ────────────────────────────────────────
    result_pl = (
        tips_pl
        .with_columns(
            (pl.col("tip") / pl.col("total_bill") * 100).alias("tip_rate")
        )
        .filter(pl.col("tip_rate") > 15)
        .group_by("day")
        .agg(
            pl.col("tip_rate").mean().alias("taux_moyen"),
            pl.col("tip_rate").median().alias("taux_median"),
            pl.len().alias("n")
        )
        .sort("taux_moyen", descending=True)
    )
    print("Taux de pourboire > 15% par jour :")
    result_pl
    return


@app.cell
def _(pl, tips_pl):
    # ── Lazy evaluation — plan d'exécution optimisé ───────────────
    query_lazy = (
        tips_pl.lazy()
        .with_columns(
            (pl.col("tip") / pl.col("total_bill") * 100).round(2).alias("tip_rate")
        )
        .filter(pl.col("size") >= 3)
        .group_by(["sex", "smoker"])
        .agg(pl.col("tip_rate").mean().round(2).alias("taux_moyen"))
        .sort("taux_moyen", descending=True)
    )

    print("Plan d'exécution optimisé :")
    print(query_lazy.explain())
    print("\nRésultat :")
    query_lazy.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4 · Seaborn & Matplotlib — Visualisation Statistique

    **Matplotlib** est la librairie de visualisation bas niveau de Python.
    **Seaborn** s'appuie dessus pour fournir des graphiques statistiques de haute qualité
    avec beaucoup moins de code.

    ### Quand utiliser quoi ?

    | Besoin | Outil |
    |--------|-------|
    | Graphique statistique standard | `seaborn` |
    | Personnalisation fine | `matplotlib` |
    | Graphique interactif | `plotly` |

    ### Principaux graphiques seaborn

    | Graphique | Fonction | Objectif |
    |-----------|----------|----------|
    | Distribution | `histplot`, `kdeplot` | Une variable numérique |
    | Comparaison | `boxplot`, `violinplot` | Numérique × Catégorielle |
    | Relation | `scatterplot`, `regplot` | Deux numériques |
    | Corrélation | `heatmap` | Matrice de corrélation |
    | Multivarié | `pairplot` | Toutes les paires |
    """)
    return


@app.cell
def _(plt, sns, tips):
    # ── Distributions ─────────────────────────────────────────────
    fig_sns1, axes_sns1 = plt.subplots(1, 3, figsize=(16, 4))

    sns.histplot(tips["total_bill"], kde=True, color="steelblue",
                 ax=axes_sns1[0], bins=25)
    axes_sns1[0].set_title("histplot — Distribution\ntotal_bill", fontweight="bold")

    sns.kdeplot(data=tips, x="total_bill", hue="sex",
                fill=True, alpha=0.4, ax=axes_sns1[1])
    axes_sns1[1].set_title("kdeplot — Distribution par sexe", fontweight="bold")

    sns.ecdfplot(data=tips, x="tip", hue="time", ax=axes_sns1[2])
    axes_sns1[2].set_title("ecdfplot — Fonction de répartition\nLunch vs Dinner", fontweight="bold")

    plt.suptitle("Seaborn — Visualisation de distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_sns1
    return


@app.cell
def _(plt, sns, tips):
    # ── Comparaison et relation ───────────────────────────────────
    fig_sns2, axes_sns2 = plt.subplots(1, 3, figsize=(16, 5))

    sns.boxplot(data=tips, x="day", y="tip",
                order=["Thur", "Fri", "Sat", "Sun"],
                palette="Set2", ax=axes_sns2[0])
    axes_sns2[0].set_title("boxplot — Pourboire par jour", fontweight="bold")

    sns.violinplot(data=tips, x="sex", y="total_bill",
                   hue="smoker", split=True,
                   inner="quartile", palette="Set1", ax=axes_sns2[1])
    axes_sns2[1].set_title("violinplot — Montant\nSexe × Fumeur", fontweight="bold")

    sns.regplot(data=tips, x="total_bill", y="tip",
                scatter_kws={"alpha": 0.5, "s": 40},
                line_kws={"color": "tomato", "lw": 2},
                ax=axes_sns2[2])
    axes_sns2[2].set_title("regplot — Régression linéaire\ntotal_bill → tip", fontweight="bold")

    plt.suptitle("Seaborn — Comparaison et relations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_sns2
    return


@app.cell
def _(plt, sns, tips):
    # ── Heatmap de corrélation ────────────────────────────────────
    corr_sns = tips[["total_bill", "tip", "size"]].corr()

    fig_sns3, ax_sns3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_sns, annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5,
                annot_kws={"size": 13}, ax=ax_sns3)
    ax_sns3.set_title("heatmap — Matrice de corrélation", fontweight="bold")
    plt.tight_layout()
    fig_sns3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5 · Statsmodels — Modélisation Statistique

    Statsmodels est la librairie de référence pour la **modélisation statistique inférentielle** en Python.
    Elle fournit des résumés détaillés (coefficients, p-values, IC, tests de diagnostic)
    proches des standards académiques (R, Stata).

    ### Deux API disponibles

    | API | Syntaxe | Avantage |
    |-----|---------|----------|
    | Formule R | `smf.ols("y ~ x1 + x2", data=df)` | Lisible, proche R |
    | Matrices | `sm.OLS(y, sm.add_constant(X))` | Contrôle total |

    ### Modèles disponibles

    `OLS` (régression linéaire), `Logit` / `Probit` (classification),
    `GLM` (modèles généralisés), `ARIMA` (séries temporelles), et bien d'autres.
    """)
    return


@app.cell
def _(smf, tips):
    # ── Régression linéaire OLS avec formule ─────────────────────
    # Question : total_bill et size expliquent-ils le montant du pourboire ?
    tips_sm = tips.copy()
    tips_sm["tip_rate"] = tips_sm["tip"] / tips_sm["total_bill"] * 100

    model_ols = smf.ols("tip ~ total_bill + size + C(time)", data=tips_sm).fit()
    print(model_ols.summary())
    return model_ols, tips_sm


@app.cell
def _(model_ols, plt):
    # ── Diagnostic des résidus ────────────────────────────────────
    residus = model_ols.resid
    fitted  = model_ols.fittedvalues

    fig_sm, axes_sm = plt.subplots(1, 3, figsize=(16, 4))

    # Résidus vs ajustés
    axes_sm[0].scatter(fitted, residus, alpha=0.5, color="steelblue", s=30)
    axes_sm[0].axhline(0, color="tomato", lw=2, linestyle="--")
    axes_sm[0].set_xlabel("Valeurs ajustées"); axes_sm[0].set_ylabel("Résidus")
    axes_sm[0].set_title("Résidus vs Ajustés\n(doit être aléatoire)", fontweight="bold")

    # Histogramme des résidus
    axes_sm[1].hist(residus, bins=25, color="coral", alpha=0.7, edgecolor="white")
    axes_sm[1].set_title(f"Distribution des résidus\nSkewness={residus.skew():.2f}",
                         fontweight="bold")

    # QQ-plot des résidus
    from scipy import stats as sp_stats
    sp_stats.probplot(residus, plot=axes_sm[2])
    axes_sm[2].set_title("Q-Q Plot des résidus\n(normalité des résidus)", fontweight="bold")
    axes_sm[2].get_lines()[0].set(markersize=4, alpha=0.6, color="steelblue")

    plt.suptitle(f"Statsmodels OLS — Diagnostic (R²={model_ols.rsquared:.3f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_sm
    return


@app.cell
def _(smf, tips_sm):
    # ── Régression logistique (Logit) ─────────────────────────────
    # Question : peut-on prédire le sexe à partir du montant et du pourboire ?
    tips_logit = tips_sm.copy()
    tips_logit["female"] = (tips_logit["sex"] == "Female").astype(int)

    model_logit = smf.logit("female ~ total_bill + tip + size", data=tips_logit).fit()

    print(f"Pseudo R² (McFadden) : {model_logit.prsquared:.3f}")
    print(f"\nOdds Ratios (exp(β)) :")
    import numpy as np_sm
    print((np_sm.exp(model_logit.params)).round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6 · Scikit-learn — Machine Learning

    Scikit-learn est la librairie de machine learning la plus utilisée en Python.
    Elle propose une **API unifiée** pour tous les modèles :

    ```python
    model = MonModele(hyperparamètres)   # 1. Instancier
    model.fit(X_train, y_train)          # 2. Entraîner
    y_pred = model.predict(X_test)       # 3. Prédire
    score  = model.score(X_test, y_test) # 4. Évaluer
    ```

    ### Pipeline typique

    ```
    Données brutes
        → Prétraitement (StandardScaler, LabelEncoder, OneHotEncoder)
        → Séparation train/test (train_test_split)
        → Entraînement (fit)
        → Évaluation (score, MSE, R², classification_report)
    ```
    """)
    return


@app.cell
def _(LabelEncoder, StandardScaler, pd, tips, train_test_split):
    # ── Prétraitement ─────────────────────────────────────────────
    tips_sk = tips.copy()

    # Encodage des variables catégorielles
    le = LabelEncoder()
    tips_sk["sex_enc"]    = le.fit_transform(tips_sk["sex"])
    tips_sk["smoker_enc"] = le.fit_transform(tips_sk["smoker"])
    tips_sk["time_enc"]   = le.fit_transform(tips_sk["time"])
    tips_sk["day_enc"]    = pd.Categorical(tips_sk["day"]).codes

    features = ["total_bill", "size", "sex_enc", "smoker_enc", "time_enc", "day_enc"]
    X = tips_sk[features].values
    y = tips_sk["tip"].values

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Jeu d'entraînement : {X_train.shape[0]} observations")
    print(f"Jeu de test        : {X_test.shape[0]} observations")
    print(f"\nMoyenne features après normalisation ≈ {X_scaled.mean():.4f}")
    print(f"Écart-type après normalisation        ≈ {X_scaled.std():.4f}")
    return X_test, X_train, tips_sk, y_test, y_train


@app.cell
def _(
    LinearRegression,
    X_test,
    X_train,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    # ── Régression linéaire sklearn ───────────────────────────────
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred_reg = reg.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred_reg)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred_reg)

    print("=" * 45)
    print("RÉGRESSION LINÉAIRE — sklearn")
    print("=" * 45)
    print(f"  RMSE : {rmse:.3f} $")
    print(f"  R²   : {r2:.3f}")
    print(f"\n  Intercept : {reg.intercept_:.3f}")
    print(f"  Coefficients :")
    features_reg = ["total_bill", "size", "sex", "smoker", "time", "day"]
    for feat, coef in zip(features_reg, reg.coef_):
        print(f"    {feat:<15} : {coef:+.3f}")
    return (y_pred_reg,)


@app.cell
def _(plt, y_pred_reg, y_test):
    # ── Visualisation prédictions vs réalité ──────────────────────
    import numpy as np_sk
    fig_sk1, axes_sk1 = plt.subplots(1, 2, figsize=(13, 5))

    axes_sk1[0].scatter(y_test, y_pred_reg, alpha=0.6, color="steelblue", s=40)
    lims = [min(y_test.min(), y_pred_reg.min()), max(y_test.max(), y_pred_reg.max())]
    axes_sk1[0].plot(lims, lims, "r--", lw=2, label="Prédiction parfaite")
    axes_sk1[0].set_xlabel("Valeurs réelles ($)"); axes_sk1[0].set_ylabel("Valeurs prédites ($)")
    axes_sk1[0].set_title("Prédictions vs Réalité", fontweight="bold")
    axes_sk1[0].legend()

    residus_sk = y_test - y_pred_reg
    axes_sk1[1].hist(residus_sk, bins=20, color="coral", alpha=0.7, edgecolor="white")
    axes_sk1[1].axvline(0, color="navy", lw=2, linestyle="--")
    axes_sk1[1].set_title(f"Résidus (μ={residus_sk.mean():.2f}, σ={residus_sk.std():.2f})",
                          fontweight="bold")
    axes_sk1[1].set_xlabel("Erreur ($)")

    plt.suptitle("Sklearn — Régression linéaire sur tips", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_sk1
    return


@app.cell
def _(LogisticRegression, classification_report, tips_sk, train_test_split):
    # ── Classification : prédire si tip > 3$ ─────────────────────
    import numpy as np_sk2
    y_clf = (tips_sk["tip"] > 3).astype(int).values
    X_clf = tips_sk[["total_bill", "size"]].values

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(random_state=42)
    clf.fit(X_tr_c, y_tr_c)
    y_pred_clf = clf.predict(X_te_c)

    print("=" * 50)
    print("RÉGRESSION LOGISTIQUE — tip > 3$ (Oui/Non)")
    print("=" * 50)
    print(classification_report(y_te_c, y_pred_clf,
                                target_names=["tip ≤ 3$", "tip > 3$"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Récapitulatif — Quelle librairie pour quel besoin ?

    | Besoin | Librairie | Fonction clé |
    |--------|-----------|-------------|
    | Calcul numérique rapide | `numpy` | `np.array`, `np.mean`, `np.percentile` |
    | Manipulation tabulaire | `pandas` | `df.groupby().agg()`, `df.merge()` |
    | Traitement haute performance | `polars` | `df.lazy().filter().collect()` |
    | Visualisation statistique | `seaborn` | `histplot`, `boxplot`, `regplot` |
    | Inférence statistique | `statsmodels` | `smf.ols().fit().summary()` |
    | Machine learning | `scikit-learn` | `fit()`, `predict()`, `score()` |

    ### Chaîne de traitement typique

    ```
    pandas/polars       → Charger, nettoyer, explorer
    numpy               → Calculs numériques intermédiaires
    seaborn/matplotlib  → Visualiser et comprendre
    statsmodels         → Tester des hypothèses, régresser avec inférence
    scikit-learn        → Modéliser, prédire, évaluer
    ```

    ### Références

    - NumPy : https://numpy.org/doc/stable/user/quickstart.html
    - Pandas : https://pandas.pydata.org/docs/user_guide/index.html
    - Polars : https://docs.pola.rs/user-guide/
    - Seaborn : https://seaborn.pydata.org/tutorial.html
    - Statsmodels : https://www.statsmodels.org/stable/index.html
    - Scikit-learn : https://scikit-learn.org/stable/user_guide.html
    """)
    return


if __name__ == "__main__":
    app.run()
