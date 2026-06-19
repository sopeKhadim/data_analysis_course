# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Quiz — Séances 1-0 & 1-1")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📝 Quiz — Séances 1-0 & 1-1
    **Institut Polytechnique de Saint-Louis (IPSL) · Analyse de Données 2025-2026**

    ---

    Ce quiz couvre les deux premières séances du cours :
    - **Séance 1-0** : Introduction à l'analyse de données (CRISP-DM, taxonomie, pandas, éthique…)
    - **Séance 1-1** : Marimo, bases Python, NumPy, pandas, polars, scikit-learn

    ⏱️ **Durée estimée : 10 à 15 minutes** · 15 questions

    > Sélectionne une réponse par question, puis clique sur **Voir mes résultats** en bas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### 🔵 PARTIE 1 — Séance 1-0 : Introduction à l'Analyse de Données
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 1.** Quelle est la différence principale entre l'*analyse de données* et la *science des données* ?
    """)
    return


@app.cell
def _(mo):
    q1 = mo.ui.radio(
        options={
            "A. L'analyse de données utilise Python, la science des données utilise R": "A",
            "B. L'analyse répond à « que s'est-il passé ? » ; la science des données construit des modèles prédictifs": "B",
            "C. Ce sont deux noms différents pour la même discipline": "C",
            "D. La science des données ne nécessite pas de programmation": "D",
        },
        label="",
    )
    q1
    return (q1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 2.** Dans le *continuum analytique*, quelle approche répond à la question « Que faut-il faire ? » ?
    """)
    return


@app.cell
def _(mo):
    q2 = mo.ui.radio(
        options={
            "A. Descriptive": "A",
            "B. Diagnostique": "B",
            "C. Prédictive": "C",
            "D. Prescriptive": "D",
        },
        label="",
    )
    q2
    return (q2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 3.** Le *niveau d'études* (L1 < L2 < M1 < M2) est une variable de type :
    """)
    return


@app.cell
def _(mo):
    q3 = mo.ui.radio(
        options={
            "A. Quantitative continue": "A",
            "B. Qualitative nominale": "B",
            "C. Qualitative ordinale": "C",
            "D. Quantitative discrète": "D",
        },
        label="",
    )
    q3
    return (q3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 4.** Quelle est la **première** phase du cycle CRISP-DM ?
    """)
    return


@app.cell
def _(mo):
    q4 = mo.ui.radio(
        options={
            "A. Collecte des données": "A",
            "B. Modélisation": "B",
            "C. Compréhension métier": "C",
            "D. Déploiement": "D",
        },
        label="",
    )
    q4
    return (q4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 5.** Quel est le principal avantage du format **Parquet** par rapport au **CSV** ?
    """)
    return


@app.cell
def _(mo):
    q5 = mo.ui.radio(
        options={
            "A. Parquet est un format texte lisible par l'humain": "A",
            "B. Parquet est colonnaire : lecture sélective, 5-10× moins d'espace, très rapide": "B",
            "C. Parquet est plus simple à créer qu'un CSV": "C",
            "D. Parquet est le seul format compatible avec pandas": "D",
        },
        label="",
    )
    q5
    return (q5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 6.** L'exemple de *Google Flu Trends* illustre principalement quel piège ?
    """)
    return


@app.cell
def _(mo):
    q6 = mo.ui.radio(
        options={
            "A. Les modèles prédictifs sont toujours fiables sur le long terme": "A",
            "B. La corrélation implique la causalité si ρ > 0.8": "B",
            "C. Corrélation ≠ causalité : les recherches « flu » sont un effet de la grippe, pas sa cause": "C",
            "D. Le Big Data résout automatiquement les biais dans les données": "D",
        },
        label="",
    )
    q6
    return (q6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 7.** Que retourne `df.shape` sur un DataFrame pandas de 1 309 lignes et 14 colonnes ?
    """)
    return


@app.cell
def _(mo):
    q7 = mo.ui.radio(
        options={
            "A. `[1309, 14]`": "A",
            "B. `(1309, 14)`": "B",
            "C. `{'rows': 1309, 'cols': 14}`": "C",
            "D. `1309`": "D",
        },
        label="",
    )
    q7
    return (q7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 8.** Dans le dataset Titanic, la colonne `Pclass` (valeurs 1, 2, 3) est :
    """)
    return


@app.cell
def _(mo):
    q8 = mo.ui.radio(
        options={
            "A. Quantitative continue, car ses valeurs sont des nombres": "A",
            "B. Quantitative discrète uniquement": "B",
            "C. Qualitative ordinale : l'ordre des classes a du sens (1ère > 2ème > 3ème)": "C",
            "D. Qualitative nominale car les numéros sont arbitraires": "D",
        },
        label="",
    )
    q8
    return (q8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### 🟢 PARTIE 2 — Séance 1-1 : Python & Librairies
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 9.** Dans **marimo**, pourquoi une variable ne peut-elle être définie que dans **une seule cellule** ?
    """)
    return


@app.cell
def _(mo):
    q9 = mo.ui.radio(
        options={
            "A. C'est une limitation technique de Python 3.13": "A",
            "B. Marimo construit un graphe de dépendances (DAG) ; plusieurs définitions créent une ambiguïté": "B",
            "C. Pour économiser la mémoire RAM": "C",
            "D. Pour être compatible avec Jupyter": "D",
        },
        label="",
    )
    q9
    return (q9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 10.** Quel est le résultat de ce code Python ?
    ```python
    notes = [12, 15, 9, 18]
    admis = [n for n in notes if n >= 10]
    print(admis)
    ```
    """)
    return


@app.cell
def _(mo):
    q10 = mo.ui.radio(
        options={
            "A. `[9, 12, 15, 18]`": "A",
            "B. `[12, 15, 18]`": "B",
            "C. `[True, True, False, True]`": "C",
            "D. `[10, 12, 15, 18]`": "D",
        },
        label="",
    )
    q10
    return (q10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 11.** Quelle structure de données Python est **immuable** (non modifiable après création) ?
    """)
    return


@app.cell
def _(mo):
    q11 = mo.ui.radio(
        options={
            "A. `list`": "A",
            "B. `dict`": "B",
            "C. `tuple`": "C",
            "D. `set`": "D",
        },
        label="",
    )
    q11
    return (q11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 12.** En **NumPy**, que signifie le terme *vectorisation* ?
    """)
    return


@app.cell
def _(mo):
    q12 = mo.ui.radio(
        options={
            "A. Convertir des données texte en vecteurs numériques": "A",
            "B. Appliquer une opération à tout un tableau sans boucle Python, en déléguant à du code C compilé": "B",
            "C. Dessiner des vecteurs dans un graphique matplotlib": "C",
            "D. Encoder des variables catégorielles avec `LabelEncoder`": "D",
        },
        label="",
    )
    q12
    return (q12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 13.** Quelle est la différence entre `df.loc[]` et `df.iloc[]` dans pandas ?
    """)
    return


@app.cell
def _(mo):
    q13 = mo.ui.radio(
        options={
            "A. `loc` sélectionne par *labels* (noms) ; `iloc` sélectionne par *indices entiers*": "A",
            "B. `loc` est réservé aux colonnes numériques ; `iloc` aux colonnes texte": "B",
            "C. `loc` est plus rapide que `iloc` sur les grands datasets": "C",
            "D. Il n'y a aucune différence, ils sont interchangeables": "D",
        },
        label="",
    )
    q13
    return (q13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 14.** Quel est le principal avantage de **Polars** par rapport à pandas pour les grands volumes ?
    """)
    return


@app.cell
def _(mo):
    q14 = mo.ui.radio(
        options={
            "A. Polars compresse les données automatiquement": "A",
            "B. Polars est écrit en Rust, utilise le multi-threading et l'évaluation lazy — 5 à 20× plus rapide": "B",
            "C. Polars est compatible avec Excel, pandas ne l'est pas": "C",
            "D. Polars permet de faire du ML sans scikit-learn": "D",
        },
        label="",
    )
    q14
    return (q14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Question 15.** Dans le pipeline **scikit-learn**, quel est l'ordre correct des étapes ?
    """)
    return


@app.cell
def _(mo):
    q15 = mo.ui.radio(
        options={
            "A. `predict` → `fit` → `score`": "A",
            "B. `fit` → `predict` → `score`": "B",
            "C. `score` → `fit` → `predict`": "C",
            "D. `fit` → `score` → `predict`": "D",
        },
        label="",
    )
    q15
    return (q15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell
def _(mo):
    submit_btn = mo.ui.button(label="✅ Voir mes résultats", kind="success")
    submit_btn
    return (submit_btn,)


@app.cell
def _(
    mo,
    q1,
    q10,
    q11,
    q12,
    q13,
    q14,
    q15,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    q8,
    q9,
    submit_btn,
):
    BONNES_REPONSES = {
        "q1": "B", "q2": "D", "q3": "C", "q4": "C", "q5": "B",
        "q6": "C", "q7": "B", "q8": "C", "q9": "B", "q10": "B",
        "q11": "C", "q12": "B", "q13": "A", "q14": "B", "q15": "B",
    }

    EXPLICATIONS = {
        "q1": "L'analyse de données (descriptive/diagnostique) répond à *ce qui s'est passé*, tandis que la science des données construit des modèles prédictifs pour l'avenir.",
        "q2": "L'approche **prescriptive** recommande des actions concrètes — ex : envoyer une offre ciblée pour réduire le churn.",
        "q3": "Le niveau d'études est **ordinal** : il y a un ordre logique entre les niveaux (L1 < L2 < M1 < M2), mais l'écart entre niveaux n'est pas uniforme.",
        "q4": "CRISP-DM commence par la **compréhension métier** : définir le problème, les objectifs et les critères de succès avant toute manipulation de données.",
        "q5": "Parquet est **colonnaire** : il lit uniquement les colonnes demandées (pas toutes les lignes), ce qui le rend 5–10× plus compact et beaucoup plus rapide que CSV.",
        "q6": "Google Flu Trends illustre que **corrélation ≠ causalité** : les recherches *flu* augmentent *parce que* les gens sont malades — c'est un effet, pas une cause.",
        "q7": "`df.shape` retourne un **tuple** `(n_lignes, n_colonnes)` — ici `(1309, 14)`.",
        "q8": "`Pclass` est **qualitative ordinale** : les valeurs 1, 2, 3 ont un ordre significatif (1ère > 2ème > 3ème), mais ce ne sont pas de vraies quantités numériques.",
        "q9": "Marimo construit un **DAG** de dépendances entre cellules. Si une variable est définie deux fois, le graphe est ambigu → erreur `multiple-definitions`.",
        "q10": "La compréhension de liste filtre uniquement les notes ≥ 10 : **[12, 15, 18]** (9 est exclu car 9 < 10).",
        "q11": "Le **tuple** `(1, 2, 3)` est immuable : on ne peut pas modifier ses éléments après création. Les listes, dicts et sets sont mutables.",
        "q12": "La **vectorisation** délègue les calculs à du code C compilé sans boucle Python explicite — c'est pourquoi NumPy est 10–100× plus rapide pour les calculs numériques.",
        "q13": "`loc` sélectionne par **labels** (noms d'index/colonnes) ; `iloc` sélectionne par **position entière** (0, 1, 2…).",
        "q14": "Polars est écrit en **Rust**, exploite le **multi-threading** et l'évaluation **lazy** (optimisation du plan avant exécution) — nativement compatible Apache Arrow.",
        "q15": "Le pipeline sklearn suit toujours : `fit(X_train, y_train)` → `predict(X_test)` → `score` ou calcul de métriques.",
    }

    reponses = {
        "q1": q1.value, "q2": q2.value, "q3": q3.value,
        "q4": q4.value, "q5": q5.value, "q6": q6.value,
        "q7": q7.value, "q8": q8.value, "q9": q9.value,
        "q10": q10.value, "q11": q11.value, "q12": q12.value,
        "q13": q13.value, "q14": q14.value, "q15": q15.value,
    }

    _ = submit_btn.value

    sans_reponse = [k for k, v in reponses.items() if v is None]

    if sans_reponse:
        result_ui = mo.callout(
            mo.md(f"⚠️ **{len(sans_reponse)} question(s) sans réponse.** Réponds à toutes les questions avant de soumettre."),
            kind="warn",
        )
    else:
        score = sum(1 for k, v in reponses.items() if v == BONNES_REPONSES[k])
        total = len(BONNES_REPONSES)
        pct = round(score / total * 100)

        if pct >= 80:
            niveau, emoji, kind = "Excellent !", "🏆", "success"
        elif pct >= 60:
            niveau, emoji, kind = "Bien — quelques points à revoir", "👍", "info"
        else:
            niveau, emoji, kind = "À retravailler — relis les séances", "📖", "warn"

        lignes = []
        for i, (k, bonne) in enumerate(BONNES_REPONSES.items(), 1):
            eleve = reponses[k]
            icone = "✅" if eleve == bonne else "❌"
            lignes.append(
                f"{icone} **Q{i}** — Ta réponse : `{eleve}` | Bonne réponse : `{bonne}`\n\n"
                f"  > {EXPLICATIONS[k]}"
            )

        result_ui = mo.vstack([
            mo.callout(
                mo.md(f"## {emoji} Score : {score} / {total} ({pct} %)\n\n**{niveau}**"),
                kind=kind,
            ),
            mo.md("---\n### 🔍 Détail question par question\n\n" + "\n\n---\n\n".join(lignes)),
        ])

    result_ui
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
