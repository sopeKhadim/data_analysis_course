import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Correction de l'exercice de nettoyage sur le dataset Titanic",
)


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    return np, pd, plt


@app.cell
def _td_titanic_chargement(pd):
    df_titanic = pd.read_csv("datasets/raw/titanic.csv")
    print(f"1. ✅ Titanic chargé : {df_titanic.shape[0]} lignes × {df_titanic.shape[1]} colonnes")
    print("\n 2. 🔍 Valeurs manquantes :")
    _mp = df_titanic.isnull().sum()
    _pct = (_mp / len(df_titanic) * 100).round(1)
    print(pd.DataFrame({"Manquants": _mp, "%": _pct}).sort_values("%", ascending=False).to_string())
    print("\n La colonne body contient presque 90 % des valeurs manquantes, elle nous renseigne les passagers qui ont survecus (corps non retrouvé).\n Il y a ~10 % qui ont été indentifiés.")
    print("\n 3. 📋 Types de données :")
    print(df_titanic.dtypes.to_string())
    print(f"\n🔁 Doublons : {df_titanic.duplicated().sum()}")
    return (df_titanic,)


@app.cell
def _(df_titanic):
    _age_by_pclass = (
        df_titanic.groupby("pclass")["age"]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .rename("% age manquant")
    )
    _age_by_pclass
    return


@app.cell(hide_code=True)
def _td_titanic_ex2(df_titanic, mo):
    # ── Preuve empirique : taux de manquants par groupe ───

    _boat_by_survived = (
        df_titanic.groupby("survived")["boat"]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .rename("% boat manquant")
    )

    _body_by_survived = (
        df_titanic.groupby("survived")["body"]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .rename("% body manquant")
    )

    _age_by_pclass = (
        df_titanic.groupby("pclass")["age"]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .rename("% age manquant")
    )

    _cabin_by_pclass = (
        df_titanic.groupby("pclass")["cabin"]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .rename("% cabin manquant")
    )

    _intro = mo.md(
        r"""
    ## Exercice 2 — Correction : Mécanismes de données manquantes

    Pour chaque colonne, le **mécanisme** détermine la stratégie d'imputation valide.
    Identifier un mécanisme incorrectement peut introduire un biais ou du *data leakage*.

    | Colonne | % Manquants | Mécanisme | Justification | Stratégie recommandée |
    |---------|:-----------:|:---------:|---------------|-----------------------|
    | `age` | ~20 % | **MAR** | Le taux varie selon `pclass` et `sex` (variables observées) — les 3e classe sont moins bien documentés | Médiane par groupe `pclass` × `sex` via `groupby().transform("median")` |
    | `cabin` | ~77 % | **MAR** | Les numéros de cabine n'existaient qu'en 1ère classe — le manquant dépend directement de `pclass` | Indicatrice `has_cabin` (1 = cabine connue) — l'absence est le signal utile, pas la valeur |
    | `embarked` | < 1 % | **MCAR** | 2 valeurs manquantes, aucune corrélation avec d'autres variables — erreurs de saisie isolées | Mode → `'S'` (Southampton, port le plus fréquent) |
    | `fare` | < 1 % | **MCAR** | 1 seule valeur manquante, aucune structure détectable | Médiane (robuste aux tarifs extrêmes de 1ère classe) |
    | `boat` | ~63 % | **MNAR** | Manque **parce que le passager est décédé** — la valeur absente encode l'issue fatale | Indicatrice `has_boat` — ne jamais imputer ; `has_boat ≈ survived` |
    | `body` | ~90 % | **MNAR** | Manque si survivant (pas de corps) ou si le corps n'a pas été retrouvé en mer | Indicatrice `body_found` — **exclure d'un modèle prédictif** (data leakage garanti) |
    """
    )

    # Preuves chiffrées : boat et body (MNAR)
    _proof_mnar = mo.vstack(
        [
            mo.md(
                "#### Preuve MNAR — `boat` et `body` : le taux de manquants dépend directement de `survived`"
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("`boat` manquant par statut de survie :"),
                            mo.plain(_boat_by_survived.to_string()),
                            mo.md(
                                "*→ Quasi-100 % de manquants chez les **décédés** (survived=0) : ils n'ont pas pris de canot. Chez les survivants, le numéro de canot est le plus souvent renseigné.*"
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("`body` manquant par statut de survie :"),
                            mo.plain(_body_by_survived.to_string()),
                            mo.md(
                                "*→ 100 % de manquants chez les **survivants** (pas de corps à identifier). Même chez les décédés, ~80 % restent manquants — la plupart des corps n'ont pas été retrouvés en mer.*"
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

    # Preuves chiffrées : age et cabin (MAR)
    _proof_mar = mo.vstack(
        [
            mo.md(
                "#### Preuve MAR — `age` et `cabin` : le taux de manquants varie selon `pclass` (variable observée)"
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("`age` manquant par classe :"),
                            mo.plain(_age_by_pclass.to_string()),
                            mo.md(
                                "*→ La 3ème classe présente davantage d'âges manquants — les passagers moins aisés étaient moins bien documentés sur les manifestes de l'époque.*"
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("`cabin` manquant par classe :"),
                            mo.plain(_cabin_by_pclass.to_string()),
                            mo.md(
                                "*→ Quasi-100 % de manquants en 3ème classe : les numéros de cabine n'étaient attribués qu'en 1ère classe (pontées et dortoirs collectifs en 2ème et 3ème).*"
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

    _synthese = mo.callout(
        mo.md(
            "`boat` et `body` sont des cas de **MNAR structurel** — la valeur manquante encode "
            "directement l'information sur la survie. Les utiliser brutes dans un modèle introduit "
            "un **data leakage**. Les indicatrices `has_boat` et `body_found` restent valides pour "
            "une **analyse historique descriptive**, mais doivent être exclues de tout modèle "
            "prédictif dont la cible est `survived`."
        ),
        kind="warn",
    )


    _output = mo.vstack([_intro, _proof_mnar, _proof_mar, _synthese])
    _output
    return


@app.cell
def _td_titanic_ex3(df_titanic, plt):
    from sklearn.impute import SimpleImputer as _SI_td

    print("=== Exercice 3 — Comparaison stratégies A et B ===\n")

    # Stratégie A : suppression des lignes avec manquants sur age, embarked, fare
    _cols_a = ["age", "embarked", "fare"]
    _df_a = df_titanic.dropna(subset=_cols_a)
    print(f"Stratégie A — suppression listwise sur {_cols_a}")
    print(
        f"  Passagers conservés : {len(_df_a)} / {len(df_titanic)} ({len(_df_a) / len(df_titanic) * 100:.1f}%)"
    )
    print(
        f"  Données perdues     : {len(df_titanic) - len(_df_a)} lignes ({(1 - len(_df_a) / len(df_titanic)) * 100:.1f}%)"
    )
    print(f"  Taux de survie      : {_df_a['survived'].mean():.3f}")
    print(f"  Répartition M/F     : {_df_a['sex'].value_counts().to_dict()}")

    # Stratégie B : imputation ciblée
    _df_b = df_titanic.copy()

    # age → médiane par pclass + sex
    _df_b["age"] = _df_b.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    # fallback si groupe entier manquant
    # _df_b["age"] = _df_b["age"].fillna(_df_b["age"].median())

    # embarked → mode
    _mode_emb = _df_b["embarked"].mode()[0]
    _df_b["embarked"] = _df_b["embarked"].fillna(_mode_emb)

    # fare → médiane
    _df_b["fare"] = _df_b["fare"].fillna(_df_b["fare"].median())

    # indicatrice cabin
    _df_b["has_cabin"] = _df_b["cabin"].notna().astype(int)

    print(f"\nStratégie B — imputation ciblée")
    print(
        f"  Passagers conservés : {len(_df_b)} / {len(df_titanic)} ({len(_df_b) / len(df_titanic) * 100:.1f}%)"
    )
    print(f"  Données perdues     : 0 lignes (0.0%)")
    print(f"  Taux de survie      : {_df_b['survived'].mean():.3f}")
    print(f"  Répartition M/F     : {_df_b['sex'].value_counts().to_dict()}")
    print(
        f"  Passagers avec cabine identifiée : {_df_b['has_cabin'].sum()} ({_df_b['has_cabin'].mean() * 100:.1f}%)"
    )

    _diff_survie = _df_a["survived"].mean() - _df_b["survived"].mean()
    print(f"\n📌 Différence de taux de survie A vs B : {abs(_diff_survie):.4f}")
    if abs(_diff_survie) > 0.01:
        print(
            "   → Écart > 1 point : la stratégie A introduit un biais mesurable."
        )
    else:
        print("   → Écart faible : biais limité sur ce dataset.")

    # Visualisation
    _fig_td, _axes_td = plt.subplots(1, 3, figsize=(15, 4))

    # Âge avant / après imputation
    _axes_td[0].hist(
        df_titanic["age"].dropna(),
        bins=30,
        alpha=0.6,
        label="Original",
        color="#3498db",
    )
    _axes_td[0].hist(
        _df_b["age"], bins=30, alpha=0.5, label="Après imputation", color="#e67e22"
    )
    _axes_td[0].axvline(
        df_titanic["age"].median(),
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Médiane globale={df_titanic['age'].median():.1f}",
    )
    _axes_td[0].set_title(
        "Distribution de l'âge\navant / après imputation", fontweight="bold"
    )
    _axes_td[0].set_xlabel("Âge")
    _axes_td[0].legend(fontsize=8)

    # Taux de survie par stratégie
    _surv_vals = [_df_a["survived"].mean(), _df_b["survived"].mean()]
    _bars_td = _axes_td[1].bar(
        ["Stratégie A\n(suppression)", "Stratégie B\n(imputation)"],
        _surv_vals,
        color=["#e74c3c", "#2ecc71"],
        alpha=0.8,
        edgecolor="white",
    )
    _axes_td[1].set_title(
        "Taux de survie selon\nla stratégie de nettoyage", fontweight="bold"
    )
    _axes_td[1].set_ylabel("Taux de survie")
    _axes_td[1].set_ylim(0, 0.65)
    for _i_td, _v_td in enumerate(_surv_vals):
        _axes_td[1].text(
            _i_td,
            _v_td + 0.01,
            f"{_v_td:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    # Médiane âge par pclass + sex (preuve MAR)
    _age_med = df_titanic.groupby(["pclass", "sex"])["age"].median().reset_index()
    _age_med["groupe"] = (
        "Classe " + _age_med["pclass"].astype(str) + " — " + _age_med["sex"]
    )
    _axes_td[2].barh(
        _age_med["groupe"],
        _age_med["age"],
        color=["#3498db" if s == "male" else "#e91e8c" for s in _age_med["sex"]],
        alpha=0.8,
        edgecolor="white",
    )
    for _i_m, _v_m in enumerate(_age_med["age"]):
        _axes_td[2].text(
            _v_m + 0.3, _i_m, f"{_v_m:.0f} ans", va="center", fontsize=9
        )
    _axes_td[2].set_title(
        "Médiane d'âge par classe × sexe\n(justification imputation MAR)",
        fontweight="bold",
    )
    _axes_td[2].set_xlabel("Âge médian")

    plt.suptitle(
        "TD Titanic — Comparaison des stratégies de nettoyage",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("/tmp/td_titanic.png", dpi=120, bbox_inches="tight")
    plt.show()

    _surv_a = _df_a["survived"].mean()
    _surv_b = _df_b["survived"].mean()
    _n_a = len(_df_a)
    _n_b = len(_df_b)
    _lost_a = len(df_titanic) - _n_a
    _sex_a = _df_a["sex"].value_counts().to_dict()
    _sex_b = _df_b["sex"].value_counts().to_dict()
    return


@app.cell(hide_code=True)
def _td_ex3_reponses(df_titanic, mo):
    # Réponses
    _cols_r3 = ["age", "embarked", "fare"]
    _df_r3a = df_titanic.dropna(subset=_cols_r3)
    _df_r3b = df_titanic.copy()
    _df_r3b["age"] = _df_r3b.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
    _df_r3b["age"] = _df_r3b["age"].fillna(_df_r3b["age"].median())
    _df_r3b["embarked"] = _df_r3b["embarked"].fillna(_df_r3b["embarked"].mode()[0])
    _df_r3b["fare"] = _df_r3b["fare"].fillna(_df_r3b["fare"].median())
    _df_r3b["has_cabin"] = _df_r3b["cabin"].notna().astype(int)

    _sa = _df_r3a["survived"].mean()
    _sb = _df_r3b["survived"].mean()
    _na = len(_df_r3a)
    _nb = len(_df_r3b)
    _lost = len(df_titanic) - _na
    _sex_a3 = _df_r3a["sex"].value_counts().to_dict()
    _sex_b3 = _df_r3b["sex"].value_counts().to_dict()
    _diff3 = abs(_sa - _sb)
    _biais3 = "**OUI**, la stratégie A introduit un biais mesurable" if _diff3 > 0.01 else "biais limité (< 1 point de différence)"

    mo.md(f"""
    ### Correction — Exercice 3 : Réponses

    #### Passagers conservés et données perdues

    | Critère | Stratégie A (suppression) | Stratégie B (imputation) |
    |---------|--------------------------|--------------------------|
    | Passagers conservés | **{_na}** / {len(df_titanic)} ({_na/len(df_titanic)*100:.1f}%) | **{_nb}** / {len(df_titanic)} (100%) |
    | Données perdues | **{_lost} lignes** ({_lost/len(df_titanic)*100:.1f}%) | **0 ligne** (0%) |
    | Taux de survie global | **{_sa:.3f}** ({_sa*100:.1f}%) | **{_sb:.3f}** ({_sb*100:.1f}%) |
    | Hommes conservés | {_sex_a3.get("male", 0)} | {_sex_b3.get("male", 0)} |
    | Femmes conservées | {_sex_a3.get("female", 0)} | {_sex_b3.get("female", 0)} |

    #### La stratégie A introduit-elle un biais ?

    {_biais3} (Δ = {_diff3:.4f}).

    La suppression listwise (stratégie A) élimine préférentiellement les passagers dont l'âge
    est manquant. Or, comme montré dans l'exercice 2, `age` est **MAR** : les manquants sont
    concentrés en **3ème classe**, qui a un **taux de survie plus faible** que les autres classes.
    En supprimant ces lignes, on sous-représente les passagers de 3ème classe → le taux de survie
    calculé sur la stratégie A est **artificiellement gonflé** par rapport à la réalité.

    La **stratégie B** préserve l'intégralité des {len(df_titanic)} passagers et impute `age` en
    tenant compte de la classe et du sexe (médiane par groupe), ce qui est plus représentatif de
    la réalité sociodémographique du bateau. Le taux de survie obtenu est donc **plus fiable**.

    > **Conclusion :** Pour une analyse de survie, la stratégie B est fortement préférable.
    > La stratégie A ne convient que si les données manquantes sont strictement MCAR, ce qui
    > n'est pas le cas ici pour `age` (mécanisme MAR prouvé dans l'exercice 2).
    """)
    return


@app.cell
def _td_titanic_ex4(df_titanic, np, plt):
    from scipy import stats as _scipy_td

    print("=== Exercice 4 — Outliers sur 'fare' ===\n")
    _fare = df_titanic["fare"].dropna()

    _Q1_f = _fare.quantile(0.25)
    _Q3_f = _fare.quantile(0.75)
    _IQR_f = _Q3_f - _Q1_f
    _b_inf_f = _Q1_f - 1.5 * _IQR_f
    _b_sup_f = _Q3_f + 1.5 * _IQR_f
    _out_iqr_f = (_fare < _b_inf_f) | (_fare > _b_sup_f)

    _z_f = np.abs(_scipy_td.zscore(_fare))
    _out_z_f = _z_f > 3

    print(f"IQR = {_IQR_f:.2f}  |  Q1 = {_Q1_f:.2f}  |  Q3 = {_Q3_f:.2f}")
    print(f"Bornes IQR : [{_b_inf_f:.2f}, {_b_sup_f:.2f}]")
    print(f"Outliers IQR    : {_out_iqr_f.sum()} ({_out_iqr_f.sum()/len(_fare)*100:.1f}%)")
    print(f"Outliers Z-score: {_out_z_f.sum()} ({_out_z_f.sum()/len(_fare)*100:.1f}%)")
    print(f"\nDifférence (IQR vs Z-score) : {_out_iqr_f.sum() - _out_z_f.sum()} outliers de plus avec IQR")
    print(f"\n5 tarifs les plus élevés :")
    print(_fare.nlargest(5).to_string())
    print(f"\nContexte des outliers — passagers à tarif élevé :")
    _top_fare = df_titanic[["pclass", "sex", "fare", "cabin", "survived"]].nlargest(10, "fare")
    print(_top_fare.to_string())

    _fig_f, _axes_f = plt.subplots(1, 3, figsize=(17, 4))

    # Boxplot
    _axes_f[0].boxplot(_fare, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#3498db", alpha=0.6),
                       flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=4, alpha=0.6))
    _axes_f[0].axhline(_b_sup_f, color="orange", linestyle="--", linewidth=1.5,
                       label=f"Borne IQR = {_b_sup_f:.0f} £")
    _axes_f[0].set_title("Boxplot — Fare (tarif)", fontweight="bold")
    _axes_f[0].set_ylabel("Fare (£)")
    _axes_f[0].legend(fontsize=8)

    # Distribution avec borne IQR
    _axes_f[1].hist(_fare[_fare <= 200], bins=40, color="#3498db", alpha=0.7, edgecolor="white",
                    label="Tarifs ≤ 200 £")
    _axes_f[1].axvline(_b_sup_f, color="orange", linestyle="--", linewidth=2,
                       label=f"Borne IQR sup = {_b_sup_f:.0f} £")
    _axes_f[1].axvline(_fare.mean(), color="purple", linestyle=":", linewidth=1.5,
                       label=f"Moyenne = {_fare.mean():.0f} £")
    _axes_f[1].set_title("Distribution des tarifs (≤ 200 £)", fontweight="bold")
    _axes_f[1].set_xlabel("Fare (£)")
    _axes_f[1].legend(fontsize=8)

    # Comparaison méthodes IQR vs Z-score
    _methods_f = ["IQR\n(1.5×IQR)", "Z-score\n(|z| > 3)"]
    _counts_f = [int(_out_iqr_f.sum()), int(_out_z_f.sum())]
    _bars_f = _axes_f[2].bar(_methods_f, _counts_f, color=["#f39c12", "#9b59b6"],
                              alpha=0.85, edgecolor="white", width=0.5)
    _axes_f[2].set_title("Nombre d'outliers détectés\nIQR vs Z-score", fontweight="bold")
    _axes_f[2].set_ylabel("Nombre d'outliers")
    for _b_f, _c_f in zip(_bars_f, _counts_f):
        _axes_f[2].text(_b_f.get_x() + _b_f.get_width()/2, _b_f.get_height() + 1,
                        f"{_c_f}\n({_c_f/len(_fare)*100:.1f}%)", ha="center",
                        fontweight="bold", fontsize=10)

    plt.suptitle("TD Titanic — Détection des outliers sur 'fare'", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/td_titanic_outliers.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _td_ex4_reponses(df_titanic, mo, np):
    from scipy import stats as _scipy_r4
    _fare_r4 = df_titanic["fare"].dropna()
    _q1_r4 = _fare_r4.quantile(0.25)
    _q3_r4 = _fare_r4.quantile(0.75)
    _iqr_r4 = _q3_r4 - _q1_r4
    _b_inf_r4 = _q1_r4 - 1.5 * _iqr_r4
    _b_sup_r4 = _q3_r4 + 1.5 * _iqr_r4
    _out_iqr_r4 = ((_fare_r4 < _b_inf_r4) | (_fare_r4 > _b_sup_r4)).sum()
    _out_z_r4 = (np.abs(_scipy_r4.zscore(_fare_r4)) > 3).sum()
    _diff_r4 = int(_out_iqr_r4) - int(_out_z_r4)
    _n_total_r4 = len(_fare_r4)
    _fare_max_r4 = _fare_r4.max()

    mo.md(f"""
    ### Correction — Exercice 4 : Réponses

    #### Question 1 — Combien d'outliers détecte la méthode IQR ?

    **Calcul pas à pas :**
    - Q1 = {_q1_r4:.2f} £  |  Q3 = {_q3_r4:.2f} £  |  IQR = Q3 − Q1 = **{_iqr_r4:.2f} £**
    - Borne inférieure = Q1 − 1.5 × IQR = {_b_inf_r4:.2f} £ → aucun tarif négatif, borne sans effet
    - Borne supérieure = Q3 + 1.5 × IQR = **{_b_sup_r4:.2f} £**

    Tout tarif supérieur à **{_b_sup_r4:.2f} £** est classé outlier par la méthode IQR.

    **→ {_out_iqr_r4} outliers détectés ({_out_iqr_r4/_n_total_r4*100:.1f}% des passagers).**

    #### Question 2 — Ces outliers sont-ils des erreurs ou des valeurs légitimes ?

    **Ce sont des valeurs légitimes.** Les tarifs les plus élevés (jusqu'à {_fare_max_r4:.0f} £)
    appartiennent tous à des passagers de **1ère classe** avec des cabines de luxe (suites privées).
    Ces tarifs reflètent la réalité tarifaire historique du Titanic (White Star Line, 1912) et
    sont vérifiables dans les archives — ce ne sont pas des erreurs de saisie.

    > **Justification :** La distribution de `fare` est fortement asymétrique à droite
    > (*right-skewed*) à cause de la structure en 3 classes. L'IQR se calibre sur la majorité
    > des passagers (2ème et 3ème classe, tarifs faibles) et signale mécaniquement les billets
    > de 1ère classe comme outliers, alors qu'ils sont parfaitement cohérents avec le contexte
    > métier. Les supprimer biaiserait l'analyse de survie.

    #### Question 3 — Comparaison IQR vs Z-score (|z| > 3)

    | Méthode | Paramètre | Outliers détectés | % du dataset |
    |---------|-----------|-------------------|-------------|
    | **IQR** | Borne sup = {_b_sup_r4:.1f} £ | **{_out_iqr_r4}** | {_out_iqr_r4/_n_total_r4*100:.1f}% |
    | **Z-score** | Seuil \\|z\\| > 3 | **{_out_z_r4}** | {_out_z_r4/_n_total_r4*100:.1f}% |
    | **Différence** | | **+{_diff_r4}** avec IQR | |

    **Oui, il y a une différence significative.** L'IQR détecte **{_diff_r4} outliers de plus**
    que le Z-score pour deux raisons :

    1. **Asymétrie de la distribution :** `fare` suit une loi quasi-exponentielle (très asymétrique).
       Le Z-score utilise la moyenne et l'écart-type qui sont eux-mêmes tirés vers le haut par les
       grands tarifs → la plage "normale" est artificiellement élargie → moins d'outliers détectés.
       L'IQR, basé sur les quartiles, est insensible à cette distorsion.

    2. **Conservatisme du seuil 3σ :** Sur une distribution normale, |z| > 3 représente 0.27% des
       données. Sur `fare` (non normale), ce seuil capte moins d'extrêmes que la règle 1.5 × IQR.

    > **Bonne pratique :** Pour des distributions asymétriques (*skewed*), l'IQR est plus adapté
    > que le Z-score qui suppose implicitement une distribution approximativement normale.

    #### Question 4 — Quelle stratégie adoptez-vous ?

    **→ Conservation des outliers sans modification, avec adaptation du prétraitement.**

    | Option | Verdict | Justification |
    |--------|---------|---------------|
    | **Suppression** | ❌ À éviter | Perd des passagers de 1ère classe → biaise l'analyse de survie |
    | **Remplacement (médiane/cap)** | ❌ À éviter | Détruit l'information sur la richesse, centrale pour la survie |
    | **Conservation + RobustScaler** | ✅ Recommandé | Préserve l'info, neutralise l'impact sur les algorithmes sensibles |
    | **Conservation + log(fare+1)** | ✅ Recommandé | Compresse l'échelle, conserve l'ordre et réduit l'asymétrie |

    > **Règle générale :** Un outlier **légitime** ne se supprime pas — on adapte le prétraitement.
    > Utiliser `RobustScaler` (médiane/IQR) ou une transformation logarithmique `np.log1p(fare)`
    > pour que l'algorithme soit moins sensible à l'amplitude des valeurs extrêmes.
    """)
    return


@app.cell
def _td_titanic_ex5(df_titanic):
    print("=== Exercice 5 — Preuves chiffrées pour la discussion ===\n")

    # Question 1 — body et survived
    print("Q1 — Lien entre 'body' (corps retrouvé) et 'survived' :")
    _body_lien = df_titanic.groupby("survived")["body"].apply(
        lambda x: x.isnull().mean() * 100
    ).rename("% body manquant").round(1)
    print(_body_lien.to_string())
    print()

    # Question 2 — biais pclass si on supprime sans cabin
    print("Q2 — Répartition pclass avec cabin vs sans cabin :")
    _cabin_pclass = df_titanic.groupby("pclass")["cabin"].apply(
        lambda x: x.notna().mean() * 100
    ).rename("% avec cabin").round(1)
    print(_cabin_pclass.to_string())
    _surv_pclass = df_titanic.groupby("pclass")["survived"].mean().round(3)
    print("\nTaux de survie par classe :")
    print(_surv_pclass.to_string())
    print()

    # Question 3 — utilité des colonnes pour un modèle
    print("Q3 — Cardinalité des colonnes à exclure :")
    for _c5 in ["boat", "body", "ticket", "name"]:
        _u5 = df_titanic[_c5].nunique()
        _m5 = df_titanic[_c5].isnull().mean() * 100
        print(f"  {_c5:<10} → {_u5:>4} valeurs uniques | {_m5:.1f}% manquants")
    print()

    # Question 4 — médiane globale vs médiane par groupe
    print("Q4 — Médiane âge globale vs par pclass × sex :")
    _med_globale = df_titanic["age"].median()
    print(f"  Médiane globale : {_med_globale:.1f} ans")
    _med_groupes = df_titanic.groupby(["pclass", "sex"])["age"].median()
    print("\n  Médiane par groupe :")
    print(_med_groupes.to_string())
    _ecart_max = (_med_groupes.max() - _med_groupes.min())
    print(f"\n  Écart max entre groupes : {_ecart_max:.1f} ans")
    print(f"  → Imputer par la médiane globale ({_med_globale:.1f} ans) introduit jusqu'à {_ecart_max:.1f} ans d'erreur")
    return


@app.cell(hide_code=True)
def _td_ex5_reponses(df_titanic, mo):
    #
    _surv_by_pclass = df_titanic.groupby("pclass")["survived"].mean().round(3)
    _cabin_by_pclass = df_titanic.groupby("pclass")["cabin"].apply(
        lambda x: x.notna().mean() * 100
    ).round(1)
    _med_glob = df_titanic["age"].median()
    _med_grp = df_titanic.groupby(["pclass", "sex"])["age"].median()
    _ecart = round(_med_grp.max() - _med_grp.min(), 1)

    # body manquant chez les survivants vs décédés
    _body_surv0 = round(df_titanic[df_titanic["survived"] == 0]["body"].isnull().mean() * 100, 1)
    _body_surv1 = round(df_titanic[df_titanic["survived"] == 1]["body"].isnull().mean() * 100, 1)

    mo.md(rf"""
    ### Correction — Exercice 5 : Réponses

    ---

    #### Question 1 — Les données manquantes dans `body` sont-elles MCAR, MAR ou MNAR ?

    **→ MNAR (Missing Not At Random).**

    | Groupe | % `body` manquant |
    |--------|------------------|
    | Décédés (`survived = 0`) | **{_body_surv0}%** |
    | Survivants (`survived = 1`) | **{_body_surv1}%** |

    Le taux de manquants est radicalement différent selon la survie :
    - Un **survivant** n'a pas de numéro de corps par définition → `body` manque à ~{_body_surv1}%.
    - Un **décédé** peut avoir un numéro si son corps a été retrouvé et identifié, mais beaucoup
      ont coulé avec le navire → `body` manque quand même à ~{_body_surv0}%.

    Le manquant dans `body` dépend directement de la **valeur de `survived`** (non observée au
    moment de la prédiction) : c'est la définition exacte du **MNAR**. Le mécanisme est
    structurel — c'est l'événement lui-même (décès + retrouvé vs non retrouvé vs survie)
    qui détermine si la valeur existe.

    > ⚠️ Utiliser `body` brut dans un modèle prédictif = **data leakage** garanti.

    ---

    #### Question 2 — Quel biais introduit-on en supprimant les passagers sans `cabin` ?

    | Classe | % passagers avec cabine connue | Taux de survie |
    |--------|-------------------------------|----------------|
    | 1ère classe | **{_cabin_by_pclass.get(1, 0):.1f}%** | **{_surv_by_pclass.get(1, 0):.3f}** ({_surv_by_pclass.get(1, 0)*100:.1f}%) |
    | 2ème classe | **{_cabin_by_pclass.get(2, 0):.1f}%** | **{_surv_by_pclass.get(2, 0):.3f}** ({_surv_by_pclass.get(2, 0)*100:.1f}%) |
    | 3ème classe | **{_cabin_by_pclass.get(3, 0):.1f}%** | **{_surv_by_pclass.get(3, 0):.3f}** ({_surv_by_pclass.get(3, 0)*100:.1f}%) |

    Supprimer les lignes sans `cabin` revient à **conserver quasi-exclusivement les passagers
    de 1ère classe** (seule classe où les cabines sont massivement renseignées).
    Cela introduit un **biais de sélection de classe** très fort :
    - On sur-représente la 1ère classe (taux de survie élevé ~{_surv_by_pclass.get(1,0)*100:.0f}%)
    - On élimine la majorité des 2ème et 3ème classe (taux de survie bien plus faibles)
    - Le taux de survie global calculé sur ce sous-ensemble serait **massivement surestimé**

    > **Bonne pratique :** Ne jamais supprimer sur `cabin`. Créer l'indicatrice `has_cabin`
    > (= proxy de `pclass`) et conserver tous les passagers.

    ---

    #### Question 3 — Faut-il conserver `boat`, `body`, `ticket`, `name` pour un modèle ?

    | Colonne | Type de problème | Verdict | Justification |
    |---------|-----------------|---------|---------------|
    | `boat` | **Data leakage** (MNAR) | ❌ Exclure | Proxy direct de `survived=1` — connu seulement après le naufrage |
    | `body` | **Data leakage** (MNAR) | ❌ Exclure | Proxy direct de `survived=0` — connu seulement après le naufrage |
    | `ticket` | **Haute cardinalité** | ❌ Exclure | Identifiant quasi-unique, aucun signal prédictif direct |
    | `name` | **Haute cardinalité** | ❌ Exclure brut | Des centaines de valeurs uniques — mais on peut en extraire le **titre** (Mr, Mrs, Miss, Master…) comme feature catégorielle utile |

    > **Exception pour `name` :** Extraction du titre via `str.extract(r' ([A-Za-z]+)\.')` crée
    > une feature catégorielle pertinente (Master → enfant garçon, Mrs → femme mariée, etc.)
    > qui apporte de l'information sur l'âge et le statut social sans les inconvénients
    > d'une colonne à haute cardinalité.

    ---

    #### Question 4 — Médiane globale vs médiane par `pclass` × `sex` pour `age`

    | Groupe | Médiane d'âge | Écart avec médiane globale ({_med_glob:.1f} ans) |
    |--------|--------------|--------------------------------------------------|
    | Classe 1 — female | {_med_grp.get((1,"female"), float("nan")):.0f} ans | {abs(_med_grp.get((1,"female"), _med_glob) - _med_glob):.1f} ans |
    | Classe 1 — male   | {_med_grp.get((1,"male"), float("nan")):.0f} ans | {abs(_med_grp.get((1,"male"), _med_glob) - _med_glob):.1f} ans |
    | Classe 2 — female | {_med_grp.get((2,"female"), float("nan")):.0f} ans | {abs(_med_grp.get((2,"female"), _med_glob) - _med_glob):.1f} ans |
    | Classe 2 — male   | {_med_grp.get((2,"male"), float("nan")):.0f} ans | {abs(_med_grp.get((2,"male"), _med_glob) - _med_glob):.1f} ans |
    | Classe 3 — female | {_med_grp.get((3,"female"), float("nan")):.0f} ans | {abs(_med_grp.get((3,"female"), _med_glob) - _med_glob):.1f} ans |
    | Classe 3 — male   | {_med_grp.get((3,"male"), float("nan")):.0f} ans | {abs(_med_grp.get((3,"male"), _med_glob) - _med_glob):.1f} ans |

    L'écart maximum entre les groupes atteint **{_ecart} ans**.
    Imputer tous les âges manquants par la médiane globale ({_med_glob:.1f} ans) introduit donc
    jusqu'à **{_ecart} ans d'erreur** sur certains groupes.

    **Conséquences concrètes :**
    - Un enfant en 3ème classe (médiane ~{_med_grp.get((3,"male"), _med_glob):.0f} ans pour un garçon)
      se verrait attribuer {_med_glob:.1f} ans → **surestimation** de son âge → mauvaise classification
      (les enfants avaient une priorité différente pour les canots).
    - Un homme de 1ère classe (médiane ~{_med_grp.get((1,"male"), _med_glob):.0f} ans) se verrait
      attribuer {_med_glob:.1f} ans → **sous-estimation** de son âge réel.

    > **Conclusion :** L'imputation par groupe (`pclass` × `sex`) est **systématiquement supérieure**
    > à la médiane globale car elle respecte la structure MAR des données — les manquants dans `age`
    > sont corrélés à la classe et au sexe, donc imputer en tenant compte de ces variables
    > réduit le biais d'imputation de façon mesurable.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
