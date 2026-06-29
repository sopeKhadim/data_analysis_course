import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Correction de l'exercice 1 sur le dataset Titanic",
)


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np


    return np, pd, plt


@app.cell
def _td_titanic_chargement(pd):
    df_titanic = pd.read_csv("datasets/raw/titanic.csv")
    print(f"✅ Titanic chargé : {df_titanic.shape[0]} lignes × {df_titanic.shape[1]} colonnes")
    print("\n📋 Types de données :")
    print(df_titanic.dtypes.to_string())
    print("\n🔍 Valeurs manquantes :")
    _mp = df_titanic.isnull().sum()
    _pct = (_mp / len(df_titanic) * 100).round(1)
    print(pd.DataFrame({"Manquants": _mp, "%": _pct}).sort_values("%", ascending=False).to_string())
    print(f"\n🔁 Doublons : {df_titanic.duplicated().sum()}")
    return (df_titanic,)


@app.cell
def _td_titanic_ex2(df_titanic):
    print("=== Exercice 2 — Mécanismes MCAR/MAR/MNAR ===\n")

    # Indice : boat manquant si décédé, body manquant si survivant ou corps non retrouvé
    print("Lien entre 'boat' (canot) et 'survived' :")
    print(df_titanic.groupby("survived")["boat"].apply(lambda x: x.isnull().mean()).rename("% manquant boat").round(3).to_string())

    print("\nLien entre 'body' (corps) et 'survived' :")
    print(df_titanic.groupby("survived")["body"].apply(lambda x: x.isnull().mean()).rename("% manquant body").round(3).to_string())

    print("\nLien entre 'age' manquant et 'pclass' (indice MAR) :")
    print(df_titanic.groupby("pclass")["age"].apply(lambda x: x.isnull().mean()).rename("% manquant age").round(3).to_string())

    print("\nLien entre 'cabin' manquant et 'pclass' (indice MAR) :")
    print(df_titanic.groupby("pclass")["cabin"].apply(lambda x: x.isnull().mean()).rename("% manquant cabin").round(3).to_string())
    return


@app.cell
def _td_titanic_ex3(df_titanic, plt):
    from sklearn.impute import SimpleImputer as _SI_td

    print("=== Exercice 3 — Comparaison stratégies A et B ===\n")

    # Stratégie A : suppression des lignes avec manquants sur age, embarked, fare
    _cols_a = ["age", "embarked", "fare"]
    _df_a = df_titanic.dropna(subset=_cols_a)
    print(f"Stratégie A — suppression listwise sur {_cols_a}")
    print(f"  Passagers conservés : {len(_df_a)} / {len(df_titanic)} ({len(_df_a)/len(df_titanic)*100:.1f}%)")
    print(f"  Taux de survie      : {_df_a['survived'].mean():.3f}")
    print(f"  Répartition M/F     : {_df_a['sex'].value_counts().to_dict()}")

    # Stratégie B : imputation ciblée
    _df_b = df_titanic.copy()

    # age → médiane par pclass + sex
    _df_b["age"] = _df_b.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    # fallback si groupe entier manquant
    _df_b["age"] = _df_b["age"].fillna(_df_b["age"].median())

    # embarked → mode
    _mode_emb = _df_b["embarked"].mode()[0]
    _df_b["embarked"] = _df_b["embarked"].fillna(_mode_emb)

    # fare → médiane
    _df_b["fare"] = _df_b["fare"].fillna(_df_b["fare"].median())

    # indicatrice cabin
    _df_b["has_cabin"] = _df_b["cabin"].notna().astype(int)

    print(f"\nStratégie B — imputation ciblée")
    print(f"  Passagers conservés : {len(_df_b)} / {len(df_titanic)} ({len(_df_b)/len(df_titanic)*100:.1f}%)")
    print(f"  Taux de survie      : {_df_b['survived'].mean():.3f}")
    print(f"  Répartition M/F     : {_df_b['sex'].value_counts().to_dict()}")
    print(f"  Passagers avec cabine identifiée : {_df_b['has_cabin'].sum()} ({_df_b['has_cabin'].mean()*100:.1f}%)")

    print(f"\n📌 Différence de taux de survie A vs B : {abs(_df_a['survived'].mean() - _df_b['survived'].mean()):.4f}")
    print("   → Un écart significatif indique un biais introduit par la stratégie A.")

    # Visualisation
    _fig_td, _axes_td = plt.subplots(1, 3, figsize=(15, 4))

    # Âge avant / après imputation
    _axes_td[0].hist(df_titanic["age"].dropna(), bins=30, alpha=0.6, label="Original", color="#3498db")
    _axes_td[0].hist(_df_b["age"], bins=30, alpha=0.5, label="Après imputation", color="#e67e22")
    _axes_td[0].set_title("Distribution de l'âge\navant / après imputation", fontweight="bold")
    _axes_td[0].set_xlabel("Âge")
    _axes_td[0].legend(fontsize=8)

    # Taux de survie par stratégie
    _axes_td[1].bar(["Stratégie A\n(suppression)", "Stratégie B\n(imputation)"],
                    [_df_a["survived"].mean(), _df_b["survived"].mean()],
                    color=["#e74c3c", "#2ecc71"], alpha=0.8, edgecolor="white")
    _axes_td[1].set_title("Taux de survie selon\nla stratégie de nettoyage", fontweight="bold")
    _axes_td[1].set_ylabel("Taux de survie")
    _axes_td[1].set_ylim(0, 0.6)
    for _i_td, _v_td in enumerate([_df_a["survived"].mean(), _df_b["survived"].mean()]):
        _axes_td[1].text(_i_td, _v_td + 0.01, f"{_v_td:.3f}", ha="center", fontweight="bold")

    # Médiane âge par pclass + sex (preuve MAR)
    _age_med = df_titanic.groupby(["pclass", "sex"])["age"].median().reset_index()
    _age_med["groupe"] = _age_med["pclass"].astype(str) + " — " + _age_med["sex"]
    _axes_td[2].barh(_age_med["groupe"], _age_med["age"],
                     color=["#3498db" if s == "male" else "#e91e8c" for s in _age_med["sex"]],
                     alpha=0.8, edgecolor="white")
    _axes_td[2].set_title("Médiane d'âge par classe × sexe\n(justification imputation MAR)", fontweight="bold")
    _axes_td[2].set_xlabel("Âge médian")

    plt.suptitle("TD Titanic — Comparaison des stratégies de nettoyage", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/td_titanic.png", dpi=120, bbox_inches="tight")
    plt.show()
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
    print(f"\n5 tarifs les plus élevés :")
    print(_fare.nlargest(5).to_string())
    print(f"\n📌 Ces tarifs correspondent à des passagers de 1ère classe avec cabines privées.")
    print("   → Valeurs légitimes (tarifs premium), pas des erreurs. À conserver.")

    _fig_f, _axes_f = plt.subplots(1, 2, figsize=(12, 4))
    _axes_f[0].boxplot(_fare, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#3498db", alpha=0.6),
                       flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=4, alpha=0.6))
    _axes_f[0].set_title("Boxplot — Fare (tarif)", fontweight="bold")
    _axes_f[0].set_ylabel("Fare (£)")

    _axes_f[1].hist(_fare[_fare <= 200], bins=40, color="#3498db", alpha=0.7, edgecolor="white")
    _axes_f[1].axvline(_b_sup_f, color="orange", linestyle="--", linewidth=2, label=f"Borne IQR sup = {_b_sup_f:.0f} £")
    _axes_f[1].set_title("Distribution des tarifs (≤ 200 £)", fontweight="bold")
    _axes_f[1].set_xlabel("Fare (£)")
    _axes_f[1].legend(fontsize=9)

    plt.suptitle("TD Titanic — Détection des outliers sur 'fare'", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/tmp/td_titanic_outliers.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
