# Analyse de Données — Cours 24h

**Institut Polytechnique de Saint-Louis (IPSL) / Analyse de Données · 2025–2026**
**Niveau :** 1ère année ingénieur (GeIT1) | **Volume :** 24h — 6 séances 

---

## Description

Ce dépôt contient les supports de cours, exercices (TD) et travaux pratiques (TP) du module **Analyse de Données**.

Le cours couvre l'intégralité du cycle de vie d'un projet data — de la collecte et du nettoyage jusqu'à la modélisation et l'industrialisation d'un pipeline — avec une approche résolument pratique en Python.

L'environnement repose sur **marimo** (notebooks réactifs, alternative moderne à Jupyter) et **uv** (gestionnaire de dépendances ultra-rapide, alternative à conda/pip).

### Plan des séances

| # | Thème | Outils                            |
|---|-------|-----------------------------------|
| 1 | Introduction à l'analyse de données | pandas, marimo, uv                |
| 2 | Collecte, nettoyage et préparation | pandas, ydata-profiling           |
| 3 | Analyse statistique et visualisation exploratoire (EDA) | scipy, seaborn, plotly, statsmodels |
| 4 | Échantillonnage et réduction de dimension | scikit-learn, PCA                 |
| 5 | Clustering — K-Means et segmentation | scikit-learn, seaborn             |
| 6 | Régression et classification supervisées | scikit-learn, SHAP                |
| 7 | Industrialisation d'un pipeline | MLflow                   |

---

## Prérequis

- Python 3.12+
- Bases en Python (fonctions, listes, dictionnaires)
- Notions de probabilités et d'algèbre linéaire

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/sopekhadim/data_analysis.git
cd data_analysis
```

### 2. Installer `uv`

`uv` est un gestionnaire de paquets et d'environnements Python très rapide (remplace pip + conda).

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Vérifier l'installation :

```bash
uv --version
```

### 3. Installer `marimo` globalement (commande disponible sans `uv run`)

```bash
uv tool install "marimo[recommended]"
```

Après cette commande, `marimo` est accessible directement dans le terminal :

```bash
marimo edit seance3_analyse_statistique_EDA.py
marimo run seance3_analyse_statistique_EDA.py
```

> **Note :** `uv tool install` installe l'outil dans un environnement isolé global,
> distinct du projet. Cela permet d'utiliser `marimo` comme une commande système,
> sans avoir à préfixer chaque commande avec `uv run`.

### 4. Installer les dépendances du projet

```bash
uv sync
```

Cette commande lit le fichier `pyproject.toml`, crée un environnement virtuel `.venv/` et installe toutes les dépendances (pandas, scipy, seaborn, plotly, statsmodels, marimo…).

---

## Utilisation

### Lancer un notebook marimo

```bash
# Avec uv tool install marimo (recommandé — commande directe)
marimo edit seance1-0_introduction_analyse_donnees.py
marimo edit seance2_collecte_nettoyage_preparation.py
marimo edit seance3_analyse_statistique_EDA.py

# Ou avec uv run (sans installation globale)
uv run marimo edit seance3_analyse_statistique_EDA.py
```

### Mode lecture seule (présentation)

```bash
marimo run seance3_analyse_statistique_EDA.py
```

### Créer un nouveau notebook

```bash
marimo new mon_notebook.py
```

### Vérifier les erreurs d'un notebook sans le lancer

```bash
marimo check seance3_analyse_statistique_EDA.py
```

---

## Commandes `uv` utiles

```bash
# Ajouter une dépendance
uv add numpy

# Ajouter une dépendance de développement
uv add --dev pytest

# Supprimer une dépendance
uv remove nom-du-paquet

# Mettre à jour toutes les dépendances
uv sync --upgrade

# Lister les paquets installés
uv pip list

# Exécuter un script Python dans l'environnement du projet
uv run python main.py
```

---

## Commandes `marimo` utiles

```bash
# Éditer un notebook (mode interactif)
marimo edit notebook.py

# Exécuter un notebook en mode application
marimo run notebook.py

# Convertir un notebook Jupyter (.ipynb) en marimo
marimo convert notebook.ipynb -o notebook.py

# Lancer le tutoriel interactif marimo
marimo tutorial intro

# Mettre à jour marimo
uv tool upgrade marimo
```

---

## Structure du dépôt

```
data_analysis/
├── seance1_introduction_analyse_donnees_v3.py   # Séance 1 — Introduction
├── seance2_collecte_nettoyage_preparation.py     # Séance 2 — Nettoyage
├── seance3_analyse_statistique_EDA.py            # Séance 3 — Stats & EDA
├── datasets/                                     # Jeux de données locaux
├── Syllabus/                                     # Syllabus détaillé du cours
├── pyproject.toml                                # Dépendances du projet
└── uv.lock                                       # Versions exactes (reproductibilité)
```

---

## Dépendances principales

| Librairie | Usage |
|-----------|-------|
| `pandas` | Manipulation de données tabulaires |
| `numpy` | Calcul numérique |
| `scipy` | Tests statistiques |
| `statsmodels` | Modèles statistiques avancés |
| `seaborn` / `matplotlib` | Visualisation statique |
| `plotly` | Visualisation interactive |
| `scikit-learn` | Machine learning |
| `marimo` | Notebooks réactifs |

---

## Références

- McKinney, W. (2022). *Python for Data Analysis*, 3e éd. O'Reilly.
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly.
- Documentation marimo : [marimo.io](https://marimo.io)
- Documentation uv : [docs.astral.sh/uv](https://docs.astral.sh/uv)