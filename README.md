# TP1 - Bonnes pratiques logicielles pour le MLOps

Ce projet transforme un entraînement exploratoire en pipeline Python reproductible.

## Installation avec Conda

```powershell
conda env create -f environment.yml
conda activate mlops_base_env
python train.py
```

Le script charge Iris, sépare les données, entraîne une forêt aléatoire, affiche son score et
enregistre le modèle dans `models/iris_model.pkl`. Le dossier `models/` est volontairement ignoré
par Git.

## Vérifications

```powershell
python train.py
git status --short --ignored
git check-ignore -v models/iris_model.pkl
```

Le dernier appel doit indiquer que la règle `models/` de `.gitignore` protège le modèle binaire.
