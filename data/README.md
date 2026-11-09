# Données

Le jeu **Credit Card Fraud Detection** (Machine Learning Group – ULB, publié sur Kaggle) n'est **pas versionné** dans ce dépôt :
il pèse 150 Mo et les conditions de sa source doivent être respectées (voir la page du jeu pour la licence et la citation).
Le fichier `data/creditcard.csv` est ignoré par Git (`.gitignore`) ; seuls de petits échantillons nommés `data/sample_*.csv` pourraient être versionnés.

## Obtenir le fichier

1. Télécharger l'archive depuis Kaggle : <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
   (ou avec la CLI : `kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`).
2. Placer `creditcard.csv` dans ce dossier : `data/creditcard.csv`.
3. Vérifier l'intégrité : l'empreinte MD5 attendue est `e90efcb83d69faf99fcab8b0255024de`
   (`md5sum data/creditcard.csv`). Le test `tests/test_full_data.py` la contrôle automatiquement.

## Description

| Colonne | Contenu |
|---|---|
| `Time` | secondes écoulées depuis la première transaction du jeu (≈ 2 jours) |
| `V1` … `V28` | composantes PCA **anonymisées** (ne pas les interpréter comme des variables métier) |
| `Amount` | montant de la transaction |
| `Class` | cible : 0 = normale, 1 = fraude |

284 807 transactions, dont 492 frauduleuses (0,173 %) ; 1 081 lignes strictement identiques (dont 19 fraudes), retirées avant la partition.

## Sans le jeu Kaggle

`python scripts/run_all.py --synthetic` exécute toute la chaîne sur un **proxy synthétique** de même schéma
(`app/synthetic.py`). Les résultats sont alors étiquetés « PROXY SYNTHÉTIQUE » et **ne prouvent aucune performance réelle**.
Les tests rapides (CI) n'ont jamais besoin du jeu réel : ils utilisent ce générateur déterministe.
