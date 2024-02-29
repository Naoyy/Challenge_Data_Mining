# Estimate CO2 emissions from cars

Projet dans le cadre du cours de Data Mining de Jeannot Jynaldo et Jsem Yoan.

Accéder à la compétition Kaggle:
[Estimate CO2 emissions from cars](https://www.kaggle.com/t/1827a6edacf742a69c7336b02e155948)

Classement du groupe: 1er sur 16 équipes.
## Table des matières

* [A Propos du Projet](#a-propos-du-projet)
* [Travailler sur le projet](#travailler-sur-le-projet)
* [Organisation du Projet](#organisation-du-projet)
* [Contact](#contact)

## A Propos du Projet

Dans le cadre de cette compétition Kaggle, la promotion MoSEF 2024 se lance dans un problème d'apprentissage supervisé. L'objectif est de prédire les émissions de CO2 en (g/km) de voitures en Europe. Pour cela on dispose de données regrouppant les caractéristiques de voitures. Les groupes sont en binômes et tentent de prédire le plus justement possible la cible. La métrique d'évaluation utilisée pour cette compétition est la Mean Absolute Error (MAE) qu'il faudra minimiser. On classe ensuite les participants, l'équipe avec la MAE la plus faible étant première. 
<a name="a-propos-du-projet"></a>

## Travailler sur le projet
1. Cloner le repository
```
git clone https://github.com/Naoyy/Projet_Mining.git
cd Projet_Mining
```

2. Travailler sur le projet en créant l'environnement virtuel

```
python3 -m venv .venv_mining
source .venv_mining/bin/activate
pip install -r requirements.txt
```
<a name="travailler-sur-le-projet"></a>

## Organisation du Projet

Dans le dossier Data il faut télécharger soit même les données nécessaire à l'apprentissage du modèle. Elles sont disponibles [ici](https://www.kaggle.com/competitions/estimate-co2-emissions-from-cars/data). Une fois les données téléchargées dezip puis placer dans le répertoire ./data

Le fichier preprocessing.py contient le preprocessing utilisé pour notre meilleur soumission Kaggle. De la même façon le fichier baseline.py contient le preprocessing qui nous a servi à calculer la baseline .

Le notebook models.ipynb contient les modèles qui ont été utilisés pour les soumissions finales.

- data/ 
    - train.csv (à ajouter soi même)
    - test.csv (à ajouter soi même)
    - sample_submission.csv (exemple type de submission kaggle)

- src/
    - preprocessing.py

- results/
    - new_final_xgb_no_erwltp.csv (csv of the best submission)
    - new_kf15_xgb.csv (csv of the 2nd best submission)

- .gitignore

- README.md

- analyse_exploratoire.ipynb 

- models.ipynb

- presentation.pdf

- requirements.txt
<a name="organisation-du-projet"></a>

## Contact

- [Jynaldo Jeannot](https://github.com/jeannoj99) - jeannotjynaldo@gmail.com
- [Yoan Jsem](https://github.com/Naoyy) - yoan.jsem@gmail.com
<a name="contact"></a>
