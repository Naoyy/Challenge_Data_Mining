# Estimate CO2 emissions from cars

Projet dans le cadre du cours de Data Mining de Jeannot Jynaldo et Jsem Yoan.

Accéder à la compétition Kaggle:
[Estimate CO2 emissions from cars](https://www.kaggle.com/t/1827a6edacf742a69c7336b02e155948)

## Table des matières

* [A Propos du Projet](#a_propos_du_projet)
* [Travailler sur le projet](#travailler_sur_le_projet)
* [Organisation du Projet](#organisation_du_projet)
* [Contact](#contact)

## A Propos du Projet

Dans le cadre de cette compétition Kaggle, la promotion MoSEF 2024 se lance dans un problème d'apprentissage supervisé. L'objectif est de prédire les émissions de CO2 en (g/km) de voitures en Europe. Pour cela on dispose de données regrouppant les caractéristiques de voitures. Les groupes sont en binômes et tentent de prédire le plus justement possible la cible. La métrique d'évaluation utilisée pour cette compétition est la Mean Absolute Error (MAE) qu'il faudra minimiser. On classe ensuite les participants, l'équipe avec la MAE la plus faible étant première. 


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


## Organisation du projet

Dans le dossier Data il faut télécharger soit même les données nécessaire à l'apprentissage du modèle. Elles sont disponibles [ici](https://www.kaggle.com/competitions/estimate-co2-emissions-from-cars/data). Une fois les données téléchargées dezip puis placer dans le répertoire ./data

Le fichier preprocessing.py contient le preprocessing utilisé pour notre meilleur soumission Kaggle. De la même façon le fichier baseline.py contient le preprocessing qui nous a servi à calculer la baseline .

Le notebook models.ipynb contient les modèles qui ont été utilisés pour les soumissions finales.

- data/ 
    - train.csv (à ajouter soi même)
    - test.csv (à ajouter soi même)
    - sample_submission.csv (exemple type de submission kaggle)

- src/
    - preprocessing.py (contient la mise en classe et le preprocessing)
    - baseline.py

- results/
    - model1.csv (csv of the best submission)
    - model2.csv (csv of the 2nd best submission)

- .gitignore

- README.md

- analyse_exploratoire.ipynb 

- models.ipynb

- presentation.pdf

- requirements.txt

## Contact

- [Jynaldo Jeannot](https://github.com/jeannoj99) - Jynaldo.Jeannot@etu.univ-paris1.fr
- [Yoan Jsem](https://github.com/Naoyy) - Yoan.Jsem@etu.univ-paris1.fr
