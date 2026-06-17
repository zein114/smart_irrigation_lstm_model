# Smart Irrigation LSTM Model

Ce projet entraîne un modèle LSTM pour prédire l'humidité du sol à partir de capteurs et de données météorologiques.

## Objectif

- Conserver le modèle LSTM existant.
- Ajouter des données NASA récentes.
- Conserver les valeurs aberrantes conformément à la demande de l'encadreur.
- Générer automatiquement des recommandations agricoles.
- Afficher un tableau de bord avec : humidité du sol, température, précipitations, NDVI et recommandation.

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage

- Télécharger les données NASA récentes :

```bash
./venv/bin/python fetch_nasa_data.py
```

- Entraîner et évaluer le modèle :

```bash
./venv/bin/python train.py
```

- Générer le tableau de bord :

```bash
./venv/bin/python dashboard.py
```

Le script crée `dashboard.html` et des graphiques PNG dans le dossier du projet.

## Données

- `data/iotsensordata.csv` contient les observations de capteurs.
- `data/nasa_recent_data.csv` contient les données météo NASA récentes et un estimateur NDVI.
