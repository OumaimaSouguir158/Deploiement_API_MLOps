# 🟡 Projet 2 — Déploiement d'un modèle de Machine Learning en API (MLOps)

## Objectif
Exposer un modèle entraîné (ici le modèle de churn du projet 1) via une API REST, conteneurisée pour un déploiement reproductible.

## Fonctionnalités
- Endpoint de prédiction (`/predict`)
- Chargement du modèle sérialisé (joblib)
- Validation des entrées (Pydantic)
- Conteneurisation Docker
- Endpoint de supervision (`/health`)

## Stack technique
Python · FastAPI · Scikit-learn (modèle sérialisé) · Docker

## Concepts démontrés
MLOps, mise en production de modèles, sérialisation de modèles, API d'inférence.

## Structure du projet
```
02_Deploiement_API_MLOps/
├── train_model.py        # entraîne et sérialise le modèle
├── model/                 # artefacts sérialisés (model, scaler, encoders)
├── app/
│   └── main.py            # application FastAPI
├── Dockerfile
├── requirements.txt
└── exemple_requete.md
```

## Comment lancer le projet

### En local
```bash
pip install -r requirements.txt
python train_model.py          # génère les artefacts dans model/
uvicorn app.main:app --reload  # démarre l'API sur http://localhost:8000
```
Documentation interactive générée automatiquement : `http://localhost:8000/docs`

### Avec Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Voir `exemple_requete.md` pour un exemple de requête/réponse.

## Ligne CV
« Déploiement ML en API — FastAPI, Docker, MLOps. »

## Question d'entretien possible
Comment géreriez-vous la mise à jour d'un modèle déjà déployé en production sans interruption de service ?

*(Réponse : déploiement blue/green ou canary — la nouvelle version du modèle est déployée en parallèle de l'ancienne, on lui envoie progressivement une part du trafic tout en surveillant ses métriques, puis on bascule totalement une fois la confiance validée. Le versionnage des modèles (model_version dans la réponse API) permet de tracer quelle version a produit quelle prédiction.)*
