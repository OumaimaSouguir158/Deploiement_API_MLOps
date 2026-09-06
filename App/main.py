"""
API REST d'inférence pour le modèle de prédiction de churn.
Expose un endpoint /predict qui charge le modèle sérialisé (RandomForest)
et retourne la probabilité de désabonnement pour un client donné.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

app = FastAPI(
    title="API de prédiction de churn",
    description="Expose un modèle de Machine Learning (RandomForest) permettant "
                "de prédire la probabilité de résiliation d'un client télécom.",
    version="1.0.0",
)

# Chargement des artefacts au démarrage du service
model = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
feature_order = joblib.load(os.path.join(MODEL_DIR, "feature_order.joblib"))


class ClientFeatures(BaseModel):
    tenure_mois: int = Field(..., ge=0, le=100, description="Ancienneté en mois")
    charges_mensuelles: float = Field(..., ge=0, description="Facture mensuelle en euros")
    type_contrat: str = Field(..., description="Mensuel | 1 an | 2 ans")
    type_internet: str = Field(..., description="Fibre | DSL | Aucun")
    support_technique: str = Field(..., description="Oui | Non")
    senior: int = Field(..., ge=0, le=1)
    nb_services: int = Field(..., ge=1, le=10)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool
    model_version: str = "1.0.0"


@app.get("/health")
def health_check():
    """Endpoint de supervision (utilisé par le load balancer / Kubernetes)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures):
    try:
        row = []
        for col in feature_order:
            value = getattr(features, col)
            if col in encoders:
                if value not in encoders[col].classes_:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Valeur inconnue pour '{col}': {value}. "
                               f"Valeurs valides: {list(encoders[col].classes_)}",
                    )
                value = encoders[col].transform([value])[0]
            row.append(value)

        X = scaler.transform([row])
        proba = model.predict_proba(X)[0, 1]

        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            churn_predicted=bool(proba >= 0.5),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
