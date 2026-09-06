"""
Entraîne le modèle de churn (réutilisé du projet 1) et le sérialise
avec joblib pour qu'il puisse être chargé par l'API FastAPI (app/main.py).
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_dataset(n=3000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly_charges = rng.normal(65, 25, n).clip(15, 150)
    contract = rng.choice(["Mensuel", "1 an", "2 ans"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fibre", "DSL", "Aucun"], n, p=[0.45, 0.35, 0.20])
    support_tech = rng.choice(["Oui", "Non"], n, p=[0.4, 0.6])
    senior = rng.choice([0, 1], n, p=[0.85, 0.15])
    num_services = rng.integers(1, 6, n)

    logit = (
        -1.2 - 0.04 * tenure + 0.015 * monthly_charges
        + (contract == "Mensuel") * 1.1 + (internet == "Fibre") * 0.4
        + (support_tech == "Non") * 0.5 + senior * 0.3 - 0.1 * num_services
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "tenure_mois": tenure,
        "charges_mensuelles": monthly_charges.round(2),
        "type_contrat": contract,
        "type_internet": internet,
        "support_technique": support_tech,
        "senior": senior,
        "nb_services": num_services,
        "churn": churn,
    })


def main():
    df = generate_dataset()

    encoders = {}
    for col in ["type_contrat", "type_internet", "support_technique"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    print(f"AUC sur le jeu de test : {auc:.3f}")

    joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_order.joblib"))
    print(f"Modèle et artefacts sauvegardés dans {MODEL_DIR}/")


if __name__ == "__main__":
    main()
