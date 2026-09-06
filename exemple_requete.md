# Exemple de requête / réponse de l'API

## Requête
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_mois": 4,
    "charges_mensuelles": 89.5,
    "type_contrat": "Mensuel",
    "type_internet": "Fibre",
    "support_technique": "Non",
    "senior": 0,
    "nb_services": 2
  }'
```

## Réponse
```json
{
  "churn_probability": 0.7423,
  "churn_predicted": true,
  "model_version": "1.0.0"
}
```
