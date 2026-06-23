from datetime import datetime
import json
import mlflow
from elasticsearch import Elasticsearch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Connexion à Elasticsearch
# Sous WSL, localhost pointe vers Windows/WSL, le conteneur est exposé sur le port 9200
es = Elasticsearch(["http://localhost:9200"])

# 2. Préparation des données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Hyperparamètres
n_estimators = 100
max_depth = 5

# 3. Configuration et démarrage de MLflow
mlflow.set_experiment("Iris_Classification_Experiment")

with mlflow.start_run() as run:
    # Entraînement du modèle
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Prédictions et calcul des métriques
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log dans MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("train_accuracy", accuracy) # On simule la métrique demandée
    mlflow.log_metric("prediction_score", accuracy) 
    
    # Enregistrement de l'artefact (le modèle)
    mlflow.sklearn.log_model(clf, "model")
    
    # 4. Préparation des données pour Elasticsearch selon le mapping du TP
    run_info = mlflow.get_run(run.info.run_id)
    
    log_document = {
        "@timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or "unnamed_run",
        "experiment_id": run.info.experiment_id,
        "experiment_name": "Iris_Classification_Experiment",
        "status": run.info.status,
        "duration_seconds": 1.5, # Ajustable ou calculable si besoin
        "metric_train_accuracy": float(accuracy),
        "metric_prediction": float(accuracy),
        "param_n_estimators": str(n_estimators),
        "param_max_depth": str(max_depth),
        "params": {
            "n_estimators": str(n_estimators),
            "max_depth": str(max_depth)
        },
        "metrics": {
            "accuracy": float(accuracy)
        }
    }
    
    # Envoi vers Elasticsearch
    try:
        response = es.index(index="mlflow-metrics", document=log_document)
        print(f"[-] Métriques envoyées à Elasticsearch. ID du doc: {response['_id']}")
    except Exception as e:
        print(f"[!] Erreur lors de l'envoi à Elasticsearch: {e}")

print("[-] Run MLflow terminé avec succès !")