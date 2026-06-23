# -*- coding: utf-8 -*-
"""
model_pipeline.py
Pipeline modulaire, validé et documenté pour la prédiction du Churn Client.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datetime import datetime, timezone
import os
from elasticsearch import Elasticsearch

def _get_elasticsearch():
    return Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))

def index_metrics_to_elasticsearch(run_id, experiment_name, params, metrics, report):
    es = _get_elasticsearch()
    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_name": experiment_name,
        "params": params,
        "metrics": {
            "accuracy": metrics["accuracy"],
        },
        "report": report,
    }
    es.index(index=os.getenv("ELASTICSEARCH_INDEX", "mlflow-metrics"), document=doc)
    
def _get_mlflow():
    """Charge MLflow uniquement lorsque les fonctionnalités de réentraînement en ont besoin."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow est requis uniquement pour le réentraînement ou le suivi des expériences."
        ) from exc

    return mlflow


def configure_mlflow(
    experiment_name: str = "Churn_Prediction",
    tracking_uri: str = "http://localhost:5000",
) -> None:
    """
    Configure MLflow avec l'URI de tracking et le nom de l'expérience.
    """
    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow configuré avec l'expérience: {experiment_name}")
    print(f"[INFO] URI de tracking: {tracking_uri}")


def prepare_data(
    data_path: str = "Churn_Modelling.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Charge, nettoie, encode et normalise le jeu de données Churn.
    """
    if not isinstance(data_path, str):
        raise ValueError("L'argument 'data_path' doit être une chaîne de caractères.")
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            "L'argument 'test_size' doit être strictement compris entre 0 et 1."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier de données '{data_path}' est introuvable.")

    print(f"[INFO] Chargement des données depuis '{data_path}'...")
    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Le fichier de données est vide.")

    columns_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns], errors="ignore"
    )

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    if "Exited" not in df.columns:
        raise KeyError("La colonne cible 'Exited' est manquante dans le dataset.")

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    model_params: Dict[str, Any] | None = None,
) -> LinearSVC:
    """
    Entraîne un classifieur LinearSVC sur les données fournies.
    """
    if len(X_train) != len(y_train):
        raise ValueError("Tailles incohérentes entre X_train et y_train.")

    params = {"random_state": random_state, "max_iter": 10000}
    if model_params:
        params.update(model_params)

    model = LinearSVC(**params)
    model.fit(X_train, y_train)
    return model


def retrain_model(
    data_path: str = "Churn_Modelling.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: Dict[str, Any] | None = None,
    model_path: str = "svc_model.pkl",
    scaler_path: str = "scaler.pkl",
) -> Dict[str, Any]:
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres, puis persiste les artefacts.
    Enregistre les métriques et paramètres dans MLflow.
    """
    mlflow = _get_mlflow()
    with mlflow.start_run():
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data_path=data_path, test_size=test_size, random_state=random_state
        )
        
        # Définir les paramètres
        params = {"random_state": random_state, "test_size": test_size, "max_iter": 10000}
        if model_params:
            params.update(model_params)
        
        # Logger les paramètres dans MLflow
        mlflow.log_params(params)
        
        model = train_model(
            X_train,
            y_train,
            random_state=random_state,
            model_params=model_params,
        )
        
        save_model(model, scaler, model_path=model_path, scaler_path=scaler_path)
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logger les métriques dans MLflow
        mlflow.log_metric("accuracy", metrics["accuracy"])
        
        # Logger le rapport de classification
        report = metrics["report"]
        print(f"[INFO] Rapport de classification:\n{report}")
        
        # Logger le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"[INFO] Exécution MLflow enregistrée avec les métriques: accuracy={metrics['accuracy']:.4f}")

        return {
            "model": model,
            "scaler": scaler,
            "metrics": metrics,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }


def evaluate_model(
    model: LinearSVC, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Évalue les performances d'un modèle LinearSVC sur un échantillon de test.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    matrix = confusion_matrix(y_test, y_pred)

    return {"accuracy": accuracy, "report": report, "matrix": matrix}


def save_model(
    model: LinearSVC,
    scaler: StandardScaler,
    model_path: str = "svc_model.pkl",
    scaler_path: str = "scaler.pkl",
) -> None:
    """
    Sérialise et sauvegarde sur le disque le modèle et son scaler associé.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(
    model_path: str = "svc_model.pkl", scaler_path: str = "scaler.pkl"
) -> Tuple[LinearSVC, StandardScaler]:
    """
    Charge un modèle et un scaler sérialisés depuis le stockage local.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Artefacts introuvables.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
