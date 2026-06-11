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

def prepare_data(data_path: str = 'Churn_Modelling.csv', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Charge, nettoie, encode et normalise le jeu de données Churn.
    """
    if not isinstance(data_path, str):
        raise ValueError("L'argument 'data_path' doit être une chaîne de caractères.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("L'argument 'test_size' doit être strictement compris entre 0 et 1.")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier de données '{data_path}' est introuvable.")
        
    print(f"[INFO] Chargement des données depuis '{data_path}'...")
    df = pd.read_csv(data_path)
    
    if df.empty:
        raise ValueError("Le fichier de données est vide.")

    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    if 'Exited' not in df.columns:
        raise KeyError("La colonne cible 'Exited' est manquante dans le dataset.")
        
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> LinearSVC:
    """
    Entraîne un classifieur LinearSVC sur les données fournies.
    """
    if len(X_train) != len(y_train):
        raise ValueError("Tailles incohérentes entre X_train et y_train.")

    model = LinearSVC(random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LinearSVC, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Évalue les performances d'un modèle LinearSVC sur un échantillon de test.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    matrix = confusion_matrix(y_test, y_pred)
    
    return {"accuracy": accuracy, "report": report, "matrix": matrix}

def save_model(model: LinearSVC, scaler: StandardScaler, model_path: str = 'svc_model.pkl', scaler_path: str = 'scaler.pkl') -> None:
    """
    Sérialise et sauvegarde sur le disque le modèle et son scaler associé.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path: str = 'svc_model.pkl', scaler_path: str = 'scaler.pkl') -> Tuple[LinearSVC, StandardScaler]:
    """
    Charge un modèle et un scaler sérialisés depuis le stockage local.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Artefacts introuvables.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler