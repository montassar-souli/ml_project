# -*- coding: utf-8 -*-
"""
pipeline_prefect.py
Orchestration Prefect (v3) intégrant le suivi de code, la qualité, la sécurité et les tests unitaires.
"""

import argparse
import sys
import subprocess
from prefect import task, flow
import main

# Liste explicite et optimisée des fichiers du projet à analyser
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

# ==========================================
#          TÂCHES DE SUIVI DU CODE
# ==========================================

@task(name="1. Installer/Vérifier les Dépendances", retries=1)
def task_install_dependencies():
    main.run_dependencies_install()

@task(name="Code - 1. Formatage du code (Black)")
def task_format_code():
    print(f"[QUALITY] Vérification du formatage sur : {PROJECT_FILES}")
    # --check renvoie un code d'erreur si le code n'est pas bien formaté sans le modifier
    result = subprocess.run(["black", "--check"] + PROJECT_FILES, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARNING] Le code n'est pas parfaitement formaté selon les standards Black.")
        print(result.stderr)
    else:
        print("[OK] Formatage du code conforme.")

@task(name="Code - 2. Qualité du code (Flake8)")
def task_lint_code():
    print(f"[QUALITY] Analyse de la qualité syntaxique sur : {PROJECT_FILES}")
    # Configuration rapide en ligne de commande pour éviter les fichiers externes
    result = subprocess.run(["flake8", "--max-line-length=120"] + PROJECT_FILES, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARNING] Des alertes de qualité de code ont été détectées :")
        print(result.stdout)
    else:
        print("[OK] Qualité du code excellente (0 anomalie Flake8).")

@task(name="Code - 3. Sécurité du code (Bandit)")
def task_security_check():
    print(f"[SECURITY] Analyse de sécurité optimisée sur : {PROJECT_FILES}")
    # Bandit scanne uniquement nos fichiers cibles (-r pour récursif si besoin, mais ici la liste suffit)
    result = subprocess.run(["bandit"] + PROJECT_FILES, capture_output=True, text=True)
    if result.returncode != 0:
        print("[SECURITY WARNING] Potentielles failles détectées :")
        print(result.stdout)
    else:
        print("[OK] Aucun problème de sécurité critique détecté par Bandit.")

@task(name="Code - 4. Exécution des tests unitaires (Pytest)")
def task_run_tests():
    print("[TESTS] Exécution de la suite de tests unitaires via Pytest...")
    result = subprocess.run(["pytest", "test_pipeline.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("La suite de tests unitaires a échoué.")
    print("[OK] Tous les tests unitaires sont passés avec succès.")


# ==========================================
#          TÂCHES ML CLASSIQUES
# ==========================================

@task(name="2. Préparation des Données")
def task_prepare_data(data_path: str = 'Churn_Modelling.csv'):
    return main.run_preparation(data_path)

@task(name="3. Entraînement du Modèle")
def task_train_model(X_train, y_train):
    return main.run_training(X_train, y_train)

@task(name="4. Sauvegarder le Modèle")
def task_save_model(model, scaler, model_path: str = 'svc_model.pkl', scaler_path: str = 'scaler.pkl'):
    main.run_saving(model, scaler, model_path, scaler_path)

@task(name="5. Charger le Modèle")
def task_load_model(model_path: str = 'svc_model.pkl', scaler_path: str = 'scaler.pkl'):
    return main.run_loading(model_path, scaler_path)

@task(name="6. Évaluer le Modèle")
def task_evaluate_model(model, X_test, y_test):
    return main.run_evaluation(model, X_test, y_test)


# ==========================================
#          DÉFINITION DES FLOWS
# ==========================================

@flow(name="code", description="Flow dédié à la qualité, au formatage, à la sécurité et aux tests du code.")
def code_pipeline_flow():
    print("=== [FLOW] Début du Flow: code ===")
    task_install_dependencies()
    task_format_code()
    task_lint_code()
    task_security_check()
    task_run_tests()
    print("=== [FLOW] Fin du Flow: code ===")


@flow(name="all", description="Flow complet exécutant le suivi du code suivi du pipeline ML complet.")
def all_pipeline_flow(data_path: str = 'Churn_Modelling.csv'):
    print("=== [FLOW] Début du Flow GLOBAL: all ===")
    
    # Étape de suivi du code (Ajoutées juste après le chargement des dépendances)
    task_install_dependencies()
    task_format_code()
    task_lint_code()
    task_security_check()
    task_run_tests()
    
    # Pipeline ML (Exécuté uniquement si les étapes du dessus n'ont pas planté le flow)
    X_train, X_test, y_train, y_test, scaler = task_prepare_data(data_path)
    model = task_train_model(X_train, y_train)
    task_save_model(model, scaler)
    metrics = task_evaluate_model(model, X_test, y_test)
    
    print("=== [FLOW] Fin du Flow GLOBAL: all ===")
    return metrics


@flow(name="entrainement", description="Flow de préparation des données et d'entraînement du modèle.")
def train_pipeline_flow(data_path: str = 'Churn_Modelling.csv'):
    print("=== [FLOW] Début du Flow: entrainement ===")
    X_train, _, y_train, _, scaler = task_prepare_data(data_path)
    model = task_train_model(X_train, y_train)
    task_save_model(model, scaler)
    print("=== [FLOW] Fin du Flow: entrainement ===")


@flow(name="evaluate", description="Flow de chargement du modèle et d'évaluation.")
def evaluate_pipeline_flow(data_path: str = 'Churn_Modelling.csv'):
    print("=== [FLOW] Début du Flow: evaluate ===")
    _, X_test, _, y_test, _ = task_prepare_data(data_path)
    model_loaded, _ = task_load_model()
    metrics = task_evaluate_model(model_loaded, X_test, y_test)
    print("=== [FLOW] Fin du Flow: evaluate ===")
    return metrics


# ==========================================
#         INTERFACE CLI (ARGPARSE)
# ==========================================

def parse_and_run():
    parser = argparse.ArgumentParser(description="Automatisation des Flows Prefect (CI/CD et ML Pipeline).")
    parser.add_argument(
        '--flow', 
        type=str, 
        required=True, 
        choices=['all', 'entrainement', 'evaluate', 'code'],
        help="Nom du flow Prefect à exécuter (all, entrainement, evaluate, code)"
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='Churn_Modelling.csv',
        help="Chemin vers le fichier de données"
    )

    args = parser.parse_args()

    if args.flow == 'all':
        all_pipeline_flow(data_path=args.data)
    elif args.flow == 'entrainement':
        train_pipeline_flow(data_path=args.data)
    elif args.flow == 'evaluate':
        evaluate_pipeline_flow(data_path=args.data)
    elif args.flow == 'code':
        code_pipeline_flow()

if __name__ == '__main__':
    parse_and_run()