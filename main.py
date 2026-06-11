# -*- coding: utf-8 -*-
"""
main.py
Point d'entrée de l'application permettant l'exécution par arguments ou par appels programmatiques directs.
"""

import argparse
import sys
import model_pipeline

def run_dependencies_install():
    """Vérifie ou simule l'installation des dépendances requis par le projet."""
    import subprocess
    print("[MAIN] Vérification et mise à niveau des dépendances...")
    # Optionnel : décommentez la ligne ci-dessous si vous souhaitez forcer l'install automatique dans la tâche
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("[MAIN] Dépendances vérifiées avec succès.")

def run_preparation(data_path='Churn_Modelling.csv'):
    print("[MAIN] Étape : Préparation des données.")
    return model_pipeline.prepare_data(data_path)

def run_training(X_train, y_train):
    print("[MAIN] Étape : Entraînement du modèle.")
    return model_pipeline.train_model(X_train, y_train)

def run_saving(model, scaler, model_path='svc_model.pkl', scaler_path='scaler.pkl'):
    print(f"[MAIN] Étape : Sauvegarde du modèle sous {model_path}.")
    model_pipeline.save_model(model, scaler, model_path, scaler_path)

def run_loading(model_path='svc_model.pkl', scaler_path='scaler.pkl'):
    print(f"[MAIN] Étape : Chargement du modèle depuis {model_path}.")
    return model_pipeline.load_model(model_path, scaler_path)

def run_evaluation(model, X_test, y_test):
    print("[MAIN] Étape : Évaluation du modèle.")
    metrics = model_pipeline.evaluate_model(model, X_test, y_test)
    print(f"Accuracy obtenue : {metrics['accuracy']:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Gestionnaire du Pipeline de Classification du Churn Client.")
    parser.add_argument('--prepare', action='store_true', help='Prépare uniquement les données.')
    parser.add_argument('--run-all', action='store_true', help='Exécute le pipeline séquentiel complet.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.prepare:
        run_preparation()

    if args.run_all:
        run_dependencies_install()
        X_train, X_test, y_train, y_test, scaler = run_preparation()
        model = run_training(X_train, y_train)
        run_saving(model, scaler)
        model_loaded, _ = run_loading()
        run_evaluation(model_loaded, X_test, y_test)

if __name__ == '__main__':
    main()