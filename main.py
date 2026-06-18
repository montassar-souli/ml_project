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


def run_preparation(data_path="Churn_Modelling.csv"):
    print("[MAIN] Étape : Préparation des données.")
    return model_pipeline.prepare_data(data_path)


def run_training(X_train, y_train):
    print("[MAIN] Étape : Entraînement du modèle.")
    return model_pipeline.train_model(X_train, y_train)


def run_retraining(
    data_path="Churn_Modelling.csv",
    test_size=0.2,
    random_state=42,
    model_params=None,
    model_path="svc_model.pkl",
    scaler_path="scaler.pkl",
):
    print("[MAIN] Étape : Réentraînement du modèle.")
    return model_pipeline.retrain_model(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        model_params=model_params,
        model_path=model_path,
        scaler_path=scaler_path,
    )


def run_saving(model, scaler, model_path="svc_model.pkl", scaler_path="scaler.pkl"):
    print(f"[MAIN] Étape : Sauvegarde du modèle sous {model_path}.")
    model_pipeline.save_model(model, scaler, model_path, scaler_path)


def run_loading(model_path="svc_model.pkl", scaler_path="scaler.pkl"):
    print(f"[MAIN] Étape : Chargement du modèle depuis {model_path}.")
    return model_pipeline.load_model(model_path, scaler_path)


def run_evaluation(model, X_test, y_test):
    print("[MAIN] Étape : Évaluation du modèle.")
    metrics = model_pipeline.evaluate_model(model, X_test, y_test)
    print(f"Accuracy obtenue : {metrics['accuracy']:.4f}")
    return metrics


def main():
    # Initialiser la configuration MLflow
    print("[MAIN] Initialisation de MLflow...")
    model_pipeline.configure_mlflow()
    
    parser = argparse.ArgumentParser(
        description="Gestionnaire du Pipeline de Classification du Churn Client."
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Prépare uniquement les données."
    )
    parser.add_argument(
        "--run-all", action="store_true", help="Exécute le pipeline séquentiel complet."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.prepare:
        run_preparation()

    if args.run_all:
        run_dependencies_install()
        result = run_retraining()
        print(f"[MAIN] Pipeline complété avec accuracy: {result['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
