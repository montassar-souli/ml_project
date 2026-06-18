# -*- coding: utf-8 -*-
"""
deploiement_prefect.py
Fichier de déploiement et de planification (Scheduling) pour les flows Prefect (v3).
"""

import argparse
from datetime import timedelta
import signal
import subprocess
import sys
import time

from prefect.client.schemas.schedules import IntervalSchedule
# Importation des flows existants depuis votre fichier d'orchestration
from pipeline_prefect import all_pipeline_flow, train_pipeline_flow, evaluate_pipeline_flow

def deploy_and_serve_pipelines():
    print("[DEPLOY] Préparation des configurations de déploiement...")

    # 1. Définition du planning pour le flow global (Une fois par jour -> Intervalle de 24 heures)
    daily_schedule = IntervalSchedule(interval=timedelta(days=1))

    # 2. Préparation du déploiement pour le flow 'all' (Avec planning journalier)
    deployment_all = all_pipeline_flow.to_deployment(
        name="ml-pipeline-all",
        version="1.0.0",
        schedule=daily_schedule,
        tags=["production", "ci-cd-ml"]
    )

    # 3. Préparation du déploiement pour le flow 'entrainement' (Déclenchement manuel uniquement)
    deployment_train = train_pipeline_flow.to_deployment(
        name="ml-pipeline-train",
        version="1.0.0",
        tags=["dev", "training-only"]
    )

    print("[DEPLOY] Lancement du processus d'écoute (Worker intégré via .serve)...")
    print("[INFO] Le flow 'all' s'exécutera automatiquement toutes les 24 heures.")
    print("[INFO] En attente d'ordres de déclenchement sur http://localhost:4200 ...")
    
    # 4. Mise en production (Maintient le script actif pour intercepter et exécuter les jobs)
    from prefect import serve
    serve(deployment_all, deployment_train)


def start_mlflow_and_prefect_servers(
    mlflow_host: str = "0.0.0.0",
    mlflow_port: int = 5000,
):
    """Lance MLflow UI et le serveur Prefect dans deux processus séparés."""

    print("[SERVERS] Démarrage de MLflow UI et du serveur Prefect...")
    mlflow_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--host",
            mlflow_host,
            "--port",
            str(mlflow_port),
        ]
    )
    prefect_process = subprocess.Popen(["prefect", "server", "start"])

    processes = [
        ("mlflow", mlflow_process),
        ("prefect", prefect_process),
    ]

    def stop_processes(*_):
        print("[SERVERS] Arrêt des services en cours...")
        for _, process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_processes)
    signal.signal(signal.SIGTERM, stop_processes)

    print(f"[SERVERS] MLflow UI disponible sur http://{mlflow_host}:{mlflow_port}")
    print("[SERVERS] Prefect Server est en cours de démarrage via 'prefect server start'.")

    try:
        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"[SERVERS] Le processus {name} s'est arrêté avec le code {exit_code}.")
                    stop_processes()
                    raise SystemExit(exit_code)
            time.sleep(1)
    except KeyboardInterrupt:
        stop_processes()
        for _, process in processes:
            process.wait()
        print("[SERVERS] Services arrêtés.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Déploiement Prefect et lancement des services MLflow/Prefect."
    )
    parser.add_argument(
        "--start-services",
        action="store_true",
        help="Lance MLflow UI et Prefect Server dans deux processus séparés.",
    )
    parser.add_argument(
        "--mlflow-host",
        default="0.0.0.0",
        help="Adresse d'écoute de MLflow UI.",
    )
    parser.add_argument(
        "--mlflow-port",
        type=int,
        default=5000,
        help="Port d'écoute de MLflow UI.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.start_services:
        start_mlflow_and_prefect_servers(
            mlflow_host=args.mlflow_host,
            mlflow_port=args.mlflow_port,
        )
    else:
        deploy_and_serve_pipelines()