"""FastAPI application exposing churn prediction."""

from contextlib import asynccontextmanager
import logging
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

import main


logger = logging.getLogger(__name__)


GEOGRAPHY_MAPPING = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAPPING = {"Female": 0, "Male": 1}

FEATURE_COLUMNS = [
	"CreditScore",
	"Geography",
	"Gender",
	"Age",
	"Tenure",
	"Balance",
	"NumOfProducts",
	"HasCrCard",
	"IsActiveMember",
	"EstimatedSalary",
]

MODEL_PATH = "svc_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None


class ChurnRequest(BaseModel):
	CreditScore: int = Field(..., ge=0, description="Credit score du client", examples=[650])
	Geography: Literal["France", "Germany", "Spain"] = Field(..., examples=["France"])
	Gender: Literal["Female", "Male"] = Field(..., examples=["Female"])
	Age: int = Field(..., ge=0, description="Age du client", examples=[40])
	Tenure: int = Field(..., ge=0, description="Nombre d'années avec la banque", examples=[3])
	Balance: float = Field(..., ge=0, description="Solde du client", examples=[0.0])
	NumOfProducts: int = Field(..., ge=0, description="Nombre de produits détenus", examples=[1])
	HasCrCard: int = Field(..., ge=0, le=1, description="0 ou 1", examples=[1])
	IsActiveMember: int = Field(..., ge=0, le=1, description="0 ou 1", examples=[1])
	EstimatedSalary: float = Field(..., ge=0, description="Salaire estimé", examples=[50000.0])


class ChurnResponse(BaseModel):
	prediction: int = Field(..., description="Classe prédite par le modèle", examples=[0])
	churn: bool = Field(..., description="Indique si le client est classé en churn")
	score: float | None = Field(None, description="Score de décision du modèle si disponible", examples=[-0.75])


class RetrainRequest(BaseModel):
	data_path: str = Field("Churn_Modelling.csv", description="Chemin vers le dataset", examples=["Churn_Modelling.csv"])
	test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Proportion du jeu de test", examples=[0.2])
	random_state: int = Field(42, description="Graine aléatoire pour le split", examples=[42])
	C: float = Field(1.0, gt=0.0, description="Paramètre de régularisation du LinearSVC", examples=[1.0])
	max_iter: int = Field(10000, gt=0, description="Nombre maximal d'itérations", examples=[10000])
	tol: float | None = Field(None, gt=0.0, description="Tolérance d'arrêt", examples=[0.0001])
	loss: Literal["hinge", "squared_hinge"] = Field("squared_hinge", description="Fonction de perte", examples=["squared_hinge"])
	dual: bool | None = Field(None, description="Mode dual ou primal", examples=[None, True, False])
	model_path: str = Field(MODEL_PATH, description="Chemin de sauvegarde du modèle", examples=["svc_model.pkl"])
	scaler_path: str = Field(SCALER_PATH, description="Chemin de sauvegarde du scaler", examples=["scaler.pkl"])


class RetrainResponse(BaseModel):
	status: str = Field(..., description="Statut de l'opération", examples=["model_retrained"])
	model_path: str = Field(..., description="Chemin du modèle sauvegardé")
	scaler_path: str = Field(..., description="Chemin du scaler sauvegardé")
	accuracy: float = Field(..., description="Accuracy obtenue après réentraînement", examples=[0.8005])
	report: str = Field(..., description="Rapport de classification textuel")
	matrix: list[list[int]] = Field(..., description="Matrice de confusion")


def encode_features(payload: ChurnRequest) -> pd.DataFrame:
	"""Transforme la requête brute en matrice numérique compatible avec le modèle."""
	try:
		row = {
			"CreditScore": payload.CreditScore,
			"Geography": GEOGRAPHY_MAPPING[payload.Geography],
			"Gender": GENDER_MAPPING[payload.Gender],
			"Age": payload.Age,
			"Tenure": payload.Tenure,
			"Balance": payload.Balance,
			"NumOfProducts": payload.NumOfProducts,
			"HasCrCard": payload.HasCrCard,
			"IsActiveMember": payload.IsActiveMember,
			"EstimatedSalary": payload.EstimatedSalary,
		}
	except KeyError as exc:
		raise HTTPException(status_code=400, detail=f"Valeur catégorielle invalide: {exc}") from exc

	return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_from_payload(payload: ChurnRequest) -> ChurnResponse:
	"""Exécute la prédiction en gérant les erreurs applicatives de façon explicite."""
	if model is None or scaler is None:
		raise HTTPException(
			status_code=503,
			detail=(
				f"Le modèle n'est pas chargé. Vérifiez que MODEL_PATH='{MODEL_PATH}' "
				f"et SCALER_PATH='{SCALER_PATH}' pointent vers des fichiers valides."
			),
		)

	try:
		features = encode_features(payload)
		scaled_features = pd.DataFrame(
			scaler.transform(features), columns=FEATURE_COLUMNS
		)
		prediction = int(model.predict(scaled_features)[0])

		score = None
		if hasattr(model, "decision_function"):
			score = float(model.decision_function(scaled_features)[0])
		return ChurnResponse(prediction=prediction, churn=bool(prediction), score=score)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=f"Données invalides pour la prédiction: {exc}") from exc
	except FileNotFoundError as exc:
		raise HTTPException(
			status_code=503,
			detail=(
				f"Artefact manquant lors du chargement du modèle: {exc}. "
				f"Vérifiez MODEL_PATH='{MODEL_PATH}' et SCALER_PATH='{SCALER_PATH}'."
			),
		) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {exc}") from exc


def retrain_from_payload(payload: RetrainRequest) -> RetrainResponse:
	"""Réentraîne le modèle avec des hyperparamètres fournis en POST."""
	global model, scaler

	model_params: dict[str, Any] = {
		"C": payload.C,
		"max_iter": payload.max_iter,
		"loss": payload.loss,
	}
	if payload.tol is not None:
		model_params["tol"] = payload.tol
	if payload.dual is not None:
		model_params["dual"] = payload.dual

	try:
		result = main.run_retraining(
			data_path=payload.data_path,
			test_size=payload.test_size,
			random_state=payload.random_state,
			model_params=model_params,
			model_path=payload.model_path,
			scaler_path=payload.scaler_path,
		)
		model = result["model"]
		scaler = result["scaler"]
		metrics = result["metrics"]
		matrix = metrics["matrix"]
		matrix_list = matrix.tolist() if hasattr(matrix, "tolist") else matrix
		return RetrainResponse(
			status="model_retrained",
			model_path=result["model_path"],
			scaler_path=result["scaler_path"],
			accuracy=float(metrics["accuracy"]),
			report=str(metrics["report"]),
			matrix=matrix_list,
		)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=f"Impossible de réentraîner: {exc}") from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=f"Paramètres invalides pour le réentraînement: {exc}") from exc
	except TypeError as exc:
		raise HTTPException(status_code=400, detail=f"Hyperparamètre invalide pour LinearSVC: {exc}") from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Erreur interne lors du réentraînement: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Charge le modèle au démarrage de l'application."""
	global model, scaler
	try:
		model, scaler = main.run_loading(MODEL_PATH, SCALER_PATH)
		logger.info("Modèle chargé avec succès depuis MODEL_PATH=%s et SCALER_PATH=%s", MODEL_PATH, SCALER_PATH)
	except FileNotFoundError:
		logger.exception(
			"Impossible de charger le modèle. Vérifiez que MODEL_PATH=%s et SCALER_PATH=%s sont corrects.",
			MODEL_PATH,
			SCALER_PATH,
		)
		model = None
		scaler = None
	except Exception:
		logger.exception(
			"Erreur inattendue lors du chargement du modèle. Vérifiez MODEL_PATH=%s et SCALER_PATH=%s.",
			MODEL_PATH,
			SCALER_PATH,
		)
		model = None
		scaler = None
	yield


app = FastAPI(
	title="Churn Prediction API",
	description=(
		"API FastAPI pour la prédiction et le réentraînement du churn client. "
		"Utilisez /predict pour obtenir une prédiction, /retrain pour réentraîner le modèle, "
		"et /health pour vérifier l'état du service."
	),
	version="1.0.0",
	openapi_tags=[
		{"name": "Health", "description": "Vérification de l'état de l'API et des artefacts."},
		{"name": "Prediction", "description": "Endpoints de prédiction du churn client."},
		{"name": "Retrain", "description": "Endpoint de réentraînement du modèle avec hyperparamètres."},
	],
	lifespan=lifespan,
)


def build_ui_html() -> str:
	"""Construit une interface HTML simple pour tester les prédictions dans le navigateur."""
	return """
<!DOCTYPE html>
<html lang="fr">
<head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<title>Churn Client - Démo API</title>
	<style>
		:root {
			--bg: #081120;
			--panel: rgba(10, 17, 32, 0.86);
			--panel-border: rgba(255, 255, 255, 0.12);
			--text: #e8eef9;
			--muted: #9bb0d1;
			--accent: #4fd1c5;
			--accent-2: #8b5cf6;
			--danger: #ff6b6b;
			--success: #4ade80;
			--shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
		}

		* { box-sizing: border-box; }

		body {
			margin: 0;
			min-height: 100vh;
			font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
			color: var(--text);
			background:
				radial-gradient(circle at top left, rgba(79, 209, 197, 0.2), transparent 32%),
				radial-gradient(circle at top right, rgba(139, 92, 246, 0.18), transparent 28%),
				linear-gradient(180deg, #050816 0%, #081120 48%, #050816 100%);
		}

		.shell {
			max-width: 1180px;
			margin: 0 auto;
			padding: 32px 20px 48px;
		}

		.hero {
			display: grid;
			grid-template-columns: 1.15fr 0.85fr;
			gap: 20px;
			align-items: stretch;
			margin-bottom: 20px;
		}

		.card {
			background: var(--panel);
			border: 1px solid var(--panel-border);
			border-radius: 24px;
			box-shadow: var(--shadow);
			backdrop-filter: blur(18px);
		}

		.hero-copy, .hero-stat, .form-card, .result-card {
			padding: 24px;
		}

		.eyebrow {
			display: inline-flex;
			align-items: center;
			gap: 10px;
			font-size: 12px;
			letter-spacing: 0.12em;
			text-transform: uppercase;
			color: var(--muted);
			margin-bottom: 14px;
		}

		.eyebrow::before {
			content: "";
			width: 10px;
			height: 10px;
			border-radius: 999px;
			background: linear-gradient(135deg, var(--accent), var(--accent-2));
			box-shadow: 0 0 22px rgba(79, 209, 197, 0.8);
		}

		h1 {
			margin: 0 0 12px;
			font-size: clamp(2.1rem, 5vw, 4.2rem);
			line-height: 0.98;
			letter-spacing: -0.05em;
		}

		.subtitle {
			margin: 0;
			color: var(--muted);
			font-size: 1rem;
			line-height: 1.7;
			max-width: 62ch;
		}

		.hero-stat {
			display: grid;
			place-content: center;
			text-align: center;
			background:
				linear-gradient(145deg, rgba(79, 209, 197, 0.16), rgba(139, 92, 246, 0.14)),
				var(--panel);
		}

		.hero-stat strong {
			font-size: 3rem;
			display: block;
			margin-bottom: 6px;
		}

		.hero-stat span {
			color: var(--muted);
		}

		.grid {
			display: grid;
			grid-template-columns: 1.05fr 0.95fr;
			gap: 20px;
			align-items: start;
		}

		.section {
			margin-top: 20px;
		}

		.section-grid {
			display: grid;
			grid-template-columns: 1.15fr 0.85fr;
			gap: 20px;
			align-items: start;
		}

		.form-grid {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 14px;
		}

		.field {
			display: flex;
			flex-direction: column;
			gap: 8px;
		}

		label {
			font-size: 0.85rem;
			color: var(--muted);
		}

		input, select {
			width: 100%;
			padding: 13px 14px;
			border-radius: 14px;
			border: 1px solid rgba(255, 255, 255, 0.12);
			background: rgba(255, 255, 255, 0.04);
			color: var(--text);
			outline: none;
			transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
		}

		input:focus, select:focus {
			border-color: rgba(79, 209, 197, 0.8);
			background: rgba(255, 255, 255, 0.06);
			transform: translateY(-1px);
		}

		.actions {
			display: flex;
			gap: 12px;
			flex-wrap: wrap;
			margin-top: 18px;
		}

		button {
			appearance: none;
			border: none;
			border-radius: 14px;
			padding: 13px 18px;
			font-weight: 700;
			cursor: pointer;
			color: #05111a;
			background: linear-gradient(135deg, var(--accent), #c6ffdf);
			box-shadow: 0 12px 32px rgba(79, 209, 197, 0.28);
		}

		button.secondary {
			color: var(--text);
			background: rgba(255, 255, 255, 0.06);
			box-shadow: none;
			border: 1px solid rgba(255, 255, 255, 0.12);
		}

		.result-card {
			min-height: 100%;
		}

		.retrain-result {
			min-height: 260px;
		}

		.helper {
			font-size: 0.86rem;
			color: var(--muted);
			line-height: 1.6;
			margin: 6px 0 0;
		}

		.small-grid {
			display: grid;
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 14px;
			margin-top: 14px;
		}

		.result-box {
			margin-top: 12px;
			padding: 18px;
			border-radius: 18px;
			background: rgba(255, 255, 255, 0.04);
			border: 1px solid rgba(255, 255, 255, 0.08);
			min-height: 160px;
			white-space: pre-wrap;
			line-height: 1.6;
		}

		.tag {
			display: inline-flex;
			align-items: center;
			gap: 8px;
			padding: 8px 12px;
			border-radius: 999px;
			font-size: 0.85rem;
			margin-bottom: 16px;
			border: 1px solid rgba(255, 255, 255, 0.12);
			background: rgba(255, 255, 255, 0.04);
			color: var(--muted);
		}

		.status-ok { color: var(--success); }
		.status-error { color: var(--danger); }

		@media (max-width: 920px) {
			.hero, .grid, .section-grid { grid-template-columns: 1fr; }
			.form-grid { grid-template-columns: 1fr; }
			.small-grid { grid-template-columns: 1fr; }
		}
	</style>
</head>
<body>
	<main class="shell">
		<section class="hero">
			<div class="card hero-copy">
				<div class="eyebrow">FastAPI · Client HTML</div>
				<h1>Prédire le churn dans le navigateur.</h1>
				<p class="subtitle">
					Renseigne les attributs du client, lance la prédiction, et affiche immédiatement le résultat ainsi que le score du modèle.
				</p>
			</div>
			<div class="card hero-stat">
				<strong>/predict</strong>
				<span>Envoi direct vers l’API FastAPI servie par le même processus</span>
			</div>
		</section>

		<section class="grid">
			<div class="card form-card">
				<div class="tag">Formulaire de prédiction</div>
				<form id="predict-form">
					<div class="form-grid">
						<div class="field"><label for="CreditScore">CreditScore</label><input id="CreditScore" type="number" value="650" min="0" required /></div>
						<div class="field"><label for="Geography">Geography</label><select id="Geography"><option>France</option><option>Germany</option><option>Spain</option></select></div>
						<div class="field"><label for="Gender">Gender</label><select id="Gender"><option>Female</option><option>Male</option></select></div>
						<div class="field"><label for="Age">Age</label><input id="Age" type="number" value="40" min="0" required /></div>
						<div class="field"><label for="Tenure">Tenure</label><input id="Tenure" type="number" value="3" min="0" required /></div>
						<div class="field"><label for="Balance">Balance</label><input id="Balance" type="number" value="0" min="0" step="0.01" required /></div>
						<div class="field"><label for="NumOfProducts">NumOfProducts</label><input id="NumOfProducts" type="number" value="1" min="0" required /></div>
						<div class="field"><label for="HasCrCard">HasCrCard</label><select id="HasCrCard"><option value="1">1</option><option value="0">0</option></select></div>
						<div class="field"><label for="IsActiveMember">IsActiveMember</label><select id="IsActiveMember"><option value="1">1</option><option value="0">0</option></select></div>
						<div class="field"><label for="EstimatedSalary">EstimatedSalary</label><input id="EstimatedSalary" type="number" value="50000" min="0" step="0.01" required /></div>
					</div>
					<div class="actions">
						<button type="submit">Lancer la prédiction</button>
						<button class="secondary" type="button" id="fill-demo">Charger un exemple</button>
					</div>
				</form>
			</div>

			<aside class="card result-card">
				<div class="tag">Résultat</div>
				<div id="result" class="result-box">Aucune prédiction pour l’instant.</div>
			</aside>
		</section>

		<section class="section section-grid">
			<div class="card form-card">
				<div class="tag">Réentraînement du modèle</div>
				<p class="helper">
					Modifie les hyperparamètres ci-dessous, puis envoie la requête POST vers /retrain pour recalculer le modèle et recharger les artefacts en mémoire.
				</p>
				<form id="retrain-form">
					<div class="form-grid">
						<div class="field"><label for="data_path">data_path</label><input id="data_path" type="text" value="Churn_Modelling.csv" required /></div>
						<div class="field"><label for="test_size">test_size</label><input id="test_size" type="number" value="0.2" min="0.01" max="0.99" step="0.01" required /></div>
						<div class="field"><label for="random_state">random_state</label><input id="random_state" type="number" value="42" step="1" required /></div>
						<div class="field"><label for="C">C</label><input id="C" type="number" value="1" min="0.0001" step="0.1" required /></div>
						<div class="field"><label for="max_iter">max_iter</label><input id="max_iter" type="number" value="10000" min="1" step="1" required /></div>
						<div class="field"><label for="loss">loss</label><select id="loss"><option value="squared_hinge">squared_hinge</option><option value="hinge">hinge</option></select></div>
					</div>
					<div class="small-grid">
						<div class="field"><label for="tol">tol (optionnel)</label><input id="tol" type="number" placeholder="Laisser vide pour None" min="0" step="0.0001" /></div>
						<div class="field"><label for="dual">dual</label><select id="dual"><option value="">auto</option><option value="true">true</option><option value="false">false</option></select></div>
						<div class="field"><label for="retrain_model_path">model_path</label><input id="retrain_model_path" type="text" value="svc_model.pkl" required /></div>
						<div class="field"><label for="scaler_path">scaler_path</label><input id="scaler_path" type="text" value="scaler.pkl" required /></div>
					</div>
					<div class="actions">
						<button type="submit">Lancer le réentraînement</button>
						<button class="secondary" type="button" id="fill-retrain-demo">Valeurs par défaut</button>
					</div>
				</form>
			</div>

			<aside class="card result-card retrain-result">
				<div class="tag">Résultat du réentraînement</div>
				<div id="retrain-result" class="result-box">Aucun réentraînement pour l’instant.</div>
			</aside>
		</section>
	</main>

	<script>
		const resultBox = document.getElementById('result');
		const form = document.getElementById('predict-form');
		const retrainForm = document.getElementById('retrain-form');
		const retrainResultBox = document.getElementById('retrain-result');

		const demo = {
			CreditScore: 650,
			Geography: 'France',
			Gender: 'Female',
			Age: 40,
			Tenure: 3,
			Balance: 0,
			NumOfProducts: 1,
			HasCrCard: '1',
			IsActiveMember: '1',
			EstimatedSalary: 50000,
		};

		function readValue(id) {
			return document.getElementById(id).value;
		}

		function setResult(message, isError = false) {
			resultBox.innerHTML = `<span class="${isError ? 'status-error' : 'status-ok'}">${isError ? 'Erreur' : 'Succès'}</span>\n\n${message}`;
		}

		function setRetrainResult(message, isError = false) {
			retrainResultBox.innerHTML = `<span class="${isError ? 'status-error' : 'status-ok'}">${isError ? 'Erreur' : 'Succès'}</span>\n\n${message}`;
		}

		document.getElementById('fill-demo').addEventListener('click', () => {
			Object.entries(demo).forEach(([key, value]) => {
				document.getElementById(key).value = value;
			});
			setResult('Exemple chargé. Tu peux lancer la prédiction.', false);
		});

		document.getElementById('fill-retrain-demo').addEventListener('click', () => {
			document.getElementById('data_path').value = 'Churn_Modelling.csv';
			document.getElementById('test_size').value = '0.2';
			document.getElementById('random_state').value = '42';
			document.getElementById('C').value = '1';
			document.getElementById('max_iter').value = '10000';
			document.getElementById('tol').value = '';
			document.getElementById('loss').value = 'squared_hinge';
			document.getElementById('dual').value = '';
			document.getElementById('retrain_model_path').value = 'svc_model.pkl';
			document.getElementById('scaler_path').value = 'scaler.pkl';
			setRetrainResult('Paramètres par défaut chargés. Tu peux lancer le réentraînement.', false);
		});

		form.addEventListener('submit', async (event) => {
			event.preventDefault();

			const payload = {
				CreditScore: Number(readValue('CreditScore')),
				Geography: readValue('Geography'),
				Gender: readValue('Gender'),
				Age: Number(readValue('Age')),
				Tenure: Number(readValue('Tenure')),
				Balance: Number(readValue('Balance')),
				NumOfProducts: Number(readValue('NumOfProducts')),
				HasCrCard: Number(readValue('HasCrCard')),
				IsActiveMember: Number(readValue('IsActiveMember')),
				EstimatedSalary: Number(readValue('EstimatedSalary')),
			};

			setResult('Prédiction en cours...');

			try {
				const response = await fetch('/predict', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(payload),
				});

				const data = await response.json();

				if (!response.ok) {
					throw new Error(data.detail || 'Erreur inconnue');
				}

				setResult(
					`Prédiction brute: ${data.prediction}\n` +
					`Churn: ${data.churn ? 'oui' : 'non'}\n` +
					`Score: ${data.score === null ? 'N/A' : data.score}`
				);
			} catch (error) {
				setResult(error.message, true);
			}
		});

		retrainForm.addEventListener('submit', async (event) => {
			event.preventDefault();

			const rawTol = document.getElementById('tol').value;
			const rawDual = document.getElementById('dual').value;

			const payload = {
				data_path: document.getElementById('data_path').value,
				test_size: Number(document.getElementById('test_size').value),
				random_state: Number(document.getElementById('random_state').value),
				C: Number(document.getElementById('C').value),
				max_iter: Number(document.getElementById('max_iter').value),
				tol: rawTol === '' ? null : Number(rawTol),
				loss: document.getElementById('loss').value,
				dual: rawDual === '' ? null : rawDual === 'true',
				model_path: document.getElementById('retrain_model_path').value,
				scaler_path: document.getElementById('scaler_path').value,
			};

			setRetrainResult('Réentraînement en cours...');

			try {
				const response = await fetch('/retrain', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(payload),
				});

				const data = await response.json();

				if (!response.ok) {
					throw new Error(data.detail || 'Erreur inconnue');
				}

				setRetrainResult(
					`Statut: ${data.status}\n` +
					`Accuracy: ${data.accuracy}\n` +
					`Modèle: ${data.model_path}\n` +
					`Scaler: ${data.scaler_path}\n\n` +
					`Matrice de confusion:\n${JSON.stringify(data.matrix, null, 2)}\n\n` +
					`Rapport:\n${data.report}`
				);
			} catch (error) {
				setRetrainResult(error.message, true);
			}
		});
	</script>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
	return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui() -> HTMLResponse:
	return HTMLResponse(build_ui_html())


@app.get(
	"/health",
	tags=["Health"],
	summary="Vérifier l'état du service",
	description="Retourne l'état du service ainsi que les chemins des artefacts chargés.",
)
def health_check() -> dict:
	return {
		"status": "ok",
		"model_loaded": model is not None and scaler is not None,
		"model_path": MODEL_PATH,
		"scaler_path": SCALER_PATH,
	}


@app.post(
	"/predict",
	response_model=ChurnResponse,
	tags=["Prediction"],
	summary="Prédire le churn d'un client",
	description="Reçoit les caractéristiques d'un client et retourne la classe prédite ainsi qu'un score si disponible.",
	responses={
		200: {"description": "Prédiction calculée avec succès."},
		400: {"description": "Données invalides fournies dans la requête."},
		503: {"description": "Le modèle ou le scaler ne sont pas chargés."},
	},
)
def predict_churn(payload: ChurnRequest) -> ChurnResponse:
	return predict_from_payload(payload)


@app.post(
	"/retrain",
	response_model=RetrainResponse,
	tags=["Retrain"],
	summary="Réentraîner le modèle",
	description="Réentraîne le modèle avec les hyperparamètres fournis, sauvegarde les artefacts, puis recharge le modèle en mémoire.",
	responses={
		200: {"description": "Modèle réentraîné avec succès."},
		400: {"description": "Hyperparamètres invalides ou incohérents."},
		404: {"description": "Dataset ou artefact introuvable."},
		500: {"description": "Erreur interne durant le réentraînement."},
	},
)
def retrain_model(payload: RetrainRequest) -> RetrainResponse:
	return retrain_from_payload(payload)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
