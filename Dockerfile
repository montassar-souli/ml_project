FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

ENV API_BASE_URL=http://localhost:5000 \
	FRONTEND_ORIGIN=http://localhost:8000 \
	FRONTEND_PORT=8000 \
	API_PORT=5000

RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir \
		fastapi \
		uvicorn \
		pandas \
		numpy \
		scikit-learn \
		joblib \
		mlflow

COPY app.py main.py model_pipeline.py front_server.py ./
COPY svc_model.pkl ./svc_model.pkl
COPY scaler.pkl ./scaler.pkl

EXPOSE 5000 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${API_PORT} & exec python3 front_server.py"]
