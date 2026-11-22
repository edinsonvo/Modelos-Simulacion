# Fase 4 — API REST con monitoreo básico

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py train.py predict.py app.py cliente_prueba.py \
     monitoring_utils.py retraining_policy.py monitor.py ./
COPY data/ ./data/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
