# Fase 3 — API REST para el modelo de predicción del IPC

FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero para aprovechar el cache de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código, los datos y el modelo ya entrenado
COPY features.py train.py predict.py app.py cliente_prueba.py ./
COPY data/ ./data/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
