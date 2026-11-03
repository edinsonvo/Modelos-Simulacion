# Fase 2 — Contenedor reproducible para entrenar y predecir el modelo de IPC

FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero para aprovechar el cache de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y los datos necesarios
COPY features.py train.py predict.py ./
COPY data/ ./data/

# La carpeta model/ se monta como volumen para persistir el modelo
# entrenado fuera del contenedor (ver README.md para el comando docker run).
RUN mkdir -p model

ENTRYPOINT ["python"]
CMD ["predict.py", "--help"]
