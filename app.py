"""
app.py — API REST para el modelo de predicción de la variación mensual
del IPC en Colombia (por ciudad y grupo de gasto).

Endpoints:
    GET  /health   -> estado del servicio y del modelo cargado
    POST /predict  -> predicción puntual a partir de features ya calculadas
    POST /train     -> reentrena el modelo y lo versiona

Ejecutar localmente:
    uvicorn app:app --reload --port 8000

Documentación interactiva autogenerada por FastAPI:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import CIUDADES_VALIDAS, GRUPOS_GASTO_VALIDOS
from predict import load_model, predict_from_payload
from train import run_training

MODEL_DIR = "model"
DEFAULT_DATA_PATH = "data/ipc_panel_sintetico.csv"

# El modelo se carga una vez al iniciar el servicio y se cachea en memoria.
# /train reemplaza esta referencia por el modelo recién entrenado.
_model_cache = {"pipe": None, "version": None}


def _try_load_latest():
    try:
        pipe = load_model(model_dir=MODEL_DIR)
        latest_path = Path(MODEL_DIR) / "latest.json"
        import json
        with open(latest_path, encoding="utf-8") as f:
            version = json.load(f)["version"]
        _model_cache["pipe"] = pipe
        _model_cache["version"] = version
    except FileNotFoundError:
        _model_cache["pipe"] = None
        _model_cache["version"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_latest()
    yield


app = FastAPI(
    title="API de predicción del IPC — Colombia",
    description=(
        "Predice la variación mensual del IPC por ciudad y grupo de gasto "
        "usando un modelo de aprendizaje supervisado entrenado sobre un "
        "panel de datos macroeconómicos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Esquemas de entrada / salida
# ---------------------------------------------------------------------

class PredictRequest(BaseModel):
    ciudad: str = Field(..., examples=["Medellín"], description=f"Una de: {CIUDADES_VALIDAS}")
    grupo_gasto: str = Field(..., examples=["Transporte"], description=f"Uno de: {GRUPOS_GASTO_VALIDOS}")
    ipc_lag1: float = Field(..., description="Variación del IPC del mes t-1 para esa serie")
    ipc_lag3: float = Field(..., description="Variación del IPC del mes t-3 para esa serie")
    ipc_lag6: float = Field(..., description="Variación del IPC del mes t-6 para esa serie")
    mes_sin: float = Field(..., description="sin(2*pi*mes/12), codificación cíclica del mes")
    mes_cos: float = Field(..., description="cos(2*pi*mes/12), codificación cíclica del mes")
    tasa_banrep: float = Field(..., description="Tasa de intervención del Banco de la República (%)")
    var_trm: float = Field(..., description="Variación porcentual mensual de la TRM")
    var_petroleo: float = Field(..., description="Variación porcentual mensual del petróleo Brent")


class PredictResponse(BaseModel):
    ciudad: str
    grupo_gasto: str
    ipc_variacion_mensual_predicha: float
    model_version: Optional[int]


class TrainRequest(BaseModel):
    data_path: str = Field(default=DEFAULT_DATA_PATH, description="Ruta al CSV de entrenamiento")
    model_type: str = Field(default="random_forest", pattern="^(ridge|random_forest)$")
    test_start: str = Field(default="2023-01-01", description="Fecha de corte train/test (YYYY-MM-DD)")


class TrainResponse(BaseModel):
    version: int
    model_type: str
    n_observations: int
    train_period: list[str]
    test_period: list[str]
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[int]


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model_cache["pipe"] is not None,
        model_version=_model_cache["version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model_cache["pipe"] is None:
        raise HTTPException(
            status_code=503,
            detail="No hay un modelo entrenado disponible. Llama primero a POST /train.",
        )

    payload = request.model_dump()
    try:
        prediccion = predict_from_payload(_model_cache["pipe"], payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        ciudad=request.ciudad,
        grupo_gasto=request.grupo_gasto,
        ipc_variacion_mensual_predicha=round(prediccion, 4),
        model_version=_model_cache["version"],
    )


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest = TrainRequest()):
    try:
        metadata = run_training(
            data_path=request.data_path,
            model_dir=MODEL_DIR,
            model_type=request.model_type,
            test_start=request.test_start,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _try_load_latest()

    return TrainResponse(
        version=metadata["version"],
        model_type=metadata["model_type"],
        n_observations=metadata["n_observations"],
        train_period=metadata["train_period"],
        test_period=metadata["test_period"],
        metrics=metadata["metrics"],
    )
