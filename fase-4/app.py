"""
app.py — API REST para el modelo de predicción de la variación mensual
del IPC en Colombia (por ciudad y grupo de gasto), con monitoreo básico.

Endpoints:
    GET  /health    -> estado del servicio y del modelo cargado
    POST /predict   -> predicción puntual a partir de features ya calculadas
    POST /train     -> reentrena el modelo y lo versiona
    POST /feedback  -> registra el valor real de IPC para una predicción previa
    GET  /metrics   -> métricas de operación y de monitoreo (Fase 4)

Toda llamada a /predict deja una sola fila en el registro de monitoreo,
incluidas las rechazadas por validación (422), por ciudad desconocida (400)
o por ausencia de modelo (503).

Ejecutar localmente:
    uvicorn app:app --reload --port 8000

Documentación interactiva autogenerada por FastAPI:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import monitoring_utils as mon
from features import CIUDADES_VALIDAS, GRUPOS_GASTO_VALIDOS
from predict import load_model, predict_from_payload
from retraining_policy import evaluate_retraining_policy
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
        "panel de datos macroeconómicos, con registro de solicitudes."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Registra los rechazos 422 de /predict antes de devolver el error.

    FastAPI valida el cuerpo antes de llamar a la función de la ruta, así que
    este manejador es el único punto donde esas solicitudes pueden quedar en
    el log. Solo se registran las dirigidas a /predict.
    """
    if request.url.path == "/predict":
        try:
            cuerpo = await request.json()
        except Exception:
            cuerpo = None
        mon.log_rejected(status_code=422, payload=cuerpo)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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
    prediction_id: str
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


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., description="Id devuelto por /predict al momento de predecir")
    actual_value: float = Field(..., description="Variación mensual del IPC realmente observada")


class FeedbackResponse(BaseModel):
    prediction_id: str
    predicted_value: float
    actual_value: float
    error_abs: float


class MetricsResponse(BaseModel):
    n_predicciones_totales: int
    n_predicciones_con_feedback: int
    predicciones_por_ciudad: dict
    predicciones_por_grupo_gasto: dict
    rolling_window: int
    mae_rodante: Optional[float]
    rmse_rodante: Optional[float]
    total_solicitudes: int
    solicitudes_fallidas: int
    porcentaje_fallos: Optional[float]
    mensaje_porcentaje: str
    model_version_actual: Optional[int]
    versiones_disponibles: list
    politica_reentrenamiento: dict


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
    payload = request.model_dump()

    if _model_cache["pipe"] is None:
        mon.log_rejected(status_code=503, payload=payload)
        raise HTTPException(
            status_code=503,
            detail="No hay un modelo entrenado disponible. Llama primero a POST /train.",
        )

    try:
        prediccion = predict_from_payload(_model_cache["pipe"], payload)
    except ValueError as e:
        mon.log_rejected(status_code=400, payload=payload)
        raise HTTPException(status_code=400, detail=str(e))

    prediccion = round(prediccion, 4)
    prediction_id = mon.log_prediction(payload, prediccion, _model_cache["version"])

    return PredictResponse(
        prediction_id=prediction_id,
        ciudad=request.ciudad,
        grupo_gasto=request.grupo_gasto,
        ipc_variacion_mensual_predicha=prediccion,
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


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    try:
        record = mon.log_feedback(request.prediction_id, request.actual_value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FeedbackResponse(
        prediction_id=record["prediction_id"],
        predicted_value=record["predicted_value"],
        actual_value=record["actual_value"],
        error_abs=record["error_abs"],
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    base = mon.compute_metrics()
    versiones = mon.list_model_versions(MODEL_DIR)
    politica = evaluate_retraining_policy(base["rmse_rodante"], base["n_predicciones_con_feedback"])

    return MetricsResponse(
        **base,
        model_version_actual=_model_cache["version"],
        versiones_disponibles=versiones,
        politica_reentrenamiento=politica,
    )
