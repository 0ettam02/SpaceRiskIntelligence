import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.exceptions import ConvergenceWarning

from app.config import CORS_ORIGINS
from app.data_store import store
from app.routers import analysis, data_quality, map as map_router, models, overview, pipeline_status

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# La regressione polinomiale non converge sempre entro max_iter su questi dati:
# è un limite noto e già presente nel notebook originale, non un errore del backend.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Unknown solver options")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(
    title="SpaceRiskIntelligence API",
    description=(
        "API sperimentale che espone gli artefatti reali della pipeline di ricerca "
        "(cambiamenti_climatici/claudia/output_definitivo) e i 5 classificatori addestrati "
        "in memoria all'avvio. Non è un sistema operativo di allerta incendi."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(overview.router)
app.include_router(map_router.router)
app.include_router(analysis.router)
app.include_router(models.router)
app.include_router(data_quality.router)
app.include_router(pipeline_status.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "cellsLoaded": len(store.cells), "modelsTrained": list(store.model_results.keys())}
