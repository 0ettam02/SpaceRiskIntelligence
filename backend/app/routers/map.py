from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_store import store

router = APIRouter(prefix="/map", tags=["map"])


@router.get("/cells")
def get_map_cells(
    riskLevel: Optional[str] = Query(default=None),
    minLastDetectionDate: Optional[str] = Query(default=None),
    metric: Optional[str] = Query(default=None),
):
    cells = store.cells

    if riskLevel and riskLevel != "all":
        cells = [c for c in cells if c["riskLevel"] == riskLevel]

    if minLastDetectionDate:
        cells = [c for c in cells if c["lastDetectionDate"] and c["lastDetectionDate"] >= minLastDetectionDate]

    if metric == "probability":
        cells = [c for c in cells if c["hasPrediction"]]

    return {"cells": cells, "total": len(cells)}


@router.get("/cells/{cell_id}")
def get_cell_details(cell_id: str):
    cell = store.cells_by_id.get(cell_id)
    if not cell:
        raise HTTPException(status_code=404, detail="Cella non trovata")

    historical_series = store.cell_historical_series(cell["lat"], cell["lon"])
    return {**cell, "historicalSeries": historical_series}
