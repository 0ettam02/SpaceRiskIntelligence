from typing import Optional

from fastapi import APIRouter, Query

from app.data_store import store

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/time-series")
def get_time_series(segmentId: Optional[str] = Query(default="all"), windowDays: Optional[int] = Query(default=None)):
    if segmentId and segmentId != "all":
        series = [row for row in store.calendar_series if row["segmentId"] == int(segmentId)]
    else:
        series = store.calendar_series

    if windowDays:
        series = series[-windowDays:]

    return {
        "series": series,
        "segments": _segment_summaries(),
        "embargoWindows": _embargo_windows(),
    }


def _segment_summaries():
    panel = store.italy_daily_panel_df
    ml = store.italy_ml_df
    summaries = []
    for _, segment in store.segments_df.iterrows():
        segment_id = int(segment.segment_id)
        rows = panel[panel.segment_id == segment_id]
        summaries.append(
            {
                "id": segment_id,
                "label": f"Segmento {segment_id}",
                "start": rows.date.min().strftime("%Y-%m-%d") if len(rows) else segment.start,
                "end": rows.date.max().strftime("%Y-%m-%d") if len(rows) else segment.end,
                "days": int(rows.date.nunique()),
                "usableForModel": bool(segment.usable_for_model),
                "reason": segment.reason.replace("_", " "),
                "rowsInSample": int(len(rows)),
                "totalDetections": int(rows["detection_count"].sum()),
                "totalFrp": round(float(rows["daily_frp_sum"].sum()), 1),
                "mlRows": int((ml.segment_id == segment_id).sum()) if ml is not None else 0,
            }
        )
    return summaries


def _embargo_windows():
    model = (store.model_results or {}).get("random_forest_italia_era5")
    if model is None:
        return []
    split = model["italyReport"]["temporal_split"]
    return [
        {
            "segmentId": "italy-temporal-split",
            "description": (
                f"Split Italia: train {split['train'][0]} – {split['train'][1]}, "
                f"validation {split['validation'][0]} – {split['validation'][1]}, "
                f"test {split['test'][0]} – {split['test'][1]}; embargo di {split['embargo_days']} giorni."
            ),
        }
    ]
