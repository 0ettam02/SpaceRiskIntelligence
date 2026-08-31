from typing import Optional

from fastapi import APIRouter, Query

from app.config import HORIZON
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
    panel = store.daily_panel_df
    summaries = []
    for _, segment in store.segments_df.iterrows():
        segment_id = int(segment.segment_id)
        rows = panel[panel.segment_id == segment_id]
        summaries.append(
            {
                "id": segment_id,
                "label": f"Segmento {segment_id}",
                "start": segment.start,
                "end": segment.end,
                "days": int(segment.days),
                "usableForModel": bool(segment.usable_for_model),
                "reason": segment.reason.replace("_", " "),
                "rowsInSample": int(len(rows)),
                "totalDetections": int(rows["detection_count"].sum()),
                "totalFrp": round(float(rows["daily_frp_sum"].sum()), 1),
                "mlRows": int(store.ml_rows_per_segment.get(segment_id, 0)),
            }
        )
    return summaries


def _embargo_windows():
    return [
        {
            "segmentId": int(segment.segment_id),
            "description": (
                f"{HORIZON} giorni di embargo applicati ai confini di training/validation/test del segmento {int(segment.segment_id)}"
            ),
        }
        for _, segment in store.segments_df.iterrows()
        if segment.usable_for_model
    ]
