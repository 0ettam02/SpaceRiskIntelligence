"""Meteo giornaliero NASA POWER (MERRA-2, 0,5° x 0,625°) per le celle campionate.

Scarica una serie per ogni nodo della griglia POWER che contiene almeno una cella
del campione, salva un JSON per nodo (download riprendibile) e poi consolida un
pannello cella/giorno in ``claudia/dati/power/meteo_celle.csv.gz`` (ignorato da git). Dati di pubblico dominio,
nessuna chiave necessaria.

Esecuzione dalla root del repository:
    backend/.venv/bin/python cambiamenti_climatici/claudia/modelli_ml/power_weather.py
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CELLS = ROOT / "cambiamenti_climatici/claudia/output_definitivo/campione_celle_v3.csv"
OUTPUT = ROOT / "cambiamenti_climatici/claudia/dati/power"
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR",
              "WS2M", "WS2M_MAX", "WS10M_MAX", "GWETTOP", "GWETROOT"]
START, END = "20250327", "20260717"   # 30 giorni prima dell'inizio del pannello FIRMS
LAT_STEP, LON_STEP = 0.5, 0.625


def power_nodes(cells):
    nodes = cells.assign(node_lat=(np.round(cells.lat_cell / LAT_STEP) * LAT_STEP).round(3),
                         node_lon=(np.round(cells.lon_cell / LON_STEP) * LON_STEP).round(4))
    nodes["node_lon"] = nodes.node_lon.where(nodes.node_lon < 180, nodes.node_lon - 360)
    return nodes


def fetch(lat, lon, path, retries=6):
    url = (f"{API}?parameters={','.join(PARAMETERS)}&community=AG&latitude={lat}&longitude={lon}"
           f"&start={START}&end={END}&format=JSON")
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as response:
                payload = json.load(response)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.rename(path)
            return path
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"download fallito per {lat},{lon}")


def download(nodes, workers):
    raw = OUTPUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    todo = [(la, lo, raw / f"{la}_{lo}.json") for la, lo in nodes[["node_lat", "node_lon"]].drop_duplicates().itertuples(index=False)]
    todo = [t for t in todo if not t[2].exists()]
    print(f"nodi da scaricare: {len(todo)}", flush=True)
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, *t) for t in todo]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except RuntimeError as exc:
                failed += 1
                print(exc, flush=True)
            if i % 200 == 0:
                print(f"{i}/{len(todo)} scaricati", flush=True)
    return failed


def consolidate(nodes):
    frames = []
    for la, lo in nodes[["node_lat", "node_lon"]].drop_duplicates().itertuples(index=False):
        payload = json.loads((OUTPUT / "raw" / f"{la}_{lo}.json").read_text())
        params = payload["properties"]["parameter"]
        frame = pd.DataFrame(params)
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
        frame = frame.replace(-999.0, np.nan).rename_axis("date").reset_index()
        frame["node_lat"], frame["node_lon"] = la, lo
        frame["elevation_m"] = payload["geometry"]["coordinates"][2]
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    panel = nodes[["lat_cell", "lon_cell", "node_lat", "node_lon"]].merge(weather, on=["node_lat", "node_lon"])
    panel = panel.drop(columns=["node_lat", "node_lon"]).sort_values(["lat_cell", "lon_cell", "date"])
    panel.to_csv(OUTPUT / "meteo_celle.csv.gz", index=False, float_format="%.3f")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    nodes = power_nodes(pd.read_csv(CELLS)[["lat_cell", "lon_cell"]].drop_duplicates())
    failed = download(nodes, args.workers)
    if failed:
        raise SystemExit(f"{failed} nodi non scaricati: rilanciare lo script per riprendere")
    panel = consolidate(nodes)
    print(f"pannello meteo: {len(panel)} righe, {panel[['lat_cell', 'lon_cell']].drop_duplicates().shape[0]} celle, "
          f"mancanti per variabile: {panel[PARAMETERS].isna().mean().round(4).to_dict()}")


if __name__ == "__main__":
    main()
