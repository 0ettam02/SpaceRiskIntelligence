"""
areeCalde_temporale_migliorato.py
---------------------------------
Versione migliorata: evidenzia la degenerazione degli incendi nel tempo.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
CSV_PATH = "dataset_incendi_FIRMS.csv"
CAMPIONE = 500_000

COL_LAT  = "latitude"
COL_LON  = "longitude"
COL_DATA = "acq_date"

# 🎨 NUOVA PALETTE (progressiva = storytelling)
FASCE = [
    ("2011–2015", 2011, 2015, "#4cc9f0", 0.05),  # blu (passato)
    ("2016–2020", 2016, 2020, "#f9c74f", 0.10),  # giallo
    ("2021–2025", 2021, 2025, "#f94144", 0.18),  # rosso
    ("2026",      2026, 2026, "#9d0208", 0.35),  # rosso scuro (critico)
]

MARKER_SIZE = 0.3

# Tema
BG_FIG  = "#0d1117"
BG_AX   = "#0d1b2a"
COL_PAE = "#1a2a3a"
COL_BOR = "#3a5a7a"
COL_GRI = "#2a3a4a"


# ════════════════════════════════════════════════════════════════════════
# 1. Caricamento
# ════════════════════════════════════════════════════════════════════════
def carica_dati(path, campione):
    p = Path(path)
    if not p.exists():
        sys.exit("File non trovato")

    df = pd.read_csv(p, low_memory=False)

    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
    df["_anno"] = pd.to_datetime(df[COL_DATA], errors="coerce").dt.year

    df = df.dropna(subset=[COL_LAT, COL_LON, "_anno"])
    df["_anno"] = df["_anno"].astype(int)

    if campione and len(df) > campione:
        df = df.sample(campione, random_state=42)

    return df


# ════════════════════════════════════════════════════════════════════════
# 2. Fasce
# ════════════════════════════════════════════════════════════════════════
def dividi_fasce(df):
    risultati = []
    for (label, y0, y1, colore, alpha) in FASCE:
        sub = df[df["_anno"].between(y0, y1)]
        if not sub.empty:
            risultati.append((label, colore, alpha, sub))
    return risultati


# ════════════════════════════════════════════════════════════════════════
# 3. Confini
# ════════════════════════════════════════════════════════════════════════
def carica_confini():
    try:
        import geodatasets, geopandas as gpd
        return gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except:
        return None


# ════════════════════════════════════════════════════════════════════════
# 4. Disegno
# ════════════════════════════════════════════════════════════════════════
def disegna(fasce_dati, world):
    fig, ax = plt.subplots(figsize=(22, 11))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # sfondo
    if world is not None:
        world.plot(ax=ax, color=COL_PAE, edgecolor=COL_BOR, linewidth=0.5)

    # 🔥 punti (vecchi sotto, nuovi sopra + più grandi)
    for label, colore, alpha, sub in fasce_dati:
        size = MARKER_SIZE

        if "2026" in label:
            size *= 3
        elif "2021" in label:
            size *= 2

        ax.scatter(
            sub[COL_LON],
            sub[COL_LAT],
            s=size,
            c=colore,
            alpha=alpha,
            linewidths=0,
            zorder=2,
            rasterized=True
        )

    # bordi sopra
    if world is not None:
        world.plot(ax=ax, color="none", edgecolor=COL_BOR, linewidth=0.4)

    # titolo
    ax.set_title(
        "🔥 Evoluzione degli incendi nel tempo (NASA FIRMS)\n"
        "Dal blu (passato) al rosso scuro (critico)",
        fontsize=15,
        color="white",
        pad=20
    )

    # assi
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(color=COL_GRI, linestyle="--", linewidth=0.3)

    # 🧠 legenda migliorata
    handles = []
    for label, colore, alpha, sub in fasce_dati:
        handles.append(
            mpatches.Patch(
                color=colore,
                label=f"{label}\n{len(sub):,} hotspot"
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(-0.18, 0.5),
        frameon=True,
        facecolor=BG_AX,
        edgecolor=COL_BOR,
        labelcolor="white",
        title="Evoluzione incendi",
        title_fontsize=12
    )

    # 🔥 messaggio visivo
    fig.text(
        0.02, 0.85,
        "⬆ Aumento progressivo\nincendi globali",
        color="#f94144",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    df = carica_dati(CSV_PATH, CAMPIONE)
    fasce = dividi_fasce(df)
    world = carica_confini()

    fig = disegna(fasce, world)
    return fig


if __name__ == "__main__":
    main()
