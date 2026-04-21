"""
areeCalde.py
------------
Heatmap mondiale degli incendi con istogramma 2D + smoothing gaussiano.
Molto più veloce del KDE puro, stesso effetto visivo sfumato.

Uso nel notebook:
  %matplotlib inline
  %run areeCalde.py

Dipendenze:
  pip install pandas geopandas matplotlib numpy scipy geodatasets
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mp
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ── Configurazione ────────────────────────────────────────────────────────────

CSV_PATH    = "dataset_incendi_FIRMS.csv"
RISOLUZIONE = 360      # celle per asse longitudine (metà per latitudine)
SMOOTH      = 3        # sigma smoothing gaussiano (↑ = più sfumato)
CAMPIONE    = None     # int per campionare N righe casuali, None = tutto il dataset
TITOLO_DATI = "NASA FIRMS"  # sorgente dati mostrata nel titolo

# Colonne attese nel CSV
COL_LAT = "latitude"
COL_LON = "longitude"
COL_DATA = "acq_date"  # opzionale — usata per il range temporale nel titolo

# ── Colormap ──────────────────────────────────────────────────────────────────
# Definita in formato uniforme (valore, hex) per chiarezza
_COLORI_CMAP = [
    (0.00, "#00000000"),   # trasparente (zero incendi)
    (0.15, "#fee08b"),     # giallo tenue
    (0.45, "#fc8d59"),     # arancione
    (0.75, "#d73027"),     # rosso
    (1.00, "#7f0000"),     # bordeaux intenso
]

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "incendi",
    [(v, mcolors.to_rgba(c)) for v, c in _COLORI_CMAP]
)

# ── Tema grafico ──────────────────────────────────────────────────────────────
BG_FIG  = "#1a1a2e"
BG_AX   = "#16213e"
COL_PAE = "#0f3460"   # fill paesi
COL_BOR = "#7a9cc4"   # bordi paesi / assi
COL_GRI = "#4a6a8a"   # griglia


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caricamento e validazione dati
# ══════════════════════════════════════════════════════════════════════════════

def carica_dati(path: str, campione: int | None = None) -> pd.DataFrame:
    """Carica il CSV FIRMS con validazione robusta delle colonne."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERRORE] File non trovato: {p.resolve()}")

    print(f"[1/4] Caricamento dati da {p.name} …", end=" ", flush=True)
    df = pd.read_csv(p, low_memory=False)

    # Verifica colonne obbligatorie
    for col in (COL_LAT, COL_LON):
        if col not in df.columns:
            sys.exit(f"\n[ERRORE] Colonna mancante nel CSV: '{col}'\n"
                     f"        Colonne disponibili: {list(df.columns)}")

    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
    df = df.dropna(subset=[COL_LAT, COL_LON])

    # Filtra coordinate fuori range
    mask_valide = (
        df[COL_LAT].between(-90, 90) &
        df[COL_LON].between(-180, 180)
    )
    n_scartati = (~mask_valide).sum()
    if n_scartati:
        print(f"\n   ⚠ {n_scartati} righe con coordinate fuori range scartate.")
    df = df[mask_valide]

    if df.empty:
        sys.exit("[ERRORE] Nessun dato valido dopo la pulizia.")

    # Campionamento opzionale (utile per dataset > 10M righe)
    if campione and len(df) > campione:
        df = df.sample(n=campione, random_state=42)
        print(f"\n   ℹ Campionamento: {campione:,} / {len(df):,} righe.")

    print(f"   {len(df):,} hotspot caricati ✓")
    return df


def range_temporale(df: pd.DataFrame) -> str:
    """Restituisce il range di date come stringa, se disponibile."""
    if COL_DATA not in df.columns:
        return ""
    try:
        date = pd.to_datetime(df[COL_DATA], errors="coerce").dropna()
        if date.empty:
            return ""
        return f" · {date.min().date()} → {date.max().date()}"
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. Istogramma 2D + smoothing gaussiano
# ══════════════════════════════════════════════════════════════════════════════

def calcola_heatmap(df: pd.DataFrame, risoluzione: int, smooth: float) -> np.ndarray:
    """
    Costruisce la griglia di densità tramite istogramma 2D (O(N), molto veloce)
    e applica uno smoothing gaussiano per sfumare i bordi.

    La risoluzione in latitudine è metà di quella in longitudine per
    mantenere celle approssimativamente quadrate in gradi.
    """
    print("[2/4] Calcolo istogramma 2D …", end=" ", flush=True)

    # n_lon : n_lat = 2 : 1 → celle quasi-quadrate (1° lon × 1° lat)
    n_lon = risoluzione
    n_lat = risoluzione // 2

    Z, _, _ = np.histogram2d(
        df[COL_LON], df[COL_LAT],
        bins=[n_lon, n_lat],
        range=[[-180, 180], [-90, 90]]
    )
    Z = Z.T  # trasposta: asse Y = latitudine

    Z = gaussian_filter(Z, sigma=smooth)
    Z = Z / Z.max()  # normalizza in [0, 1]

    print(f"   griglia {n_lon}×{n_lat} ✓")
    return Z


# ══════════════════════════════════════════════════════════════════════════════
# 3. Caricamento confini del mondo
# ══════════════════════════════════════════════════════════════════════════════

def carica_confini():
    """
    Tenta di caricare i confini dei paesi con le seguenti priorità:
      1. geodatasets (metodo moderno, rimpiazza gpd.datasets deprecato)
      2. geopandas naturalearth_lowres (legacy, deprecato ma funziona ancora)
      3. URL CDN esterno di fallback
      4. None (nessun confine — rettangolo di sfondo)
    """
    print("[3/4] Caricamento confini …", end=" ", flush=True)

    # ① geodatasets — preferito da GeoPandas ≥ 0.12
    try:
        import geodatasets
        import geopandas as gpd
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        print("   geodatasets ✓")
        return world
    except Exception:
        pass

    # ② geopandas legacy
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        print("   geopandas legacy ✓")
        return world
    except Exception:
        pass

    # ③ CDN esterno
    try:
        import geopandas as gpd
        url = ("https://naciscdn.org/naturalearth/110m/cultural/"
               "ne_110m_admin_0_countries.zip")
        world = gpd.read_file(url)
        print("   CDN esterno ✓")
        return world
    except Exception:
        pass

    print("   non disponibili (solo rettangolo di sfondo)")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Visualizzazione
# ══════════════════════════════════════════════════════════════════════════════

def disegna(Z: np.ndarray, world, titolo_sub: str) -> plt.Figure:
    """Assembla la figura finale con heatmap, confini e colorbar."""
    print("[4/4] Rendering …", end=" ", flush=True)

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # ── Sfondo paesi (sotto la heatmap) ──────────────────────────────────────
    if world is not None:
        world.plot(ax=ax, color=COL_PAE, edgecolor=COL_BOR,
                   linewidth=0.6, zorder=1)
    else:
        ax.add_patch(mp.Rectangle(
            (-180, -90), 360, 180,
            linewidth=1.5, edgecolor=COL_BOR, facecolor=BG_AX, zorder=1
        ))

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap = ax.imshow(
        Z,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap=CMAP,
        alpha=0.90,
        aspect="auto",          # 'equal' distorce meno ma richiede Cartopy
        vmin=0, vmax=1,
        zorder=2
    )

    # ── Bordi paesi (sopra la heatmap) ───────────────────────────────────────
    if world is not None:
        world.plot(ax=ax, color="none", edgecolor=COL_BOR,
                   linewidth=0.6, zorder=3)

    # ── Titolo dinamico ───────────────────────────────────────────────────────
    ax.set_title(
        f"🔥 Heatmap Incendi — Densità Hotspot Globale\n"
        f"({TITOLO_DATI}{titolo_sub} · smoothing gaussiano σ={SMOOTH})",
        fontsize=16, fontweight="bold", color="white", pad=15
    )

    # ── Etichette assi ────────────────────────────────────────────────────────
    ax.set_xlabel("Longitudine", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Latitudine",  color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90,  90)

    # ── Griglia ───────────────────────────────────────────────────────────────
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color=COL_GRI, linewidth=0.4, linestyle="--", alpha=0.5, zorder=4)

    # ── Bordo esterno ─────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(COL_BOR)
        spine.set_linewidth(1.5)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = plt.colorbar(heatmap, ax=ax, orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Densità relativa di hotspot", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.outline.set_edgecolor(COL_BOR)

    plt.tight_layout()
    print("   Fatto ✓")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df    = carica_dati(CSV_PATH, campione=CAMPIONE)
    sub   = range_temporale(df)
    Z     = calcola_heatmap(df, RISOLUZIONE, SMOOTH)
    world = carica_confini()
    fig   = disegna(Z, world, titolo_sub=sub)
    plt.show()
    return fig


if __name__ == "__main__":
    main()