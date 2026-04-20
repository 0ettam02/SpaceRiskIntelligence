"""
areeCalde.py
------------
Visualizza su mappa statica MONDIALE le aree con maggiore densità di hotspot
rilevati dai satelliti NASA FIRMS.
 
Logica:
  - Legge il CSV già salvato dal notebook
  - Raggruppa gli hotspot in celle di griglia (grid clustering)
  - Calcola la densità per cella e assegna un livello di rischio
  - Visualizza la mappa mondiale con colori dal giallo (basso) al bordeaux (critico)
 
Uso nel notebook:
  %run areeCalde.py
"""
 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import box
import warnings
warnings.filterwarnings("ignore")
 
# ── Configurazione ────────────────────────────────────────────────────────────
 
CSV_PATH    = "dataset_incendi_FIRMS.csv"
GRID_DEG    = 1.0                           # cella 1° (~111 km) — ottimale per scala mondiale
 
# Soglie densità per livello di rischio (hotspot per cella)
SOGLIE = {
    "Basso":   (1,  5),
    "Medio":   (6,  20),
    "Alto":    (21, 60),
    "Critico": (61, float("inf")),
}
COLORI = {
    "Basso":   "#fee08b",   # giallo chiaro
    "Medio":   "#fc8d59",   # arancione
    "Alto":    "#d73027",   # rosso
    "Critico": "#7f0000",   # bordeaux scuro
}
 
# ── 1. Caricamento dati ───────────────────────────────────────────────────────
 
df = pd.read_csv(CSV_PATH, low_memory=False)
 
df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])
 
# ── 2. Grid clustering mondiale ───────────────────────────────────────────────
 
 
df["cell_lat"] = np.floor(df["latitude"]  / GRID_DEG) * GRID_DEG
df["cell_lon"] = np.floor(df["longitude"] / GRID_DEG) * GRID_DEG
 
grid = (
    df.groupby(["cell_lat", "cell_lon"])
    .size()
    .reset_index(name="hotspot_count")
)
 
def assegna_rischio(n):
    for livello, (low, high) in SOGLIE.items():
        if low <= n <= high:
            return livello
    return "Basso"
 
grid["rischio"] = grid["hotspot_count"].apply(assegna_rischio)
 

 
# ── 3. Costruzione GeoDataFrame celle ────────────────────────────────────────
 
grid["geometry"] = grid.apply(
    lambda r: box(r["cell_lon"], r["cell_lat"],
                  r["cell_lon"] + GRID_DEG, r["cell_lat"] + GRID_DEG),
    axis=1
)
gdf_grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326")
 
# ── 4. Confini del mondo ──────────────────────────────────────────────────────
 
try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except Exception:
    try:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
    except Exception:
        world = None
        print("   (confini non disponibili)")
 
# ── 5. Visualizzazione ────────────────────────────────────────────────────────
 
print("🖼️  Genero la mappa mondiale...")
 
fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")
 
# Sfondo paesi con bordi ben visibili
if world is not None:
    world.plot(ax=ax, color="#0f3460", edgecolor="#7a9cc4", linewidth=0.7, zorder=1)
else:
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((-180, -90), 360, 180, linewidth=1.5,
                              edgecolor="#7a9cc4", facecolor="#16213e", zorder=1))
 
# Celle di rischio
ordine_rischio = ["Basso", "Medio", "Alto", "Critico"]
for livello in ordine_rischio:
    subset = gdf_grid[gdf_grid["rischio"] == livello]
    if not subset.empty:
        subset.plot(
            ax=ax,
            color=COLORI[livello],
            alpha=0.80,
            edgecolor="none",
            zorder=2 + ordine_rischio.index(livello)
        )
 
# Titolo e assi
ax.set_title(
    "🔥 Aree Calde — Rischio Incendio Globale\n(densità hotspot NASA FIRMS · griglia 1°)",
    fontsize=16, fontweight="bold", color="white", pad=15
)
ax.set_xlabel("Longitudine", color="#aaaaaa", fontsize=9)
ax.set_ylabel("Latitudine",  color="#aaaaaa", fontsize=9)
ax.tick_params(colors="#aaaaaa")
 
ax.set_xlim(-180, 180)
ax.set_ylim(-90,  90)
 
# Griglia coordinate
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range(-90, 91, 30))
ax.grid(color="#4a6a8a", linewidth=0.5, linestyle="--", alpha=0.6)
 
# Bordo esterno della mappa
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#7a9cc4")
    spine.set_linewidth(1.5)
 
# Legenda
legenda = [
    mpatches.Patch(
        color=COLORI[l],
        label=f"{l}  ({SOGLIE[l][0]}–"
              + (f"{int(SOGLIE[l][1])} hotspot/cella)" if SOGLIE[l][1] != float('inf') else "61+ hotspot/cella)")
    )
    for l in ordine_rischio
]
ax.legend(
    handles=legenda,
    loc="lower left",
    framealpha=0.85,
    facecolor="#1a1a2e",
    edgecolor="#e94560",
    labelcolor="white",
    title="Livello di Rischio",
    title_fontsize=10,
    fontsize=9
)
 
plt.tight_layout()
plt.show()