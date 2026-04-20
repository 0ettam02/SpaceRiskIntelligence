"""
areeCalde.py
------------
Heatmap mondiale degli incendi con istogramma 2D + smoothing gaussiano.
Molto più veloce del KDE puro, stesso effetto visivo sfumato.

Uso nel notebook:
  %matplotlib inline
  %run areeCalde.py
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ── Configurazione ────────────────────────────────────────────────────────────

CSV_PATH    = "dataset_incendi_FIRMS.csv"
RISOLUZIONE = 360   # celle per asse (360 lon x 180 lat = 1 cella per grado)
SMOOTH      = 3     # sigma dello smoothing gaussiano (più alto = più sfumato)

# ── 1. Caricamento dati ───────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, low_memory=False)
df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])

# ── 2. Istogramma 2D + smoothing ─────────────────────────────────────────────

Z, _, _ = np.histogram2d(
    df["longitude"], df["latitude"],
    bins=[RISOLUZIONE, RISOLUZIONE // 2],
    range=[[-180, 180], [-90, 90]]
)
Z = Z.T  # trasposta per avere lat sull'asse Y

# Smoothing gaussiano per sfumare i bordi
Z = gaussian_filter(Z, sigma=SMOOTH)

# Normalizza tra 0 e 1
Z = Z / Z.max()
print("   Fatto ✓")

# ── 3. Confini del mondo ──────────────────────────────────────────────────────

world = None
try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except Exception:
    try:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
    except Exception:
        print("   (confini non disponibili)")

# ── 4. Colormap ───────────────────────────────────────────────────────────────

cmap = mcolors.LinearSegmentedColormap.from_list("incendi", [
    (0.00, (0, 0, 0, 0)),       # trasparente
    (0.15, "#fee08b"),           # giallo
    (0.45, "#fc8d59"),           # arancione
    (0.75, "#d73027"),           # rosso
    (1.00, "#7f0000"),           # bordeaux
])

# ── 5. Visualizzazione ────────────────────────────────────────────────────────


fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# Sfondo paesi
if world is not None:
    world.plot(ax=ax, color="#0f3460", edgecolor="#7a9cc4", linewidth=0.6, zorder=1)
else:
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((-180, -90), 360, 180, linewidth=1.5,
                               edgecolor="#7a9cc4", facecolor="#16213e", zorder=1))

# Heatmap
heatmap = ax.imshow(
    Z,
    extent=[-180, 180, -90, 90],
    origin="lower",
    cmap=cmap,
    alpha=0.90,
    aspect="auto",
    zorder=2
)

# Bordi paesi sopra la heatmap
if world is not None:
    world.plot(ax=ax, color="none", edgecolor="#7a9cc4", linewidth=0.6, zorder=3)

# Titolo e assi
ax.set_title(
    "🔥 Heatmap Incendi — Densità Hotspot Globale\n(NASA FIRMS · smoothing gaussiano)",
    fontsize=16, fontweight="bold", color="white", pad=15
)
ax.set_xlabel("Longitudine", color="#aaaaaa", fontsize=9)
ax.set_ylabel("Latitudine",  color="#aaaaaa", fontsize=9)
ax.tick_params(colors="#aaaaaa")
ax.set_xlim(-180, 180)
ax.set_ylim(-90,  90)

# Griglia
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range(-90, 91, 30))
ax.grid(color="#4a6a8a", linewidth=0.4, linestyle="--", alpha=0.5, zorder=4)

# Bordo esterno
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#7a9cc4")
    spine.set_linewidth(1.5)

# Colorbar
cbar = plt.colorbar(heatmap, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Densità relativa di hotspot", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.outline.set_edgecolor("#7a9cc4")

plt.tight_layout()
plt.show()