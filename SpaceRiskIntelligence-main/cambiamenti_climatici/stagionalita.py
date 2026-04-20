"""
stagionalita_lineare_log.py
--------------------------
Dashboard stagionalità incendi con grafico lineare + scala log.

Uso:
  %matplotlib inline
  %run stagionalita_lineare_log.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────

CSV_PATH  = "dataset_incendi_FIRMS.csv"
NOMI_MESI = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
             "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

PALETTE_ANNI = plt.cm.plasma

# ── 1. Load dati ─────────────────────────────────────────────────────────────

print("📂 Carico il dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)

colonne_data = ["acq_date", "date", "Date", "ACQ_DATE"]
col_data = next((c for c in colonne_data if c in df.columns), None)

if col_data is None:
    print("⚠️ Colonna data non trovata")
    exit()

df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
df = df.dropna(subset=[col_data])

df["mese"] = df[col_data].dt.month
df["anno"] = df[col_data].dt.year

anni = sorted(df["anno"].unique())
print(f"Range: {df[col_data].min().date()} → {df[col_data].max().date()}")

# ── 2. Aggregazioni ──────────────────────────────────────────────────────────

pivot = (
    df.groupby(["anno", "mese"]).size()
    .unstack(level="mese")
    .reindex(columns=range(1, 13), fill_value=0)
)

totale_anno = pivot.sum(axis=1)
media_mese  = pivot.mean(axis=0)

# ── 3. Layout ────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#1a1a2e")

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[5, 1.2],
    height_ratios=[3, 2],
    hspace=0.15,
    wspace=0.05
)

ax_line  = fig.add_subplot(gs[0, 0])
ax_tot   = fig.add_subplot(gs[0, 1])
ax_media = fig.add_subplot(gs[1, 0])

for ax in [ax_line, ax_tot, ax_media]:
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#7a9cc4")

# ── 4. STAGIONALITÀ LINEARE (LOG SCALE) ─────────────────────────────────────

for i, anno in enumerate(anni):
    colore = PALETTE_ANNI(i / max(len(anni) - 1, 1))
    ax_line.plot(
        range(12),
        pivot.loc[anno].values.astype(float) + 1,  # evita log(0)
        color=colore,
        linewidth=1.5,
        alpha=0.7,
        marker="o",
        markersize=4,
        label=str(anno)
    )

# Media
ax_line.plot(
    range(12),
    media_mese.values + 1,
    color="white",
    linewidth=3,
    linestyle="--",
    label="Media"
)

# 🔥 Scala log
ax_line.set_yscale("log")

ax_line.set_xticks(range(12))
ax_line.set_xticklabels(NOMI_MESI, color="white")

ax_line.set_ylabel("Hotspot (scala log)", color="#aaaaaa")
ax_line.tick_params(axis="y", colors="#aaaaaa")

# Stagioni
for x in [2.5, 5.5, 8.5, 10.5]:
    ax_line.axvline(x, color="#7a9cc4", linestyle="--", alpha=0.3)

ax_line.set_title(
    "🔥 Stagionalità Incendi (scala log)",
    color="white",
    fontsize=14,
    loc="left"
)

ax_line.grid(axis="y", color="#4a6a8a", linestyle="--", alpha=0.4)

# Legenda
handles, labels = ax_line.get_legend_handles_labels()
ax_line.legend(
    handles, labels,
    fontsize=7,
    ncol=max(1, len(anni)//4),
    facecolor="#1a1a2e",
    edgecolor="#7a9cc4",
    labelcolor="white"
)

# ── 5. Totale per anno ───────────────────────────────────────────────────────

colori_anni = [PALETTE_ANNI(i / max(len(anni) - 1, 1)) for i in range(len(anni))]

ax_tot.barh(range(len(anni)), totale_anno.values,
            color=colori_anni)

ax_tot.set_yticks([])
ax_tot.set_xlabel("Totale hotspot", color="#aaaaaa")
ax_tot.tick_params(axis="x", colors="#aaaaaa")

for i, val in enumerate(totale_anno.values):
    ax_tot.text(val + totale_anno.max()*0.02, i, f"{val:,}",
                va="center", color="white", fontsize=7)

ax_tot.set_xlim(0, totale_anno.max()*1.3)
ax_tot.axvline(totale_anno.mean(),
               color="#aaaaaa", linestyle="--", alpha=0.6)

ax_tot.set_title("Totale annuale", color="white")

# ── 6. Media mensile ─────────────────────────────────────────────────────────

bar_colors = []
for m in range(1, 13):
    if m in [12,1,2]: bar_colors.append("#4a90d9")
    elif m in [3,4,5]: bar_colors.append("#6abf69")
    elif m in [6,7,8,9]: bar_colors.append("#fc8d59")
    else: bar_colors.append("#f4a736")

ax_media.bar(range(12), media_mese.values,
             color=bar_colors, alpha=0.4)

ax_media.set_xticks(range(12))
ax_media.set_xticklabels(NOMI_MESI, color="white")

ax_media.set_ylabel("Media hotspot", color="#aaaaaa")
ax_media.tick_params(axis="y", colors="#aaaaaa")

ax_media.set_title("Media mensile incendi", color="white")

ax_media.grid(axis="y", color="#4a6a8a",
              linestyle="--", alpha=0.4)

# ── Render ───────────────────────────────────────────────────────────────────

plt.show()
