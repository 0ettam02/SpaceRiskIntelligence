"""
stagionalita_lineare_log.py
--------------------------
Dashboard stagionalità incendi con grafico lineare + scala log.

Uso:
  %matplotlib inline
  %run stagionalita_lineare_log.py

Dipendenze:
  pip install pandas matplotlib
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")




# ── Configurazione ────────────────────────────────────────────────────────────

CSV_PATH  = "dataset_incendi_FIRMS.csv"

NOMI_MESI = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
             "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

# Stagioni meteorologiche (indici 0-based dei mesi)
STAGIONI = {
    "Inverno": {"mesi": [11, 0, 1],  "colore": "#4a90d9"},
    "Primav.": {"mesi": [2, 3, 4],   "colore": "#6abf69"},
    "Estate":  {"mesi": [5, 6, 7, 8],"colore": "#fc8d59"},
    "Autunno": {"mesi": [9, 10],     "colore": "#f4a736"},
}

# Colonne data candidate (in ordine di priorità)
COLONNE_DATA = ["acq_date", "date", "Date", "ACQ_DATE"]

# Tema grafico
BG_FIG = "#1a1a2e"
BG_AX  = "#16213e"
COL_BOR = "#7a9cc4"
COL_GRI = "#4a6a8a"

PALETTE_ANNI = plt.cm.plasma


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caricamento e validazione
# ══════════════════════════════════════════════════════════════════════════════

def carica_dati(path: str) -> tuple[pd.DataFrame, str]:
    """
    Carica il CSV e restituisce (dataframe, nome_colonna_data).
    Esce con messaggio chiaro se il file è assente o le colonne mancano.
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERRORE] File non trovato: {p.resolve()}")

    print(f"[1/3] Caricamento dati da {p.name} …", end=" ", flush=True)
    df = pd.read_csv(p, low_memory=False)
    
        # ── DIAGNOSTICA TEMPORANEA ─────────────────────────────
    print("\n=== DIAGNOSI CSV ===")
    print(f"Righe totali: {len(df):,}")
    print(f"Colonne: {list(df.columns)}")
    
    col_data_temp = next((c for c in COLONNE_DATA if c in df.columns), None)
    if col_data_temp:
        print(f"Valori data (primi 5): {df[col_data_temp].head().tolist()}")
        print(f"Valori data (ultimi 5): {df[col_data_temp].tail().tolist()}")
        print(f"Tipo colonna: {df[col_data_temp].dtype}")
    print("====================\n")

    col_data = next((c for c in COLONNE_DATA if c in df.columns), None)
    if col_data is None:
        sys.exit(
            f"\n[ERRORE] Nessuna colonna data trovata.\n"
            f"         Cercate: {COLONNE_DATA}\n"
            f"         Presenti: {list(df.columns)}"
        )

    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    n_invalid = df[col_data].isna().sum()
    if n_invalid:
        print(f"\n   ⚠ {n_invalid} righe con data non valida scartate.")
    df = df.dropna(subset=[col_data])

    if df.empty:
        sys.exit("[ERRORE] Nessun dato valido dopo la pulizia delle date.")

    df["mese"] = df[col_data].dt.month
    df["anno"] = df[col_data].dt.year

    print(f"   {len(df):,} righe · "
          f"{df[col_data].min().date()} → {df[col_data].max().date()} ✓")
    return df, col_data


# ══════════════════════════════════════════════════════════════════════════════
# 2. Aggregazione
# ══════════════════════════════════════════════════════════════════════════════

def aggrega(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Restituisce:
      pivot       — righe=anni, colonne=mesi (1–12), valori=conteggio hotspot
      totale_anno — serie anno → totale annuale
      media_mese  — serie mese → media mensile su tutti gli anni
    """
    print("[2/3] Aggregazione …", end=" ", flush=True)

    pivot = (
        df.groupby(["anno", "mese"]).size()
        .unstack(level="mese")
        .reindex(columns=range(1, 13), fill_value=0)
    )

    if pivot.empty:
        sys.exit("[ERRORE] Nessun dato aggregabile (pivot vuoto).")

    totale_anno = pivot.sum(axis=1)
    media_mese  = pivot.mean(axis=0)

    anni = sorted(pivot.index.tolist())
    print(f"   {len(anni)} anni ({anni[0]}–{anni[-1]}) ✓")
    return pivot, totale_anno, media_mese


# ══════════════════════════════════════════════════════════════════════════════
# 3. Helpers di stile
# ══════════════════════════════════════════════════════════════════════════════

def _stile_ax(ax: plt.Axes) -> None:
    """Applica il tema scuro a un asse."""
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_BOR)


def _colore_anno(i: int, n: int) -> tuple:
    """Colore plasma normalizzato per l'i-esimo anno su n totali."""
    return PALETTE_ANNI(i / max(n - 1, 1))


def _colore_mese(mese_1based: int) -> str:
    """Restituisce il colore stagionale per un mese (1-based)."""
    idx = mese_1based - 1  # 0-based
    for stagione in STAGIONI.values():
        if idx in stagione["mesi"]:
            return stagione["colore"]
    return "#aaaaaa"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Pannello A — Stagionalità lineare (log scale)
# ══════════════════════════════════════════════════════════════════════════════

def disegna_lineare(ax: plt.Axes, pivot: pd.DataFrame,
                    media_mese: pd.Series) -> None:
    """
    Linea per ogni anno (palette plasma) + media in bianco tratteggiato.
    Scala logaritmica sull'asse Y per evidenziare variazioni relative.
    Sfondo stagionale semitrasparente invece delle linee verticali.
    """
    anni = pivot.index.tolist()
    n    = len(anni)

    # Sfondo stagioni (sostituisce le axvline arbitrarie dell'originale)
    for stagione, info in STAGIONI.items():
        for m_idx in info["mesi"]:
            ax.axvspan(m_idx - 0.5, m_idx + 0.5,
                       color=info["colore"], alpha=0.07, zorder=0)

    # Una linea per anno
    for i, anno in enumerate(anni):
        ax.plot(
            range(12),
            pivot.loc[anno].values.astype(float) + 1,  # +1 evita log(0)
            color=_colore_anno(i, n),
            linewidth=1.4,
            alpha=0.75,
            marker="o",
            markersize=3,
        )

    # Media storica
    ax.plot(
        range(12),
        media_mese.values + 1,
        color="white",
        linewidth=2.5,
        linestyle="--",
        label="Media storica",
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xticks(range(12))
    ax.set_xticklabels(NOMI_MESI, color="white", fontsize=9)
    ax.set_ylabel("Hotspot  (scala log)", color="#aaaaaa", fontsize=9)
    ax.tick_params(axis="y", colors="#aaaaaa")
    ax.grid(axis="y", color=COL_GRI, linestyle="--", alpha=0.4)
    ax.set_title("🔥 Stagionalità incendi — scala log",
                 color="white", fontsize=13, loc="left", pad=8)
    ax.legend(fontsize=8, facecolor=BG_FIG,
              edgecolor=COL_BOR, labelcolor="white")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pannello B — Colorbar anni (sostituisce la legenda)
# ══════════════════════════════════════════════════════════════════════════════

def disegna_colorbar_anni(ax: plt.Axes, anni: list) -> None:
    """
    Colorbar verticale con gli anni al posto di una legenda testuale
    (evita il problema con 20+ anni che riempiono tutta la figura).
    """
    n = len(anni)
    norm = mcolors.Normalize(vmin=anni[0], vmax=anni[-1])
    sm   = cm.ScalarMappable(cmap=PALETTE_ANNI, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax, orientation="vertical")
    cbar.set_label("Anno", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.outline.set_edgecolor(COL_BOR)

    # Tick solo sugli anni presenti
    step = max(1, n // 10)  # al massimo ~10 tick
    cbar.set_ticks([anni[i] for i in range(0, n, step)])


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pannello C — Totale annuale (barre orizzontali)
# ══════════════════════════════════════════════════════════════════════════════

def disegna_totale_anno(ax: plt.Axes, totale_anno: pd.Series) -> None:
    """
    Barre orizzontali con etichetta anno sull'asse Y e valore a fine barra.
    Media storica come linea verticale tratteggiata.
    """
    anni = totale_anno.index.tolist()
    n    = len(anni)
    colori = [_colore_anno(i, n) for i in range(n)]

    ax.barh(range(n), totale_anno.values, color=colori, height=0.7)

    # Etichette anno sull'asse Y
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(a) for a in anni], color="white", fontsize=7)

    ax.set_xlabel("Totale hotspot", color="#aaaaaa", fontsize=8)
    ax.tick_params(axis="x", colors="#aaaaaa", labelsize=7)

    # Valore numerico a fine barra
    max_val = totale_anno.max()
    for i, val in enumerate(totale_anno.values):
        ax.text(val + max_val * 0.02, i, f"{val:,}",
                va="center", color="white", fontsize=6.5)

    ax.set_xlim(0, max_val * 1.28)
    ax.axvline(totale_anno.mean(),
               color="#aaaaaa", linestyle="--", alpha=0.6,
               label=f"Media {totale_anno.mean():,.0f}")
    ax.legend(fontsize=7, facecolor=BG_FIG,
              edgecolor=COL_BOR, labelcolor="white")
    ax.grid(axis="x", color=COL_GRI, linestyle="--", alpha=0.3)
    ax.set_title("Totale annuale", color="white", fontsize=10, pad=6)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Pannello D — Media mensile (barre colorate per stagione)
# ══════════════════════════════════════════════════════════════════════════════

def disegna_media_mensile(ax: plt.Axes, media_mese: pd.Series) -> None:
    """
    Barre mensili colorate per stagione meteorologica con legenda stagioni.
    """
    bar_colors = [_colore_mese(m) for m in range(1, 13)]

    ax.bar(range(12), media_mese.values, color=bar_colors, alpha=0.75)

    ax.set_xticks(range(12))
    ax.set_xticklabels(NOMI_MESI, color="white", fontsize=9)
    ax.set_ylabel("Media hotspot / anno", color="#aaaaaa", fontsize=9)
    ax.tick_params(axis="y", colors="#aaaaaa")
    ax.grid(axis="y", color=COL_GRI, linestyle="--", alpha=0.4)
    ax.set_title("Media mensile per stagione", color="white",
                 fontsize=11, loc="left", pad=6)

    # Legenda stagioni
    patch_stagioni = [
        plt.Rectangle((0, 0), 1, 1,
                       fc=info["colore"], alpha=0.75, label=nome)
        for nome, info in STAGIONI.items()
    ]
    ax.legend(handles=patch_stagioni, fontsize=8,
              facecolor=BG_FIG, edgecolor=COL_BOR, labelcolor="white",
              loc="upper left")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Assembly figura
# ══════════════════════════════════════════════════════════════════════════════

def disegna_dashboard(pivot: pd.DataFrame,
                      totale_anno: pd.Series,
                      media_mese: pd.Series,
                      range_date: str) -> plt.Figure:
    """Assembla i 4 pannelli in una figura unica."""
    print("[3/3] Rendering …", end=" ", flush=True)

    anni = pivot.index.tolist()

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(BG_FIG)

    # Layout: [lineare | colorbar] / [media_mese | totale_anno]
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[5, 1.0],
        height_ratios=[3, 2],
        hspace=0.22,
        wspace=0.08,
    )

    ax_line  = fig.add_subplot(gs[0, 0])
    ax_cbar  = fig.add_subplot(gs[0, 1])   # colorbar anni (ex legenda)
    ax_media = fig.add_subplot(gs[1, 0])
    ax_tot   = fig.add_subplot(gs[1, 1])

    for ax in [ax_line, ax_media, ax_tot]:
        _stile_ax(ax)

    disegna_lineare(ax_line, pivot, media_mese)
    disegna_colorbar_anni(ax_cbar, anni)
    disegna_media_mensile(ax_media, media_mese)
    disegna_totale_anno(ax_tot, totale_anno)

    fig.suptitle(
        f"🔥 Dashboard Stagionalità Incendi  ·  {range_date}",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("   Fatto ✓")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df, col_data = carica_dati(CSV_PATH)
    range_date   = f"{df[col_data].min().date()} → {df[col_data].max().date()}"
    pivot, totale_anno, media_mese = aggrega(df)
    fig = disegna_dashboard(pivot, totale_anno, media_mese, range_date)
    plt.show()
    return fig


if __name__ == "__main__":
    main()