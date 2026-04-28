import requests
import pandas as pd
import time
from pathlib import Path
from io import StringIO

MAP_KEY  = 'dfe9b1d91d73e30117538437ffdcf5bb'
ANNI     = list(range(2011, 2027))
SENSORE  = 'MODIS_SP'
OUTPUT   = Path('dataset_incendi_FIRMS_storico.csv')

# Endpoint per paese (piu affidabile per dati storici)
BASE_URL_PAESE = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv'
# Endpoint area come fallback
BASE_URL_AREA  = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv'

# Paesi con piu incendi al mondo (copertura globale pratica)
PAESI = [
    'BRA','USA','AUS','RUS','CAN','CHN','COD','AGO','MOZ','ZMB',
    'TZA','ETH','MDG','VEN','COL','BOL','ARG','PER','MEX','IND',
    'IDN','MYS','MMR','THA','LAO','KHM','VNM','PHL','NGA','SDN',
    'CAF','CMR','GHA','CIV','MLI','NER','TCD','SOM','KEN','ZWE',
    'ZAF','NAM','BWA','SWZ','MWI','UGA','FIN','SWE','NOR','FRA',
    'ESP','PRT','ITA','GRC','TUR','IRN','IRQ','SAU','PAK','AFG',
]

def scarica_paese_anno(paese, anno):
    url = f'{BASE_URL_PAESE}/{MAP_KEY}/{SENSORE}/{paese}/12months/{anno}-01-01'
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.text) > 100 and 'acq_date' in r.text:
            df = pd.read_csv(StringIO(r.text))
            if not df.empty and 'acq_date' in df.columns:
                # Filtra solo l'anno richiesto
                df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
                df = df[df['acq_date'].dt.year == anno]
                return df
        elif r.status_code == 429:
            print(' [rate limit 60s]', flush=True)
            time.sleep(60)
            return scarica_paese_anno(paese, anno)
    except Exception as e:
        pass
    return None

def scarica_anno(anno):
    frames = []
    print(f'  {anno}: ', end='', flush=True)
    for paese in PAESI:
        df_p = scarica_paese_anno(paese, anno)
        if df_p is not None and not df_p.empty:
            frames.append(df_p)
            print('#', end='', flush=True)
        else:
            print('.', end='', flush=True)
        time.sleep(0.8)
    if frames:
        df_anno = pd.concat(frames, ignore_index=True)
        df_anno = df_anno.drop_duplicates(subset=['latitude','longitude','acq_date','acq_time'])
        print(f'  -> {len(df_anno):,} hotspot')
        return df_anno
    print('  -> nessun dato')
    return None

def main():
    print('FIRMS MODIS - Download per paese 2011-2026')
    print('Legenda: # = dati OK   . = vuoto/errore')
    tutti_frame = []
    anni_da_scaricare = list(ANNI)
    if OUTPUT.exists():
        print('File parziale trovato - riprendo...')
        df_esistente = pd.read_csv(OUTPUT, low_memory=False)
        df_esistente['acq_date'] = pd.to_datetime(df_esistente['acq_date'])
        anni_presenti = set(df_esistente['acq_date'].dt.year.unique())
        anni_da_scaricare = [a for a in ANNI if a not in anni_presenti]
        print(f'Anni gia presenti: {sorted(anni_presenti)}')
        print(f'Anni rimanenti: {anni_da_scaricare}')
        tutti_frame.append(df_esistente)
    for anno in anni_da_scaricare:
        df_anno = scarica_anno(anno)
        if df_anno is not None:
            tutti_frame.append(df_anno)
            df_tmp = pd.concat(tutti_frame, ignore_index=True)
            df_tmp.to_csv(OUTPUT, index=False)
            print(f'  Salvato - totale: {len(df_tmp):,} righe')
        time.sleep(2)
    if tutti_frame:
        df_finale = pd.concat(tutti_frame, ignore_index=True)
        df_finale = df_finale.drop_duplicates(subset=['latitude','longitude','acq_date','acq_time'])
        df_finale['acq_date'] = pd.to_datetime(df_finale['acq_date'])
        df_finale = df_finale.sort_values('acq_date').reset_index(drop=True)
        df_finale.to_csv(OUTPUT, index=False)
        anni_u = sorted(df_finale['acq_date'].dt.year.unique())
        print('COMPLETATO!')
        print(f'Righe totali : {len(df_finale):,}')
        print(f'Anni coperti : {anni_u[0]} -> {anni_u[-1]}')
        print(f'Ora in stagionalita.py imposta:')
        print(f'  CSV_PATH = "dataset_incendi_FIRMS_storico.csv"')
    else:
        print('ERRORE: nessun dato scaricato.')

if __name__ == '__main__':
    main()
