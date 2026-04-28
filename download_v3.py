import urllib.request
import pandas as pd
from pathlib import Path

ANNI   = list(range(2011, 2026))
OUTPUT = Path('dataset_incendi_FIRMS_storico.csv')

tutti = []

for anno in ANNI:
    url = f'https://firms.modaps.eosdis.nasa.gov/data/country/modis/{anno}/modis_{anno}_all_countries.csv'
    dest = Path(f'modis_{anno}.csv')
    
    print(f'{anno}: scarico...', end=' ', flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        df = pd.read_csv(dest, low_memory=False)
        tutti.append(df)
        print(f'{len(df):,} righe OK')
    except Exception as e:
        print(f'ERRORE: {e}')

if tutti:
    df_finale = pd.concat(tutti, ignore_index=True)
    df_finale.to_csv(OUTPUT, index=False)
    print(f'Salvato: {len(df_finale):,} righe totali')
    print(f'Ora imposta CSV_PATH = \"{OUTPUT}\" in stagionalita.py')
