import os
import requests
import pandas as pd
import numpy as np

LAT = 18.0735
LON = -15.9582

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,RH2M,PRECTOTCORR,WS2M,PS"
    "&community=AG"
    "&longitude=-15.9582"
    "&latitude=18.0735"
    "&start=20250101"
    "&end=20250601"
    "&format=JSON"
)

def estimate_ndvi(df):
    """Estime un NDVI basé sur la température, l'humidité et la pluie."""
    temp_norm = np.clip((35.0 - df['T2M']) / 35.0, 0.0, 1.0)
    humidity_norm = np.clip(df['RH2M'] / 100.0, 0.0, 1.0)
    rain_norm = np.clip(df['PRECTOTCORR'] / (df['PRECTOTCORR'].max() + 1e-6), 0.0, 1.0)
    ndvi = 0.35 * humidity_norm + 0.30 * temp_norm + 0.35 * rain_norm
    return np.clip(ndvi, 0.0, 1.0)

data = requests.get(url).json()["properties"]["parameter"]

df = pd.DataFrame(data)
df = df.reset_index().rename(columns={
    "index": "date",
    "T2M": "Air temperature (C)",
    "RH2M": "Air humidity (%)",
    "PRECTOTCORR": "rainfall",
    "WS2M": "Wind speed (Km/h)",
    "PS": "Pressure (KPa)"
})

# Calcul du NDVI estimé pour le tableau de bord
ndvi_df = df.rename(columns={
    'Air temperature (C)': 'T2M',
    'Air humidity (%)': 'RH2M',
    'rainfall': 'PRECTOTCORR'
})
df['NDVI'] = estimate_ndvi(ndvi_df)

os.makedirs("data", exist_ok=True)
df.to_csv("data/nasa_recent_data.csv", index=False)
print("✅ Données NASA sauvegardées dans data/nasa_recent_data.csv")
