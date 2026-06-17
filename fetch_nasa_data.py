import requests
import pandas as pd

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

df.to_csv("data/nasa_recent_data.csv", index=False)
print("Données NASA sauvegardées")














