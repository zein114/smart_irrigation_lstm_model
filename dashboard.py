import os
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import estimate_ndvi
from recommendations import generate_recommendation

DATA_DIR = "data"
SENSOR_CSV = os.path.join(DATA_DIR, "iotsensordata.csv")
NASA_CSV = os.path.join(DATA_DIR, "nasa_recent_data.csv")


def load_sensor_data():
    if not os.path.exists(SENSOR_CSV):
        raise FileNotFoundError(f"Fichier de capteur introuvable : {SENSOR_CSV}")
    df = pd.read_csv(SENSOR_CSV)
    df.columns = df.columns.str.strip()
    return df


def load_nasa_data():
    if os.path.exists(NASA_CSV):
        df = pd.read_csv(NASA_CSV)
        df.columns = df.columns.str.strip()
        return df
    return None


def ensure_ndvi(df):
    if df is None:
        return None
    if 'NDVI' not in df.columns:
        df['NDVI'] = estimate_ndvi(df)
    return df


def plot_series(df, x, y, title, output_name, ylabel=None):
    plt.figure(figsize=(10, 4))
    plt.plot(df[x], df[y], marker='o', linestyle='-', alpha=0.8)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(ylabel or y)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()


def create_dashboard(sensor, nasa):
    if 'timestamp' not in sensor.columns:
        sensor = sensor.copy()
        sensor['timestamp'] = range(len(sensor))

    plot_series(sensor, 'timestamp', 'Soil Moisture', 'Soil Moisture', 'dashboard_soil_moisture.png', 'Soil Moisture (%)')
    plot_series(sensor, 'timestamp', 'Temperature', 'Air Temperature', 'dashboard_temperature.png', 'Température (°C)')
    plot_series(sensor, 'timestamp', 'rainfall', 'Precipitation', 'dashboard_precipitation.png', 'Précipitation (mm)')
    plot_series(sensor, 'timestamp', 'NDVI', 'NDVI Estimé', 'dashboard_ndvi.png', 'NDVI')

    latest_sensor = sensor.iloc[-1].to_dict()
    latest_nasa = nasa.iloc[-1].to_dict() if nasa is not None else None
    recommendation = generate_recommendation(latest_sensor, latest_nasa)

    html = f"""
    <html lang='fr'>
    <head>
      <meta charset='UTF-8'>
      <title>Tableau de bord Smart Irrigation</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f9fb; color: #1f2937; }}
        .card {{ background: white; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 18px rgba(31, 41, 55, 0.08); }}
        .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
        h1 {{ margin-bottom: 8px; }}
        h2 {{ margin-top: 0; }}
        .metric {{ font-size: 2rem; margin: 8px 0; color: #0f766e; }}
        .recommendation {{ white-space: pre-wrap; line-height: 1.6; }}
      </style>
    </head>
    <body>
      <h1>Smart Irrigation Dashboard</h1>
      <div class='grid'>
        <div class='card'>
          <h2>Dernière lecture</h2>
          <div class='metric'>{latest_sensor.get('Soil Moisture', 'N/A'):.1f}%</div>
          <p>Humidité du sol</p>
          <div class='metric'>{latest_sensor.get('Temperature', 'N/A'):.1f}°C</div>
          <p>Température</p>
          <div class='metric'>{latest_sensor.get('rainfall', 'N/A'):.1f} mm</div>
          <p>Précipitations</p>
          <div class='metric'>{latest_sensor.get('NDVI', 'N/A'):.2f}</div>
          <p>NDVI estimé</p>
        </div>
        <div class='card'>
          <h2>Recommandation agricole</h2>
          <div class='recommendation'>{recommendation}</div>
        </div>
      </div>
      <div class='card'>
        <h2>Visualisations</h2>
        <img src='dashboard_soil_moisture.png' alt='Soil Moisture' width='100%'>
        <img src='dashboard_temperature.png' alt='Temperature' width='100%'>
        <img src='dashboard_precipitation.png' alt='Precipitation' width='100%'>
        <img src='dashboard_ndvi.png' alt='NDVI' width='100%'>
      </div>
    </body>
    </html>
    """

    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("✅ Dashboard créé : dashboard.html")
    print("✅ Graphiques générés : dashboard_soil_moisture.png, dashboard_temperature.png, dashboard_precipitation.png, dashboard_ndvi.png")


def main():
    sensor = load_sensor_data()
    sensor = ensure_ndvi(sensor)
    nasa = ensure_ndvi(load_nasa_data())
    create_dashboard(sensor, nasa)


if __name__ == '__main__':
    main()
