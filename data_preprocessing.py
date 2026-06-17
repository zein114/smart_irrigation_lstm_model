import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config

def create_sequences(data, time_steps):
    """Crée des séquences glissantes pour le LSTM"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, 0])  # Cible: Soil Moisture (index 0)
    return np.array(X), np.array(y)

def estimate_ndvi(df):
    """Estime un indice de végétation (NDVI) à partir des données disponibles."""
    soil_norm = np.clip(df['Soil Moisture'] / 100.0, 0.0, 1.0)
    humidity_norm = np.clip(df['Air humidity (%)'] / 100.0, 0.0, 1.0)
    temp_norm = np.clip((35.0 - df['Temperature']) / 35.0, 0.0, 1.0)
    rain_norm = np.clip(df['rainfall'] / (df['rainfall'].max() + 1e-6), 0.0, 1.0)
    return np.clip(0.35 * soil_norm + 0.30 * humidity_norm + 0.20 * rain_norm + 0.15 * temp_norm, 0.0, 1.0)

def preprocess_pipeline(df):
    # 1. Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()

    # 2. Conserver les valeurs aberrantes : pas de suppression des extrêmes
    # L'encadreur a demandé de garder les outliers.

    # 3. Créer un timestamp artificiel basé sur l'index
    df['timestamp'] = range(len(df))

    # 4. Trier par Soil Moisture pour créer une pseudo-séquence logique
    df = df.sort_values('Soil Moisture', ascending=True).reset_index(drop=True)

    # 5. Gérer les valeurs manquantes
    if 'rainfall' in df.columns:
        df['rainfall'] = df['rainfall'].fillna(0.0)

    df = df.dropna()

    # 6. Estimer le NDVI si nécessaire
    if 'NDVI' not in df.columns:
        df['NDVI'] = estimate_ndvi(df)

    # 7. Sélectionner les caractéristiques
    feature_cols = [
        'Soil Moisture',
        'Air temperature (C)',
        'Air humidity (%)',
        'rainfall',
        'Wind speed (Km/h)',
        'Pressure (KPa)',
        'Temperature',
        'NDVI'
    ]

    # Vérification
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    df_clean = df[feature_cols].copy()

    # 8. Normalisation MinMax
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # 9. Création des séquences
    X, y = create_sequences(scaled_data, config.TIME_STEPS)

    # 10. Split chronologique
    total_samples = len(X)
    train_end = int(total_samples * config.TRAIN_SPLIT)
    val_end = train_end + int(total_samples * config.VAL_SPLIT)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
