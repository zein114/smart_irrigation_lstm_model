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

def preprocess_pipeline(df):
    # 1. Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # 2. Créer un timestamp artificiel basé sur l'index
    df['timestamp'] = range(len(df))
    
    # 3. Trier par Soil Moisture pour créer une pseudo-séquence logique
    # Cela simule un cycle d'assèchement et d'arrosage
    df = df.sort_values('Soil Moisture', ascending=True).reset_index(drop=True)
    
    # 4. Gérer les valeurs manquantes
    if 'rainfall' in df.columns:
        df['rainfall'] = df['rainfall'].fillna(0.0)
    
    df = df.dropna()
    
    # 5. Sélectionner les caractéristiques
    feature_cols = [
        'Soil Moisture',
        'Air temperature (C)',
        'Air humidity (%)',
        'rainfall',
        'Wind speed (Km/h)',
        'Pressure (KPa)',
        'Temperature'
    ]
    
    # Vérification
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")
    
    df_clean = df[feature_cols].copy()
    
    # 6. Normalisation MinMax
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean)
    
    # 7. Création des séquences
    X, y = create_sequences(scaled_data, config.TIME_STEPS)
    
    # 8. Split chronologique
    total_samples = len(X)
    train_end = int(total_samples * config.TRAIN_SPLIT)
    val_end = train_end + int(total_samples * config.VAL_SPLIT)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler