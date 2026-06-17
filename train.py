import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import config

from data_preprocessing import preprocess_pipeline
from model import build_lstm_model

def main():
    print("🔄 Chargement des données...")
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"⚠️ Fichier non trouvé : {config.RAW_DATA_PATH}")
        return
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    print("⚙️ Prétraitement (Création timestamp artificiel + Tri logique)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = preprocess_pipeline(df)
    print(f"✅ Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("🏗️ Construction du modèle LSTM amélioré...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()
    
    print("🚀 Démarrage de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("📊 Évaluation sur l'ensemble de Test...")
    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    
    print("\n--- Résultats Finaux (Données Normalisées) ---")
    for name, value in zip(metric_names, test_metrics):
        print(f"{name.upper()}: {value:.4f}")
    
    # Évaluation en données réelles
    y_test_pred = model.predict(X_test)
    dummy_features = np.zeros((len(y_test), config.NUM_FEATURES - 1))
    
    y_test_denorm = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), dummy_features], axis=1)
    )[:, 0]
    
    y_pred_denorm = scaler.inverse_transform(
        np.concatenate([y_test_pred.reshape(-1, 1), dummy_features], axis=1)
    )[:, 0]
    
    print("\n--- Résultats Finaux (Données Réelles) ---")
    print(f"MAE réel: {mean_absolute_error(y_test_denorm, y_pred_denorm):.4f}")
    print(f"RMSE réel: {np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)):.4f}")
    print(f"R² réel: {r2_score(y_test_denorm, y_pred_denorm):.4f}")
    
    tolerance_abs = 5.0
    accurate_preds = np.sum(np.abs(y_test_denorm - y_pred_denorm) <= tolerance_abs)
    tolerance_accuracy_real = (accurate_preds / len(y_test_denorm)) * 100
    print(f"Précision de Tolérance (±{tolerance_abs}%): {tolerance_accuracy_real:.2f}%")
    
    print("💾 Sauvegarde du modèle et du scaler...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    model.save(os.path.join(config.DATA_DIR, "lstm_soil_moisture_model.keras"))
    joblib.dump(scaler, config.SCALER_PATH)
    
    print("📈 Génération des courbes d'entraînement...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (Huber)')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['tolerance_accuracy'], label='Train Tol. Acc.')
    plt.plot(history.history['val_tolerance_accuracy'], label='Val Tol. Acc.')
    plt.title('Tolerance Accuracy (±5%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("✅ Terminé !")

if __name__ == "__main__":
    main()