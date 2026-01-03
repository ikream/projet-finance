"""
Préparation des données pour le modèle LSTM
- Charge les données brutes (crypto et stocks)
- Nettoie les données (gestion NaN, normalisation)
- Crée des séquences pour LSTM
- Sauvegarde les données préparées
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean_data():
    """Charge et nettoie les données brutes"""
    print("📊 Chargement et nettoyage des données...")
    
    # Charger les données crypto
    crypto_df = pd.read_csv('prj_finance/data_pipeline/crypto_clean.csv')
    print(f"  Crypto data shape: {crypto_df.shape}")
    
    # Charger les données stocks (format spécial avec colonnes multiples)
    stocks_df = pd.read_csv('prj_finance/data_pipeline/stocks_raw.csv')
    print(f"  Stocks raw shape: {stocks_df.shape}")
    
    # Nettoyer les données stocks (supprimer colonnes inutiles, pivoter si nécessaire)
    # Le CSV a des colonnes comme Date, Close, High, Low, Open, Volume, Symbol, puis répétition pour MSFT
    # Garder seulement AAPL pour simplicité
    stocks_df = stocks_df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Symbol']]
    stocks_df = stocks_df[stocks_df['Symbol'] == 'AAPL'].drop(columns=['Symbol'])
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df = stocks_df.sort_values('Date').reset_index(drop=True)
    print(f"  Stocks AAPL cleaned shape: {stocks_df.shape}")
    
    # Nettoyer les données crypto
    # Supposer que crypto_clean.csv a déjà des colonnes Date, Close, etc.
    crypto_df['Date'] = pd.to_datetime(crypto_df['Date'])
    crypto_df = crypto_df.sort_values('Date').reset_index(drop=True)
    print(f"  Crypto cleaned shape: {crypto_df.shape}")
    
    # Fusionner les données (utiliser stocks comme principale, ajouter crypto si pertinent)
    # Pour simplicité, utiliser seulement les données stocks AAPL
    df = stocks_df.copy()
    df = df.dropna()  # Supprimer les lignes avec NaN
    df = df.drop_duplicates(subset=['Date'])
    
    print(f"  Données finales shape: {df.shape}")
    print(f"  Période: {df['Date'].min()} à {df['Date'].max()}")
    
    return df

def create_sequences(data, seq_length=30, target_col='Close'):
    """Crée des séquences pour LSTM"""
    print(f"🔄 Création de séquences (longueur: {seq_length})...")
    
    # Features: utiliser toutes les colonnes numériques sauf Date
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Séquence des features
        seq = data[feature_cols].iloc[i:i+seq_length].values
        X.append(seq)
        
        # Target: prix de clôture suivant
        target = data[target_col].iloc[i+seq_length]
        y.append(target)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"  Séquences créées: {X.shape[0]}")
    print(f"  Shape X: {X.shape}, y: {y.shape}")
    
    return X, y

def prepare_data(seq_length=30, test_size=0.2, val_size=0.1):
    """Pipeline complet de préparation"""
    
    # Création des dossiers
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Charger et nettoyer
    df = load_and_clean_data()
    
    # 2. Normalisation
    scaler = MinMaxScaler()
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Sauvegarder le scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler sauvegardé: models/scaler.pkl")
    
    # 3. Créer séquences
    X, y = create_sequences(df_scaled, seq_length=seq_length)
    
    # 4. Split train/val/test
    # D'abord split train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Puis train / val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, shuffle=False
    )
    
    print(f"  Train: {X_train.shape[0]} séquences")
    print(f"  Val: {X_val.shape[0]} séquences")
    print(f"  Test: {X_test.shape[0]} séquences")
    
    # 5. Sauvegarder
    np.save('data/processed/X_seq.npy', X_train)
    np.save('data/processed/y_seq.npy', y_train)
    np.save('data/processed/X_val_seq.npy', X_val)
    np.save('data/processed/y_val_seq.npy', y_val)
    np.save('data/processed/X_test_seq.npy', X_test)
    np.save('data/processed/y_test_seq.npy', y_test)
    
    print("✅ Données préparées sauvegardées dans data/processed/")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    prepare_data(seq_length=30, test_size=0.2, val_size=0.1)
    print("\n🎉 Préparation des données terminée!")