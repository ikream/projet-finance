"""
Fonctions utilitaires pour le module IA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_returns(prices):
    """Calcule les rendements à partir des prix"""
    returns = np.diff(prices) / prices[:-1]
    return returns

def calculate_volatility(prices, window=20):
    """Calcule la volatilité mobile"""
    returns = calculate_returns(prices)
    volatility = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
    return volatility.values

def create_lag_features(df, columns, lags=[1, 2, 3, 5, 7]):
    """Crée des features avec décalages temporels"""
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    
    return df_lagged

def calculate_technical_indicators(df):
    """Calcule des indicateurs techniques supplémentaires"""
    df = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df

def split_sequences_multivariate(sequences, targets, n_steps):
    """Divise une séquence multivariée en échantillons"""
    X, y = [], []
    
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix]
        seq_y = targets[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

def calculate_confidence_interval(predictions, confidence_level=0.95):
    """Calcule l'intervalle de confiance pour des prédictions"""
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Z-score pour le niveau de confiance
    z_score = 1.96  # pour 95% de confiance
    
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    return mean_pred, lower_bound, upper_bound

def format_currency(value):
    """Formate un nombre en devise"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Formate un nombre en pourcentage"""
    return f"{value:.2f}%"

def save_plot(fig, filename, dpi=300):
    """Sauvegarde une figure avec des paramètres par défaut"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def log_message(message, level="INFO"):
    """Journalise un message avec horodatage"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_config(config_file="config.json"):
    """Charge la configuration depuis un fichier JSON"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        log_message(f"Configuration chargée depuis {config_file}")
        return config
    except FileNotFoundError:
        log_message(f"Fichier de configuration {config_file} non trouvé", "WARNING")
        return {}

class Timer:
    """Classe utilitaire pour mesurer le temps d'exécution"""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_message(f"Début: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        elapsed = self.end_time - self.start_time
        log_message(f"Fin: {self.name} - Temps écoulé: {elapsed}")
    
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        else:
            return timedelta(0)
        



#         ai_modeling/
# ├── prepare_dataset.py # Préparation des datasets
# ├── prophet_model.py # Modèle Prophet
# ├── lstm_model.py # Modèle LSTM
# ├── recommendation_engine.py # Moteur de recommandation
# ├── evaluate_models.py # Évaluation et comparaison
# ├── predict.py # Pipeline de prédiction
# ├── utils.py # Fonctions utilitaires
# ├── requirements.txt # Dépendances
# └── README.md # Documentation