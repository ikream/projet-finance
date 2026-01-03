"""
Script principal pour faire des prédictions avec les modèles entraînés
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from prophet.serialize import model_from_json
import tensorflow as tf
import joblib
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# Ajouter le chemin du projet pour pouvoir importer recommendation_engine
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from recommendation_engine import RecommendationEngine
except ImportError:
    print("⚠️  Module recommendation_engine non trouvé, création d'une version simple")
    
    class RecommendationEngine:
        def __init__(self):
            pass
        
        def generate_signal(self, price_change, confidence):
            # Normaliser le changement de prix
            normalized_change = min(max(price_change, -0.5), 0.5)  # Limiter à ±50%
            
            if normalized_change > 0.05:  # +5%
                action = 'BUY'
                message = f'Forte tendance haussière (+{normalized_change*100:.1f}%)'
            elif normalized_change > 0.02:  # +2% à +5%
                action = 'BUY'
                message = f'Tendance haussière modérée (+{normalized_change*100:.1f}%)'
            elif normalized_change < -0.05:  # -5%
                action = 'SELL'
                message = f'Forte tendance baissière ({normalized_change*100:.1f}%)'
            elif normalized_change < -0.02:  # -2% à -5%
                action = 'SELL'
                message = f'Tendance baissière modérée ({normalized_change*100:.1f}%)'
            else:
                action = 'HOLD'
                message = f'Tendance neutre ({normalized_change*100:.1f}%)'
            
            return {
                'action': action,
                'confidence': round(min(max(confidence, 0), 100), 1),
                'message': message,
                'predicted_change': round(normalized_change * 100, 2)
            }
        
        def generate_portfolio_recommendation(self, signals, portfolio_value=10000):
            if not signals:
                return {
                    'overall_action': 'HOLD',
                    'total_confidence': 50,
                    'portfolio_value': portfolio_value,
                    'allocations': []
                }
            
            # Calculer la moyenne pondérée des signaux
            buy_weight = 0
            sell_weight = 0
            hold_weight = 0
            total_weight = 0
            
            for signal in signals:
                weight = signal['confidence'] / 100
                if signal['action'] == 'BUY':
                    buy_weight += weight
                elif signal['action'] == 'SELL':
                    sell_weight += weight
                else:
                    hold_weight += weight
                total_weight += weight
            
            if total_weight > 0:
                buy_weight /= total_weight
                sell_weight /= total_weight
                hold_weight /= total_weight
            
            # Déterminer l'action globale
            if buy_weight > sell_weight and buy_weight > hold_weight:
                overall_action = 'BULLISH'
            elif sell_weight > buy_weight and sell_weight > hold_weight:
                overall_action = 'BEARISH'
            else:
                overall_action = 'NEUTRAL'
            
            # Calculer la confiance moyenne
            total_confidence = np.mean([s['confidence'] for s in signals]) if signals else 50
            
            # Générer les allocations
            allocations = []
            if signals:
                for signal in signals:
                    if signal['action'] == 'BUY':
                        allocation_percent = min(40, signal['confidence'] / 2.5)
                        allocation_type = 'BUY'
                    elif signal['action'] == 'SELL':
                        allocation_percent = min(30, signal['confidence'] / 3.3)
                        allocation_type = 'SELL'
                    else:
                        allocation_percent = min(30, signal['confidence'] / 3.3)
                        allocation_type = 'HOLD'
                    
                    allocations.append({
                        'action': allocation_type,
                        'allocation_percent': round(allocation_percent, 1),
                        'allocation_amount': round(portfolio_value * allocation_percent / 100, 2),
                        'confidence': signal['confidence'],
                        'predicted_change': signal['predicted_change']
                    })
            
            return {
                'overall_action': overall_action,
                'total_confidence': round(total_confidence, 1),
                'portfolio_value': portfolio_value,
                'allocations': allocations
            }

class Predictor:
    def __init__(self):
        """Initialise le prédicteur avec les modèles chargés"""
        self.models = {}
        self.scaler = None
        self.recommendation_engine = RecommendationEngine()
        
    def load_models(self):
        """Charge tous les modèles et artefacts nécessaires"""
        print("📂 Chargement des modèles...")
        
        try:
            # Chargement Prophet
            with open('models/prophet_model.json', 'r') as fin:
                self.models['prophet'] = model_from_json(fin.read())
            print("✅ Modèle Prophet chargé")
        except FileNotFoundError:
            print("⚠️  Modèle Prophet non trouvé")
            self.models['prophet'] = None
        
        try:
            # Chargement LSTM - ESSAYER LE FORMAT .keras D'ABORD
            lstm_path = 'models/lstm_best_model.keras'
            if os.path.exists(lstm_path):
                self.models['lstm'] = tf.keras.models.load_model(lstm_path)
                print("✅ Modèle LSTM chargé (meilleur modèle)")
            else:
                # Essayer le modèle final
                lstm_path = 'models/lstm_model.keras'
                if os.path.exists(lstm_path):
                    self.models['lstm'] = tf.keras.models.load_model(lstm_path)
                    print("✅ Modèle LSTM chargé (modèle final)")
                else:
                    # Essayer le format H5 comme fallback
                    lstm_path = 'models/lstm_model.h5'
                    self.models['lstm'] = tf.keras.models.load_model(lstm_path)
                    print("✅ Modèle LSTM chargé (format H5)")
        except FileNotFoundError:
            print("⚠️  Modèle LSTM non trouvé")
            self.models['lstm'] = None
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement du LSTM: {e}")
            print("   Tentative de chargement du modèle LSTM avec custom_objects...")
            try:
                # Tentative avec des objets personnalisés
                self.models['lstm'] = tf.keras.models.load_model(
                    'models/lstm_model.keras',
                    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
                )
                print("✅ Modèle LSTM chargé avec custom_objects")
            except Exception as e2:
                print(f"❌ Impossible de charger le modèle LSTM: {e2}")
                self.models['lstm'] = None
        
        try:
            # Chargement du scaler - CORRECTION: vérifier s'il existe et s'il a le bon nombre de features
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
                print(f"✅ Scaler chargé (attend {self.scaler.n_features_in_} features)")
            else:
                print("⚠️  Scaler non trouvé")
                self.scaler = None
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement du scaler: {e}")
            self.scaler = None
        
        try:
            # Chargement des métadonnées
            with open('data/processed/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            print("✅ Métadonnées chargées")
        except FileNotFoundError:
            print("⚠️  Métadonnées non trouvées")
            self.metadata = {}
    
    def prepare_recent_data(self, n_days=30):
        """
        Prépare les données récentes pour la prédiction
        """
        print(f"\n📊 Préparation des {n_days} derniers jours...")
        
        try:
            # Chemin vers les données - plusieurs options
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'prj_finance', 'data_pipeline', 'stocks_raw.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'prj_finance', 'data_pipeline', 'stocks_data.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'final_dataset_ready_for_ai.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'stocks_data.csv'),
                'stocks_raw.csv',
                'data/stocks_raw.csv'
            ]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"  Trouvé: {path}")
                    try:
                        df = pd.read_csv(path)
                        print(f"  Données chargées: {df.shape}")
                        break
                    except Exception as e:
                        print(f"  ❌ Erreur de lecture pour {path}: {e}")
                        continue
            
            if df is None:
                print("❌ Aucun fichier de données trouvé")
                return None
            
            # Afficher les premières lignes pour debug
            print(f"\n  Aperçu des données (5 premières lignes):")
            print(df.head())
            print(f"\n  Types de données:")
            print(df.dtypes)
            
            # Nettoyer les données - supprimer la première ligne si elle contient des en-têtes
            if df['Date'].iloc[0] == 'AAPL' or pd.isna(df['Date'].iloc[0]):
                print(f"  ⚠️  Première ligne semble être un en-tête, suppression...")
                df = df.iloc[1:].reset_index(drop=True)
                print(f"  Données après suppression de la première ligne: {df.shape}")
            
            # Convertir la date si elle existe
            date_column = 'Date'
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                df = df.sort_values(date_column)
                print(f"  Date column: {date_column}")
                
                # Supprimer les lignes où Date est NaT
                initial_rows = len(df)
                df = df.dropna(subset=[date_column])
                print(f"  Lignes après suppression des dates invalides: {len(df)}/{initial_rows}")
            else:
                print("⚠️  Aucune colonne 'Date' trouvée")
                # Créer une date factice basée sur l'index
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                date_column = 'Date'
            
            # Convertir les colonnes numériques
            numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    # Convertir en numérique, forcer les erreurs à NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  Converti {col} en numérique")
            
            # Filtrer pour AAPL seulement si la colonne existe
            if 'Symbol' in df.columns:
                print(f"  Symboles disponibles: {df['Symbol'].unique()[:10]}")  # Afficher seulement les 10 premiers
                
                # Vérifier s'il y a des valeurs non-AAPL
                unique_symbols = df['Symbol'].dropna().unique()
                if 'AAPL' in unique_symbols:
                    df = df[df['Symbol'] == 'AAPL']
                    print(f"  Filtrage pour le symbole: AAPL")
                    print(f"  Lignes après filtrage AAPL: {len(df)}")
                elif len(unique_symbols) > 0:
                    print(f"  ⚠️  AAPL non trouvé, utilisation du premier symbole: {unique_symbols[0]}")
                    df = df[df['Symbol'] == unique_symbols[0]]
                else:
                    print(f"  ⚠️  Aucun symbole valide trouvé, utilisation de toutes les données")
            else:
                print("  ⚠️  Colonne 'Symbol' non trouvée, utilisation de toutes les données")
            
            # Supprimer les NaN seulement dans les colonnes numériques importantes
            initial_rows = len(df)
            required_columns = ['Close', 'Date']
            columns_to_check = [col for col in required_columns if col in df.columns]
            
            if columns_to_check:
                df = df.dropna(subset=columns_to_check)
                print(f"  Lignes après suppression NaN dans {columns_to_check}: {len(df)}/{initial_rows}")
            else:
                print(f"  ⚠️  Aucune colonne requise trouvée")
            
            # Supprimer les doublons de date
            df = df.drop_duplicates(subset=[date_column])
            print(f"  Lignes après suppression doublons: {len(df)}")
            
            # Vérifier qu'on a des données
            if len(df) == 0:
                print("❌ Données vides après nettoyage")
                return None
            
            # Récupération des n derniers jours (ou toutes si moins)
            n_days_actual = min(n_days, len(df))
            recent_data = df.tail(n_days_actual).copy()
            print(f"  Utilisation des {n_days_actual} derniers jours (demandé: {n_days})")
            
            # Préparation des features - avec fallback
            default_features = ['Close', 'Volume', 'Open', 'High', 'Low']
            if 'features' in self.metadata:
                features = self.metadata.get('features')
                print(f"  Features des métadonnées: {features}")
            else:
                features = default_features
                print(f"  Features par défaut: {features}")
            
            # Trouver les features disponibles
            available_features = []
            for f in features:
                if f in recent_data.columns:
                    available_features.append(f)
                else:
                    print(f"  ⚠️  Feature '{f}' non trouvée dans les données")
            
            # S'assurer qu'on a au moins les features de base
            if len(available_features) < 2:
                print("  ⚠️  Pas assez de features, recherche de colonnes numériques...")
                # Ajouter d'autres colonnes numériques si disponibles
                numeric_cols = recent_data.select_dtypes(include=[np.number]).columns.tolist()
                print(f"  Colonnes numériques disponibles: {numeric_cols}")
                
                # Priorité à 'Close' et 'Volume'
                priority_features = ['Close', 'close', 'CLOSE', 'price', 'Price', 'PRICE']
                volume_features = ['Volume', 'volume', 'VOLUME', 'vol', 'Vol']
                
                for f in priority_features:
                    if f in numeric_cols and f not in available_features:
                        available_features.append(f)
                        print(f"  Ajout feature prioritaire: {f}")
                        break
                
                for f in volume_features:
                    if f in numeric_cols and f not in available_features:
                        available_features.append(f)
                        print(f"  Ajout volume: {f}")
                        break
                
                # Ajouter d'autres colonnes jusqu'à 4 features
                for col in numeric_cols:
                    if col not in available_features and col != date_column:
                        available_features.append(col)
                        if len(available_features) >= 4:  # Limiter à 4 features
                            break
            
            if len(available_features) == 0:
                print("❌ Aucune feature numérique trouvée")
                return None
            
            # CORRECTION: Pour LSTM, nous n'avons besoin que de la colonne 'Close' pour prédire
            lstm_features = ['Close']  # On ne prédit que le prix de clôture
            
            print(f"  Features utilisées pour Prophet: {available_features}")
            print(f"  Features utilisées pour LSTM: {lstm_features}")
            
            # Extraction des données
            X_recent = recent_data[available_features].values
            X_recent_lstm = recent_data[lstm_features].values
            
            # Vérifier que X_recent n'est pas vide et ne contient pas de NaN
            if len(X_recent) == 0:
                print("❌ Données X_recent vides")
                return None
            
            # Remplacer les NaN restants par la moyenne de la colonne
            if np.isnan(X_recent).any():
                print(f"  ⚠️  NaN détectés dans les données, remplacement par moyenne...")
                for i in range(X_recent.shape[1]):
                    col_mean = np.nanmean(X_recent[:, i])
                    nan_mask = np.isnan(X_recent[:, i])
                    X_recent[nan_mask, i] = col_mean
            
            if np.isnan(X_recent_lstm).any():
                print(f"  ⚠️  NaN détectés dans les données LSTM, remplacement par moyenne...")
                for i in range(X_recent_lstm.shape[1]):
                    col_mean = np.nanmean(X_recent_lstm[:, i])
                    nan_mask = np.isnan(X_recent_lstm[:, i])
                    X_recent_lstm[nan_mask, i] = col_mean
            
            print(f"  Données récentes shape: {X_recent.shape}")
            print(f"  Données LSTM shape: {X_recent_lstm.shape}")
            
            # Obtenir le prix de clôture pour l'affichage
            close_price = None
            close_columns = ['Close', 'close', 'CLOSE', 'Price', 'price', 'PRICE']
            for col in close_columns:
                if col in recent_data.columns:
                    close_price = recent_data[col].iloc[-1]
                    break
            
            if close_price is not None:
                print(f"  Dernière date: {recent_data[date_column].iloc[-1].strftime('%Y-%m-%d')}")
                print(f"  Dernier prix: ${close_price:.2f}")
            else:
                # Utiliser la première colonne numérique
                if len(available_features) > 0:
                    close_price = recent_data[available_features[0]].iloc[-1]
                    print(f"  Dernière date: {recent_data[date_column].iloc[-1].strftime('%Y-%m-%d')}")
                    print(f"  Dernier prix (colonne {available_features[0]}): ${close_price:.2f}")
                else:
                    print(f"  Dernière date: {recent_data[date_column].iloc[-1].strftime('%Y-%m-%d')}")
                    print(f"  Dernier prix: N/A")
            
            return {
                'X_raw': X_recent,  # Pour Prophet
                'X_raw_lstm': X_recent_lstm,  # Pour LSTM (juste Close)
                'dates': recent_data[date_column].values,
                'prices': recent_data[lstm_features[0]].values if len(lstm_features) > 0 else np.zeros(len(recent_data)),
                'features': available_features,
                'lstm_features': lstm_features,
                'full_data': recent_data
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la préparation des données: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_prophet(self, recent_data, days_ahead=7):
        """Prédit avec le modèle Prophet"""
        if self.models.get('prophet') is None:
            print("⚠️  Modèle Prophet non disponible")
            return None
        
        print(f"\n🔮 Prédiction Prophet pour les {days_ahead} prochains jours...")
        
        try:
            model = self.models['prophet']
            
            # Dernière date des données
            last_date = pd.Timestamp(recent_data['dates'][-1])
            print(f"  Dernière date connue: {last_date.strftime('%Y-%m-%d')}")
            
            # Vérifier si le modèle a été entraîné avec des données récentes
            # Si les prédictions sont dans le passé, ajuster les dates
            
            # Création du dataframe futur
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Extraire seulement les prédictions futures (après la dernière date connue)
            future_mask = forecast['ds'] > last_date
            future_predictions = forecast[future_mask].head(days_ahead)
            
            # Si pas assez de prédictions futures, prendre les dernières disponibles
            if len(future_predictions) < days_ahead:
                print(f"  ⚠️  Pas assez de prédictions futures, utilisation des dernières prédictions")
                future_predictions = forecast.tail(days_ahead)
            
            print(f"  Première date prédite: {future_predictions['ds'].iloc[0].strftime('%Y-%m-%d')}")
            print(f"  Dernière date prédite: {future_predictions['ds'].iloc[-1].strftime('%Y-%m-%d')}")
            
            # Formatage des résultats
            predictions = []
            for _, row in future_predictions.iterrows():
                # CORRECTION: S'assurer que le prix est réaliste
                predicted_price = float(row['yhat'])
                
                # Vérifier si le prix est réaliste (entre 50% et 200% du prix actuel)
                current_price = recent_data['prices'][-1] if len(recent_data['prices']) > 0 else 250
                realistic_min = current_price * 0.5
                realistic_max = current_price * 2.0
                
                if predicted_price < realistic_min or predicted_price > realistic_max:
                    print(f"  ⚠️  Prix Prophet irréaliste: ${predicted_price:.2f}, ajustement...")
                    # Ajuster vers un prix plus réaliste (moyenne mobile)
                    predicted_price = current_price * 1.02  # Légère hausse de 2%
                
                predicted_price = max(10, predicted_price)  # Minimum 10$
                
                lower_bound = max(10, float(row['yhat_lower']))
                upper_bound = max(10, float(row['yhat_upper']))
                
                # Ajuster la confiance basée sur l'intervalle
                confidence = min(95, max(60, 100 * (1 - (upper_bound - lower_bound) / predicted_price)))
                
                predictions.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'predicted_price': round(predicted_price, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'confidence_interval': f"[{round(lower_bound, 2)}, {round(upper_bound, 2)}]",
                    'model': 'prophet',
                    'confidence': round(confidence, 1)
                })
            
            print(f"✅ {len(predictions)} prédictions générées")
            
            # Afficher un aperçu
            for i, pred in enumerate(predictions[:3]):
                print(f"    Jour {i+1}: {pred['date']} - ${pred['predicted_price']:.2f} (conf: {pred['confidence']}%)")
            if len(predictions) > 3:
                print(f"    ... et {len(predictions)-3} autres jours")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction Prophet: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_lstm(self, recent_data, days_ahead=7):
        """Prédit avec le modèle LSTM"""
        if self.models.get('lstm') is None or self.scaler is None:
            print("⚠️  Modèle LSTM ou scaler non disponible")
            return None
        
        print(f"\n🔮 Prédiction LSTM pour les {days_ahead} prochains jours...")
        
        try:
            model = self.models['lstm']
            
            # Vérifier le nombre de features attendu par le scaler
            n_features_expected = getattr(self.scaler, 'n_features_in_', 1)
            print(f"  Scaler attend {n_features_expected} features")
            print(f"  Nous avons {recent_data['X_raw_lstm'].shape[1]} features pour LSTM")
            
            # CORRECTION: Vérifier si le scaler est compatible
            if n_features_expected != recent_data['X_raw_lstm'].shape[1]:
                print(f"  ⚠️  Incompatibilité de features, création d'un nouveau scaler...")
                # Créer un nouveau scaler basé sur les données récentes
                lstm_scaler = MinMaxScaler()
                lstm_scaler.fit(recent_data['X_raw_lstm'])
                print(f"  ✅ Nouveau scaler créé pour LSTM")
            else:
                # Utiliser le scaler existant
                lstm_scaler = self.scaler
            
            X_for_scaling = recent_data['X_raw_lstm']
            
            # Préparation de la dernière séquence
            sequence_length = self.metadata.get('sequence_length', 30)
            print(f"  Sequence length utilisée: {sequence_length}")
            
            # Vérifier si nous avons assez de données
            if len(X_for_scaling) < sequence_length:
                print(f"  ⚠️  Pas assez de données: {len(X_for_scaling)} < {sequence_length}")
                # Utiliser toutes les données disponibles
                sequence_length = min(sequence_length, len(X_for_scaling))
                print(f"  Nouvelle sequence length: {sequence_length}")
            
            if sequence_length < 1:
                print("❌ Pas assez de données pour créer une séquence")
                return None
            
            # Appliquer le scaling
            X_recent_scaled = lstm_scaler.transform(X_for_scaling)
            
            # Création de la séquence - reshape pour LSTM
            last_sequence = X_recent_scaled[-sequence_length:]
            # Reshape pour LSTM: (samples, time_steps, features)
            n_features_lstm = recent_data['X_raw_lstm'].shape[1]
            last_sequence_reshaped = last_sequence.reshape(1, sequence_length, n_features_lstm)
            
            print(f"  Dernière séquence shape: {last_sequence_reshaped.shape}")
            
            # Vérifier la compatibilité des dimensions
            expected_input_shape = model.input_shape
            print(f"  Shape d'entrée attendue par le modèle: {expected_input_shape}")
            
            # Prédictions itératives
            predictions = []
            current_sequence = last_sequence.copy()
            
            for day in range(days_ahead):
                try:
                    # Reshape pour la prédiction
                    current_sequence_reshaped = current_sequence.reshape(1, sequence_length, n_features_lstm)
                    
                    # Prédiction pour le prochain jour
                    pred_scaled = model.predict(current_sequence_reshaped, verbose=0)
                    
                    # Inverse scaling
                    pred_price = lstm_scaler.inverse_transform(pred_scaled)[0, 0]
                    
                    # CORRECTION: Vérifier que le prix est réaliste
                    current_price = recent_data['prices'][-1] if len(recent_data['prices']) > 0 else 250
                    realistic_min = current_price * 0.7  # -30% maximum
                    realistic_max = current_price * 1.3  # +30% maximum
                    
                    if pred_price < realistic_min or pred_price > realistic_max:
                        print(f"  ⚠️  Prix LSTM irréaliste: ${pred_price:.2f}, ajustement...")
                        # Ajuster vers un prix plus réaliste
                        # Petite tendance basée sur les derniers mouvements
                        recent_trend = np.mean(np.diff(recent_data['prices'][-5:])) if len(recent_data['prices']) > 5 else 0
                        pred_price = current_price + recent_trend * 0.5
                        pred_price = max(realistic_min, min(realistic_max, pred_price))
                    
                    pred_price = max(10, pred_price)  # Minimum 10$
                    
                    # Calcul de la date
                    if len(recent_data['dates']) > 0:
                        last_date = pd.Timestamp(recent_data['dates'][-1])
                    else:
                        # Date par défaut si pas de dates
                        last_date = pd.Timestamp.now()
                    
                    prediction_date = (last_date + timedelta(days=day+1)).strftime('%Y-%m-%d')
                    
                    # Calcul de la confiance (diminue avec l'horizon)
                    confidence = max(95 - (day * 8), 60)
                    
                    # Mise à jour de la séquence pour la prédiction suivante
                    new_value = pred_scaled[0, 0]
                    new_sequence = np.vstack([current_sequence[1:], [[new_value]]])
                    current_sequence = new_sequence
                    
                    predictions.append({
                        'date': prediction_date,
                        'predicted_price': round(float(pred_price), 2),
                        'confidence': confidence,
                        'model': 'lstm'
                    })
                    
                    print(f"    Jour {day+1}: {prediction_date} - ${pred_price:.2f} (conf: {confidence}%)")
                    
                except Exception as e:
                    print(f"    ❌ Erreur pour le jour {day+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            if predictions:
                print(f"✅ {len(predictions)} prédictions générées")
                return predictions
            else:
                print("❌ Aucune prédiction LSTM générée")
                return None
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction LSTM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_recommendations(self, recent_data, predictions):
        """Génère des recommandations basées sur les prédictions"""
        print("\n🎯 Génération des recommandations...")
        
        if not predictions or len(predictions) == 0:
            print("⚠️  Aucune prédiction disponible pour générer des recommandations")
            return None
        
        # Prix actuel - utiliser la première colonne de prix
        if len(recent_data['prices']) > 0:
            current_price = recent_data['prices'][-1]
        else:
            # Si pas de prix, utiliser la dernière valeur de X_raw
            if len(recent_data['X_raw']) > 0:
                current_price = recent_data['X_raw'][-1, 0]
            else:
                current_price = 0
        
        print(f"  Prix actuel: ${current_price:.2f}")
        
        # Regroupement des prédictions par modèle
        predictions_by_model = {}
        for pred in predictions:
            model = pred['model']
            if model not in predictions_by_model:
                predictions_by_model[model] = []
            predictions_by_model[model].append(pred)
        
        print(f"  Prédictions par modèle: {list(predictions_by_model.keys())}")
        
        # Génération des signaux pour chaque modèle
        signals = []
        
        for model_name, model_predictions in predictions_by_model.items():
            # Utilisation de la première prédiction pour générer le signal
            if len(model_predictions) > 0:
                first_prediction = model_predictions[0]
                predicted_price = first_prediction['predicted_price']
                
                # Calcul du changement (limité à ±30% pour éviter des valeurs extrêmes)
                price_change = (predicted_price - current_price) / current_price if current_price > 0 else 0
                price_change = min(max(price_change, -0.3), 0.3)  # Limiter à ±30%
                
                # Confiance du modèle
                confidence = first_prediction.get('confidence', 80)
                
                # Génération du signal
                signal = self.recommendation_engine.generate_signal(price_change, confidence)
                signal['model'] = model_name
                signal['current_price'] = round(current_price, 2)
                signal['predicted_price'] = predicted_price
                signal['actual_change'] = round(((predicted_price - current_price) / current_price * 100), 2)
                
                signals.append(signal)
                
                print(f"  Signal {model_name}: {signal['action']} (changement: {signal['actual_change']}%)")
        
        # Recommandation de portefeuille
        portfolio_recommendation = self.recommendation_engine.generate_portfolio_recommendation(
            signals, portfolio_value=10000
        )
        
        # Prédictions consolidées
        consolidated_predictions = self._consolidate_predictions(predictions)
        
        return {
            'signals': signals,
            'portfolio_recommendation': portfolio_recommendation,
            'consolidated_predictions': consolidated_predictions,
            'current_price': current_price,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _consolidate_predictions(self, predictions):
        """Consolide les prédictions de différents modèles"""
        # Groupement par date
        predictions_by_date = {}
        
        for pred in predictions:
            date = pred['date']
            if date not in predictions_by_date:
                predictions_by_date[date] = []
            predictions_by_date[date].append(pred)
        
        # Consolidation
        consolidated = []
        
        for date, date_predictions in predictions_by_date.items():
            # Calcul de la moyenne des prédictions
            prices = [p['predicted_price'] for p in date_predictions]
            avg_price = np.mean(prices)
            
            # Calcul de la confiance moyenne
            confidences = [p.get('confidence', 80) for p in date_predictions]
            avg_confidence = np.mean(confidences)
            
            # Calcul de l'intervalle de confiance
            if len(prices) > 1:
                std_price = np.std(prices)
                lower_bound = max(10, avg_price - 1.96 * std_price)  # Minimum 10$
                upper_bound = avg_price + 1.96 * std_price
            else:
                lower_bound = max(10, avg_price * 0.9)  # Minimum 10$
                upper_bound = avg_price * 1.1
            
            # Trier les prédictions par confiance
            date_predictions_sorted = sorted(date_predictions, key=lambda x: x.get('confidence', 0), reverse=True)
            
            consolidated.append({
                'date': date,
                'predicted_price': round(avg_price, 2),
                'confidence': round(avg_confidence, 1),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'models_used': [p['model'] for p in date_predictions_sorted],
                'individual_predictions': date_predictions_sorted
            })
        
        return consolidated
    
    def save_results(self, recent_data, predictions, recommendations):
        """Sauvegarde tous les résultats"""
        print("\n💾 Sauvegarde des résultats...")
        
        os.makedirs('outputs', exist_ok=True)
        
        # 1. Données récentes
        try:
            recent_data_df = pd.DataFrame({
                'Date': recent_data['dates'],
                'Price': recent_data['prices']
            })
            recent_data_df.to_csv('outputs/recent_data.csv', index=False, encoding='utf-8')
            print("✅ Données récentes sauvegardées: outputs/recent_data.csv")
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde des données récentes: {e}")
        
        # 2. Prédictions
        if predictions:
            try:
                predictions_df = pd.DataFrame(predictions)
                predictions_df.to_csv('outputs/all_predictions.csv', index=False, encoding='utf-8')
                print("✅ Toutes les prédictions sauvegardées: outputs/all_predictions.csv")
            except Exception as e:
                print(f"⚠️  Erreur lors de la sauvegarde des prédictions: {e}")
        
        # 3. Prédictions consolidées
        if recommendations and 'consolidated_predictions' in recommendations:
            try:
                consolidated_df = pd.DataFrame(recommendations['consolidated_predictions'])
                consolidated_df.to_csv('outputs/consolidated_predictions.csv', index=False, encoding='utf-8')
                print("✅ Prédictions consolidées sauvegardées: outputs/consolidated_predictions.csv")
            except Exception as e:
                print(f"⚠️  Erreur lors de la sauvegarde des prédictions consolidées: {e}")
        
        # 4. Recommandations complètes
        if recommendations:
            try:
                with open('outputs/final_recommendations.json', 'w', encoding='utf-8') as f:
                    json.dump(recommendations, f, indent=2, default=self._json_serializer)
                print("✅ Recommandations finales sauvegardées: outputs/final_recommendations.json")
            except Exception as e:
                print(f"⚠️  Erreur lors de la sauvegarde des recommandations: {e}")
        
        # 5. Rapport exécutif
        self._generate_executive_report(recommendations)
    
    def _generate_executive_report(self, recommendations):
        """Génère un rapport exécutif"""
        if not recommendations:
            return
        
        try:
            report_lines = [
                "="*60,
                "RAPPORT DE PRÉDICTION - FINSENSE AI",
                f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "="*60,
                "",
                "RÉSUMÉ EXÉCUTIF",
                ""
            ]
            
            # Prix actuel
            current_price = recommendations.get('current_price', 0)
            report_lines.append(f"Prix actuel: ${current_price:,.2f}")
            
            # Signaux
            signals = recommendations.get('signals', [])
            if signals:
                report_lines.append(f"\nSIGNALS DE TRADING:")
                for signal in signals:
                    action_symbol = "[ACHAT]" if signal['action'] == 'BUY' else "[VENTE]" if signal['action'] == 'SELL' else "[MAINTIEN]"
                    report_lines.append(f"  • {signal['model'].upper()}: {action_symbol} "
                                  f"(Confiance: {signal['confidence']}%)")
                    report_lines.append(f"    {signal['message']}")
                    report_lines.append(f"    Prix prédit: ${signal['predicted_price']:,.2f} "
                                  f"(Changement: {signal.get('actual_change', signal['predicted_change'])}%)")
            
            # Prédictions consolidées
            consolidated = recommendations.get('consolidated_predictions', [])
            if consolidated:
                report_lines.append(f"\nPRÉDICTIONS POUR LES 7 PROCHAINS JOURS:")
                for pred in consolidated[:7]:  # Limité aux 7 premiers jours
                    report_lines.append(f"  • {pred['date']}: ${pred['predicted_price']:,.2f} "
                                  f"(Confiance: {pred['confidence']}%)")
                    report_lines.append(f"    Intervalle: [${pred['lower_bound']:,.2f}, ${pred['upper_bound']:,.2f}]")
                    report_lines.append(f"    Modèles utilisés: {', '.join(pred['models_used'])}")
            
            # Recommandation de portefeuille
            portfolio = recommendations.get('portfolio_recommendation', {})
            if portfolio:
                report_lines.append(f"\nRECOMMANDATION DE PORTEFEUILLE:")
                report_lines.append(f"  Tendance générale: {portfolio.get('overall_action', 'N/A')}")
                report_lines.append(f"  Confiance totale: {portfolio.get('total_confidence', 0)}%")
                
                allocations = portfolio.get('allocations', [])
                if allocations:
                    report_lines.append(f"  Allocations recommandées:")
                    for alloc in allocations:
                        action_text = "ACHAT" if alloc['action'] == 'BUY' else "VENTE" if alloc['action'] == 'SELL' else "MAINTIEN"
                        report_lines.append(f"    • {action_text}: {alloc['allocation_percent']}% "
                                      f"(${alloc['allocation_amount']:,.2f})")
                        report_lines.append(f"      Confiance: {alloc['confidence']}%, "
                                      f"Prédiction: {alloc['predicted_change']}%")
            
            report_lines.extend([
                "",
                "="*60,
                "DISCLAIMER:",
                "Ces prédictions sont basées sur des modèles d'IA et des données historiques.",
                "Elles ne constituent pas un conseil financier. Investissez avec prudence.",
                "="*60
            ])
            
            # Écrire avec encodage UTF-8
            with open('outputs/executive_report.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            print("✅ Rapport exécutif généré: outputs/executive_report.txt")
        except Exception as e:
            print(f"⚠️  Erreur lors de la génération du rapport exécutif: {e}")
    
    def _json_serializer(self, obj):
        """Helper pour sérialiser des objets non-standard en JSON"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return str(obj)
    
    def display_results(self, recommendations):
        """Affiche les résultats de manière lisible"""
        if not recommendations:
            print("❌ Aucune recommandation disponible")
            return
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE PRÉDICTION")
        print("="*60)
        
        # Prix actuel
        current_price = recommendations.get('current_price', 0)
        print(f"\n💰 Prix actuel: ${current_price:,.2f}")
        
        # Signaux
        signals = recommendations.get('signals', [])
        if signals:
            print(f"\n🎯 SIGNALS DE TRADING:")
            print("-" * 60)
            for signal in signals:
                action_icon = "🟢" if signal['action'] == 'BUY' else "🔴" if signal['action'] == 'SELL' else "🟡"
                print(f"{action_icon} {signal['model'].upper()}:")
                print(f"  Action: {signal['action']}")
                print(f"  Confiance: {signal['confidence']}%")
                print(f"  Message: {signal['message']}")
                print(f"  Prix actuel: ${signal['current_price']:,.2f}")
                print(f"  Prix prédit: ${signal['predicted_price']:,.2f}")
                actual_change = signal.get('actual_change', signal['predicted_change'])
                change_icon = "📈" if actual_change > 0 else "📉" if actual_change < 0 else "➡️"
                print(f"  Changement prédit: {change_icon} {actual_change}%")
                print()
        
        # Prédictions consolidées
        consolidated = recommendations.get('consolidated_predictions', [])
        if consolidated:
            print(f"\n🔮 PRÉDICTIONS POUR LES 7 PROCHAINS JOURS:")
            print("-" * 70)
            print(f"{'Date':<12} {'Prix prédit':<15} {'Confiance':<12} {'Intervalle':<25} {'Modèles'}")
            print("-" * 70)
            
            for pred in consolidated[:7]:
                interval = f"[${pred['lower_bound']:,.0f}, ${pred['upper_bound']:,.0f}]"
                models_str = ', '.join(pred['models_used'][:2])
                if len(pred['models_used']) > 2:
                    models_str += f" (+{len(pred['models_used'])-2})"
                
                # Indicateur de tendance
                trend_icon = "📈" if pred['predicted_price'] > current_price else "📉" if pred['predicted_price'] < current_price else "➡️"
                
                print(f"{pred['date']:<12} {trend_icon} ${pred['predicted_price']:<13,.0f} "
                      f"{pred['confidence']:<11}% {interval:<25} {models_str}")
        
        # Recommandation de portefeuille
        portfolio = recommendations.get('portfolio_recommendation', {})
        if portfolio:
            print(f"\n💰 RECOMMANDATION DE PORTEFEUILLE:")
            print("-" * 60)
            print(f"Valeur du portefeuille: ${portfolio.get('portfolio_value', 0):,.2f}")
            
            overall_action = portfolio.get('overall_action', '')
            if overall_action == 'BULLISH':
                trend_icon = "📈"
            elif overall_action == 'BEARISH':
                trend_icon = "📉"
            else:
                trend_icon = "➡️"
            
            print(f"Tendance générale: {trend_icon} {overall_action}")
            print(f"Confiance totale: {portfolio.get('total_confidence', 0)}%")
            
            allocations = portfolio.get('allocations', [])
            if allocations:
                print(f"\nAllocations recommandées:")
                for alloc in allocations:
                    action_icon = "🟢" if alloc['action'] == 'BUY' else "🔴" if alloc['action'] == 'SELL' else "🟡"
                    print(f"{action_icon} {alloc['action']}:")
                    print(f"  Allocation: {alloc['allocation_percent']}% "
                          f"(${alloc['allocation_amount']:,.2f})")
                    print(f"  Confiance: {alloc['confidence']}%")
                    change_icon = "📈" if alloc['predicted_change'] > 0 else "📉" if alloc['predicted_change'] < 0 else "➡️"
                    print(f"  Prédiction: {change_icon} {alloc['predicted_change']}% de changement")
        
        print("\n" + "="*60)
        print("✅ PRÉDICTION TERMINÉE")
        print("="*60)

def main():
    """Pipeline principal de prédiction"""
    
    print("="*60)
    print("🤖 SYSTÈME DE PRÉDICTION FINSENSE AI")
    print("="*60)
    
    # Initialisation
    predictor = Predictor()
    
    # 1. Chargement des modèles
    predictor.load_models()
    
    # Vérifier si au moins un modèle est chargé
    if predictor.models.get('prophet') is None and predictor.models.get('lstm') is None:
        print("❌ Aucun modèle n'a pu être chargé")
        print("Veuillez d'abord entraîner les modèles avec:")
        print("python modeles/prophet_model.py")
        print("python modeles/lstm_model.py")
        return None, None
    
    # 2. Préparation des données récentes
    recent_data = predictor.prepare_recent_data(n_days=60)
    
    if recent_data is None:
        print("❌ Impossible de préparer les données récentes")
        return None, None
    
    # 3. Prédictions avec différents modèles
    all_predictions = []
    
    # Prophet
    prophet_predictions = predictor.predict_with_prophet(recent_data, days_ahead=7)
    if prophet_predictions:
        all_predictions.extend(prophet_predictions)
        print(f"  Prophet: {len(prophet_predictions)} prédictions")
    else:
        print("  Prophet: Aucune prédiction")
    
    # LSTM
    lstm_predictions = predictor.predict_with_lstm(recent_data, days_ahead=7)
    if lstm_predictions:
        all_predictions.extend(lstm_predictions)
        print(f"  LSTM: {len(lstm_predictions)} prédictions")
    else:
        print("  LSTM: Aucune prédiction")
    
    if not all_predictions:
        print("❌ Aucune prédiction générée")
        return predictor, None
    
    print(f"\n✅ Total des prédictions: {len(all_predictions)}")
    
    # 4. Génération des recommandations
    recommendations = predictor.generate_recommendations(recent_data, all_predictions)
    
    if recommendations:
        # 5. Sauvegarde des résultats
        predictor.save_results(recent_data, all_predictions, recommendations)
        
        # 6. Affichage des résultats
        predictor.display_results(recommendations)
    else:
        print("❌ Impossible de générer des recommandations")
    
    return predictor, recommendations

if __name__ == "__main__":
    predictor, recommendations = main()