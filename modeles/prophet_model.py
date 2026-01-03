"""
Modèle Prophet pour la prédiction de séries temporelles
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import os
from datetime import datetime, timedelta

def prepare_prophet_data():
    """Prépare les données pour Prophet"""
    print(" Préparation des données pour Prophet...")
    
    # Chargement des données
    df = pd.read_csv('data/processed/prophet_data.csv')
    
    # Vérifications minimales
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("Le fichier prophet_data.csv doit contenir les colonnes 'ds' et 'y'.")

    # Forcer le type date et numérique, supprimer lignes invalides
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Supprimer lignes avec date ou target manquante
    df = df.dropna(subset=['ds', 'y']).reset_index(drop=True)

    if df.empty:
        raise ValueError("Aucune ligne valide après nettoyage de prophet_data.csv (ds/y).")

    # Si plusieurs enregistrements par date (ex: plusieurs symbols), agréger par moyenne
    if df.duplicated(subset=['ds']).any():
        df = df.groupby('ds', as_index=False)['y'].mean()

    # Tri chronologique
    df = df.sort_values('ds').reset_index(drop=True)

    # Division train/test
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"  Train: {len(train_df)} échantillons")
    print(f"  Test: {len(test_df)} échantillons")
    
    return train_df, test_df

def train_prophet_model(train_df, periods=30):
    """
    Entraîne un modèle Prophet
    """
    print(" Entraînement du modèle Prophet...")
    
    # Initialisation du modèle
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    # Entraînement — n'utiliser que les colonnes attendues par Prophet
    model.fit(train_df[['ds', 'y']])
    
    # Création du dataframe pour les prédictions futures
    future = model.make_future_dataframe(periods=periods)
    
    # Prédiction
    forecast = model.predict(future)
    
    return model, forecast

def evaluate_prophet_model(model, train_df, test_df):
    """
    Évalue le modèle Prophet — aligne par date et calcule métriques en toute sécurité
    """
    print(" Évaluation du modèle Prophet...")
    
    # Prédiction sur la période de test (on demande autant de périodes que la taille du test)
    future = model.make_future_dataframe(periods=len(test_df))
    forecast_full = model.predict(future)
    
    # S'assurer que 'ds' est en datetime
    forecast_full['ds'] = pd.to_datetime(forecast_full['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])
    
    # Fusionner sur les dates pour être sûr d'aligner y_true et y_pred
    cols_needed = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    if not set(cols_needed).issubset(forecast_full.columns):
        raise RuntimeError("Prévision Prophet ne contient pas toutes les colonnes attendues.")
    
    merged = pd.merge(
        test_df[['ds', 'y']].rename(columns={'y': 'y_true'}),
        forecast_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='inner'
    ).sort_values('ds').reset_index(drop=True)
    
    if merged.empty:
        raise RuntimeError("Aucun enregistrement commun entre les dates de test et les prédictions.")
    
    y_true = merged['y_true'].values
    y_pred = merged['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE en évitant division par zéro
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('nan')
    
    print(f"\n📊 Métriques d'évaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'y_true': y_true,
        'y_pred': y_pred,
        'forecast': merged,        # dataframe aligné ds / y_true / yhat / bounds
        'forecast_full': forecast_full
    }

def plot_prophet_results(model, forecast, train_df, test_df, eval_results):
    """
    Visualise les résultats de Prophet
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prédictions vs réalité
    ax1 = axes[0, 0]
    ax1.plot(train_df['ds'], train_df['y'], 'b-', label='Train', alpha=0.7)
    ax1.plot(test_df['ds'], test_df['y'], 'g-', label='Test réel', alpha=0.7)
    ax1.plot(eval_results['forecast']['ds'], eval_results['y_pred'], 
             'r--', label='Prédictions', alpha=0.8)
    ax1.fill_between(eval_results['forecast']['ds'],
                     eval_results['forecast']['yhat_lower'],
                     eval_results['forecast']['yhat_upper'],
                     alpha=0.2, color='gray', label='Intervalle de confiance')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prix')
    ax1.set_title('Prédictions Prophet vs Réalité')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Composantes du modèle — rendre la figure Prophet et afficher comme image dans ax2
    ax2 = axes[0, 1]
    comp_fig = model.plot_components(forecast)
    try:
        canvas = FigureCanvas(comp_fig)
        canvas.draw()
        # obtenir dimensions et buffer RGBA puis convertir en RGB
        w, h = comp_fig.canvas.get_width_height()
        buf = canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)
        img = arr[..., :3]
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('Composantes de la prédiction Prophet')
    finally:
        plt.close(comp_fig)
    
    # 3. Erreurs de prédiction
    ax3 = axes[1, 0]
    errors = eval_results['y_true'] - eval_results['y_pred']
    ax3.plot(eval_results['forecast']['ds'], errors, 'o-', alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Erreur (réel - prédit)')
    ax3.set_title('Erreurs de prédiction')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution des erreurs
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Erreur')
    ax4.set_ylabel('Fréquence')
    ax4.set_title(f'Distribution des erreurs\nMAE: {eval_results["mae"]:.2f}, MAPE: {eval_results["mape"]:.2f}%')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/prophet_results.png', dpi=300, bbox_inches='tight')
    print(" Graphiques sauvegardés: outputs/prophet_results.png")
    
    return fig

def save_prophet_model(model, eval_results):
    """
    Sauvegarde le modèle Prophet et ses métriques
    """
    print(" Sauvegarde du modèle Prophet...")
    
    # s'assurer que le dossier models existe
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarde du modèle
    import json
    from prophet.serialize import model_to_json
    
    with open('models/prophet_model.json', 'w') as fout:
        fout.write(model_to_json(model))
    
    # Sauvegarde des métriques
    metrics = {
        'mae': float(eval_results['mae']),
        'rmse': float(eval_results['rmse']),
        'mape': float(eval_results['mape']),
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/prophet_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(" Modèle Prophet sauvegardé: models/prophet_model.json")
    print(" Métriques sauvegardées: models/prophet_metrics.json")
    
    return metrics

def predict_future(model, days_ahead=7):
    """
    Prédit les prix futurs
    """
    print(f"🔮 Prédiction des {days_ahead} prochains jours...")
    
    # Création du dataframe futur
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    # Extraction des prédictions futures
    future_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)
    
    # Conversion en format plus lisible
    predictions = []
    for _, row in future_predictions.iterrows():
        predictions.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'predicted_price': round(float(row['yhat']), 2),
            'lower_bound': round(float(row['yhat_lower']), 2),
            'upper_bound': round(float(row['yhat_upper']), 2),
            'confidence_interval': f"[{round(float(row['yhat_lower']), 2)}, {round(float(row['yhat_upper']), 2)}]"
        })
    
    # Sauvegarde des prédictions
    with open('outputs/prophet_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f" Prédictions sauvegardées: outputs/prophet_predictions.json")
    
    return predictions

def main():
    """Pipeline principal Prophet"""
    
    # Création des dossiers
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Préparation des données
    train_df, test_df = prepare_prophet_data()
    
    # 2. Entraînement du modèle
    model, forecast = train_prophet_model(train_df, periods=len(test_df))
    
    # 3. Évaluation
    eval_results = evaluate_prophet_model(model, train_df, test_df)
    
    # 4. Visualisation
    plot_prophet_results(model, forecast, train_df, test_df, eval_results)
    
    # 5. Sauvegarde
    metrics = save_prophet_model(model, eval_results)
    
    # 6. Prédiction future
    future_predictions = predict_future(model, days_ahead=7)
    
    print("\n" + "="*50)
    print(" RÉSUMÉ DU MODÈLE PROPHET")
    print("="*50)
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"\n Prédictions pour les 7 prochains jours:")
    for pred in future_predictions:
        print(f"  {pred['date']}: ${pred['predicted_price']} ({pred['confidence_interval']})")
    
    return model, metrics, future_predictions

if __name__ == "__main__":
    model, metrics, predictions = main()